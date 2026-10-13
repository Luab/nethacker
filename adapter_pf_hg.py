from __future__ import annotations

import contextlib
import ctypes
import queue
import threading
import time
import traceback
from collections.abc import Mapping
from typing import Any

import nle.nethack as nh
from pf_hg import agent as autoascend_agent
from pf_hg import jf_log

_ACTIONS = tuple(nh.ACTIONS)
_ACTION_TO_INDEX = {int(action): index for index, action in enumerate(_ACTIONS)}


class AgentHang(autoascend_agent.AgentPanic):
    """Injected into an agent thread that stopped producing actions (a livelock)."""


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copy = getattr(value, "copy", None)
        copied[key] = copy() if callable(copy) else value
    return copied


def _action_index(action: Any) -> int:
    try:
        return _ACTION_TO_INDEX[int(action)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"AutoAscend produced invalid action {action!r}") from error


def _raise_in_thread(thread: threading.Thread, exc_type: type) -> bool:
    ident = thread.ident
    if ident is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), ctypes.py_object(exc_type))
    if res > 1:  # affected more than one thread: undo
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
        return False
    return res == 1


class ArenaEnvAdapter:
    """Minimal env-like object for AutoAscend.

    AutoAscend's blocking env.step(action) call is translated into one action returned from
    Bot.act(...), then resumed when the arena supplies the next observation.
    """

    def __init__(self) -> None:
        self._actions: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._observations: queue.Queue[Mapping[str, Any] | None] = queue.Queue(maxsize=1)
        self._closed = threading.Event()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._closed.is_set():
            raise autoascend_agent.AgentFinished()
        self._actions.put(action)
        observation = self._observations.get()
        if observation is None or self._closed.is_set():
            raise autoascend_agent.AgentFinished()
        return _copy_observation(observation), 0.0, False, {}

    def next_action_index(self, timeout: float) -> int:
        action = self._actions.get(timeout=timeout)
        return _action_index(action)

    def provide_observation(self, observation: Mapping[str, Any]) -> bool:
        if self._closed.is_set():
            return False
        try:
            self._observations.put_nowait(observation)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        self._closed.set()
        with contextlib.suppress(queue.Full):
            self._observations.put_nowait(None)

    def debug_tiles(self, *args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def debug_log(self, *args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class AutoAscendDriver:
    """Runs AutoAscend in a thread and keeps it alive for the whole episode.

    - The first action after a (re)start may include a cold numba JIT compile, so it gets
      ``action_timeout`` (just under the arena's 120s hang guard).
    - Afterwards an action that takes longer than ``hang_timeout`` is treated as a livelock:
      an AgentHang panic is injected into the agent thread, which recovers like any panic.
    - If the thread still does not answer, or dies, a fresh agent takes over the same game.
    Every fallback returns ESC, which is always a legal, harmless action.
    """

    # `nethackers eval` kills the whole batch after 3600 s per wave: one runaway episode would void
    # every score. Past this budget the driver only sends ESC, so NLE's no-progress cutoff ends the
    # game within seconds and the progress reached so far is kept.
    WALL_BUDGET_SECONDS = 50 * 60

    # Timing: the arena kills a bot (and zeroes the episode) when one act() takes over 120 s.
    # Only the episode's first action may need a cold numba JIT compile (up to FIRST_ACTION_TIMEOUT);
    # afterwards the worst case is HANG + RECOVER + RESTART = 85 + 5 + 20 = 110 s. Compilation may occur after the initial ESC, so every action
    # needs compilation headroom. 110 seconds leaves ten seconds below the
    # arena deadline, including hang recovery and restart.
    FIRST_ACTION_TIMEOUT = 100.0
    RECOVER_TIMEOUT = 5.0
    RESTART_TIMEOUT = 20.0

    def __init__(self, action_timeout: float = 100.0, hang_timeout: float = 85.0) -> None:
        self._action_timeout = action_timeout
        self._hang_timeout = hang_timeout
        self._warm = False  # set once any agent in this process produced an action
        self._episode_start = time.monotonic()
        self._gave_up = False
        self._env: ArenaEnvAdapter | None = None
        self._agent: autoascend_agent.Agent | None = None
        self._thread: threading.Thread | None = None
        self._thread_error: str | None = None
        self._sent_first_action = False
        self._restarts = 0
        self._fallback_action = _ACTION_TO_INDEX[int(autoascend_agent.A.Command.ESC)]

    def reset(self, initial_observation: Mapping[str, Any]) -> None:
        del initial_observation
        self.close()
        self._restarts = 0
        self._episode_start = time.monotonic()
        self._gave_up = False
        self._start(fresh_game=True)

    def _start(self, fresh_game: bool) -> None:
        previous = self._agent
        self._thread_error = None
        self._sent_first_action = False
        self._env = ArenaEnvAdapter()
        self._agent = autoascend_agent.Agent(self._env, panic_on_errors=True)
        self._agent.resumed_game = not fresh_game
        if not fresh_game and previous is not None:
            self._agent.previous_character = previous.character
        self._thread = threading.Thread(target=self._run_agent, args=(self._agent,), name="autoascend",
                                        daemon=True)
        self._thread.start()

    def _restart(self, reason: str) -> None:
        jf_log.log(f"DRIVER restart #{self._restarts + 1}: {reason}")
        if self._env is not None:
            self._env.close()
        old = self._thread
        if old is not None and old.is_alive():
            _raise_in_thread(old, SystemExit)  # a livelocked thread would keep burning CPU
        self._restarts += 1
        self._start(fresh_game=False)

    def act(self, observation: Mapping[str, Any]) -> int:
        try:
            return self._act(observation)
        except Exception:
            jf_log.log("DRIVER error: " + traceback.format_exc(limit=5))
            return self._fallback_action

    def _act(self, observation: Mapping[str, Any]) -> int:
        if self._env is None or self._gave_up:
            return self._fallback_action
        if time.monotonic() - self._episode_start > self.WALL_BUDGET_SECONDS:
            jf_log.log("DRIVER wall-clock budget exhausted: ending the episode")
            self._gave_up = True
            self.close()
            return self._fallback_action
        if self._thread is None or not self._thread.is_alive() or self._thread_error is not None:
            if self._restarts >= 50:
                return self._fallback_action
            self._restart(f"agent thread died: {(self._thread_error or '')[-500:]}")
        if self._sent_first_action and not self._env.provide_observation(observation):
            self._restart("observation not accepted")
        if not self._sent_first_action:
            timeout = self.RESTART_TIMEOUT if self._warm else self._action_timeout
        else:
            timeout = self._hang_timeout
        try:
            action = self._env.next_action_index(timeout=timeout)
        except queue.Empty:
            if not self._sent_first_action:
                return self._fallback_action
            # livelock: interrupt the agent thread and give it a moment to recover
            jf_log.log("DRIVER hang: injecting AgentHang")
            if self._thread is not None:
                _raise_in_thread(self._thread, AgentHang)
            try:
                action = self._env.next_action_index(timeout=self.RECOVER_TIMEOUT)
            except queue.Empty:
                self._restart("agent did not recover from hang")
                try:
                    action = self._env.next_action_index(timeout=self.RESTART_TIMEOUT)
                except queue.Empty:
                    return self._fallback_action
                self._sent_first_action = True
                return action
            # the recovered agent's ESC answers the observation it already has
        self._sent_first_action = True
        self._warm = True
        return action

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._agent = None
        self._thread = None

    @property
    def thread_error(self) -> str | None:
        return self._thread_error

    def _run_agent(self, agent: autoascend_agent.Agent) -> None:
        try:
            agent.main()
        except autoascend_agent.AgentFinished:
            pass
        except BaseException:
            if agent is self._agent:
                self._thread_error = traceback.format_exc(limit=20)[-8_000:]
