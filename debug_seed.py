"""Run the bot directly against NLE for one seed and print metrics.

Usage: python debug_seed.py <seed> [max_steps]

Uses the same trajectory_spec("public", "local", seed) the arena uses, and
constructs Agent(env, seed=0, ...) directly (the Agent's rng is always
RandomState(0) regardless of the arena bot_seed).
"""
import sys
import time
import traceback

import nle.nethack as nh
from nle.env.tasks import NetHackChallenge

from nethackers.arena.environment import NLEEnvironment, PUBLIC_OBSERVATION_KEYS
from nethackers.arena.seeds import trajectory_spec
from nethackers.arena.progress import NetHackProgress

from autoascend.agent import Agent, AgentFinished


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    spec = trajectory_spec("public", "local", seed)
    env = NLEEnvironment(max_steps, 10000, "wiz-gno-neu-mal")
    env._env.install_seeds(spec)
    obs, info = env._env.reset()

    agent = Agent(env, seed=0, verbose=False, panic_on_errors=False)
    # Feed the initial observation through the same path the arena uses.
    # Agent.main() expects to be driven by env.step(); we run it in a thread
    # like the arena adapter, or simpler: run main() and let it block on
    # env.step.  We need to provide the first observation to the env adapter.
    # Easiest: use the ArenaEnvAdapter.
    from arena_adapter import ArenaEnvAdapter
    adapter = ArenaEnvAdapter()
    agent = Agent(adapter, seed=0, verbose=False, panic_on_errors=False)

    import threading
    thread_error = []

    def run_agent():
        try:
            agent.main()
        except AgentFinished:
            pass
        except BaseException:
            thread_error.append(traceback.format_exc(limit=30))

    t = threading.Thread(target=run_agent, daemon=True)
    t.start()

    steps = 0
    started = time.monotonic()
    observation = obs
    # NOTE: the arena does NOT feed the initial observation to the agent.
    # The agent's first action (ESC) is applied to the initial observation,
    # and the *result* is the first observation the agent sees.
    try:
        while steps < max_steps:
            action = adapter.next_action_index(timeout=120)
            observation, reward, terminated, truncated = env.step(action)
            steps += 1
            if not adapter.provide_observation(observation):
                break
            if terminated or truncated:
                break
    except Exception as e:
        print(f"DRIVER ERROR: {e!r}")
        traceback.print_exc()
    finally:
        adapter.close()
        t.join(timeout=5)

    if thread_error:
        print("AGENT THREAD ERROR:")
        print(thread_error[0][-8000:])

    m = env.metrics()
    print(f"seed={seed} steps={steps} wall={time.monotonic()-started:.2f}")
    print(f"progress={m.progress} turns={m.turns} max_depth={m.max_depth} "
          f"end_status={m.end_status} milestone={m.milestone} cause={m.cause_of_death}")
    print(f"thread_error={bool(thread_error)}")

    # Print final inventory and message history if available
    try:
        print("--- last message ---")
        print(agent.message)
        print("--- message history (last 500) ---")
        for msg in agent._message_history[:]:
            print(repr(msg))
        print("--- inventory ---")
        for item in agent.inventory.items:
            print(repr(item), 'objs=', [o.name for o in item.objs], 'status=', item.status)
        print("--- blstats ---")
        print(agent.blstats)
    except Exception as e:
        print(f"could not print agent state: {e!r}")


if __name__ == "__main__":
    main()
