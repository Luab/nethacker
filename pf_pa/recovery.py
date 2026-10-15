"""Bounded, interruptible natural HP recovery between encounters."""
import sys
from dataclasses import dataclass

from .exceptions import AgentPanic
from .glyph import G, Hunger, SS
from .strategy import Strategy
from . import utils


class SearchStalled(AgentPanic):
    """A recovery command could not finish its prompt response chain."""


@dataclass(frozen=True)
class SearchResult:
    """Evidence of a completed command exchange, not proof of elapsed time."""
    status: str
    sent: bool = True

    @classmethod
    def from_response(cls, response, previous, parsed, received, before_turn,
                      after_turn, message, popup, reading):
        if response is None or response is previous or not received:
            return cls('stale_observation')
        if response['misc'].any() or popup or reading:
            return cls('prompt')
        if parsed is not response:
            # A menu response or a stale parsed state cannot confirm this search.
            return cls('unconfirmed_observation')
        text = message.lower().strip()
        if any(part in text for part in ("unknown command", "you can't", "you cannot")):
            return cls('refused')
        if after_turn < before_turn:
            return cls('time_reversed')
        if after_turn == before_turn and text and not text.startswith('you find '):
            # Silent search is normal. Other unchanged-turn messages could be
            # unrecognized refusals, so conservatively stop this recovery session.
            return cls('unconfirmed_message')
        return cls('completed')


class Recovery:
    def __init__(self, agent):
        self.agent = agent
        self._health = None
        self.blocked_until = -1
        self.sessions = 0

    def adopt(self, previous):
        """Carry game-owned guards across a worker restart, never its generator."""
        self._health = previous._health
        self.blocked_until = previous.blocked_until
        self.sessions = previous.sessions

    def observe(self, stats):
        # Called on parsed observations, including non-turn-consuming menus.
        # A changed HP pool (level gain or polymorph) is not incoming damage.
        health = (stats.hitpoints, stats.max_hitpoints, stats.monster_level)
        if self._health is not None:
            hp, maximum, form = self._health
            if health[1:] == (maximum, form) and health[0] < hp:
                self.blocked_until = max(self.blocked_until, stats.time + 5)
        self._health = health

    def _unsafe_reason(self):
        agent = self.agent
        stats = agent.blstats
        if stats.time < self.blocked_until:
            return 'recent_damage_or_backoff'
        if stats.hunger_state >= Hunger.HUNGRY:
            return 'hungry'
        if stats.carrying_capacity > 0:
            return 'burdened'
        # Receiver-specific descent policy already budgets rest around fast
        # digging. Do not turn a short escape or half-dug hole into a long rest.
        dive = agent.global_logic.dive
        if dive._in_own_pit() or (dive.diving and
                (dive.digging_tool() is not None or dive.digging_wand() is not None)):
            return 'descent_owned'
        prop = agent.character.prop
        if prop.blind or prop.hallu or prop.confusion or prop.stun or prop.polymorph:
            return 'impaired'
        # Unlike get_visible_monsters(), this includes distant, unreachable and
        # warning/invisible occupants; none is a good reason to assume safety.
        tracker = agent.monster_tracker
        if ((tracker.monster_mask & ~tracker.peaceful_monster_mask).any() or
                utils.any_in(agent.glyphs, G.WARNING, G.INVISIBLE_MON)):
            return 'monster'
        if utils.any_in(agent.glyphs, G.SWALLOW):
            return 'engulfed'
        terrain = agent.current_level().objects[stats.y, stats.x]
        if terrain in {SS.S_pool, SS.S_water, SS.S_lava}:
            return 'wet_terrain'
        if terrain in G.TRAPS:
            return 'trap'
        return None

    @staticmethod
    def _position(stats):
        return stats.dungeon_number, stats.level_number, stats.y, stats.x

    @Strategy.wrap
    def rest(self):
        agent = self.agent
        stats = agent.blstats
        # Hysteresis: start at a meaningful injury, finish with a reserve.
        if stats.max_hitpoints <= 0 or stats.hitpoints >= 0.7 * stats.max_hitpoints \
                or self._unsafe_reason() is not None:
            yield False
            return
        yield True

        self.sessions += 1
        agent.stats_logger.log_event('recovery_started')
        start = agent.blstats
        position = self._position(start)
        last_gain_turn = start.time
        best_hp = start.hitpoints
        reason = 'interrupted'
        searches = 0
        confirmed = 0
        same_turn = 0
        stalled = 0
        try:
            while True:
                stats = agent.blstats
                if (self._position(stats) != position or stats.max_hitpoints != start.max_hitpoints
                        or stats.monster_level != start.monster_level):
                    reason = 'state_changed'
                    break
                reason = self._unsafe_reason()
                if reason is not None:
                    break
                if stats.hitpoints >= 0.9 * stats.max_hitpoints:
                    reason = 'recovered'
                    agent.stats_logger.log_event('recovery_completed')
                    break
                if stats.hitpoints > best_hp:
                    best_hp = stats.hitpoints
                    last_gain_turn = stats.time
                if stats.time - last_gain_turn >= 30 or stats.time - start.time >= 200 or searches >= 200:
                    reason = 'budget_or_no_healing'
                    self.blocked_until = max(self.blocked_until, stats.time + 30)
                    break
                previous_turn = stats.time
                reason = 'interrupted'
                # One search only: normal observation updates and all outer
                # survival preemptions run before another recovery action.
                # Count attempted calls separately: preemption may unwind search
                # after dispatch but before its completion can be confirmed.
                searches += 1
                agent.stats_logger.log_event('recovery_search')
                try:
                    result = agent.search(return_result=True)
                except SearchStalled:
                    reason = 'prompt_stalled'
                    self.blocked_until = max(self.blocked_until, agent.blstats.time + 30)
                    # Let Agent.main cancel/resynchronize the pending interaction
                    # using its existing AgentPanic path, not stale recovery state.
                    raise
                if result.status != 'completed':
                    reason = result.status
                    self.blocked_until = max(self.blocked_until, agent.blstats.time + 30)
                    break
                confirmed += 1
                agent.stats_logger.log_event('recovery_search_confirmed')
                if agent.blstats.time == previous_turn:
                    same_turn += 1
                    stalled += 1
                    agent.stats_logger.log_event('recovery_search_same_turn')
                else:
                    stalled = 0
                # Identical fresh observations can mean a legitimate speed action
                # or a silent refused/stale response. Allow a short run, never an
                # unbounded one. HP gains cannot reset this independent clock.
                if stalled >= 4:
                    reason = 'turn_stalled'
                    self.blocked_until = max(self.blocked_until, agent.blstats.time + 30)
                    break
        finally:
            # Bounded per-game diagnostics make actual activation and observed
            # HP changes inspectable without action-sized logs or hidden state.
            if self.sessions <= 12:
                end = agent.blstats
                print(f'RECOVERY session={self.sessions} role={agent.character.role} '
                      f'turn={start.time}:{end.time} hp={start.hitpoints}:{end.hitpoints} '
                      f'maxhp={start.max_hitpoints} level={start.dungeon_number}:{start.level_number} '
                      f'hunger={start.hunger_state}:{end.hunger_state} searches={searches} confirmed={confirmed} '
                      f'same_turn={same_turn} reason={reason}',
                      file=sys.stderr, flush=True)
