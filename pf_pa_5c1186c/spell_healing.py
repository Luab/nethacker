"""Current-observation self-healing with bounded menu traversal and diagnostics."""
import json
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass

from .exceptions import AgentPanic


@dataclass(frozen=True)
class HealingSpell:
    letter: str
    name: str
    level: int
    failure: int
    retention: int  # lower endpoint, not an estimate of exact turns remaining


# NLE's spellretention() prints e.g. '91%-100%' after the initial turn.
# https://github.com/facebookresearch/nle/blob/main/src/spell.c
_ROW = re.compile(
    r'^\s*([a-zA-Z])\s*[-+]\s*(extra healing|healing)\s+'
    r'(\d+)\s+healing\s+(\d+)%\s+(\d+%(?:-\d+%)?|\(gone\))\s*$'
)
_PAGE = re.compile(r'\((\d+) of (\d+)\)')


def parse_healing_spells(lines):
    """Parse exact/ranged retention; unknown or invalid ranges fail closed."""
    spells = []
    for line in lines:
        match = _ROW.fullmatch(line)
        if match is None:
            continue
        letter, name, level, failure, retention = match.groups()
        bounds = [0] if retention == '(gone)' else [int(n) for n in re.findall(r'\d+', retention)]
        if not 0 <= bounds[0] <= bounds[-1] <= 100:
            continue
        spells.append(HealingSpell(letter, name, int(level), int(failure), bounds[0]))
    return spells


def rejection_reason(spell, energy, missing_hp):
    if spell.retention <= 0:
        return 'forgotten'
    if not 1 <= spell.level <= 7:
        return 'invalid_level'
    if not 0 <= spell.failure <= (15 if spell.name == 'extra healing' else 20):
        return 'unreliable'
    if spell.level * 5 > energy:
        return 'energy'
    if spell.name == 'extra healing' and missing_hp < 25:
        return 'small_wound'
    return None


def choose_healing_spell(spells, energy, missing_hp):
    usable = [s for s in spells if rejection_reason(s, energy, missing_hp) is None]
    preferred = ('extra healing', 'healing') if missing_hp >= 25 else ('healing',)
    return next((s for name in preferred for s in usable if s.name == name), None)


def current_page(agent):
    """Read pagination from this observation, never accumulated popup history."""
    obs = getattr(agent, '_observation', None)
    if obs is None:
        return None
    for row in obs['tty_chars']:
        match = _PAGE.search(bytes(row).decode('ascii', errors='replace'))
        if match:
            return tuple(map(int, match.groups()))
    return None


def current_more(agent):
    """Acknowledge only a message page in the current raw observation."""
    obs = getattr(agent, '_observation', None)
    return obs is not None and (bool(obs['misc'][2]) or any(
        b'--More--' in bytes(row) for row in obs['tty_chars']))


def current_prompt(agent):
    obs = getattr(agent, '_observation', None)
    return (bool(agent.single_popup) or
            'In what direction?' in agent.single_message or
            (obs is not None and any(obs['misc'])))


class SpellHealing:
    MAX_PAGES = 8
    MAX_MESSAGES = 8
    MAX_CANCELS = 3
    TRACE_LIMIT = 64

    def __init__(self):
        self.next_check_turn = -1
        self.diagnostics = deque(maxlen=self.TRACE_LIMIT)
        self.trace_count = 0

    def adopt(self, previous):
        """Copy game-level guards/diagnostics without adopting a pending cast."""
        self.next_check_turn = previous.next_check_turn
        self.diagnostics.extend(previous.diagnostics)
        self.trace_count = previous.trace_count

    def ready(self, hp, max_hp, energy, turn):
        return (turn >= self.next_check_turn and energy >= 5 and
                max_hp - hp >= 6 and hp * 100 <= max_hp * 65)

    def _record(self, record):
        self.diagnostics.append(record)
        if self.trace_count < self.TRACE_LIMIT:
            self.trace_count += 1
            # The normal agent logger is disabled in the arena. Only observed
            # game data are emitted; no evaluator identity or seed is accessed.
            try:
                print('HEAL_MENU ' + json.dumps(record, sort_keys=True), file=sys.stderr)
            except OSError:
                pass

    def cast(self, agent, command):
        start_turn = int(agent.blstats.time)
        self.next_check_turn = start_turn + 20
        selected = None
        directed = False
        completed = False
        message_pages = 0
        before_hp = int(agent.blstats.hitpoints)
        missing_hp = int(agent.blstats.max_hitpoints) - before_hp
        energy = int(agent.blstats.energy)
        pages = []
        reason = 'interrupted'
        actions = []
        prompt = ''

        def messages():
            nonlocal message_pages, reason
            # Keep the iterator alive through hunger/recall messages. Ending it
            # here hands the later direction prompt to unrelated policy code.
            while current_more(agent):
                if message_pages >= self.MAX_MESSAGES:
                    reason = 'message_limit'
                    raise AgentPanic('healing message chain did not finish')
                message_pages += 1
                actions.append('MORE')
                yield ' '

        def cancel():
            nonlocal reason
            for _ in range(self.MAX_CANCELS):
                yield from messages()
                actions.append('ESC')
                yield '\x1b'
                yield from messages()
                if not current_prompt(agent):
                    return
            reason = 'cancel_stalled'
            raise AgentPanic('healing prompt did not cancel')

        def inputs():
            nonlocal selected, directed, completed, reason, prompt
            yield from messages()
            msg = agent.single_message.lower()
            if not agent.single_popup and 'cast' in msg and 'spell' in msg and '?' in msg:
                actions.append('*')
                yield '*'
                yield from messages()
            seen = set()
            for _ in range(self.MAX_PAGES):
                rows = agent.single_popup
                spells = parse_healing_spells(rows)
                page = current_page(agent)
                signature = (tuple(rows), page)
                pages.append(dict(page=page, message=agent.single_message[:180],
                                  rows=[r[:140] for r in rows[:8]],
                                  spells=[dict(**asdict(s), rejection=rejection_reason(s, energy, missing_hp))
                                          for s in spells[:8]]))
                if signature in seen:
                    reason = 'repeated_page'
                    break
                seen.add(signature)
                selected = choose_healing_spell(spells, energy, missing_hp)
                if selected is not None:
                    break
                reason = ('ineligible' if spells else 'unparsed_menu' if rows else 'no_menu')
                if page is None or not 1 <= page[0] < page[1] <= self.MAX_PAGES:
                    break
                # SPACE advances a menu only while an observed later page exists.
                actions.append(' ')
                yield ' '
                yield from messages()
            if selected is None:
                yield from cancel()
                return
            actions.append(selected.letter)
            yield selected.letter
            yield from messages()
            prompt = agent.single_message[:180]
            if 'In what direction?' in agent.single_message:
                directed = True
                actions.append('.')
                yield '.'
                yield from messages()
                completed = not current_prompt(agent)
                if completed:
                    reason = 'directed'
                else:
                    reason = 'incomplete_direction'
                    yield from cancel()
            else:
                reason = 'no_direction'
                yield from cancel()

        with agent.atom_operation():
            try:
                agent.step(command, inputs())
                if selected is not None:
                    event = 'cast_' if completed else 'cast_fail_'
                    agent.stats_logger.log_event(event + selected.name)
                    if completed and agent.blstats.time > start_turn:
                        self.next_check_turn = int(agent.blstats.time)
                    else:
                        agent.last_cast_fail_turn[selected.name] = agent.blstats.time
                agent.log('SPELL_HEAL spell={} directed={} hp={}->{} retry={}'.format(
                    selected.name if selected else 'unavailable', directed, before_hp,
                    agent.blstats.hitpoints, self.next_check_turn))
            finally:
                # Net HP gain is an observation, not proof of spell replenishment:
                # regeneration and concurrent damage can affect the same turn.
                self._record(dict(turn=start_turn, pages=pages, reason=reason, actions=actions,
                                  spell=asdict(selected) if selected else None, directed=directed,
                                  completed=completed, message_pages=message_pages,
                                  prompt=prompt, message=agent.single_message[:180] if hasattr(agent, 'single_message') else '',
                                  hp_before=before_hp, hp=int(agent.blstats.hitpoints),
                                  energy_before=energy, energy=int(agent.blstats.energy),
                                  elapsed=int(agent.blstats.time) - start_turn, retry=int(self.next_check_turn)))
        return completed
