"""Bounded public-state divination practice and nondirectional identification.

Practice has a concrete goal: make a known identify spell usable on carried
unknown equipment or consumables. It never waits for power or food, and its
lifetime budget cannot restart when a route, level, or inventory changes.
"""
import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import objects as O, power_route
from .character import Character
from .glyph import Hunger
from .item import Item
from .item.utils import public_buc
from .strategy import Strategy


UTILITY_SPELLS = ('detect monsters', 'identify')
TRAIN_LIMIT = 24
TRAIN_NUTRITION_LIMIT = 240
IDENTIFY_LIMIT = 64
FOOD_RESERVE = 800
_FAILED = re.compile(r"fail to cast|too impaired|arms are not free|lack the strength|"
                     r"too hungry|unable to chant|don't know any spells|knowledge.*twisted|"
                     r"don't have enough energy|concentration falters", re.I)


def enabled(agent):
    ch = agent.character
    return getattr(ch, 'role', None) == Character.WIZARD and getattr(ch, 'race', None) == Character.GNOME


def state(agent):
    if not hasattr(agent, '_utility_spells'):
        agent._utility_spells = dict(training=0, training_nutrition=0, identifies=0,
            successes=0, picks=0, retry_after=-1, target_failures={}, last=None)
    return agent._utility_spells


def _safe(agent):
    bl, prop = agent.blstats, agent.character.prop
    inv = agent.inventory
    if any(getattr(inv, method, lambda: False)() for method in
           ('shop_debt_pending', '_credit_clone_reserved', '_preparation_shop_reserved')):
        return False
    if getattr(inv, '_armor_fitting', None) is not None or any(
            'Amulet of Yendor' in (item.text or '') for item in inv.items):
        return False
    if getattr(getattr(inv, '_metal_armor', None), 'pending', None) is not None:
        return False
    logic = getattr(agent, 'global_logic', None)
    if getattr(getattr(logic, 'wizard_route', None), 'phase', None) in ('return', 'recover'):
        return False
    for owner in (getattr(agent, 'equipment_recovery', None),
                  getattr(logic, 'equipment_identification', None),
                  getattr(logic, 'altar_preparation', None)):
        if getattr(owner, 'state', None) is not None:
            return False
    return (enabled(agent) and bl.hunger_state <= Hunger.NOT_HUNGRY and
        bl.hitpoints == bl.max_hitpoints and not bl.carrying_capacity and
        not any((prop.blind, prop.confusion, prop.stun, prop.hallu, prop.polymorph)) and
        not agent.hands_welded() and not agent.get_visible_monsters() and
        not agent.current_level().shop[bl.y, bl.x] and
        agent.inventory.carried_nutrition() >= FOOD_RESERVE)


def _key(item):
    return (tuple(item.glyphs), tuple(o.name for o in item.objs), item.naming,
            public_buc(item), item.modifier)


def identify_value(agent, item):
    if item.shop_status != Item.NOT_SHOP:
        return 0
    if not item.is_unambiguous():
        return {nh.WAND_CLASS: 90, nh.AMULET_CLASS: 85, nh.RING_CLASS: 80,
                nh.SPBOOK_CLASS: 75, nh.ARMOR_CLASS: 70, nh.POTION_CLASS: 60,
                nh.SCROLL_CLASS: 55}.get(item.category, 0)
    if item.is_armor() and (public_buc(item) == Item.UNKNOWN or item.modifier is None):
        return 45
    if item.category in (nh.SPBOOK_CLASS, nh.RING_CLASS, nh.AMULET_CLASS) and \
            public_buc(item) == Item.UNKNOWN:
        return 35
    return 0


def _targets(agent):
    failures = state(agent)['target_failures']
    return [item for item in agent.inventory.items if identify_value(agent, item) > 0 and
            failures.get(_key(item), 0) < 4]


def _available(agent, name, max_failure):
    catalog = agent.character.spell_catalog or {}
    spell = catalog.get(name)
    return (spell is not None and name in agent.character.known_spells and
            spell.retention != (0, 0) and spell.remaining_upper(agent.blstats.time) > 0 and
            spell.failure <= max_failure)


def choice(agent):
    st = state(agent)
    if not _safe(agent) or agent.blstats.time < st['retry_after'] or not _targets(agent):
        return None
    bl, ch = agent.blstats, agent.character
    if (_available(agent, 'identify', .35) and bl.energy >= 25 and
            st['identifies'] < IDENTIFY_LIMIT):
        return 'identify'
    # Stop at Basic: higher ranks need much greater practice and are not an
    # ordinary early-game resource investment. No unavailable skill-slot grind.
    if (not _available(agent, 'identify', 1.0) or
            ch.spell_catalog['identify'].failure <= .35 or
            ch.skill_levels[O.P_DIVINATION_SPELL] != ch.name_to_skill_level['Unskilled'] or
            bl.experience_level < 2 or not _available(agent, 'detect monsters', .15) or
            st['training'] >= TRAIN_LIMIT or st['training_nutrition'] >= TRAIN_NUTRITION_LIMIT or
            bl.energy < 5 + max(10, (bl.max_energy + 1) // 2)):
        return None
    return 'detect monsters'


def cast_nondirectional(agent, name):
    """Run one actual spell/menu transaction; never queue a direction key.

    Completion is separated from effect evidence. Identification records public
    inventory changes; detection success requires a completed turn and either
    an explicit detection message or substantial power expenditure. Power can
    regenerate on the casting turn, so it is not an exact spell cost meter.
    """
    if name not in UTILITY_SPELLS or name not in agent.character.known_spells:
        return dict(spell=name, outcome='unavailable', success=False, picks=[])
    before_turn, before_power = int(agent.blstats.time), int(agent.blstats.energy)
    before = {agent.inventory.items.get_letter(i): i.text for i in agent.inventory.items}
    record = dict(spell=name, turn=before_turn, outcome='interrupted', success=False,
                  selected=False, picks=[], messages=[])
    picked, pages = set(), {}

    def value(item):
        return 0 if agent.inventory.items.get_letter(item) in picked else identify_value(agent, item)

    def on_pick(item):
        letter = agent.inventory.items.get_letter(item)
        picked.add(letter)
        record['picks'].append(dict(letter=letter, before=item.text))

    def answers():
        head = ' '.join(agent.single_popup[:2])
        if 'Choose which spell to cast' not in head:
            record['outcome'] = 'no_spell_menu'
            return
        record['selected'] = True
        yield agent.character.known_spells[name]
        pending = []
        for _ in range(80):
            msg = agent.single_message or ''
            if msg and (not record['messages'] or msg != record['messages'][-1]):
                record['messages'].append(msg)
            if _FAILED.search(msg):
                record['outcome'] = 'cast_failed'
                return
            if pending:
                yield pending.pop(0)
                continue
            head = msg + ' ' + ' '.join(agent.single_popup[:3])
            if 'What would you like to identify' in head:
                if name != 'identify':
                    record['outcome'] = 'unexpected_identify_menu'
                    yield A.Command.ESC
                    return
                keys = power_route._identify_step(agent, pages, value=value, on_pick=on_pick)
                if keys is None:
                    record['outcome'] = 'no_useful_menu_choice'
                    yield A.Command.ESC
                    return
                if keys.endswith('\r'):
                    pages.clear()
                pending = list(keys)
                yield pending.pop(0)
                continue
            misc = agent._observation['misc']
            if misc[0] or misc[1]:
                record['outcome'] = 'unexpected_prompt'
                yield A.Command.ESC
                return
            if misc[2]:
                yield A.TextCharacters.SPACE
                continue
            record['outcome'] = 'completed'
            return
        record['outcome'] = 'response_budget'
        yield A.Command.ESC

    with agent.atom_operation():
        agent.step(A.Command.CAST, answers())
    agent.inventory.items.update()
    after = {agent.inventory.items.get_letter(i): i.text for i in agent.inventory.items}
    record.update(turn_after=int(agent.blstats.time), power_before=before_power,
                  power_after=int(agent.blstats.energy))
    for pick in record['picks']:
        pick['after'] = after.get(pick['letter'])
        pick['changed'] = pick['after'] is not None and pick['after'] != pick['before']
    record['changed_letters'] = [letter for letter in before if letter in after and after[letter] != before[letter]]
    spent = before_power - int(agent.blstats.energy)
    cost = 5 * agent.character.spell_catalog[name].level
    record['effect_evidence'] = (
        'inventory_changed' if name == 'identify' and record['changed_letters'] else
        'detection_message' if name == 'detect monsters' and any(
            re.search(r'strange feeling for a moment|You feel threatened|You sense the presence of monsters', msg)
            for msg in record['messages']) else
        'power_expenditure' if name == 'detect monsters' and spent >= cost - 1 else None)
    record['success'] = (record['selected'] and record['outcome'] == 'completed' and
                         agent.blstats.time > before_turn and record['effect_evidence'] is not None)
    agent.stats_logger.log_event(('cast_' if record['success'] else 'cast_fail_') + name)
    return record


@Strategy.wrap
def use(agent):
    if choice(agent) is None:
        yield False
        return
    yield True
    name = choice(agent)
    if name is None:
        return
    st, bl = state(agent), agent.blstats
    st['retry_after'] = int(bl.time) + 25
    # Equipment, skill, level and memory may have changed since the last menu.
    # Observe current failure/memory before paying for this optional cast.
    agent.character.parse_spellcast_view()
    if not _safe(agent) or not _available(agent, name, .35 if name == 'identify' else .15):
        return
    current_targets = _targets(agent)
    if not current_targets:
        return
    targets = {agent.inventory.items.get_letter(i): _key(i) for i in current_targets}
    primary = max(current_targets, key=lambda item: identify_value(agent, item))
    primary_key = _key(primary)
    if name == 'detect monsters':
        st['training'] += 1
        st['training_nutrition'] += 0 if bl.intelligence >= 17 else 2 if bl.intelligence == 16 else 5 if bl.intelligence == 15 else 10
    else:
        st['identifies'] += 1
    agent._utility_divination_goal = True
    st['last'] = dict(spell=name, turn=int(agent.blstats.time), outcome='interrupted',
                      success=False, picks=[])
    try:
        record = st['last'] = cast_nondirectional(agent, name)
        st['successes'] += int(record['success'])
        st['picks'] += sum(bool(p.get('changed')) for p in record['picks'])
        if not record['success']:
            st['retry_after'] = int(agent.blstats.time) + 125
        if name == 'identify':
            attempted = [targets[p['letter']] for p in record['picks'] if p['letter'] in targets]
            for key in attempted or [primary_key]:
                st['target_failures'][key] = st['target_failures'].get(key, 0) + 1
        if record['success']:
            before_failure = agent.character.spell_fail_chance.get('identify')
            agent.character.parse_enhance_view()
            agent.character.parse_spellcast_view()
            record['identify_failure_before'] = before_failure
            record['identify_failure_after'] = agent.character.spell_fail_chance.get('identify')
    finally:
        agent._utility_divination_goal = False
        agent.log(f'UTILITY_SPELL {st["last"]}')


def snapshot(agent):
    return {key: value for key, value in state(agent).items() if key != 'target_failures'}
