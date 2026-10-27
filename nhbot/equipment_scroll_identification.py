"""Spend known identify scrolls on armor useful to the current Wizard.

This supplements free nearby altar testing; it never reads an unknown scroll or
wears an unknown item. Native read.c gives even a cursed identify scroll at
least one full inventory identification while unconfused. Inventory's ordinary
preparation remains responsible for curse checks, fitting and observed AC.
"""
import re

from nle.nethack import actions as A

from . import objects as O, power_route
from .character import Character
from .glyph import Hunger
from .item import Item
from .item.utils import public_buc, public_noncursed
from .strategy import Strategy


READ_BACKOFF = 100
MAX_RESPONSES = 80
MAX_REMEMBERED_TARGETS = 128
_SLOTS = {O.ARM_SHIELD: 'off_hand', O.ARM_HELM: 'helm', O.ARM_GLOVES: 'gloves',
          O.ARM_BOOTS: 'boots', O.ARM_SHIRT: 'shirt', O.ARM_SUIT: 'suit', O.ARM_CLOAK: 'cloak'}
_BAD_ARMOR = frozenset(('dunce cap', 'helm of opposite alignment', 'gauntlets of fumbling',
                        'fumble boots', 'levitation boots'))


def _state(agent):
    result = getattr(agent, '_equipment_scroll_id', None)
    if result is None:
        result = agent._equipment_scroll_id = dict(attempted={}, retry_after=-1, reads=0,
                                                   picks=0, last=None)
    return result


def _identity(item):
    # Letters may change after an inventory arrangement. Knowledge changes are
    # meaningful; simply relabeling the same still-unknown armor is not.
    return (item.object.name, tuple(item.glyphs), item.naming, item.count,
            item.modifier, public_buc(item))


def _remember(agent, item):
    attempted = _state(agent)['attempted']
    attempted[_identity(item)] = int(agent.blstats.time)
    while len(attempted) > MAX_REMEMBERED_TARGETS:
        del attempted[next(iter(attempted))]


def _usable_kind(item):
    if (not item.is_armor() or not item.is_unambiguous() or item.count != 1 or
            item.shop_status != Item.NOT_SHOP or item.object.name in _BAD_ARMOR or
            re.search(r'\b(rusty|corroded|burnt|rotted)\b', item.text or '')):
        return False
    obj = item.object
    # Match the caster preparation policy; do not spend identification on an
    # iron helm/boots that the Wizard currently refuses to put on.
    if O.IRON <= obj.metal <= O.MITHRIL and obj.name != 'helm of brilliance':
        return False
    return obj.sub != O.ARM_SHIELD or obj.wt <= 30 or obj.name == 'shield of reflection'


def _known_removable(item):
    return (item.is_armor() and item.is_unambiguous() and item.count == 1 and
            item.shop_status == Item.NOT_SHOP and public_noncursed(item) and
            item.modifier is not None and '(being worn' in (item.text or ''))


def armor_value(agent, item):
    """A conservative investment score, not a claim about hidden enchantment."""
    if (not _usable_kind(item) or item.equipped or item.at_ready or
            public_buc(item) == Item.CURSED):
        return 0
    inv, obj = agent.inventory, item.object
    current = getattr(inv.items, _SLOTS[obj.sub])
    if public_noncursed(item):
        if item.modifier is not None or current is None:
            return 0  # already usable; wearing in an empty slot reveals enchantment
    if current is not None:
        if not _known_removable(current):
            return 0
        if (current.object.mgc or current.object.power or current.object.blk) and \
                current.object.name != obj.name:
            return 0  # the fitting policy preserves a different magical item
    if obj.sub == O.ARM_SHIELD and inv.items.main_hand is not None and \
            getattr(inv.items.main_hand.objs[0], 'bi', False):
        return 0
    if obj.sub == O.ARM_GLOVES and any(i.equipped and i.is_corpse() for i in inv.items):
        return 0
    # A body/shirt upgrade is useful only if every covering layer can safely
    # come off and be restored by the existing fitting transaction.
    covers = (('cloak', 'suit') if obj.sub == O.ARM_SHIRT else
              ('cloak',) if obj.sub == O.ARM_SUIT else ())
    if any((layer := getattr(inv.items, name)) is not None and not _known_removable(layer)
           for name in covers):
        return 0
    candidate_ac = item.get_ac()  # an unknown modifier is conservatively treated as zero
    current_ac = current.get_ac() if current is not None else 0
    gain = current_ac - candidate_ac
    utility = current is None and bool(obj.mgc or obj.power or obj.blk)
    if gain <= 0 and not utility:
        return 0  # no lottery on a worse base item or an ordinary zero-AC shirt
    # Do not identify a spare when an owned, already-safe piece can fill the
    # same slot just as well. A second unknown copy is handled by the attempt
    # identity and by rechecking this rule after the first identification.
    for other in inv.items:
        if other is item or not _usable_kind(other) or other.object.sub != obj.sub or \
                not public_noncursed(other):
            continue
        if other.modifier is None and current is not None:
            continue
        if other.get_ac() <= candidate_ac and (not utility or
                other.object.name == obj.name or other.object.power == obj.power):
            return 0
    return 200 + 10 * max(0, gain) + (5 if utility else 0)


def _safe(agent):
    inv, bl, prop = agent.inventory, agent.blstats, agent.character.prop
    logic = getattr(agent, 'global_logic', None)
    if (agent.character.role != Character.WIZARD or agent.character.race != Character.GNOME or
            any(getattr(prop, name, False) for name in
                ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) or
            bl.hitpoints < 0.8 * bl.max_hitpoints or bl.hunger_state >= Hunger.HUNGRY or
            bl.carrying_capacity >= 2 or agent.get_visible_monsters() or agent.hands_welded() or
            inv.shop_debt_pending() or inv._credit_clone_reserved() or inv._preparation_shop_reserved() or
            getattr(inv, '_armor_fitting', None) is not None or
            getattr(getattr(logic, 'wizard_route', None), 'phase', None) in ('return', 'recover')):
        return False
    for owner in (getattr(agent, 'equipment_recovery', None),
                  getattr(logic, 'equipment_identification', None),
                  getattr(logic, 'altar_preparation', None)):
        if getattr(owner, 'state', None) is not None:
            return False
    pet = getattr(agent, '_pet_care', None)
    if pet is not None and pet.get('pending') is not None:
        return False
    return not agent.current_level().shop[bl.y, bl.x]


def _scrolls(agent):
    # BUC is deliberately unrestricted for this exact effect, unlike armor
    # enchantment. Ownership and observed object identity remain mandatory.
    return [i for i in power_route.known_identify(agent) if i.shop_status == Item.NOT_SHOP and
            i.count > 0 and not i.equipped and not i.at_ready]


def _choice(agent):
    st = _state(agent)
    if agent.blstats.time < st['retry_after']:
        return None
    scrolls = _scrolls(agent)
    if not scrolls or not _safe(agent):
        return None
    candidates = [(armor_value(agent, i), i) for i in agent.inventory.items if
                  i.is_armor() and i.is_unambiguous() and _identity(i) not in st['attempted']]
    candidates = [(value, item) for value, item in candidates if value > 0]
    if not candidates:
        return None
    _, target = max(candidates, key=lambda pair: pair[0])
    # Preserve a blessed multi-identify scroll if a single-item scroll suffices.
    scrolls.sort(key=lambda i: (public_buc(i) == Item.BLESSED, i.count))
    return scrolls[0], target


@Strategy.wrap
def identify_useful_armor(agent):
    if _choice(agent) is None:
        yield False
        return
    yield True
    choice = _choice(agent)  # completed combat/eating/ownership changes preempt selection
    if choice is None:
        return
    scroll, target = choice
    inv, st = agent.inventory, _state(agent)
    letter, target_key = inv.items.get_letter(scroll), _identity(target)
    st['retry_after'] = int(agent.blstats.time) + READ_BACKOFF
    st['reads'] += 1
    record = st['last'] = dict(turn=int(agent.blstats.time), scroll=scroll.text,
                               target=target.text, selected=False, picks=[], outcome='interrupted')
    picked = set()
    picked_armor_slots = set()
    pages = {}
    agent.log(f'EQUIPMENT_SCROLL_ID begin scroll={scroll.text!r} target={target.text!r}')

    def value(item):
        if inv.items.get_letter(item) in picked:
            return 0
        if item.is_armor():
            if item.is_unambiguous() and item.object.sub in picked_armor_slots:
                return 0  # inventory knowledge is refreshed only after the menu closes
            return armor_value(agent, item)
        # Extra identifications are already paid for. Keep the existing useful
        # non-armor priorities, below an immediately wearable armor upgrade.
        return min(90, power_route._id_value(agent, item))

    def on_pick(item):
        picked.add(inv.items.get_letter(item))
        record['picks'].append(item.text)
        st['picks'] += 1
        if item.is_armor() and item.is_unambiguous():
            picked_armor_slots.add(item.object.sub)
            _remember(agent, item)

    def answers():
        if 'What do you want to read?' not in (agent.single_message or ''):
            record['outcome'] = 'no_read_prompt'
            return
        if (not _safe(agent) or scroll not in _scrolls(agent) or
                target not in list(inv.items) or _identity(target) != target_key or
                armor_value(agent, target) <= 0):
            record['outcome'] = 'selection_changed'
            yield A.Command.ESC
            return
        # Record acceptance before sending the letter: a completed action may
        # interrupt the atom's exit. Never spend again on that unchanged target.
        record['selected'] = True
        _remember(agent, target)
        yield letter
        pending = []
        for _ in range(MAX_RESPONSES):
            if pending:
                yield pending.pop(0)
                continue
            head = (agent.single_message or '') + ' ' + ' '.join(agent.single_popup[:3])
            if 'What would you like to identify' in head:
                keys = power_route._identify_step(agent, pages, value=value, on_pick=on_pick)
                if keys is None:
                    yield A.Command.ESC
                    record['outcome'] = 'no_useful_menu_choice'
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

    try:
        with agent.atom_operation():
            agent.step(A.Command.READ, answers())
        inv.items.update()  # public inventory only; do not open containers
        matches = [i for i in inv.items if i.is_armor() and i.is_unambiguous() and
                   _identity(i)[:4] == target_key[:4]]
        record['observed'] = [dict(text=i.text, buc=public_buc(i), modifier=i.modifier) for i in matches]
        record['ac_after'] = int(agent.blstats.armor_class)
    finally:
        agent.log(f'EQUIPMENT_SCROLL_ID result={record}')


def snapshot(agent):
    st = getattr(agent, '_equipment_scroll_id', None)
    if st is None:
        return dict(reads=0, picks=0, attempted=0, retry_after=-1, last=None)
    return {key: st[key] for key in ('reads', 'picks', 'retry_after', 'last')} | \
        dict(attempted=len(st['attempted']))
