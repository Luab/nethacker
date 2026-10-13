"""Ring/amulet handling for an autoascend-family Inventory, written by CleverShovel from NetHack 3.6.6's do_wear.c,
mkobj.c, eat.c and pray.c (CleverShovel/nethacker@9dc0822; this is the module version of 29ab0a7, which adds the
identify-menu paging fix and the harm-gated scroll reading in scroll_identify.py). Installed onto Inventory by
install(); Inventory.gather_items() runs the strategies.

Sources: do_wear.c Amulet_on/Amulet_off/Ring_on (strangulation says "It constricts your throat!" and
sets Strangled = 6, removal clears it; no ring has an on-wear message for its dangerous types),
mkobj.c (strangulation/restful sleep/change amulets are cursed 9 in 10), the #remove command
refusing a cursed worn item, eat.c gethungry() (1 nutrition per 20 turns per worn ring/amulet),
pray.c (strangulation is major trouble a prayer fixes).

Port changes (nhbot): on/off switches and role gates are jf_config RING_* flags (default: Wizards only); RINGSTAT
stderr lines only with ring_amulet_config.RING_DEBUG; a ring/amulet is put on only when the game printed its BUC as
uncursed/blessed and a chargeable ring only at a known positive enchantment (the item parser reads an unknown BUC as
uncursed); a failed put on (no game time) is not retried in a loop; the combat-only put on needs a free finger/neck;
the hunger shedding touches only what this module put on (never the teleport-control or levitation rings of
power_route/tele_route/dive_logic) and never slow digestion; #remove answers its prompt whenever it is asked
(a worn blindfold also counts as an accessory). The Item.can_be_dropped_from_inventory patch is not needed here
(jf_config.WORN_KEEP)."""
import os
import re
import sys

import nle.nethack as nh
from nle.nethack import actions as A

from .. import jf_config, objects as O, utils
from ..glyph import Hunger
from ..strategy import Strategy
from . import ring_amulet_config as cfg
from .item import Item

# do_wear.c: these change combat resolution only (to-hit, damage, AC, reflecting a ray) and do
# nothing between fights, unlike a resistance or a sustain ability
_COMBAT_ONLY_RING_NAMES = ('increase accuracy', 'increase damage', 'protection')
_COMBAT_ONLY_AMULET_NAMES = ('amulet of reflection',)


def buc_of(item):
    """'cursed' / 'uncursed' / 'blessed' as the game printed it on the item line, None when the line carries no
    such word, i.e. the BUC is not known (the host parser turns an unknown status into UNCURSED, so the
    status field cannot tell; for a Priest every item is known and 'uncursed' is simply not printed)."""
    m = re.search(r'\b(cursed|uncursed|blessed)\b', item.text or '')
    return m.group(1) if m else None


def _role_in(agent, roles):
    if roles is None:
        return True
    from ..character import Character
    role = getattr(agent.character, 'role', None)
    return any(role == getattr(Character, str(name).upper(), object()) for name in roles)


def _safe_to_put_on(inv, item):
    """PORT: only a ring/amulet the game calls uncursed or blessed (a Priest is never told 'uncursed'), and a
    chargeable ring only at a known positive enchantment."""
    buc = buc_of(item)
    if buc == 'cursed':
        return False
    if buc is None:
        from ..character import Character
        if getattr(inv.agent.character, 'role', None) != Character.PRIEST:
            return False
    if item.category == nh.RING_CLASS and item.is_unambiguous() and item.object.name in cfg.CHARGED_RINGS:
        return item.modifier is not None and item.modifier > 0
    return True


def _is_combat_only_item(item):
    if not item.is_unambiguous() or item.status == Item.CURSED:
        return False
    if item.category == nh.RING_CLASS:
        return item.object.name in _COMBAT_ONLY_RING_NAMES
    if item.category == nh.AMULET_CLASS:
        return item.object.name in _COMBAT_ONLY_AMULET_NAMES
    return False


def _log(inv, what):
    if cfg.RING_DEBUG:
        print(f'RINGSTAT pid={os.getpid()} {what} depth={inv.agent.blstats.depth} turn={inv.agent.blstats.time}',
              file=sys.stderr, flush=True)


def _can_pray(agent, certain_death=False):
    try:
        return agent.is_safe_to_pray(certain_death=certain_death)
    except TypeError:  # trees without the certain_death gate
        return agent.is_safe_to_pray()


def _allowed(inv, roles=None):
    """No ring/amulet logic where the host tree wears and removes them on purpose (castle,
    Gehennom) or while floating (a levitation ring must stay on); only for the configured roles."""
    agent = inv.agent
    if not jf_config.RING_MODULE or not _role_in(agent, roles):
        return False
    if agent.blstats.depth >= cfg.MAX_DEPTH:
        return False
    if int(agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & nh.BL_MASK_LEV:
        return False
    return True


def _wear_allowed(inv):
    return _allowed(inv, jf_config.RING_WEAR_ROLES)


def _hostile_within(inv, radius):
    return any(dist <= radius for dist, *_ in inv.agent.get_visible_monsters())


def _free_slot(inv, item):
    worn = [i for i in inv.items if i.equipped and i.category == item.category]
    return len(worn) < (2 if item.category == nh.RING_CLASS else 1)


def _puton_blocked(inv, item):
    fails, turn = inv._puton_failures.get(item.glyphs[0], (0, -10 ** 9))
    if fails < cfg.PUTON_FAIL_LIMIT:
        return False
    if inv.agent.blstats.time - turn >= cfg.PUTON_FAIL_COOLDOWN:
        inv._puton_failures.pop(item.glyphs[0], None)
        return False
    return True


def put_on(self, item, finger='l'):
    """#puton a ring or amulet (armor uses wear(): a different command). True when it is worn afterwards; a failure
    (it takes no game time) is counted per glyph, see _puton_blocked."""
    assert item is not None and item.category in (nh.RING_CLASS, nh.AMULET_CLASS), item
    letter = self.items.get_letter(item)
    glyph = item.glyphs[0]
    before = sum(1 for i in self.items if i.equipped and i.glyphs[0] == glyph)

    def gen():
        if 'What do you want to put on?' not in self.agent.single_message:
            return
        yield letter
        if 'Which ring-finger' in self.agent.single_message:
            yield finger

    with self.agent.atom_operation():
        self.agent.step(A.Command.PUTON, gen())
    self.items.update(force=True)
    ok = sum(1 for i in self.items if i.equipped and i.glyphs[0] == glyph) > before
    if ok:
        self._module_worn.add(glyph)
        self._puton_failures.pop(glyph, None)
    else:
        fails, _ = self._puton_failures.get(glyph, (0, 0))
        self._puton_failures[glyph] = (fails + 1, self.agent.blstats.time)
    _log(self, f'puton {item.text!r} ok={ok} msg={self.agent.message[:80]!r}')
    return ok


def remove_ring_or_amulet(self, item):
    """#remove a worn ring or amulet; False when it stays worn (cursed, or cursed gloves / a welded weapon in the
    way). With a single worn accessory #remove selects it itself and never asks "What do you want to remove?"."""
    assert item is not None and item.equipped and item.category in (nh.RING_CLASS, nh.AMULET_CLASS), item
    letter = self.items.get_letter(item)
    glyph = item.glyphs[0]
    before = sum(1 for i in self.items if i.equipped and i.glyphs[0] == glyph)

    def gen():
        if 'What do you want to remove?' in self.agent.single_message:
            yield letter

    with self.agent.atom_operation():
        self.agent.step(A.Command.REMOVE, gen())

    self.items.update(force=True)
    # PORT: judged by the inventory, not by 'cursed' in the message ('You cannot remove your gloves...')
    ok = sum(1 for i in self.items if i.equipped and i.glyphs[0] == glyph) < before
    if ok:
        self._module_worn.discard(glyph)
    _log(self, f'remove {item.text!r} ok={ok} msg={self.agent.message[:80]!r}')
    return ok


def _try_remove(self, item):
    """remove_ring_or_amulet() remembering a cursed-stuck item: every failed #remove costs a step but
    no game turn, so retrying it each tick would stall the episode on the no-progress timeout."""
    if self.remove_ring_or_amulet(item):
        self._known_stuck_ring_amulet_glyphs.discard(item.glyphs[0])
        return True
    self._known_stuck_ring_amulet_glyphs.add(item.glyphs[0])
    return False


@utils.debug_log('inventory.identify_amulet_by_wear')
@Strategy.wrap
def identify_amulet_by_wear(self):
    """Try an unidentified amulet on (strangulation/restful sleep/change are the bad ones). A cursed
    amulet cannot be #removed, so a strangulation one is only survivable by praying (pray.c:
    major trouble): a prayer must be available before the trial, and is the fallback."""
    if not jf_config.RING_AMULET_TRIAL or not _wear_allowed(self):
        yield False
    agent = self.agent
    prop = agent.character.prop
    if prop.blind or prop.confusion or prop.stun or prop.hallu:
        yield False
    if agent.blstats.hitpoints < cfg.AMULET_WEAR_MIN_HP_FRAC * agent.blstats.max_hitpoints:
        yield False
    if _hostile_within(self, cfg.AMULET_WEAR_SAFE_RADIUS):
        yield False
    if not _can_pray(agent):
        yield False

    stuck = self._known_stuck_ring_amulet_glyphs
    # "unidentified" is the TYPE (several objs possible); BUC status defaults to uncursed in the parser
    worn_unknown = next((i for i in self.items if i.category == nh.AMULET_CLASS and i.equipped
                          and not i.is_unambiguous() and i.glyphs[0] not in stuck), None)
    if worn_unknown is not None:
        yield True
        if not self._try_remove(worn_unknown) and _can_pray(agent, certain_death=True):
            _log(self, 'pray (stuck amulet)')
            agent.pray()
        return

    # a trial does not identify the amulet (most types have no on-wear message), so it stays
    # "unidentified" afterwards: without this memory the same amulet would be put on and taken
    # off again forever (one merged bot did that 466k times in five games)
    tested = self._tested_amulet_glyphs
    candidate = next((i for i in self.items if i.category == nh.AMULET_CLASS and not i.equipped
                       and not i.is_unambiguous() and i.glyphs[0] not in stuck
                       and i.glyphs[0] not in tested and buc_of(i) != 'cursed'
                       and not _puton_blocked(self, i) and _free_slot(self, i)), None)
    if candidate is None:
        yield False

    yield True
    tested.add(candidate.glyphs[0])
    self.put_on(candidate)
    if 'constricts your throat' in agent.message:
        _log(self, 'strangulation')
        self.items.update(force=True)
        worn = next((i for i in self.items if i.category == nh.AMULET_CLASS and i.equipped), None)
        if worn is not None and not self._try_remove(worn) and _can_pray(agent, certain_death=True):
            _log(self, 'pray (strangulation)')
            agent.pray()


@utils.debug_log('inventory.wear_combat_only_rings_amulets')
@Strategy.wrap
def wear_combat_only_rings_amulets(self):
    """Wear identified combat-only rings/amulets only while a hostile is close (they cost nutrition
    and do nothing between fights)."""
    if not jf_config.RING_COMBAT_ONLY_WEAR or not _wear_allowed(self):
        yield False

    combat_only = [i for i in self.items if _is_combat_only_item(i)]
    if not combat_only:
        yield False

    now = self.agent.blstats.time
    hostile_near = _hostile_within(self, cfg.ENGAGE_RADIUS)
    if hostile_near:
        self._last_hostile_near_turn = now
        candidate = next((i for i in combat_only if not i.equipped and _safe_to_put_on(self, i)
                          and _free_slot(self, i) and not _puton_blocked(self, i)), None)
        if candidate is not None:
            yield True
            self.put_on(candidate)
            return
    else:
        since = now - (self._last_hostile_near_turn if self._last_hostile_near_turn is not None else -10 ** 9)
        if since >= cfg.DISENGAGE_COOLDOWN:
            worn = next((i for i in combat_only if i.equipped and i.glyphs[0] in self._module_worn
                          and i.glyphs[0] not in self._known_stuck_ring_amulet_glyphs), None)
            if worn is not None:
                yield True
                self._try_remove(worn)
                return

    yield False


@utils.debug_log('inventory.shed_rings_amulets_when_hungry')
@Strategy.wrap
def shed_rings_amulets_when_hungry(self):
    """Hungry or worse: take off rings/amulets this module put on and has no specific reason to keep (not a known
    amulet of life saving, not slow digestion, not a combat-only item the other strategy manages), put them back
    once fed."""
    if not jf_config.RING_NUTRITION_REMOVE or not _wear_allowed(self):
        yield False

    hungry = self.agent.blstats.hunger_state >= Hunger.HUNGRY
    life_saving = O.from_name('amulet of life saving', nh.AMULET_CLASS)

    if hungry:
        def sheddable(i):
            if i.category not in (nh.RING_CLASS, nh.AMULET_CLASS) or not i.equipped:
                return False
            if i.glyphs[0] not in self._module_worn:
                return False  # PORT: never the TC / levitation rings the dive and the power route wear on purpose
            if i.status == Item.CURSED or i.glyphs[0] in self._known_stuck_ring_amulet_glyphs:
                return False
            if i.is_unambiguous() and (i.object == life_saving or i.object.name in cfg.NEVER_SHED):
                return False
            from .scroll_identify import _always_wear
            if _always_wear(i) and self.agent.blstats.hunger_state < Hunger.WEAK:
                return False  # free action & co: they save more than the 1/20 nutrition they cost
            return not _is_combat_only_item(i)

        candidate = next((i for i in self.items if sheddable(i)), None)
        if candidate is not None:
            yield True
            if self._try_remove(candidate):
                self._shed_for_hunger.add(candidate.text)
            return
    elif self._shed_for_hunger:
        text = next(iter(self._shed_for_hunger))
        candidate = next((i for i in self.items if not i.equipped and i.text == text), None)
        if candidate is not None and _free_slot(self, candidate) and not _puton_blocked(self, candidate):
            yield True
            self.put_on(candidate)
            self._shed_for_hunger.discard(text)
            return
        self._shed_for_hunger.discard(text)

    yield False


@utils.debug_log('inventory.wear_starting_rings')
@Strategy.wrap
def wear_starting_rings(self):
    """Wear the rings the game started us with (see jf_config.RING_STARTING_WEAR) and keep
    them on; a polymorph status takes them off again for good."""
    agent = self.agent
    if self._starting_ring_glyphs is None:  # snapshot once, from the first inventory of the game
        self._starting_ring_glyphs = ({i.glyphs[0] for i in self.items if i.category == nh.RING_CLASS}
                                      if agent.blstats.time <= 30 else set())
    if not jf_config.RING_STARTING_WEAR or not self._starting_ring_glyphs or not _wear_allowed(self):
        yield False

    stuck = self._known_stuck_ring_amulet_glyphs
    starting = [i for i in self.items if i.category == nh.RING_CLASS and i.glyphs[0] in self._starting_ring_glyphs
                and not _is_combat_only_item(i)]

    if agent.character.prop.polymorph:
        worn = next((i for i in starting if i.equipped and i.glyphs[0] not in stuck), None)
        if worn is not None:
            yield True
            _log(self, 'polymorph: starting rings off')
            self._try_remove(worn)
            return
        self._starting_ring_glyphs = set()  # never wear them again this game
        yield False

    if agent.blstats.hunger_state >= Hunger.HUNGRY or agent.character.prop.blind:
        yield False
    if _hostile_within(self, cfg.AMULET_WEAR_SAFE_RADIUS):
        yield False

    candidate = next((i for i in starting if not i.equipped and i.glyphs[0] not in stuck
                       and self._starting_ring_tries.get(i.glyphs[0], 0) < 2 and _free_slot(self, i)), None)
    if candidate is None:
        yield False

    yield True
    self._starting_ring_tries[candidate.glyphs[0]] = self._starting_ring_tries.get(candidate.glyphs[0], 0) + 1
    self.put_on(candidate)


def install(inventory_cls):
    """Attach the primitives and strategies to Inventory and add their per-episode state."""
    from . import scroll_identify as si
    for name, fn in (('read_scrolls_to_identify', si.read_scrolls_to_identify),
                     ('wear_identified_beneficial', si.wear_identified_beneficial),
                     ('put_on', put_on), ('remove_ring_or_amulet', remove_ring_or_amulet),
                     ('_try_remove', _try_remove), ('identify_amulet_by_wear', identify_amulet_by_wear),
                     ('wear_combat_only_rings_amulets', wear_combat_only_rings_amulets),
                     ('wear_starting_rings', wear_starting_rings),
                     ('shed_rings_amulets_when_hungry', shed_rings_amulets_when_hungry)):
        assert not hasattr(inventory_cls, name), f'{name} already defined on {inventory_cls}'
        setattr(inventory_cls, name, fn)

    orig_init = inventory_cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._last_hostile_near_turn = None
        self._shed_for_hunger = set()
        self._known_stuck_ring_amulet_glyphs = set()
        self._tested_amulet_glyphs = set()
        self._starting_ring_glyphs = None
        self._starting_ring_tries = {}
        self._last_scroll_read_turn = -10 ** 9
        self._wear_benefit_tries = {}
        self._module_worn = set()       # PORT: glyphs this module put on (the only ones it takes off again)
        self._puton_failures = {}       # PORT: glyph -> (failed put ons, turn of the last)

    inventory_cls.__init__ = __init__
