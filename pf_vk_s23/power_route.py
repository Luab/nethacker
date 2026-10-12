"""TC_ROUTE (team member 'power-route'): teleport control plus a level-teleport trigger from what the dive carries.

Why: every hub program stops at the castle (Dlvl 25-29, 0.647 at most). With teleport control, one controlled level
teleport anywhere in the Dungeons lands in the Valley (castle + 1, teleport.c find_hell) and one more from Gehennom
reaches its bottom-1 (LEVELPORT_DEEP answers 50: 0.78-0.81). The 61 real castle arrivals (breach/kititems.txt,
--reveal) carried a ring of teleport control 4 times and a scroll of teleportation 34 times (cursed 4), and used
none of it: the bot never reads unknown scrolls (identify is 18% of all scrolls -- jf16-s14's Dlvl-29 kit held
4 identify scrolls, the TC ring and a cursed teleport scroll) and never learns which ring gives TC.

NetHack 3.6.6 rules this relies on:
* read.c SCR_TELEPORTATION: read confused or cursed -> level_tele(); uncursed and unconfused -> scrolltele():
  with TC 'Where do you want to be teleported?' (ESC aborts, the scroll is identified either way), without TC a
  random teleport, on a noteleport level (the castle) 'A mysterious force prevents you from teleporting!'.
* teleport.c level_tele(): with TC 'To what level do you want to teleport?'; answered while Confused,
  rnl(5) != 0 (4 times in 5 at Luck 0) says 'Oops...' and makes it a random level teleport. So a sure controlled
  jump is a CURSED teleport scroll read unconfused, or a level teleporter trap (used up by the jump). No level
  teleport in Sokoban ('very disoriented'); from the Mines a request for 50 clamps to Mines' End (get_level).
* read.c SCR_CONFUSE_MONSTER: 'You feel confused.' for rnd(100) turns only from a cursed scroll (or when the form
  isn't S_HUMAN: the hero's form is the role's monster, a valkyrie '@', not the race -- u_init.c; an uncursed one
  said 'Your hands begin to glow red.' in harness pr-tca7b).
* read.c SCR_IDENTIFY: 'What would you like to identify first?' is a PICK_ANY menu of the not fully identified
  items, asked again ('next') while the count lasts (1 item 80% of the time for an uncursed scroll).
* trap.c tele_trap(): a teleportation trap is not used up; eat.c: a tengu corpse gives TC 1 time in 6 ('You feel
  in control of yourself.').

State lives on the agent (agent._proute). Entry points:
  note_message(agent)          every observation (agent.update): TC prompts, the tengu message, 'You feel confused'
  read_identify(agent) -> bool one identify-by-reading step (castle-first-pass calls it at the castle landing)
  levelport_strategy(agent)    preempt: the controlled jump (anywhere), the Valley's second jump, the castle gamble
"""

import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from . import objects as O
from .glyph import SS
from .level import Level
from .strategy import Strategy

TC_RING = O.from_name('teleport control', nh.RING_CLASS)
LEV_RING = O.from_name('levitation', nh.RING_CLASS)
TELE_SCROLL = O.from_name('teleportation', nh.SCROLL_CLASS)
IDENTIFY_SCROLL = O.from_name('identify', nh.SCROLL_CLASS)
CONFUSE_SCROLL = O.from_name('confuse monster', nh.SCROLL_CLASS)
SCARE_SCROLL = O.from_name('scare monster', nh.SCROLL_CLASS)
BLANK_SCROLL = O.from_name('blank paper', nh.SCROLL_CLASS)
LEV_POTION = O.from_name('levitation', nh.POTION_CLASS)
CONFUSION_POTION = O.from_name('confusion', nh.POTION_CLASS)
BOOZE_POTION = O.from_name('booze', nh.POTION_CLASS)
WISH_WAND = O.from_name('wishing', nh.WAND_CLASS)
WATER_POTION = O.from_name('water', nh.POTION_CLASS)

DUNGEONS_OF_DOOM, GEHENNOM = 0, 1

_TC_PROMPTS = ('Where do you want to be teleported?', 'To what level do you want to teleport?')
_TC_INTRINSIC = ('You feel in control of yourself.', 'You feel centered in your personal space.')
_MENU_LINE = re.compile(r'^([a-zA-Z$#]) - (.+)$')
_PAGE = re.compile(r'\((\d+) of (\d+)\)')


class RouteState:
    def __init__(self):
        self.tc_intrinsic = False     # a tengu corpse gave it
        self.tc_glyphs = set()        # ring appearance glyphs proven to give TC (the only unknown ring worn at a prompt)
        self.tc_sets = []             # frozensets of worn unknown ring glyphs at a TC prompt: one of them gives TC
        self.not_tc = set()           # ring glyphs proven not to (worn through a level teleport with no prompt)
        self.confuse_glyphs = set()   # scroll glyphs that confused us when read (a cursed confuse monster stack)
        self.confuse_stacks = set()   # (glyph, letter) of the rest of such a stack
        self.read_glyphs = {}         # scroll glyph -> turn it was last read by us
        self.reading = None           # the read in progress: dict(glyph, text, worn, step, turn, prompt, why)
        self.last_prompt_step = -10 ** 9
        self.levelports = []          # (turn, from (dnum, dlevel, depth), controlled, text)
        self.gamble_log = None        # last castle-gamble state logged
        self.altar_tried = {}         # level key -> signature of the items last BUC-tested on its altar
        self.altar_turn = -10 ** 9
        self.puton_block_until = -1   # no ring put-ons until this turn (a welded weapon keeps the hand busy)
        self.cooldown_until = -1      # the strategy stays out until this turn (actions that pass no game time)
        self.idle_actions = 0         # consecutive actions of ours that passed no game time
        self.ring_tested = set()      # ring glyphs tried on by the wear test
        self.no_pick_rounds = 0


def state(agent):
    st = getattr(agent, '_proute', None)
    if st is None:
        st = agent._proute = RouteState()
    return st


def _log(agent, msg):
    agent.log(f'PROUTE {msg}')


# ------------------------------------------------------------------------------------------------ items

def _items(agent):
    return list(agent.inventory.items)


def _glyph(item):
    return item.glyphs[0] if item.glyphs else None


def _words(item):
    return f' {item.text or ""} '


def known_cursed(item):
    """The display says cursed (the parser turns an unknown BUC into UNCURSED, so read the text)."""
    return ' cursed ' in _words(item) and ' uncursed ' not in _words(item)


def buc_known(item):
    w = _words(item)
    return ' cursed ' in w or ' uncursed ' in w or ' blessed ' in w or ' holy water' in w


def worn_rings(agent):
    return [i for i in _items(agent) if i.category == nh.RING_CLASS and i.equipped]


def is_tc_ring(agent, item):
    if item.category != nh.RING_CLASS:
        return False
    if item.is_unambiguous():
        return item.object == TC_RING
    return _glyph(item) in state(agent).tc_glyphs


def may_be_tc(agent, item):
    return item.category == nh.RING_CLASS and TC_RING in item.objs and _glyph(item) not in state(agent).not_tc


def tc_active(agent):
    st = state(agent)
    if st.tc_intrinsic:
        return True
    worn = worn_rings(agent)
    if any(is_tc_ring(agent, r) for r in worn):
        return True
    glyphs = {_glyph(r) for r in worn}
    return any(s <= glyphs for s in st.tc_sets)


def tc_known(agent):
    """TC is ours to switch on: the intrinsic, or a carried ring known (or proven) to give it."""
    st = state(agent)
    if st.tc_intrinsic or tc_active(agent):
        return True
    return any(is_tc_ring(agent, i) for i in _items(agent))


def tele_scrolls(agent):
    return [i for i in _items(agent) if i.category == nh.SCROLL_CLASS and i.is_unambiguous() and
            i.object == TELE_SCROLL]


def cursed_tele_scrolls(agent):
    """Known cursed scrolls of teleportation, plus WISH_TELEPORT_ROUTE's wished stack (cursed, BUC not shown)."""
    out = [i for i in tele_scrolls(agent) if known_cursed(i)]
    wished = getattr(agent, '_tele_letter', None)
    if wished is not None:
        for i in tele_scrolls(agent):
            if agent.inventory.items.get_letter(i) == wished and i not in out:
                out.append(i)
    return out


def unknown_scrolls(agent):
    """Top-level scrolls of an unknown type (a letter to read them by), blank paper excluded."""
    return [i for i in _items(agent) if i.category == nh.SCROLL_CLASS and not i.is_unambiguous()]


def _p(item, targets):
    total = sum(getattr(o, 'prob', 0) or 1 for o in item.objs)
    return sum(getattr(o, 'prob', 0) or 1 for o in item.objs if o in targets) / max(total, 1)


def _level_teleporters(agent):
    level = agent.current_level()
    return [(int(y), int(x)) for y, x in zip(*(level.objects == SS.S_level_teleporter).nonzero())]


def _teleporter_approach(agent):
    """((trap), (square next to it we can reach)) for the nearest known level teleporter, or None. bfs never enters a
    level-exit trap square (agent.LEVEL_EXIT_TRAPS), so walk to a neighbour and step on from there."""
    lts = _level_teleporters(agent)
    if not lts:
        return None
    dis = agent.bfs()
    bl = agent.blstats
    best = None
    for ty, tx in lts:
        for ny in range(ty - 1, ty + 2):
            for nx in range(tx - 1, tx + 2):
                if (ny, nx) == (ty, tx) or not (0 <= ny < dis.shape[0] and 0 <= nx < dis.shape[1]):
                    continue
                d = 0 if (ny, nx) == (bl.y, bl.x) else dis[ny, nx]
                if d == -1:
                    continue
                if best is None or d < best[0]:
                    best = (d, (ty, tx), (ny, nx))
    return None if best is None else (best[1], best[2])


def _hostiles_within(agent, radius):
    bl = agent.blstats
    return [m for m in agent.get_visible_monsters() if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= radius]


def _dive(agent):
    gl = getattr(agent, 'global_logic', None)
    return getattr(gl, 'dive', None)


def _on_castle(agent):
    dive = _dive(agent)
    if dive is None:
        return False
    key = dive.castle.castle_key
    return key is not None and agent.current_level().key() == key


def on_castle_level(agent):
    """The castle, or a level that may be it: the main line at depth 25+ other than Medusa's (the castle is known only
    after a failed dig, and the landing minotaur makes CASTLE_POLY zap before any dig: harness pr-polyx answered the
    form prompt with ESC twice). Below Medusa the other candidates are her level's maze neighbours, where a xorn is
    a good form too (AC -2, four attacks, walks through the maze walls)."""
    try:
        if agent.blstats.dungeon_number != DUNGEONS_OF_DOOM or agent.current_level().dungeon_number != \
                Level.DUNGEONS_OF_DOOM:
            return False
        if _on_castle(agent):
            return True
        dive = _dive(agent)
        return agent.blstats.depth >= 25 and dive is not None and not dive.on_medusa_level()
    except Exception:
        return False


def _jump_dungeon_ok(agent):
    """Where a controlled level teleport goes deep: the main line (-> the Valley) or Gehennom (-> bottom-1)."""
    return agent.blstats.dungeon_number in (DUNGEONS_OF_DOOM, GEHENNOM) and \
        agent.current_level().dungeon_number != Level.SOKOBAN


# ------------------------------------------------------------------------------------------------ messages

def note_message(agent):
    """agent.update, every observation: learn from TC prompts, the tengu intrinsic and read effects. No steps."""
    if not jf_config.TC_ROUTE:
        return
    msg = agent.single_message or ''
    if not msg:
        return
    try:
        st = state(agent)
        if any(p in msg for p in _TC_PROMPTS) and agent.step_count != st.last_prompt_step:
            st.last_prompt_step = agent.step_count
            if st.reading is not None:
                st.reading['prompt'] = True
            _tc_proven(agent, 'level' if 'To what level' in msg else 'teleport')
        if any(p in msg for p in _TC_INTRINSIC) and not st.tc_intrinsic:
            st.tc_intrinsic = True
            _log(agent, 'teleport control intrinsic (tengu corpse)')
        if 'You feel confused.' in msg and st.reading is not None and agent.step_count - st.reading['step'] <= 8 and \
                (st.reading['glyph'], st.reading.get('letter')) not in st.confuse_stacks:
            st.confuse_glyphs.add(st.reading['glyph'])
            st.confuse_stacks.add((st.reading['glyph'], st.reading.get('letter')))
            _log(agent, f'{st.reading["text"]!r} confused us: a cursed confuse monster scroll (read.c)')
        if 'Oops...' in msg and st.reading is not None and agent.step_count - st.reading['step'] <= 8:
            _log(agent, 'confused answer to the level prompt: Oops, a random level teleport')
    except Exception as e:   # learning must never break the step loop
        agent.log(f'PROUTE note_message failed: {e!r}')


def _tc_proven(agent, what):
    """A TC prompt came up: the intrinsic or a worn ring gives TC. Pin it on the only candidate ring if we can."""
    st = state(agent)
    if st.tc_intrinsic:
        return
    worn = worn_rings(agent)
    if any(r.is_unambiguous() and r.object == TC_RING for r in worn):
        return
    if st.reading is not None and agent.step_count - st.reading['step'] <= 8:
        glyphs = set(st.reading['worn'])
    else:
        glyphs = {_glyph(r) for r in worn}
    known_other = {_glyph(r) for r in worn if r.is_unambiguous() and r.object != TC_RING}
    cands = {g for g in glyphs if g is not None and g not in known_other and g not in st.not_tc}
    if cands & st.tc_glyphs:
        return
    if len(cands) == 1:
        g = next(iter(cands))
        st.tc_glyphs.add(g)
        _log(agent, f'{what} prompt: ring glyph {g} gives teleport control')
    elif cands:
        s = frozenset(cands)
        if s not in st.tc_sets:
            st.tc_sets.append(s)
            _log(agent, f'{what} prompt: one of ring glyphs {sorted(cands)} gives teleport control')
    else:
        # no candidate ring on: TC from somewhere we didn't see (a tengu eaten while the message scrolled by)
        st.tc_intrinsic = True
        _log(agent, f'{what} prompt with no candidate ring worn: teleport control is intrinsic')


# ------------------------------------------------------------------------------------------------ identify

def _id_value(agent, item):
    """How much an identify pick is worth for the ceiling: rings (teleport control, levitation) first, then what
    makes or finds a trigger (teleport scrolls and their BUC), then wands, lifts, the rest."""
    st = state(agent)
    cat = item.category
    if cat == nh.RING_CLASS:
        if not item.is_unambiguous():
            v = 100 + 40 * _p(item, {TC_RING}) * 28 + 20 * _p(item, {LEV_RING}) * 28
            return v - (60 if _glyph(item) in st.tc_glyphs else 0)
        return 20 if not buc_known(item) else 0
    if cat == nh.SCROLL_CLASS:
        if item.is_unambiguous():
            if item.object == TELE_SCROLL and not buc_known(item):
                return 95        # a cursed one is a sure controlled jump
            return 0
        return 50 + 10 * min(item.count, 3) + 60 * _p(item, {TELE_SCROLL})
    if cat == nh.WAND_CLASS:
        return 45 if not item.is_unambiguous() else 0
    if cat == nh.POTION_CLASS:
        if item.is_unambiguous():
            return 0
        return 35 + 40 * _p(item, {LEV_POTION, CONFUSION_POTION, BOOZE_POTION})
    if cat == nh.AMULET_CLASS:
        return 30 if not item.is_unambiguous() else 0
    if item.is_armor() and not item.is_unambiguous():
        return 25
    return 1 if not item.is_unambiguous() else 0


def _menu_entries(lines):
    out = []
    for line in lines:
        m = _MENU_LINE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _identify_step(agent, menu_pages):
    """One key for an identify menu page on screen: collect the pages, then select the best item and confirm.
    Returns the key(s) to send (a string) or None to decline (ESC)."""
    popup = agent.single_popup
    text = ' '.join(popup)
    m = _PAGE.search(text)
    page, pages = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
    by_letter = {}
    for it in _items(agent):
        try:
            by_letter[agent.inventory.items.get_letter(it)] = it
        except Exception:
            pass
    menu_pages[page] = [(l, by_letter[l]) for l, _ in _menu_entries(popup) if l in by_letter]
    if len(menu_pages) < pages and page < pages:
        return '>'
    best = None
    for pg, entries in menu_pages.items():
        for letter, it in entries:
            v = _id_value(agent, it)
            if v > 0 and (best is None or v > best[0]):
                best = (v, pg, letter, it)
    if best is None:
        return None
    _, pg, letter, it = best
    if pg != page:
        return '<' if pg < page else '>'
    _log(agent, f'identify pick {letter} {it.text!r} (value {best[0]:.0f})')
    return letter + '\r'


def _read(agent, item, why):
    """Read a scroll, answering what it may ask: the identify menu (_id_value picks), a stinking cloud's aim and a
    charging scroll's wand (declined). Level prompts are agent.update's (LEVELPORT_DEEP), naming prompts too."""
    st = state(agent)
    letter = agent.inventory.items.get_letter(item)
    st.reading = dict(glyph=_glyph(item), text=item.text, why=why, step=agent.step_count, turn=agent.blstats.time,
                      letter=letter,
                      worn=frozenset(_glyph(r) for r in worn_rings(agent)), prompt=False,
                      key=(agent.blstats.dungeon_number, agent.blstats.level_number, agent.blstats.depth))
    st.read_glyphs[_glyph(item)] = agent.blstats.time
    _log(agent, f'reading {item.text!r} ({letter}): {why}')
    menu_pages = {}

    def gen():
        if 'What do you want to read?' not in agent.single_message:
            return
        yield letter
        for _ in range(60):
            msg = agent.single_message or ''
            head = msg + ' ' + ' '.join(agent.single_popup[:3])
            if 'What would you like to identify' in head:
                keys = _identify_step(agent, menu_pages)
                if keys is None:
                    yield A.Command.ESC
                    continue
                if keys.endswith('\r'):
                    menu_pages.clear()
                for k in keys:
                    yield k
                continue
            if 'Where do you want to center' in msg or 'What do you want to charge?' in msg:
                yield A.Command.ESC
                continue
            misc = agent._observation['misc']
            if misc[2] and not misc[1]:
                yield ' '
                continue
            return

    with agent.atom_operation():
        agent.step(A.Command.READ, gen())
    _log(agent, f'read {st.reading["text"]!r}: {(agent.message or "")[:220]!r}')
    agent.inventory.items.update(force=True)


def known_identify(agent):
    """Scrolls of identify we know (a cursed one still identifies one item)."""
    return [i for i in _items(agent) if i.category == nh.SCROLL_CLASS and i.is_unambiguous() and
            i.object == IDENTIFY_SCROLL]


def _identify_candidates(agent, unknown=True):
    """Scrolls to read to identify things: known identify scrolls, then (unknown=True) unknown stacks, the largest
    first (identify is the most common scroll, 18%) -- never one known cursed that may be teleportation (a cursed
    teleport scroll without TC on is a random level teleport: up, off the castle), never a label we already read,
    never one that confused us."""
    st = state(agent)
    out = list(known_identify(agent))
    if not unknown:
        return out
    rest = []
    for it in unknown_scrolls(agent):
        g = _glyph(it)
        if g in st.read_glyphs or g in st.confuse_glyphs:
            continue
        if known_cursed(it) and TELE_SCROLL in it.objs:
            continue
        rest.append(it)
    rest.sort(key=lambda i: (-min(i.count, 4), -_p(i, {IDENTIFY_SCROLL})))
    return out + rest


def _worth_identifying(agent):
    # rings (teleport control, levitation), wands, potions (levitation for the lift plan, confusion for a trigger)
    return any(_id_value(agent, i) >= 35 for i in _items(agent) if i.category != nh.SCROLL_CLASS) or \
        any(i.category == nh.SCROLL_CLASS and i.is_unambiguous() and i.object == TELE_SCROLL and not buc_known(i)
            for i in _items(agent))


def read_identify(agent, force=False, unknown=True):
    """One identify-by-reading step (acted -> True). Safe to call from the castle's west courtyard before the lift
    plan: nothing with a hostile within 3 or below half HP, blind, confused, stunned, hallucinating, levitating
    (the lift plan's turn), in a shop, or polymorphed. Reads the likeliest identify stack while there is something
    worth identifying (unknown rings, wands, potions, a teleport scroll of unknown BUC)."""
    if not jf_config.TC_ROUTE:
        return False
    bl = agent.blstats
    prop = agent.character.prop
    if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph:
        return False
    if not force and (bl.hitpoints < 0.5 * bl.max_hitpoints or _hostiles_within(agent, 3)):
        return False
    if bl.carrying_capacity >= 4:   # Overtaxed: reading is refused without a turn passing
        return False
    level = agent.current_level()
    if level.shop_interior[bl.y, bl.x]:
        return False
    if int(agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & nh.BL_MASK_LEV and not force:
        return False
    if not _worth_identifying(agent):
        return False
    cands = _identify_candidates(agent, unknown=unknown)
    if not cands:
        return False
    item = cands[0]
    _read(agent, item, f'identify by reading (P(identify)={_p(item, {IDENTIFY_SCROLL}):.2f}, x{item.count})')
    return True


def _dive_id_ready(agent):
    """TC_DIVE_ID: identify by reading outside the castle, at a quiet moment: known identify scrolls, and unknown
    stacks of 2+ (identify is 18% of all scrolls and stacks lean to the common types), while something worth
    identifying is carried. The castle landing then knows whether it holds TC and a sure trigger (the landing
    minotaur kills in 3-5 turns)."""
    if not jf_config.TC_DIVE_ID:
        return None
    # only once the dive has begun: ~90% of the kit's scrolls and potions come from the Dlvl 1-3 grind anyway, and a
    # read during the grind reshuffled whole games (pr-tc15/15b: 14 of 15 jf16 games diverged, some at XL 1-2)
    dive = _dive(agent)
    if dive is None or not dive.diving:
        return None
    # cheap tests first: this runs on every step (the preempt chain)
    cands = list(known_identify(agent))
    cands += [i for i in _identify_candidates(agent) if not i.is_unambiguous() and i.count >= 2]
    if not cands:
        return None
    bl = agent.blstats
    level = agent.current_level()
    if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES) or _on_castle(agent):
        return None
    prop = agent.character.prop
    if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph:
        return None
    if bl.hitpoints < 0.6 * bl.max_hitpoints or bl.hunger_state >= 3 or bl.carrying_capacity >= 3:
        return None
    if level.shop_interior[bl.y, bl.x] or int(agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & nh.BL_MASK_LEV:
        return None
    if not _worth_identifying(agent):
        return None
    if _hostiles_within(agent, 6):
        return None
    return cands[0]


_ALTAR_CLASSES = (nh.RING_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS, nh.AMULET_CLASS)


def _altar_items(agent):
    """Carried rings, potions, scrolls and amulets whose BUC we don't know (the display shows no B/U/C word)."""
    out = []
    for it in _items(agent):
        if it.category not in _ALTAR_CLASSES or it.equipped or buc_known(it):
            continue
        if it.category == nh.SCROLL_CLASS and jf_config.SCARE_KEEP:
            continue   # SCARE_KEEP never drops a possible scare monster scroll (and never picks one up again)
        if it.can_be_dropped_from_inventory():
            out.append(it)
    return out


def _altar_ready(agent):
    """TC_ALTAR: a known reachable altar on this level and carried items of unknown BUC (pray.c/do.c doaltarobj: a
    drop shows a black flash for cursed, amber for blessed, and sets bknown either way). The identify_items_on_altar
    of the tour never fires: the parser turns an unknown BUC into UNCURSED, so no item has status UNKNOWN (base10:
    25 of 90 games stood on an altar, 0 BUC flashes). A known cursed teleport scroll is the sure trigger, a ring known
    uncursed can be tried on and taken off, cursed water is unholy water."""
    if not jf_config.TC_ALTAR:
        return None
    level = agent.current_level()
    if not level.altars or level.dungeon_number == Level.SOKOBAN:
        return None
    # the dive only (see _dive_id_ready): the grind at XL 1-4 walked to its altar for every new scroll (pr-tc15b-jf16
    # s10: 5 trips in 500 turns at XL 1-3)
    dive = _dive(agent)
    if dive is None or not dive.diving:
        return None
    items = _altar_items(agent)
    if not items:
        return None
    st = state(agent)
    bl = agent.blstats
    sig = frozenset((_glyph(i), i.count) for i in items)
    if st.altar_tried.get(level.key()) == sig or bl.time - st.altar_turn < 1000:
        return None
    prop = agent.character.prop
    if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph:
        return None
    if bl.hitpoints < 0.6 * bl.max_hitpoints or bl.hunger_state >= 3 or level.shop_interior[bl.y, bl.x]:
        return None
    if _hostiles_within(agent, 5):
        return None
    dis = agent.bfs()
    alts = [p for p in level.altars if p == (bl.y, bl.x) or dis[p] != -1]
    if not alts:
        return None
    return min(alts, key=lambda p: 0 if p == (bl.y, bl.x) else dis[p])


def _altar_step(agent, pos):
    st = state(agent)
    bl = agent.blstats
    if (bl.y, bl.x) != tuple(pos):
        _log(agent, f'to the altar at {tuple(pos)} to learn the BUC of {len(_altar_items(agent))} items')
        agent.go_to(*pos)
        return
    items = _altar_items(agent)
    st.altar_tried[agent.current_level().key()] = frozenset((_glyph(i), i.count) for i in items)
    st.altar_turn = bl.time
    _log(agent, f'BUC test on the altar: {[i.text for i in items]}')
    agent.inventory.drop(items)
    _log(agent, f'altar: {(agent.message or "")[:240]!r}')


def _ring_test_ready(agent):
    """TC_RING_TEST: an unknown ring whose BUC we know is not cursed (altar, identify) is tried on and taken off at a
    quiet moment of the dive: do_wear.c Ring_on names a levitation ring at once ('You start to float in the air!',
    learnring) -- the castle's lift plan then has a certain lift, and Medusa's islands too. Only non-cursed ones: a
    cursed teleportitis/polymorph/hunger/aggravate ring (90% of those are cursed) would stay on."""
    if not jf_config.TC_RING_TEST:
        return None
    dive = _dive(agent)
    if dive is None or not dive.diving or dive.in_gehennom() or _on_castle(agent):
        return None
    st = state(agent)
    cands = [i for i in _items(agent) if i.category == nh.RING_CLASS and not i.equipped and not i.is_unambiguous()
             and buc_known(i) and not known_cursed(i) and _glyph(i) not in st.ring_tested]
    if not cands:
        return None
    bl = agent.blstats
    prop = agent.character.prop
    if bl.time < st.puton_block_until or prop.polymorph or prop.confusion or prop.stun or prop.blind:
        return None
    if len(worn_rings(agent)) >= 2 or bl.hitpoints < 0.6 * bl.max_hitpoints:
        return None
    if int(agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & nh.BL_MASK_LEV:
        return None
    if _hostiles_within(agent, 6):
        return None
    return cands[0]


def _ring_test_step(agent, ring):
    st = state(agent)
    g = _glyph(ring)
    st.ring_tested.add(g)
    if not _put_on(agent, ring, 'wear test (a levitation ring names itself)'):
        return
    msg = agent.message or ''
    worn = next((r for r in worn_rings(agent) if _glyph(r) == g), None)
    if 'float in the air' in msg:
        _log(agent, f'ring glyph {g} is levitation: taking it off for the castle')
    if worn is not None:
        _remove(agent, worn)


# ------------------------------------------------------------------------------------------------ rings

def _put_on(agent, ring, why):
    """Put a ring on a free finger, freeing one first if needed (never a proven TC ring or a known cursed one)."""
    st = state(agent)
    if agent.hands_welded():
        st.puton_block_until = agent.blstats.time + 1000
        _log(agent, f'no ring on: the hands are welded ({ring.text!r})')
        return False
    worn = worn_rings(agent)
    if len(worn) >= 2:
        spare = next((r for r in worn if not is_tc_ring(agent, r) and not known_cursed(r) and
                      not (r.is_unambiguous() and r.object == LEV_RING)), None)
        if spare is None:
            _log(agent, f'no free finger for {ring.text!r}')
            return False
        _remove(agent, spare)
        return True
    letter = agent.inventory.items.get_letter(ring)
    left_used = any('(on left' in (r.text or '') for r in worn)

    def gen():
        if 'What do you want to put on?' not in agent.single_message:
            return
        yield letter
        if 'Which ring-finger' in agent.single_message:
            yield 'r' if left_used else 'l'

    _log(agent, f'putting on {ring.text!r}: {why}')
    with agent.atom_operation():
        agent.step(A.Command.PUTON, gen())
    msg = agent.message or ''
    _log(agent, f'put on: {msg[:160]!r}')
    if 'cannot free your weapon hand' in msg or 'cannot remove your' in msg or "You can't" in msg:
        # do_wear.c: a welded weapon keeps the weapon hand's finger (both, for a two-hander) -- no game time passes,
        # and the castle gamble retried it 31,060 times (pr-tc15-jf14 s6: an unknown-cursed pick-axe in hand)
        st.puton_block_until = agent.blstats.time + 1000
        _log(agent, 'ring put-ons blocked for 1000 turns')
    agent.inventory.items.update(force=True)
    return True


def _remove(agent, ring):
    letter = agent.inventory.items.get_letter(ring)

    def gen():
        if 'What do you want to remove?' in agent.single_message:
            yield letter

    with agent.atom_operation():
        agent.step(A.Command.REMOVE, gen())
    _log(agent, f'remove {ring.text!r}: {(agent.message or "")[:120]!r}')
    agent.inventory.items.update(force=True)


def _tc_ring_to_wear(agent):
    """A carried, unworn ring that switches TC on: a proven one, or the rest of a proven set."""
    st = state(agent)
    worn = {_glyph(r) for r in worn_rings(agent)}
    for it in _items(agent):
        if it.category == nh.RING_CLASS and not it.equipped and is_tc_ring(agent, it):
            return it
    for s in st.tc_sets:
        missing = s - worn
        if missing and len(worn) + len(missing) <= 2:
            g = next(iter(missing))
            ring = next((i for i in _items(agent) if i.category == nh.RING_CLASS and not i.equipped and
                         _glyph(i) == g), None)
            if ring is not None:
                return ring
    return None


def _spare_ring(agent):
    """A worn ring we may take off for a TC candidate: known to be something else, and not known cursed."""
    st = state(agent)
    for r in worn_rings(agent):
        if known_cursed(r) or is_tc_ring(agent, r):
            continue
        if (r.is_unambiguous() and r.object != TC_RING) or _glyph(r) in st.not_tc:
            return r
    return None


def _wear_tc_ready(agent):
    """A ring known (or proven) to give TC, not worn: put it on and keep it on -- every level teleporter stepped on is
    then a Valley ticket (LEVELPORT_DEEP) and teleport traps/teleportitis only ask (ESC keeps us in place)."""
    if not jf_config.TC_WEAR or tc_active(agent) or agent.character.prop.polymorph:
        return None
    if agent.blstats.time < state(agent).puton_block_until:
        return None
    ring = _tc_ring_to_wear(agent)
    if ring is None:
        return None
    if len(worn_rings(agent)) >= 2 and _spare_ring(agent) is None:
        return None
    if _hostiles_within(agent, 2):
        return None
    return ring


def _gamble_rings(agent):
    """Unknown rings that may give TC, best first: not known cursed (it would stay on a finger), then untried."""
    st = state(agent)
    rings = [i for i in _items(agent) if i.category == nh.RING_CLASS and not i.equipped and may_be_tc(agent, i) and
             not i.is_unambiguous()]
    rings.sort(key=lambda i: (known_cursed(i), -_p(i, {TC_RING})))
    return rings


# ------------------------------------------------------------------------------------------------ triggers

def _read_trigger(agent, scroll, why):
    st = state(agent)
    bl = agent.blstats
    before = (bl.dungeon_number, bl.level_number, bl.depth)
    try:
        _read(agent, scroll, why)
    finally:
        # (in a finally: the new level's first update may switch strategies out of the read)
        after = agent.blstats
        now = (after.dungeon_number, after.level_number, after.depth)
        controlled = bool(st.reading and st.reading['prompt'])
        if now != before:
            st.levelports.append((after.time, before, controlled, why))
            _log(agent, f'LEVELPORT {"controlled" if controlled else "random"}: {before} -> {now}')
            if not controlled:
                # a level teleport with no prompt: none of the rings worn then gives TC
                for g in (st.reading['worn'] if st.reading else ()):
                    if g not in st.tc_glyphs:
                        st.not_tc.add(g)


def _unholy_water(agent):
    """Known unholy water (cursed water: a dip curses what is dipped, the whole stack)."""
    return [i for i in _items(agent) if i.category == nh.POTION_CLASS and i.is_unambiguous() and
            i.object == WATER_POTION and (known_cursed(i) or 'unholy' in _words(i))]


def _unknown_water(agent):
    """Water of unknown BUC: 1 in 8 is unholy (mksobj blessorcurse(4)), 3 in 4 plain -- plain water blanks a scroll."""
    return [i for i in _items(agent) if i.category == nh.POTION_CLASS and i.is_unambiguous() and
            i.object == WATER_POTION and not buc_known(i)]


def _dip(agent, item, potion, why):
    s_letter = agent.inventory.items.get_letter(item)
    p_letter = agent.inventory.items.get_letter(potion)
    _log(agent, f'dipping {item.text!r} ({s_letter}) into {potion.text!r} ({p_letter}): {why}')

    def gen():
        if 'What do you want to dip?' not in (agent.single_message or ''):
            return
        yield s_letter
        for _ in range(10):
            msg = agent.single_message or ''
            if 'into the fountain?' in msg or 'into the water?' in msg or 'into the pool?' in msg:
                yield 'n'
                continue
            if 'What do you want to dip' in msg and 'into?' in msg:
                yield p_letter
                continue
            misc = agent._observation['misc']
            if misc[2] and not misc[1]:
                yield ' '
                continue
            return

    with agent.atom_operation():
        agent.step(A.Command.DIP, gen())
    _log(agent, f'dip: {(agent.message or "")[:200]!r}')
    agent.inventory.items.update(force=True)


def _dip_ready(agent, gamble=False):
    """TC known, teleport scrolls not known to be cursed, and unholy water (or, gamble: water of unknown BUC)."""
    if not jf_config.TC_DIP or not tc_known(agent) or agent.character.prop.polymorph:
        return None
    scrolls = [i for i in tele_scrolls(agent) if not known_cursed(i)]
    if not scrolls or cursed_tele_scrolls(agent):
        return None
    waters = _unholy_water(agent) or (_unknown_water(agent) if gamble else [])
    if not waters:
        return None
    return max(scrolls, key=lambda i: i.count), waters[0]


def _confusion_sources(agent):
    """(kind, item): known ways to get confused -- a stack that confused us before (cursed confuse monster) or a
    known potion of confusion or booze."""
    st = state(agent)
    out = []
    for it in _items(agent):
        # the rest of a stack that confused us is cursed too (only same-BUC scrolls stack); another stack of that
        # label may be uncursed ('Your hands begin to glow')
        if it.category == nh.SCROLL_CLASS and \
                ((_glyph(it), agent.inventory.items.get_letter(it)) in st.confuse_stacks or
                 (it.is_unambiguous() and it.object == CONFUSE_SCROLL and known_cursed(it))):
            out.append(('read', it))
        elif it.category == nh.POTION_CLASS and it.is_unambiguous() and it.object in (CONFUSION_POTION, BOOZE_POTION):
            out.append(('quaff', it))
    return out


def _get_confused(agent, why):
    srcs = _confusion_sources(agent)
    if not srcs:
        return False
    kind, it = srcs[0]
    if kind == 'read':
        _read(agent, it, f'get confused for a level teleport: {why}')
    else:
        _log(agent, f'quaffing {it.text!r} to get confused: {why}')
        agent.inventory.quaff(it)
    return True


# ------------------------------------------------------------------------------------------------ strategy

def _jump_step(agent):
    """TC known: the controlled jump. Cursed teleport scroll (unconfused), a level teleporter on this level; in the
    Valley (dies within ~60 turns: vr-base) the confused 1-in-5 lottery too."""
    st = state(agent)
    prop = agent.character.prop
    if not _jump_dungeon_ok(agent) or prop.stun or prop.polymorph or _wish_route_pending(agent):
        return False
    ring = None if tc_active(agent) else _tc_ring_to_wear(agent)
    if not tc_active(agent) and ring is None:
        return False
    cursed = cursed_tele_scrolls(agent)
    lts = _level_teleporters(agent)
    dive = _dive(agent)
    valley = dive is not None and dive.in_gehennom()
    lottery = jf_config.TC_LOTTERY and tele_scrolls(agent) and (_confusion_sources(agent) or prop.confusion) and \
        _lottery_place(agent)
    if not cursed and not lts and not lottery:
        return False
    if ring is not None:
        if agent.blstats.time < st.puton_block_until:
            return False
        _put_on(agent, ring, 'teleport control for the jump')
        return True
    if prop.confusion:
        if cursed or lts:
            # confused, the level prompt's answer is random 4 times in 5 (Oops): wait it out for the sure jump
            return False
        scroll = tele_scrolls(agent)[0]
        _read_trigger(agent, scroll, 'confused teleport scroll with TC: 1 in 5 controlled')
        return True
    if cursed:
        _read_trigger(agent, cursed[0], 'cursed teleport scroll with TC: controlled level teleport')
        return True
    if lts:
        target = _teleporter_approach(agent)
        if target is not None:
            (ty, tx), (ny, nx) = target
            bl = agent.blstats
            if (bl.y, bl.x) != (ny, nx):
                _log(agent, f'walking to the level teleporter at {(ty, tx)} with TC (via {(ny, nx)})')
                agent.go_to(ny, nx)
            else:
                _log(agent, f'stepping onto the level teleporter at {(ty, tx)} with TC')
                agent.move(ty, tx)
            return True
    if lottery and _get_confused(agent, 'TC known, uncursed teleport scrolls only'):
        return True
    return False


def _castle_gamble_step(agent):
    """The castle after the lift plan gave up (castle_logic.given_up): dying here costs nothing and an uncontrolled
    level teleport only goes up. Identify by reading, put candidate rings on, then every teleport trigger."""
    dive = _dive(agent)
    if not _castle_gamble_on(agent):
        return False
    st = state(agent)
    prop = agent.character.prop
    if prop.polymorph or prop.stun:
        return False
    if st.gamble_log is None:
        st.gamble_log = agent.blstats.time
        _log(agent, f'castle gamble starts (lift plan gave up: {getattr(dive.castle, "_given_up_why", "?")!r})')
    # 1. known identify scrolls first: the menu names the TC ring and a teleport scroll's BUC. Unknown scrolls wait
    # until the rings are on: one may be a cursed teleport scroll, which without TC on is a random level teleport
    # (pr-tca7: 4 of 5 salts read the cursed ZLORFIK stack before the clay TC ring was on and left for Dlvl 10-28)
    if not prop.confusion and read_identify(agent, force=True, unknown=False):
        return True
    # 2. rings that may give TC on (two fingers; one known to be something else makes room)
    if not tc_active(agent) and agent.blstats.time >= st.puton_block_until:
        cands = _gamble_rings(agent)
        ring = _tc_ring_to_wear(agent) or (cands[0] if cands else None)
        if ring is not None:
            worn = worn_rings(agent)
            if len(worn) < 2:
                _put_on(agent, ring, 'castle gamble: a ring that may give TC')
                return True
            spare = _spare_ring(agent)
            if spare is not None:
                _log(agent, f'castle gamble: {spare.text!r} is not TC, making room for {ring.text!r}')
                _remove(agent, spare)
                return True
    # 3. the triggers (TC known or not: an uncontrolled jump from here costs nothing we still had)
    if not prop.confusion and cursed_tele_scrolls(agent):
        _read_trigger(agent, cursed_tele_scrolls(agent)[0], 'castle gamble: cursed teleport scroll')
        return True
    if tele_scrolls(agent):
        if prop.confusion:
            _read_trigger(agent, tele_scrolls(agent)[0], 'castle gamble: teleport scroll read confused')
            return True
        if _get_confused(agent, 'castle gamble'):
            return True
    # 3b. TC known, uncursed teleport scrolls, no confusion: water of unknown BUC is unholy 1 time in 8
    dip = _dip_ready(agent, gamble=True)
    if dip is not None and not prop.confusion:
        _dip(agent, dip[0], dip[1], 'castle gamble: TC known, water of unknown BUC may be unholy')
        return True
    # 4. unknown scrolls, rings on: identify-likely stacks first, then the likeliest teleport scrolls (a cursed one is
    # a level teleport from here, controlled if a worn ring gives TC; an uncursed one is identified by the castle's
    # 'mysterious force')
    if not prop.confusion and not prop.blind:
        left = [i for i in unknown_scrolls(agent) if _glyph(i) not in st.confuse_glyphs]
        if _gamble_rings(agent) and not tc_active(agent):
            # more TC candidates than fingers: an identify scroll (likelier in a bigger stack) names them first
            left.sort(key=lambda i: (-min(i.count, 4), -_p(i, {IDENTIFY_SCROLL})))
        else:
            left.sort(key=lambda i: (not known_cursed(i), -_p(i, {TELE_SCROLL})))
        if left:
            _read(agent, left[0], 'castle gamble: unknown scroll (identify, or a teleport trigger)')
            return True
    return False


def _ready(agent):
    if agent.blstats.time < state(agent).cooldown_until:
        return False
    try:
        return bool(_jump_ready(agent) or _dip_ready(agent) or _wear_tc_ready(agent) or _gamble_ready(agent) or
                    _altar_ready(agent) or _dive_id_ready(agent) or _ring_test_ready(agent))
    except Exception as e:   # a readiness check must never break the preempt loop
        agent.log(f'PROUTE check failed: {e!r}')
        return False


def _one_step(agent):
    """The first thing to do, in priority order; True if we acted."""
    if _jump_step(agent):
        return True
    dip = _dip_ready(agent)
    if dip is not None:
        _dip(agent, dip[0], dip[1], 'TC known: unholy water curses the teleport stack (sure controlled jumps)')
        return True
    ring = _wear_tc_ready(agent)
    if ring is not None:
        worn = worn_rings(agent)
        if len(worn) >= 2:
            _remove(agent, _spare_ring(agent))
        else:
            _put_on(agent, ring, 'teleport control, kept on')
        return True
    if _castle_gamble_step(agent):
        return True
    altar = _altar_ready(agent)
    if altar is not None:
        _altar_step(agent, altar)
        return True
    scroll = _dive_id_ready(agent)
    if scroll is not None:
        _read(agent, scroll, f'identify by reading at a quiet moment (x{scroll.count})')
        return True
    ring = _ring_test_ready(agent)
    if ring is not None:
        _ring_test_step(agent, ring)
        return True
    return False


def levelport_strategy(agent):
    """Preempt (above the castle crossing): the controlled jump whenever TC and a sure trigger are in hand; the
    castle gamble once the lift plan has given up. The body loops while there is work (like HOLD_LOOP): after a
    one-action return agent.preempt runs one step of the lower chain, and on the castle that step was
    CASTLE_SCARE dropping the unknown scrolls -- the cursed teleport stack among them -- at a master lich
    (harness pr-tca7b tca~5)."""
    def f():
        if not jf_config.TC_ROUTE or not _ready(agent):
            yield False
            return
        yield True
        st = state(agent)
        for _ in range(40):
            steps, turn0 = agent.step_count, agent.blstats.time
            if not _one_step(agent):
                return
            # watchdog (kept across calls): actions that pass no game time (a refused put-on, a misread menu) must
            # not spin -- pr-tc15-jf14 s6 repeated a refused put-on 31,060 times
            if agent.blstats.time == turn0:
                st.idle_actions += 1
                if st.idle_actions >= 8:
                    st.idle_actions = 0
                    st.cooldown_until = agent.blstats.time + 30
                    _log(agent, 'no game time passed in 8 actions: stepping aside for 30 turns')
                    return
            else:
                st.idle_actions = 0
            if agent.step_count == steps or not _ready(agent):
                return

    return Strategy(f)


def _wish_route_pending(agent):
    """WISH_TELEPORT_ROUTE (tele_route.py) is still collecting wishes from a wand in the Dungeons: it reads both scrolls
    in one go (wresting the last charge takes up to hundreds of zaps -- not in the Valley), so don't jump ahead."""
    try:
        from . import tele_route
    except ImportError:
        return False
    return agent.blstats.dungeon_number == DUNGEONS_OF_DOOM and tele_route.wishing_wand(agent) is not None and \
        tele_route.route_wish(agent) is not None


def _jump_ready(agent):
    """Cheap test for _jump_step (no steps): TC known and a trigger in hand."""
    if not _jump_dungeon_ok(agent) or agent.character.prop.stun or agent.character.prop.polymorph:
        return False
    if _wish_route_pending(agent):
        return False
    if not tc_known(agent):
        return False
    if agent.character.prop.confusion and (cursed_tele_scrolls(agent) or _level_teleporters(agent)):
        return False
    if cursed_tele_scrolls(agent):
        return True
    if _level_teleporters(agent) and _teleporter_approach(agent) is not None:
        return True
    if jf_config.TC_LOTTERY and tele_scrolls(agent) and \
            (_confusion_sources(agent) or agent.character.prop.confusion):
        dive = _dive(agent)
        return _lottery_place(agent)
    return False


def _lottery_place(agent):
    """Where the 1-in-5 confused jump may be tried: Gehennom (the Valley kills in ~60 turns), the castle only once the
    lift plan is over (a random level teleport would end it: from the bottom level it only goes up), and the main
    line from TC_LOTTERY_DEPTH down."""
    dive = _dive(agent)
    if dive is not None and dive.in_gehennom():
        return True
    if _on_castle(agent):
        return _castle_gamble_on(agent)
    return agent.blstats.depth >= jf_config.TC_LOTTERY_DEPTH


def _castle_gamble_on(agent):
    """On the castle with the lift plan over: given up, or no passage logic at all (CASTLE_PASSAGE off)."""
    dive = _dive(agent)
    if not jf_config.TC_CASTLE_GAMBLE or dive is None or not _on_castle(agent):
        return False
    return dive.castle.given_up or not jf_config.CASTLE_PASSAGE


def _gamble_ready(agent):
    if not _castle_gamble_on(agent):
        return False
    st = state(agent)
    prop = agent.character.prop
    if prop.polymorph or prop.stun:
        return False
    if not prop.confusion and known_identify(agent) and _worth_identifying(agent):
        return True
    if not tc_active(agent) and (_gamble_rings(agent) or _tc_ring_to_wear(agent)) and \
            (len(worn_rings(agent)) < 2 or _spare_ring(agent) is not None) and \
            agent.blstats.time >= st.puton_block_until:
        return True
    if not prop.confusion and cursed_tele_scrolls(agent):
        return True
    if tele_scrolls(agent) and (prop.confusion or _confusion_sources(agent)):
        return True
    if _dip_ready(agent, gamble=True) is not None and not prop.confusion:
        return True
    if not prop.confusion and not prop.blind and \
            [i for i in unknown_scrolls(agent) if _glyph(i) not in st.confuse_glyphs]:
        return True
    return False


# ------------------------------------------------------------------------------------------------ wands

VANISH_WANDS = frozenset(('cancellation', 'teleportation', 'make invisible'))


def vanish_wand(agent):
    """An unknown wand that may be teleportation, for castle_cross.blocker_zap's last resort at a moat blocker (a shark
    in the moat ring's one-wide channel): the engrave test's 'The engraving on the floor vanishes!' leaves exactly
    cancellation / teleportation / make invisible (inventory._determine_possible_wands). Zapped at a monster, a wand of
    teleportation sends it elsewhere on the level -- zap.c u_teleport_mon() -> rloc() works on no-teleport levels like
    the castle, which stop only the hero -- and we see it vanish. Of the other two, cancellation does nothing to a
    shark's bite and make invisible hides it (the bad third). pr-a25s1-m1: the kit's wand of teleportation stayed
    unknown after its 'vanishes' test while a demilich spent 7 moat turns on a shark at (0,3) with an emptied wand of
    striking and then melee. Returns the candidate with the fewest possibilities, or None."""
    best = None
    for it in _items(agent):
        if it.category != nh.WAND_CLASS or it.is_unambiguous():
            continue
        names = {getattr(o, 'name', '') for o in it.objs}
        if 'teleportation' not in names or not names <= VANISH_WANDS:
            continue
        if agent.inventory.is_known_empty(it) or it.comment == 'EMPT':
            continue
        if best is None or len(names) < len(best[0]):
            best = (names, it)
    return None if best is None else best[1]
