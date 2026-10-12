"""Read unidentified scrolls at a safe moment to identify rings and amulets (a scroll of identify is 18% of
all scrolls), then wear the identified ones worth wearing. Written by CleverShovel from NetHack 3.6.6 read.c, invent.c
(identify_pack/menu_identify), spell.c (losespells), mkobj.c (blessorcurse(otmp, 4) for scrolls) and the
scroll probability table in objects/data.py (CleverShovel/nethacker@29ab0a7 version: identify-menu paging read off the
screen, harm gate, stack odds).

Risk model (per read of a scroll whose type is one of the still-possible candidates):
  * hard gates make the physical dangers survivable: earth drops a boulder for up to d20 (no metal helmet),
    a blessed fire explodes on us for 15-25, a cursed create monster brings 13 monsters, a cursed genocide
    (we answer ESC) creates several -> full HP, at least MIN_HP, nobody in sight, a prayer available;
  * what no gate removes is weighed with the real odds: punishment (uncursed/cursed only), a cursed create
    monster / genocide, and amnesia (forget(): the current level's map, 1 in 3 for 0-24% of the level maps
    and of the object discoveries, and losespells(): rn2(n + 1) of the n known spells);
  * a read happens only when P(identify) x value >= SEVERE_COST x P(severe outcome), and (harm gate) only when the
    harmful/wasteful outcomes (HARMFUL) are at most SCROLL_MAX_HARM of the still possible types."""
import re

import nle.nethack as nh
from nle.nethack import actions as A

from .. import jf_config, utils
from ..strategy import Strategy
from . import ring_amulet_config as cfg
from .item import Item
from .ring_amulet_logic import _log

P_CURSED = 0.125   # mkobj.c: blessorcurse(otmp, 4) = 1 in 4 to be blessed or cursed, half each
P_BLESSED = 0.125


def _screen(agent, limit=900):
    """The non-empty rows of the terminal (the identify menu's page marker, diagnostics)."""
    rows = [bytes(r).decode(errors='replace').rstrip() for r in agent._observation['tty_chars']]
    return ' / '.join(r.strip() for r in rows if r.strip())[:limit]


def scroll_odds(candidates, count=1):
    """{scroll name: probability} over the still possible scroll objects, weighted by generation probability.
    A stack of `count` scrolls of one appearance is `count` independent draws of the same type, so the
    weight is prob ** count (a pair is 44% identify, a triple 70%, against 18.5% for a single scroll)."""
    k = max(1, min(int(count), 3))
    total = sum(o.prob ** k for o in candidates)
    return {o.name: o.prob ** k / total for o in candidates} if total else {}


# outcomes that cost something or waste the scroll (destroy armor takes a cloak/armor/helm/shield; scare monster
# is the Castle's scroll; teleportation moves us blind): 30% of an unpriced single scroll in the xid11/xid13 logs
HARMFUL = ('destroy armor', 'amnesia', 'fire', 'earth', 'punishment', 'create monster', 'genocide',
           'scare monster', 'teleportation')


def harm_mass(odds):
    return sum(odds.get(n, 0.0) for n in HARMFUL)


def amnesia_weight(role, character_cls):
    """How much an amnesia hurts: the map and some discoveries for everyone (a little), plus the spells for
    the casters (losespells: uniform 0..n of the n known spells; a Wizard has one, force bolt, his weapon)."""
    if role == character_cls.WIZARD:
        return cfg.AMNESIA_WEIGHT_WIZARD
    if role in (character_cls.PRIEST, character_cls.HEALER, character_cls.MONK):
        return cfg.AMNESIA_WEIGHT_CASTER
    return cfg.AMNESIA_WEIGHT_PLAIN


def severe_risk(odds, amnesia_w):
    """Probability-weighted cost of the outcomes no gate removes (see module docstring)."""
    p = odds.get
    punishment = p('punishment', 0.0) * (1 - P_BLESSED)          # blessed/confused only 'You feel guilty'
    cursed_summon = (p('create monster', 0.0) + p('genocide', 0.0)) * P_CURSED
    return punishment + cursed_summon + p('amnesia', 0.0) * amnesia_w


def wants_read(odds, has_target, amnesia_w):
    """(read?, P(identify), severe risk) -- the decision rule of the module docstring."""
    p_id = odds.get('identify', 0.0)
    sev = severe_risk(odds, amnesia_w)
    value = 1.0 if has_target else 0.0
    ok = p_id >= cfg.SCROLL_MIN_P_IDENTIFY and p_id * value >= cfg.SCROLL_SEVERE_COST * sev
    if ok and cfg.SCROLL_MAX_HARM is not None and harm_mass(odds) > cfg.SCROLL_MAX_HARM:
        ok = False
    return ok, p_id, sev


def _unidentified_wearables(inv):
    return [i for i in inv.items if i.category in (nh.RING_CLASS, nh.AMULET_CLASS) and not i.is_unambiguous()]


def _scroll_candidates(inv):
    from .. import objects as O
    out = []
    for i in inv.items:
        if i.category != nh.SCROLL_CLASS:
            continue
        objs = [o for o in i.objs if isinstance(o, O.Scroll)]
        if objs:
            out.append((i, objs))
    return out


def _safe_moment(inv):
    """Full HP, nobody in sight, a prayer available, nothing wrong with us, not where a scroll costs extra."""
    from ..glyph import G
    from ..level import Level
    from .ring_amulet_logic import _allowed, _can_pray
    agent = inv.agent
    bl = agent.blstats
    if not _allowed(inv, jf_config.RING_SCROLL_ROLES):
        return False
    if bl.hitpoints < max(cfg.SCROLL_MIN_HP, cfg.SCROLL_MIN_HP_FRAC * bl.max_hitpoints):
        return False
    prop = agent.character.prop
    if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph:
        return False
    if agent.get_visible_monsters():
        return False
    if agent.current_level().dungeon_number == Level.SOKOBAN:  # earth there costs Luck (sokoban_guilt)
        return False
    if utils.isin(agent.glyphs, G.SHOPKEEPER).any():
        return False
    if any(getattr(i, 'shop_status', Item.NOT_SHOP) != Item.NOT_SHOP for i in inv.items):
        return False
    if bl.time - inv._last_scroll_read_turn < cfg.SCROLL_READ_COOLDOWN:
        return False
    return _can_pray(agent)


@utils.debug_log('inventory.read_scrolls_to_identify')
@Strategy.wrap
def read_scrolls_to_identify(self):
    if not jf_config.RING_SCROLL_IDENTIFY:
        yield False
    from ..character import Character
    agent = self.agent
    targets = _unidentified_wearables(self)
    if not targets:
        yield False
    if not _safe_moment(self):
        yield False

    amn_w = amnesia_weight(agent.character.role, Character)
    best = None
    for item, objs in _scroll_candidates(self):
        ok, p_id, sev = wants_read(scroll_odds(objs, item.count if cfg.SCROLL_STACK_ODDS else 1), True, amn_w)
        if ok and (best is None or p_id > best[1]):
            best = (item, p_id, sev)
    if best is None:
        yield False

    scroll, p_id, sev = best
    # amulets first (a cursed strangulation/restful sleep one is the dangerous find), then rings
    order = sorted(targets, key=lambda i: (i.category != nh.AMULET_CLASS,))
    letters = [self.items.get_letter(i) for i in order]
    scroll_letter = self.items.get_letter(scroll)
    before = {i.glyphs[0] for i in targets}

    trace = {'exit': '', 'menu': 0, 'screen': ''}

    def gen():
        if 'What do you want to read?' not in agent.single_message:
            trace['exit'] = 'no read prompt'
            return
        yield scroll_letter
        chosen, guard = set(), 0
        while guard < 40:
            guard += 1
            text = '\n'.join(list(agent.popup) + list(agent.single_popup)) + '\n' + agent.single_message
            if 'Where do you want to center' in text or 'What do you want to charge' in text:
                yield A.Command.ESC          # stinking cloud / blessed fire / charging: decline
                return
            if 'What would you like to identify' not in text:
                # the identify menu comes after "This is an identify scroll." and its --More--: keep
                # paging until it shows (a scroll that is not identify ends without a menu)
                if b'--More--' in bytes(agent._observation['tty_chars'].reshape(-1)):
                    yield A.TextCharacters.SPACE
                    continue
                trace['exit'] = 'no menu'
                trace['screen'] = _screen(agent)
                return
            trace['menu'] += 1
            pick = next((l for l in letters if l not in chosen and f'{l} - ' in text), None)
            if pick is not None:
                chosen.add(pick)
                yield pick
                yield A.MiscAction.MORE
                continue
            # the page marker is cut out of agent.popup, so read it off the screen: the rings and amulets come
            # last in the menu and usually sit on page 2 ('(1 of 2)')
            m = re.search(r'\((\d+) of (\d+)\)', _screen(agent, 4000))
            if m and int(m.group(1)) < int(m.group(2)):
                yield '>'
                continue
            trace['exit'] = 'no target in menu'
            trace['screen'] = _screen(agent)
            yield A.Command.ESC              # nothing to pick here: the scroll is spent
            return

    yield True
    self._last_scroll_read_turn = agent.blstats.time
    with agent.atom_operation():
        agent.step(A.Command.READ, gen())
    self.items.update(force=True)
    after = {i.glyphs[0] for i in _unidentified_wearables(self)}
    tail = ' | '.join(agent._message_history[-10:])[-260:]
    _log(self, f'scroll_read p_id={p_id:.2f} severe={sev:.3f} identified={len(before - after)} msgs={tail!r}')
    if cfg.SCROLL_DEBUG and not (before - after):
        _log(self, f"scroll_dbg exit={trace['exit']!r} menu_pages={trace['menu']} screen={trace['screen']!r}")


def _always_wear(item):
    if not item.is_unambiguous() or item.status == Item.CURSED:
        return False
    if item.category == nh.RING_CLASS:
        return item.object.name in cfg.ALWAYS_WEAR_RINGS
    if item.category == nh.AMULET_CLASS:
        return item.object.name in cfg.ALWAYS_WEAR_AMULETS
    return False


@utils.debug_log('inventory.wear_identified_beneficial')
@Strategy.wrap
def wear_identified_beneficial(self):
    """Wear an identified ring/amulet that helps whenever it is worn (see cfg.ALWAYS_WEAR_*); the
    combat-only ones are the other strategy's."""
    from .ring_amulet_logic import _wear_allowed, _hostile_within, _safe_to_put_on, _puton_blocked
    if not jf_config.RING_WEAR_IDENTIFIED or not _wear_allowed(self):
        yield False
    worn_rings = [i for i in self.items if i.category == nh.RING_CLASS and i.equipped]
    worn_amulet = any(i.category == nh.AMULET_CLASS and i.equipped for i in self.items)
    stuck = self._known_stuck_ring_amulet_glyphs
    candidate = next((i for i in self.items if not i.equipped and _always_wear(i) and i.glyphs[0] not in stuck
                       and self._wear_benefit_tries.get(i.glyphs[0], 0) < 2
                       and _safe_to_put_on(self, i) and not _puton_blocked(self, i)
                       and (len(worn_rings) < 2 if i.category == nh.RING_CLASS else not worn_amulet)), None)
    if candidate is None or _hostile_within(self, cfg.AMULET_WEAR_SAFE_RADIUS):
        yield False
    yield True
    self._wear_benefit_tries[candidate.glyphs[0]] = self._wear_benefit_tries.get(candidate.glyphs[0], 0) + 1
    self.put_on(candidate)
    _log(self, f'wear_identified {candidate.text!r}')
