"""What the character carries for the Castle and below (team member 'power').

The Castle (bottom of the Dungeons of Doom, Dlvl 25-29) can't be dug through. A fall lands on its sealed west
side; the trap doors to the Valley lie behind a locked back door ((56,08) in castle.des) that is reached only
across ~26 moat squares. Passing needs levitation (ring, boots, potion), water walking boots, magical breathing
(walk the moat's bottom) or cold rays (wand of cold, frost horn) to freeze a path. Across 160 castle arrivals in
our runs about 16% (XL-10 era) to 26% (XL-8 dive) carried such an item, nearly always unidentified: 6 unknown
potion types, 1-2 unknown rings, a wand or two the engrave-test left ambiguous.

This module is side-effect free (no env steps). castle_logic executes the plan; the rest of the bot uses the
predicates behind jf_config flags (KEEP_MAGIC_BOOTS, SPARE_WISHES, SCARE_KEEP).

  passage_plan(agent, tested)  the ordered identify-by-use plan for the moat crossing
  door_tools(agent)            ways to open the locked back door
  wish_text(agent, purpose)    what a wish asks for
  scare_scrolls(agent)         carried scrolls that are (or may be) scare monster
"""

import re

import nle.nethack as nh

from . import jf_config, tele_route
from . import objects as O
from .item import Item, flatten_items

LEV_RING = O.from_name('levitation', nh.RING_CLASS)
LEV_POTION = O.from_name('levitation', nh.POTION_CLASS)
LEV_BOOTS = O.from_name('levitation boots')
WW_BOOTS = O.from_name('water walking boots')
FUMBLE_BOOTS = O.from_name('fumble boots')
SPEED_BOOTS = O.from_name('speed boots')
COLD_WAND = O.from_name('cold', nh.WAND_CLASS)
WISH_WAND = O.from_name('wishing', nh.WAND_CLASS)
OPENING_WAND = O.from_name('opening', nh.WAND_CLASS)
STRIKING_WAND = O.from_name('striking', nh.WAND_CLASS)
DIGGING_WAND = O.from_name('digging', nh.WAND_CLASS)
FROST_HORN = O.from_name('frost horn')
FIRE_HORN = O.from_name('fire horn')
MB_AMULET = O.from_name('amulet of magical breathing')
LS_AMULET = O.from_name('amulet of life saving')
STRANGLE_AMULET = O.from_name('amulet of strangulation')
SCARE = O.from_name('scare monster', nh.SCROLL_CLASS)
POLY_POTION = O.from_name('polymorph', nh.POTION_CLASS)
UNLOCKERS = frozenset(O.from_name(n) for n in ('skeleton key', 'lock pick', 'credit card'))
PASSAGE_BOOTS = frozenset((LEV_BOOTS, WW_BOOTS))

# What makes identify-by-use risky at the Castle (0 harmless .. 3 can kill): a random polymorph breaks the
# armour; sleeping and paralysis (this NLE build has the potion) leave us helpless for 25-34 turns.
_RING_DANGER = {'polymorph': 3, 'hunger': 1, 'aggravate monster': 1}
_POTION_DANGER = {'sleeping': 3, 'paralysis': 3, 'polymorph': 3, 'blindness': 2, 'hallucination': 2,
                  'confusion': 2, 'booze': 2, 'sickness': 1, 'acid': 1}

WISH_GDSM = 'blessed greased +2 gray dragon scale mail'
WISH_LEV_RING = 'blessed ring of levitation'
WISH_LS = 'blessed amulet of life saving'
WISH_SPEED = 'blessed greased +2 speed boots'
# wishes whose object has a random appearance (gray dragon scale mail is known on sight)
WISH_OBJECTS = {WISH_LEV_RING: LEV_RING, WISH_LS: LS_AMULET, WISH_SPEED: SPEED_BOOTS}
# WISH_TELEPORT_ROUTE (tele_route.py): a ring of teleport control and cursed scrolls of teleportation
WISH_OBJECTS.update({tele_route.WISH_TC_RING: tele_route.TC_RING, tele_route.WISH_TELE_SCROLLS: tele_route.TELE_SCROLL})


def _prob(obj):
    """Generation weight of an object (rings have none: all equally likely)."""
    p = getattr(obj, 'prob', None)
    return p if p else 1


def p_of(item, targets):
    """Probability that an item of this appearance is one of `targets`, by generation weight, among the objects
    the bot's knowledge (price ranges, engrave tests, discoveries) still allows."""
    total = sum(_prob(o) for o in item.objs)
    if total <= 0:
        return 0.0
    return sum(_prob(o) for o in item.objs if o in targets) / total


def _danger(item, table):
    total = sum(_prob(o) for o in item.objs)
    return sum(_prob(o) * table.get(o.name, 0) for o in item.objs) / max(total, 1)


def _certain(item, obj):
    return item.is_unambiguous() and item.object == obj


def _empty(agent, item):
    return agent.inventory.is_known_empty(item) or item.comment == 'EMPT' or \
        bool(re.search(r'\(\d+:(0|-1)\)', item.text or ''))


def _buc(item):
    return {Item.BLESSED: 'blessed', Item.CURSED: 'cursed'}.get(item.status, 'uncursed?')


def is_passage_boots(item):
    """Boots that may be levitation or water walking boots and aren't known to be anything else."""
    return item.is_armor() and all(getattr(o, 'sub', None) == O.ARM_BOOTS for o in item.objs) and \
        any(o in PASSAGE_BOOTS for o in item.objs)


def is_passage_candidate(item):
    """An item the passage plan could use: a potion or ring that may be levitation, an amulet that may be
    magical breathing, a wand that may be cold (or wishing), a horn that may be frost, candidate boots."""
    if item.is_container() or item.is_possible_container():
        return False
    cat = item.category
    if cat == nh.POTION_CLASS:
        return LEV_POTION in item.objs
    if cat == nh.RING_CLASS:
        return LEV_RING in item.objs
    if cat == nh.AMULET_CLASS:
        return MB_AMULET in item.objs
    if cat == nh.WAND_CLASS:
        return COLD_WAND in item.objs or WISH_WAND in item.objs
    if cat == nh.TOOL_CLASS:
        return FROST_HORN in item.objs
    return is_passage_boots(item)


def never_wear(item):
    """Armour the bot must not put on by itself before the Castle: levitation boots (cursed 90% of the time,
    mkobj.c) float the dig-dive off the floor for good, fumble boots trip it."""
    return item.is_armor() and any(o in (LEV_BOOTS, FUMBLE_BOOTS) for o in item.objs)


def passage_plan(agent, tested=()):
    """Ordered identify-by-use plan for crossing the Castle moat: [(action, item, why)], best first.

    action: 'puton' (ring/amulet), 'wear' (boots), 'zap' (known wand of cold), 'apply' (horn), 'quaff',
    'engrave' (engrave-test an unknown wand), 'wish' (zap a wand of wishing; `why` is the wish text).
    Items whose glyphs[0] is in `tested` are skipped. Only top-level inventory items (they have letters).
    """
    tested = set(tested)
    # a tested item that turned out to be a passage item (a ring of levitation taken off for a rest stop) is still
    # one: `tested` only removes the unknown ones already tried (pwc-dp5 jf16-s13 gave up with its lev ring)
    certain_kinds = (LEV_RING, LEV_BOOTS, WW_BOOTS, LEV_POTION, COLD_WAND, FROST_HORN, WISH_WAND)
    items = [i for i in agent.inventory.items if i.category != nh.COIN_CLASS and
             (i.glyphs[0] not in tested or (i.is_unambiguous() and i.object in certain_kinds))]
    engraved = agent.inventory.item_manager._already_engraved_glyphs
    plan = []

    # 1. certain passage items
    for it in items:
        if _certain(it, LEV_RING) and not it.equipped:
            plan.append(('puton', it, f'ring of levitation (certain, {_buc(it)})'))
    for it in items:
        if _certain(it, LEV_BOOTS) and not it.equipped:
            plan.append(('wear', it, f'levitation boots (certain, {_buc(it)})'))
        elif _certain(it, WW_BOOTS) and not it.equipped:
            plan.append(('wear', it, f'water walking boots (certain, {_buc(it)})'))
    for it in items:
        if _certain(it, COLD_WAND) and not _empty(agent, it):
            plan.append(('zap', it, 'wand of cold (certain): freeze a path'))
        elif _certain(it, FROST_HORN):
            plan.append(('apply', it, 'frost horn (certain): freeze a path'))
    for it in sorted((i for i in items if _certain(i, LEV_POTION)), key=lambda i: -i.status):
        plan.append(('quaff', it, f'potion of levitation (certain, {_buc(it)}, x{it.count})'))
    for it in items:
        if _certain(it, WISH_WAND) and not _empty(agent, it):
            plan.append(('wish', it, wish_text(agent, 'passage')))

    # 2. unknown items that may be passage items, the likeliest and safest first
    boots = [i for i in items if is_passage_boots(i) and not i.is_unambiguous() and not i.equipped]
    for it in sorted(boots, key=lambda i: -p_of(i, PASSAGE_BOOTS)):
        plan.append(('wear', it, f'unknown boots: P(levitation)={p_of(it, {LEV_BOOTS}):.2f} '
                                 f'P(water walking)={p_of(it, {WW_BOOTS}):.2f}'))
    rings = [i for i in items if i.category == nh.RING_CLASS and not i.is_unambiguous() and not i.equipped and
             LEV_RING in i.objs]
    for it in sorted(rings, key=lambda i: (-p_of(i, {LEV_RING}), _danger(i, _RING_DANGER))):
        plan.append(('puton', it, f'unknown ring: P(levitation)={p_of(it, {LEV_RING}):.2f} '
                                  f'danger={_danger(it, _RING_DANGER):.2f}'))
    for it in items:
        if it.is_wand() and not it.is_unambiguous() and COLD_WAND in it.objs and \
                it.glyphs[0] not in engraved and not _empty(agent, it):
            plan.append(('engrave', it, f'untested wand: P(cold)={p_of(it, {COLD_WAND}):.2f}'))
    for it in items:
        if it.category == nh.TOOL_CLASS and not it.is_unambiguous() and FROST_HORN in it.objs:
            plan.append(('apply', it, f'unknown horn: P(frost)={p_of(it, {FROST_HORN}):.2f} '
                                      f'P(fire)={p_of(it, {FIRE_HORN}):.2f} (a fire ray may bounce back)'))
    if jf_config.CASTLE_POLY:
        # a known potion of polymorph: a random form flies, swims or breathes water 20% of the time (castle_power)
        for it in items:
            if _certain(it, POLY_POTION):
                plan.append(('quaff', it, 'potion of polymorph: P(crossing form)=0.20'))
    potions = [i for i in items if i.category == nh.POTION_CLASS and not i.is_unambiguous() and LEV_POTION in i.objs]
    for it in sorted(potions, key=lambda i: (-p_of(i, {LEV_POTION}), _danger(i, _POTION_DANGER), -i.count)):
        plan.append(('quaff', it, f'unknown potion: P(levitation)={p_of(it, {LEV_POTION}):.2f} '
                                  f'danger={_danger(it, _POTION_DANGER):.2f} x{it.count}'))
    amulets = [i for i in items if i.category == nh.AMULET_CLASS and not i.is_unambiguous() and not i.equipped and
               MB_AMULET in i.objs]
    for it in amulets:
        plan.append(('puton', it, f'unknown amulet: P(magical breathing)={p_of(it, {MB_AMULET}):.2f} '
                                  f'P(strangulation)={p_of(it, {STRANGLE_AMULET}):.2f} (cursed 90%: pray)'))
    return plan


_PASSAGE_P = (
    ('wear', PASSAGE_BOOTS), ('puton', {LEV_RING, MB_AMULET}), ('quaff', {LEV_POTION}), ('engrave', {COLD_WAND}),
    ('apply', {FROST_HORN}),
)


def passage_odds(agent):
    """(P(some carried item is a passage item), number of certain ones, plan length): the chance that the plan
    finds levitation / water walking / magical breathing / cold, treating the candidates as independent."""
    plan = passage_plan(agent)
    p_none = 1.0
    certain = 0
    for action, item, why in plan:
        if item.is_unambiguous() or action in ('wish', 'zap'):
            certain += 1
            p_none = 0.0
            continue
        targets = dict(_PASSAGE_P).get(action, ())
        p_none *= 1 - p_of(item, targets)
    return 1 - p_none, certain, len(plan)


def kit_summary(agent):
    p, certain, n = passage_odds(agent)
    known, cands = scare_scrolls(agent)
    p_scare = 1.0
    for _, q in cands:
        p_scare *= 1 - q
    return (f'P(passage)={p:.2f} certain={certain} steps={n} scare_known={len(known)} '
            f'P(scare among candidates)={1 - p_scare:.2f} door_tools={len(door_tools(agent))}')


def door_tools(agent):
    """Ways to open the Castle's locked back door, best first: [(action, item, why)]. Kicking needs no item."""
    items = list(agent.inventory.items)
    out = []
    for obj, why in ((OPENING_WAND, 'wand of opening: unlocks'), (STRIKING_WAND, 'wand of striking: breaks it'),
                     (DIGGING_WAND, 'wand of digging: breaks it')):
        for it in items:
            if _certain(it, obj) and not _empty(agent, it):
                out.append(('zap', it, why))
    for it in items:
        if it.is_unambiguous() and it.object in UNLOCKERS:
            out.append(('apply', it, f'{it.object.name}: unlocks'))
    return out


def has_object(agent, obj):
    return any(_certain(i, obj) for i in flatten_items(agent.inventory.items))


_WISH_GOT = re.compile(r'(?:^|\s)([a-zA-Z]) - ((?:an?|\d+) [^.]+?)\.(?=\s|$)')


def note_wish(agent, text):
    """WISH_LEARN: the wish prompt was answered with `text` (agent.update); learn_wished names the result."""
    if jf_config.WISH_LEARN:
        agent._wish_pending = (WISH_OBJECTS.get(text), text, agent.step_count)


def learn_wished(agent):
    """WISH_LEARN, at the end of agent.update: 'p - a granite ring.' after a wish for a ring of levitation.
    makewish leaves the object unidentified; record its appearance glyph as the wished type so the bot uses it
    (castle_logic's certain ring, inventory.wear_life_saving) and wish_text moves on to the next wish."""
    pend = getattr(agent, '_wish_pending', None)
    if pend is None:
        return
    obj, text, step = pend
    msg = agent.message or ''
    tail = msg.rsplit(text, 1)[-1] if text in msg else msg
    m = _WISH_GOT.search(tail)
    if m is None:
        if agent.step_count - step > 30:
            agent._wish_pending = None
        return
    agent._wish_pending = None
    if obj is None:
        return
    letter = m.group(1)
    tele_route.note_wished(agent, text, letter)
    obs = agent.last_observation
    glyph = next((int(g) for l, g in zip(obs['inv_letters'], obs['inv_glyphs']) if chr(l) == letter), None)
    im = agent.inventory.item_manager
    if glyph is None or not nh.glyph_is_normal_object(glyph) or glyph in im.glyph_to_object or \
            obj in im.object_to_glyph:
        return
    if obj not in O.possibilities_from_glyph(glyph):
        agent.log(f'POWER wish {text!r} gave {m.group(2)!r}: not a {obj.name}')
        return
    im.glyph_to_object[glyph] = obj
    im.object_to_glyph[obj] = glyph
    agent.inventory.items._previous_inv_strs = None   # re-parse the inventory with the new name
    agent.log(f'POWER wished {obj.name}: {letter} - {m.group(2)}')


def wish_text(agent, purpose=None):
    """The next wish. GDSM first (magic resistance and AC for reaching the Castle), then life saving (worn at
    once: the dive's next death is survived; only ~19% of games reach the Castle, where a passage wish pays
    +0.045), then the Castle passage (a ring of levitation can be taken off to drop through the trap door),
    then speed boots. For purpose='passage' (castle_logic zapping at the moat) the ring comes first."""
    # WISH_TELEPORT_ROUTE first, even at the moat: two controlled level teleports beat one crossing (tele_route.py)
    route = tele_route.route_wish(agent)
    if route is not None:
        return route
    if purpose == 'passage' and not has_object(agent, LEV_RING):
        return WISH_LEV_RING
    if not any(i.is_armor() and i.is_unambiguous() and 'dragon scale' in i.object.name
               for i in flatten_items(agent.inventory.items)):
        return WISH_GDSM
    if not has_object(agent, LS_AMULET):
        return WISH_LS
    if not has_object(agent, LEV_RING):
        return WISH_LEV_RING
    return WISH_SPEED


# --------------------------------------------------------------------------------------------- scare monster
# A scroll of scare monster lying on our square stops every melee attacker but Rodney, minions, Angels, the
# Riders and shopkeepers/priests at home (monmove.c onscary) -- minotaurs and @ soldiers included, and in
# Gehennom, where Elbereth fails. pickup.c: the first pickup marks it, the second turns it to dust (a cursed
# one crumbles at once). The tour's arrange_items dropped all light loot for a heavy armour swap and picked it
# up again: 250 of 3158 games in our runs lost one ('The scroll turns to dust as you pick it up').

def is_scare_candidate(item):
    return item.category == nh.SCROLL_CLASS and SCARE in item.objs


def is_known_scare(agent, item):
    if item.category != nh.SCROLL_CLASS:
        return False
    if _certain(item, SCARE):
        return True
    labels = getattr(agent.inventory, 'scare_labels', ())
    return any(f'labeled {label}' in (item.text or '') for label in labels)


def keep_scroll(item):
    """SCARE_KEEP: arrange_items treats scrolls that may be scare monster as undroppable."""
    return jf_config.SCARE_KEEP and is_scare_candidate(item)


def scare_scrolls(agent):
    """(known, candidates): carried scrolls that are scare monster, and unknown ones that may be, each candidate
    as (item, P(scare)). To use one, drop it with agent.inventory.drop(item) and stand on it; the drop is
    remembered, so the bot won't pick it up again (that would turn it to dust)."""
    known, candidates = [], []
    for it in agent.inventory.items:
        if is_known_scare(agent, it):
            known.append(it)
        elif is_scare_candidate(it) and not it.is_unambiguous():
            candidates.append((it, p_of(it, {SCARE})))
    return known, candidates


_DUST_CALL = re.compile(r'Call an? scrolls? labeled ([A-Z0-9 ]+?):\s*$')


def note_dust_prompt(agent):
    """'The scroll turns to dust as you pick it up.' followed by 'Call a scroll labeled X:' names the label
    of scare monster in this game (pickup.c docall). Remember it."""
    m = _DUST_CALL.search(agent.single_message)
    if m is None or 'to dust as you' not in agent.message:
        return
    labels = agent.inventory.scare_labels
    label = m.group(1).strip()
    if label not in labels:
        labels.add(label)
        agent.log(f'POWER scare monster is labeled {label!r} (a scroll turned to dust on pickup)')
