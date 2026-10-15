"""opp-items lane: small levers from items the dive already meets (jf_config.GENOCIDE_POLICY, HORN_SCARE, TENGU_EAT).

GENOCIDE_POLICY -- NetHack 3.6.6 read.c:
* seffects SCR_GENOCIDE: a blessed scroll asks the CLASS prompt ('What class of monsters do you wish to genocide?',
  do_class_genocide: always real); uncursed and cursed ones ask the same SPECIES prompt ('What monster do you want to
  genocide? [type the name]'); a cursed one REVERSE-genocides: makemon() rn1(3,4) = 4-6 of the named species next to
  us ('Sent in some <plural>.'). ESC / 'none' wastes an uncursed scroll; a cursed one then sends in rndmonst()s.
  Confused and not blessed, it genocides our own role ('genocidal confusion'): never read an unknown scroll confused.
* Before this, agent.update ESCaped every genocide prompt (power_route's read tests, LAST RESORT reads): jf14 s0 wasted
  one of its two genocide scrolls on Dlvl 2 at T9960 and died to a minotaur (ledger A077/H018). mino_guard answered
  'minotaur' / 'H' itself.
* The policy (research/castle-gehennom-research.md 2.4): the class prompt means blessed -> 'H' (minotaurs, giants,
  titans), then 'B' (Medusa-3's ravens), 'X', ';'. The species prompt, by what we know of the scroll:
    - a minotaur within MINO_CONSUME_RANGE: 'minotaur' (6 times in 7 the species prompt is not cursed; next to one we
      live ~2 turns) -- mino_guard's rule, kept;
    - at critical HP with a genocidable hostile next to us (LAST RESORT reads): its species, same odds;
    - BUC known not cursed (the display, or the same stack's first read said 'Wiped out': a stack shares its BUC):
      'minotaur', then raven, xorn, giant eel, shark (cmp-main causes of death), skipping species already wiped out;
    - a stack proven cursed ('Sent in' on its first read): 'newt' (4-6 newts);
    - BUC unknown on the castle (or a lone scroll at castle depth, where it may be the castle): 'minotaur' (its depth
      is banked: a reversed scroll costs only the slim pass chance);
    - BUC unknown: the probe 'giant eel' -- real, it removes the castle moat's and Medusa's eels (an eel's wrap drowns
      even a levitating hero); reversed, eels land on dry squares (teleport.c goodpos: 1 in 13 per square) where they
      lose HP and flee every turn (mon.c minliquid) and cannot drown us (mhitu.c AD_WRAP drowns only from a pool) --
      but 'newt' when water is within 3 squares (a reversed eel in a pool next to us drowns us).
  Never our race or role ('dwarf', 'valkyrie': 'You feel dead inside').
* A known scroll of genocide proven not cursed is read at once when safe (read_strategy): every live minotaur dies
  and the filler mazes and the castle generated later have none (makemon returns NULL for a genocided species).

HORN_SCARE -- music.c / apply.c / uhitm.c (3.6.6):
* do_play_instrument: horns ask 'Improvise? [ynq]' (not while stunned/confused/hallucinating; drums never ask);
  do_improvisation: a tooled horn (or a frost/fire horn with no charges, or any horn played stunned/confused)
  'You produce a frightful, grave sound.' -> awaken_monsters(XL*30): every monster within distu < XL*10 that fails
  resist(TOOL_CLASS) (minotaurs, soldiers, eels: MR 0, never) and isn't immune to musical scaring (onscary(0,0):
  only Rodney, minions, Angels, Riders, shopkeepers/priests at home) flees with no timer (monflee(m, 0, ...)); a
  leather drum XL*40 (distu < XL*40/3), an empty drum of earthquake XL*5, a charged one the whole level (+ pits).
  A fleeing monster that can move away doesn't attack (monmove.c dochug: 'case 1: monster moved ... return 0'); an
  untimed flee ends 1 time in 25 per move only at full HP. Charged frost/fire horns ask 'In what direction?' and fire
  a 6d6 ray (rn1(6,6) squares) instead; a horn of plenty isn't an instrument (no prompt).
* use_camera + flash_hits_mon: a straight line to the first monster; not resists_blnd -> blinded; within dist2 < 9 it
  flees 3 times in 4 (rnd(100) turns, or untimed 1 in 4); dist2 < 3 blinds it for good. A sleeping target wakes.

TENGU_EAT -- eat.c cpostfx: a tengu corpse offers poison resistance, teleportitis and teleport control, one picked at
random, then givit(): TC 'if (ptr->mlevel <= rn2(12)) fail' = 6/12 -> 1/6 per corpse ('You feel in control of
yourself.'); mon.c corpse_chance: 1 kill in 2 leaves one. Tengu are lawful (alignment 7): makemon.c peace_minded makes
~85% of them peaceful to our lawful Valkyrie ('A tengu blocks your path.' in 3 of 4 dive sightings); killing a
peaceful costs Luck -1 half the time (mon.c xkilled) and prayers fail at Luck < 0 -- only hostile ones are fought.
"""
import re

import nle.nethack as nh

from . import jf_config, utils
from . import objects as O
from .glyph import MON, SS, Hunger

from .strategy import Strategy

GENOCIDE = O.from_name('genocide', nh.SCROLL_CLASS)
TOOLED_HORN = O.from_name('tooled horn')
FROST_HORN = O.from_name('frost horn')
FIRE_HORN = O.from_name('fire horn')
HORN_OF_PLENTY = O.from_name('horn of plenty')
LEATHER_DRUM = O.from_name('leather drum')
EARTHQUAKE_DRUM = O.from_name('drum of earthquake')
CAMERA = O.from_name('expensive camera')
BUGLE = O.from_name('bugle')
# BUGLE_SCARE: the mercenaries a bugle wakes and turns hostile (music.c awaken_soldiers: is_mercenary)
_MERCENARIES = frozenset(('soldier', 'sergeant', 'lieutenant', 'captain', 'watchman', 'watch captain'))
HORNS = frozenset((TOOLED_HORN, FROST_HORN, FIRE_HORN, HORN_OF_PLENTY))
DRUMS = frozenset((LEATHER_DRUM, EARTHQUAKE_DRUM))

G_GENO = 0x0020
SPECIES_ORDER = ('minotaur', 'raven', 'xorn', 'giant eel', 'shark')
CLASS_ORDER = ('H', 'B', 'X', ';')
PROBE = 'giant eel'
HARMLESS = 'newt'
NEVER = frozenset(('dwarf', 'valkyrie'))   # our race and role (read.c Your_Own_Race / Your_Own_Role)
_PLURAL = {'minotaur': 'minotaurs', 'raven': 'ravens', 'xorn': 'xorns', 'giant eel': 'giant eels',
           'shark': 'sharks', 'newt': 'newts'}
_SINGULAR = {v: k for k, v in _PLURAL.items()}
_WET = frozenset({SS.S_pool, SS.S_water, SS.S_lava})
_WIPED = re.compile(r'Wiped out (?:all )?([a-zA-Z ]+?)\.')
_SENT = re.compile(r'Sent in (?:some |an? )([a-zA-Z ]+?)\.')
SCARE_SOUNDS = _SCARE_SOUNDS = ('You produce a frightful, grave sound', 'You beat a deafening row', 'You pound on the drum',
                 'You blow into the horn', 'heavy, thunderous rolling') + \
    (('You extract a loud noise from', 'You blow into the bugle') if jf_config.BUGLE_SCARE else ())
_DIRS = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1), 'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1),
         'sw': (1, -1)}
_DIR_KEYS = {'n': 'k', 's': 'j', 'e': 'l', 'w': 'h', 'ne': 'u', 'nw': 'y', 'se': 'n', 'sw': 'b', '>': '>', '<': '<',
             '.': '.'}


class OppState:
    def __init__(self):
        self.reading = None          # the read in progress: dict(letter, glyph, text, step, turn, why)
        self.answer = None           # (step, kind, text) of the last genocide answer we typed
        self.prompt_answers = 0      # answers given to the genocide prompt now on screen
        self.proven = {}             # inventory letter -> ('noncursed' | 'cursed', scroll glyph)
        self.wiped = set()           # species we genocided ('Wiped out all ...')
        self.classes = set()         # classes we answered at a class prompt
        self.log = []                # (turn, what) for the botlog summary
        self.horn_glyphs = {}        # instrument glyph -> 'scare' | 'ray' | 'plenty' (what blowing it did)
        self.playing = None          # (step, glyph) of the instrument just played
        self.empty = set()           # camera texts that said 'Nothing happens'
        self.last_msg_step = -1


def state(agent):
    st = getattr(agent, '_opp', None)
    if st is None:
        st = agent._opp = OppState()
    return st


def _log(agent, msg):
    agent.log(f'OPP {msg}')


# ------------------------------------------------------------------------------------------------ messages

def note_message(agent):
    """agent.update, every observation (flags on only): genocide outcomes, what an instrument turned out to be."""
    msg = agent.single_message or ''
    st = state(agent)
    if not msg or st.last_msg_step == agent.step_count:
        return
    st.last_msg_step = agent.step_count
    try:
        if st.answer is not None and agent.step_count - st.answer[0] <= 12:
            m = _WIPED.search(msg)
            if m:
                name = m.group(1).strip()
                st.wiped.add(_SINGULAR.get(name, name))
                if st.answer[1] == 'class':
                    st.classes.add(st.answer[2])
                rd = st.reading
                if st.answer[1] == 'species' and rd is not None and rd.get('letter'):
                    # a real species genocide: this scroll was not cursed, and neither is the rest of its stack
                    st.proven[rd['letter']] = ('noncursed', rd['glyph'])
                _log(agent, f'genocide: {msg[:120]!r}')
                st.answer = None
            m = _SENT.search(msg)
            if m:
                rd = st.reading
                if rd is not None and rd.get('letter'):
                    st.proven[rd['letter']] = ('cursed', rd['glyph'])
                _log(agent, f'reverse genocide (the scroll was cursed): {msg[:120]!r}')
                st.answer = None
        if st.playing is not None and agent.step_count - st.playing[0] <= 6 and \
                any(s in msg for s in _SCARE_SOUNDS):
            st.horn_glyphs.setdefault(st.playing[1], 'scare')
    except Exception as e:   # learning must never break the step loop
        agent.log(f'OPP note_message failed: {e!r}')


def note_read(agent, item, why=''):
    """Called by every scroll reader just before it reads: the genocide answer depends on which scroll this is."""
    if not jf_config.GENOCIDE_POLICY:
        return
    try:
        letter = agent.inventory.items.get_letter(item)
    except Exception:
        letter = None
    state(agent).reading = dict(letter=letter, glyph=item.glyphs[0] if item.glyphs else None, text=item.text or '',
                                count=int(getattr(item, 'count', 1) or 1), step=agent.step_count,
                                turn=agent.blstats.time, why=why)


def not_prompting(agent):
    """agent.update: no text entry on screen -- the next genocide prompt starts its answers afresh."""
    st = getattr(agent, '_opp', None)
    if st is not None:
        st.prompt_answers = 0


# ------------------------------------------------------------------------------------------------ genocide

def _words(text):
    return f' {text or ""} '


def _buc(text):
    w = _words(text)
    if ' cursed ' in w and ' uncursed ' not in w:
        return 'cursed'
    if ' uncursed ' in w or ' blessed ' in w:
        return 'noncursed'
    return None


def _read_status(agent, rd):
    """'noncursed' / 'cursed' / None (unknown) for the scroll being read."""
    if rd is None:
        return None
    st = state(agent)
    proven = st.proven.get(rd['letter'])
    if proven is not None and proven[1] == rd['glyph']:
        return proven[0]
    return _buc(rd['text'])


def _mino_near(agent):
    """A minotaur on the screen within mino_guard's consume range (Chebyshev)."""
    try:
        from .mino_guard import MINO_CONSUME_RANGE
        guard = getattr(agent.global_logic.dive, 'mino_guard', None)
        minos = guard._minos() if guard is not None else []
        return bool(minos) and minos[0][0] <= MINO_CONSUME_RANGE
    except Exception:
        return False


def _adjacent_threat(agent):
    """At critical HP (below a third), the species of a genocidable hostile next to us, else None."""
    bl = agent.blstats
    if agent.character.prop.hallu or bl.hitpoints * 3 >= bl.max_hitpoints:
        return None
    try:
        for m in agent.get_visible_monsters():
            if max(abs(int(m[1]) - int(bl.y)), abs(int(m[2]) - int(bl.x))) != 1:
                continue
            pm = m[3]
            name = getattr(pm, 'mname', '')
            if not name or name in NEVER or 'dwarf' in name or not (getattr(pm, 'geno', 0) & G_GENO):
                continue
            return name
    except Exception:
        return None
    return None


def _water_near(agent, radius=3):
    level = agent.current_level()
    y, x = int(agent.blstats.y), int(agent.blstats.x)
    box = level.objects[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    seen = agent.glyphs[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    return any(int(g) in _WET for g in box.flat) or any(int(g) in _WET for g in seen.flat)


def _on_castle(agent):
    try:
        from . import power_route
        return power_route._on_castle(agent)
    except Exception:
        return False


def _castle_depth(agent):
    """The castle or a level that may be it (power_route.on_castle_level: main line, depth 25+, not Medusa's)."""
    try:
        from . import power_route
        return power_route.on_castle_level(agent)
    except Exception:
        return False


def _in_gehennom(agent):
    try:
        return agent.current_level().dungeon_number == 1
    except Exception:
        return False


def genocide_answer(agent, cls):
    """The text to type at a genocide prompt (class prompt: cls=True), and why."""
    st = state(agent)
    n = st.prompt_answers
    st.prompt_answers += 1
    if cls:
        # only a blessed scroll asks for a class (read.c seffects: sblessed -> do_class_genocide)
        order = [c for c in CLASS_ORDER if c not in st.classes] or list(CLASS_ORDER)
        if jf_config.GENO_EXTRAS and 'L' not in st.classes and (_on_castle(agent) or _in_gehennom(agent)):
            # veterans' class: liches (covetous master liches teleport next to the castle landing; they drive the
            # Valley pile-ups), ahead of 'H' once the mazes' minotaurs are behind us
            order = ['L'] + [c for c in order if c != 'L']
        text = order[min(n, len(order) - 1)]
        st.answer = (agent.step_count, 'class', text)
        return text, 'class prompt (a blessed scroll)'
    rd = st.reading if st.reading is not None and agent.step_count - st.reading['step'] <= 40 else None
    status = _read_status(agent, rd)
    cands = []
    why = ''
    if status == 'cursed':
        cands, why = [HARMLESS], 'a cursed scroll: 4-6 newts'
    else:
        if _mino_near(agent) and 'minotaur' not in st.wiped:
            cands.append('minotaur')
            why = 'a minotaur within reach'
        threat = _adjacent_threat(agent)
        if threat is not None and threat not in cands:
            cands.append(threat)
            why = why or f'critical HP, {threat} next to us'
        if status == 'noncursed':
            cands += [s for s in SPECIES_ORDER if s not in st.wiped and s not in cands]
            why = why or 'not cursed'
        elif not cands and 'minotaur' not in st.wiped and \
                (_on_castle(agent) or (_castle_depth(agent) and rd is not None and rd.get('count', 1) == 1)):
            # on the castle its depth is banked: a reversed scroll (1 in 7) costs only the slim chance of a pass,
            # a real one removes the landing's #1 killer (castle-first-pass H016: 10 of 18 early castle deaths). The
            # castle is known only after a failed dig, so a lone scroll at castle depth (25+, below Medusa: the
            # castle or its maze neighbours) goes for the minotaurs too; a stack there still probes first (its
            # second scroll is then sure: harness ocg1on 5/5 'Wiped out all minotaurs.' by the second read)
            cands, why = ['minotaur'], 'BUC unknown at castle depth (the castle banks its depth)'
        elif not cands:
            if _water_near(agent):
                cands, why = [HARMLESS], 'BUC unknown, water near: newt'
            elif PROBE not in st.wiped:
                cands, why = [PROBE], 'BUC unknown: the giant eel probe'
            else:
                cands, why = [HARMLESS], 'BUC unknown'
    cands = cands or [HARMLESS]
    text = cands[min(n, len(cands) - 1)]
    st.answer = (agent.step_count, 'species', text)
    return text, f'{why} (BUC {status or "unknown"}, try {n + 1})'


def answer_prompt(agent):
    """agent.update's text-entry branch: the genocide prompt on screen -> type the answer. True if answered."""
    msg = agent.single_message or ''
    cls = 'class of monsters do you wish to genocide' in msg
    if not cls and 'do you want to genocide' not in msg:
        return False
    st = state(agent)
    if st.prompt_answers >= 3:
        return False   # the generic handler ESCapes it
    text, why = genocide_answer(agent, cls)
    rd = st.reading
    _log(agent, f'genocide prompt ({"class" if cls else "species"}): answering {text!r} -- {why}; '
                f'scroll {rd["text"] if rd else "?"!r}')
    st.log.append((agent.blstats.time, text))
    agent.step(text[0], iter(text[1:] + '\r'))
    return True


def proven_cursed(agent, item):
    """This genocide stack sent monsters in when read: cursed (mino_guard must not read it at a minotaur)."""
    st = getattr(agent, '_opp', None)
    if st is None:
        return False
    try:
        letter = agent.inventory.items.get_letter(item)
    except Exception:
        return False
    p = st.proven.get(letter)
    return p is not None and p[0] == 'cursed' and item.glyphs and p[1] == item.glyphs[0]


def _known_genocide(agent):
    """A scroll of genocide we know is not cursed (display or proven), not unpaid; else None."""
    st = state(agent)
    for it in agent.inventory.items:
        if it.category != nh.SCROLL_CLASS or not it.is_unambiguous() or it.object != GENOCIDE:
            continue
        if 'unpaid' in (it.text or ''):
            continue
        status = _buc(it.text)
        if status is None:
            try:
                p = st.proven.get(agent.inventory.items.get_letter(it))
            except Exception:
                p = None
            if p is not None and it.glyphs and p[1] == it.glyphs[0]:
                status = p[0]
        if status == 'noncursed':
            return it
    return None


def read_scroll(agent, item, why):
    """Read a scroll; the genocide prompt is agent.update's (answer_prompt), --More-- is dismissed there too."""
    from nle.nethack import actions as A
    note_read(agent, item, why)
    letter = agent.inventory.items.get_letter(item)
    _log(agent, f'reading {item.text!r} ({letter}): {why}')

    def gen():
        if 'What do you want to read?' not in agent.single_message:
            return
        yield letter

    with agent.atom_operation():
        agent.step(A.Command.READ, gen())
    _log(agent, f'read -> {(agent.message or "")[:200]!r}')
    agent.inventory.items.update(force=True)


def read_strategy(agent):
    """GENOCIDE_POLICY: a known genocide scroll proven not cursed is read at once (a quiet step, able to read)."""
    def f():
        if not jf_config.GENOCIDE_POLICY:
            yield False
            return
        item = _known_genocide(agent)
        why = 'known genocide, not cursed: read at once'
        if item is None:
            yield False
            return
        prop = agent.character.prop
        if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph:
            yield False
            return
        st = state(agent)
        if st.reading is not None and agent.blstats.time - st.reading['turn'] < 2:
            yield False   # (one read a turn: a refused read passes no time)
            return
        yield True
        read_scroll(agent, item, why)

    return Strategy(f)


# ------------------------------------------------------------------------------------------------ instruments

def instrument_kind(agent, item):
    """'scare' (tooled horn, any drum, a horn that scared when blown), 'horn' (an unknown horn: tooled 5/11, frost
    2/11, fire 2/11, plenty 2/11), 'camera', or None."""
    if item.category != nh.TOOL_CLASS or not item.objs or 'unpaid' in (item.text or ''):
        return None
    objs = set(item.objs)
    if objs <= DRUMS:
        return 'scare'
    if item.is_unambiguous():
        if item.object == TOOLED_HORN:
            return 'scare'
        if item.object == BUGLE:
            return 'scare' if jf_config.BUGLE_SCARE and bugle_ok(agent) else None
        if item.object == CAMERA:
            return None if item.text in state(agent).empty else 'camera'
        return None
    if objs <= HORNS:
        what = state(agent).horn_glyphs.get(item.glyphs[0] if item.glyphs else None)
        if what == 'scare':
            return 'scare'
        if what is not None:
            return None   # a horn of plenty, or a ray horn the game has named by now
        return 'horn'
    return None


def bugle_ok(agent):
    """BUGLE_SCARE: a bugle also wakes every mercenary on the level and makes them hostile (music.c awaken_soldiers):
    never on the castle level (its barracks sleep ~40 soldiers) nor with a peaceful watchman or soldier in view (a
    town's watch turns on us)."""
    try:
        dive = agent.global_logic.dive
        key = agent.current_level().key()
        if dive.castle.castle_key is not None and key == dive.castle.castle_key:
            # CL_BUGLE_WEST (castle-lift, main's idea): the castle's soldiers wake and turn hostile, but on the west side
            # before any crossing they can't reach us -- the drawbridge is raised, the towers open inward, the barracks
            # and the trap-door hall behind locked doors -- while a minotaur flees the bugle (R176: 14/20 vs 1/20)
            if not jf_config.CL_BUGLE_WEST:
                return False
            castle = dive.castle
            mx, _ = castle._pos()
            from .castle_logic import WEST_COURTYARD
            if castle.committed() or not (mx < 0 or castle._pos() in WEST_COURTYARD):
                return False
        peaceful = agent.monster_tracker.peaceful_monster_mask
        for _, y, x, mon, _ in agent.get_visible_monsters():
            if getattr(mon, 'mname', '') in _MERCENARIES and peaceful[y, x]:
                return False
        return True
    except Exception:
        return False


def keep_kind(agent, item):
    """ItemPriority (HORN_KEEP): one of each worth carrying -- 'horn' (tooled / unknown / known frost: a cold source
    for MEDUSA_FREEZE and the castle), 'drum', 'camera'; else None."""
    if item.category != nh.TOOL_CLASS or not item.objs:
        return None
    objs = set(item.objs)
    if objs <= DRUMS:
        return 'drum'
    if item.is_unambiguous():
        if item.object in (TOOLED_HORN, FROST_HORN):
            return 'horn'
        if item.object == CAMERA:
            return 'camera'
        return None
    if objs <= HORNS:
        return 'horn'
    return None


def scare_radius2(agent, item):
    """distu below which blowing it makes a monster flee (music.c: awaken_monsters(d) scares within d/3)."""
    xl = int(agent.blstats.experience_level)
    objs = set(item.objs)
    if objs <= DRUMS:
        if item.is_unambiguous() and item.object == EARTHQUAKE_DRUM:
            return 10 ** 6   # a charged one shakes the level (an empty one is a mundane drum: XL*5)
        return xl * 40 // 3
    return xl * 10


def play(agent, item, ray_dir):
    """Apply an instrument: 'Improvise?' -> y (music.c: 'n' goes to the tune prompt), 'In what direction?' (a charged
    frost/fire horn's ray) -> ray_dir. Returns what happened: dict(improvise, direction, msg)."""
    from nle.nethack import actions as A
    st = state(agent)
    letter = agent.inventory.items.get_letter(item)
    seen = dict(improvise=False, direction=False)
    st.playing = (agent.step_count, item.glyphs[0] if item.glyphs else None)

    def gen():
        if 'What do you want to use or apply?' not in agent.single_message:
            return
        yield letter
        for _ in range(8):
            msg = agent.single_message or ''
            if 'Improvise?' in msg and not seen['improvise']:
                seen['improvise'] = True
                yield 'y'
                continue
            if 'In what direction?' in msg and not seen['direction']:
                seen['direction'] = True
                yield _DIR_KEYS.get(ray_dir, '>')
                continue
            misc = agent._observation['misc']
            if misc[2] and not misc[1]:
                yield ' '
                continue
            return

    with agent.atom_operation():
        agent.step(A.Command.APPLY, gen())
    msg = agent.message or ''
    g = item.glyphs[0] if item.glyphs else None
    if not set(item.objs) <= DRUMS and not item.is_unambiguous():
        if seen['direction']:
            st.horn_glyphs[g] = 'ray'
        elif not seen['improvise'] and not agent.character.prop.confusion and not agent.character.prop.stun and \
                not agent.character.prop.hallu:
            st.horn_glyphs[g] = 'plenty'   # no instrument prompt: a horn of plenty (apply.c hornoplenty)
        elif any(s in msg for s in _SCARE_SOUNDS):
            st.horn_glyphs[g] = 'scare'
    _log(agent, f'played {item.text!r} ({letter}) ray {ray_dir}: improvise={seen["improvise"]} '
                f'direction={seen["direction"]} -> {msg[:160]!r}')
    agent.inventory.items.update(force=True)
    return dict(seen, msg=msg)


def use_camera(agent, item, direction):
    """Flash it along direction (apply.c use_camera: the first monster in line)."""
    from nle.nethack import actions as A
    letter = agent.inventory.items.get_letter(item)

    def gen():
        if 'What do you want to use or apply?' not in agent.single_message:
            return
        yield letter
        for _ in range(4):
            msg = agent.single_message or ''
            if 'In what direction?' in msg:
                yield _DIR_KEYS[direction]
                return
            misc = agent._observation['misc']
            if misc[2] and not misc[1]:
                yield ' '
                continue
            return

    with agent.atom_operation():
        agent.step(A.Command.APPLY, gen())
    msg = agent.message or ''
    if 'Nothing happens' in msg:
        state(agent).empty.add(item.text)
    _log(agent, f'camera {item.text!r} ({letter}) {direction} -> {msg[:160]!r}')
    return msg


def free_run(agent, sy, sx, limit=20):
    """Walkable squares from us in direction (sy, sx) before a wall or the edge (a ray's room to die out)."""
    level = agent.current_level()
    y, x = int(agent.blstats.y), int(agent.blstats.x)
    h, w = level.walkable.shape
    n = 0
    while n < limit:
        y, x = y + sy, x + sx
        if not (0 <= y < h and 0 <= x < w) or not level.walkable[y, x]:
            break
        n += 1
    return n


def longest_free_dir(agent):
    """The compass direction with the longest free run (a possible ray's safest way out), and that run."""
    best = max(_DIRS.items(), key=lambda kv: free_run(agent, *kv[1]))
    return best[0], free_run(agent, *best[1])


# ------------------------------------------------------------------------------------------------ tengu

_TENGU_ID = None


def _tengu_id():
    global _TENGU_ID
    if _TENGU_ID is None:
        _TENGU_ID = MON.id_from_name('tengu')
    return _TENGU_ID


def _cheb(a, b):
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _fresh_tengu_corpse(agent, max_dist):
    level = agent.current_level()
    tid = _tengu_id()
    now = agent.blstats.time
    dis = agent.bfs()
    best = None
    for (y, x), mapping in level.corpses_to_eat.items():
        age = mapping.get(tid)
        if age is None or now - age > jf_config.TENGU_CORPSE_AGE or level.shop[y, x]:
            continue
        d = int(dis[y, x])
        if d < 0 and (y, x) != (agent.blstats.y, agent.blstats.x):
            continue
        d = max(d, 0)
        if d <= max_dist and (best is None or d < best[0]):
            best = (d, int(y), int(x), age)
    return best


def _tengu_next(agent):
    """TENGU_EAT's next action: ('hit', (y, x)) a hostile tengu next to us (HP >= 60%, nothing else within 3),
    ('walk', (y, x, age)) to / ('eat', item) a fresh tengu corpse (not Satiated, nothing hostile within 4), or None."""
    dive = agent.global_logic.dive
    if not dive.diving or dive.in_gehennom():
        return None
    # cheap gate first: a tengu on the screen or a tengu corpse recorded on this level. No bfs() call otherwise --
    # its per-step cache is shared, and filling it early in a step changed later decisions (harness ot2on s3, a
    # tengu never engaged, diverged from ot1off)
    tid = _tengu_id()
    if not utils.isin(agent.glyphs, [nh.GLYPH_MON_OFF + tid]).any() and \
            not any(tid in m for m in agent.current_level().corpses_to_eat.values()):
        return None
    from . import power_route
    if power_route.state(agent).tc_intrinsic:
        return None
    prop = agent.character.prop
    if prop.hallu or prop.blind or prop.polymorph or prop.stun or prop.confusion:
        return None
    bl = agent.blstats
    me = (int(bl.y), int(bl.x))
    mons = agent.get_visible_monsters()
    tengu = [m for m in mons if getattr(m[3], 'mname', '') == 'tengu' and _cheb(me, (m[1], m[2])) == 1]
    others = [m for m in mons if getattr(m[3], 'mname', '') != 'tengu' and _cheb(me, (m[1], m[2])) <= 3]
    if tengu and not others and bl.hitpoints >= 0.6 * bl.max_hitpoints:
        return ('hit', (int(tengu[0][1]), int(tengu[0][2])))
    if bl.hunger_state == Hunger.SATIATED or any(_cheb(me, (m[1], m[2])) <= 4 for m in mons):
        return None
    target = _fresh_tengu_corpse(agent, 6)
    if target is None:
        return None
    _, y, x, age = target
    if (y, x) != me:
        return ('walk', (y, x, age))
    for item in agent.inventory.items_below_me:
        if item.is_corpse() and item.monster_id == tid and item.count == 1:
            return ('eat', item)
    # no corpse was left here after all (1 kill in 2 leaves one): forget it
    agent.current_level().corpses_to_eat[y, x].pop(tid, None)
    return None


def tengu_strategy(agent):
    """TENGU_EAT, in the dive: hit a hostile tengu next to us until it dies or leaves, then eat its fresh corpse (not
    Satiated). The body loops while there is work: a one-action return let the dive's Elbereth-and-dig run a step
    between blows (harness ot1on: 1 kill in 14 hostile tengu next to the landing)."""
    def f():
        if not jf_config.TENGU_EAT or _tengu_next(agent) is None:
            yield False
            return
        yield True
        for _ in range(30):
            act = _tengu_next(agent)
            if act is None:
                return
            steps = agent.step_count
            bl = agent.blstats
            kind, arg = act
            if kind == 'hit':
                agent.log(f'TENGU hitting the hostile tengu at {arg}, hp {bl.hitpoints}/{bl.max_hitpoints}')
                agent.melee_attack(arg[0], arg[1])
            elif kind == 'walk':
                agent.log(f'TENGU walking to the tengu corpse at {arg[:2]} (age {bl.time - arg[2]})')
                agent.go_to(arg[0], arg[1])
            else:
                agent.log(f'TENGU eating {arg.text!r}')
                agent.inventory.eat(arg)
                return
            if agent.step_count == steps:
                return

    return Strategy(f)
