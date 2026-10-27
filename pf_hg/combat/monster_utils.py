# heuristic monster types lists
from .. import jf_config

# the Oracle: passive magic missiles on every melee hit (a hallucinating XL9 angered, hit and died to her)
ONLY_RANGED_SLOW_MONSTERS = ['floating eye', 'blue jelly', 'brown mold', 'gas spore', 'acid blob', 'Oracle']
if jf_config.HAZARD_FIXES:
    # passive acid / paralysis on touch (astra: no melee); a spotted jelly killed an XL9 elite game
    ONLY_RANGED_SLOW_MONSTERS += ['spotted jelly', 'ochre jelly', 'gelatinous cube']
if jf_config.LATE_FIXES:
    # passive stun / acid / fire molds: common from Dlvl 1, so this one reshuffles early games
    ONLY_RANGED_SLOW_MONSTERS += ['yellow mold', 'green mold', 'red mold']
EXPLODING_MONSTERS = ['yellow light', 'gas spore', 'flaming sphere', 'freezing sphere', 'shocking sphere']
INSECTS = ['giant ant', 'killer bee', 'soldier ant', 'fire ant', 'giant beetle', 'queen bee']
WEAK_MONSTERS = ['lichen', 'newt', 'shrieker', 'grid bug']
WEIRD_MONSTERS = ['leprechaun', 'nymph']


def is_monster_faster(agent, monster):
    _, y, x, mon, _ = monster
    # TOOD: implement properly
    return 'bat' in mon.mname or 'dog' in mon.mname or 'cat' in mon.mname \
           or 'kitten' in mon.mname or 'pony' in mon.mname or 'horse' in mon.mname \
           or 'bee' in mon.mname or 'fox' in mon.mname


def imminent_death_on_melee(agent, monster):
    if is_dangerous_monster(monster):
        return agent.blstats.hitpoints <= 16
    # hypothesis: retreating from ordinary monsters below 10 HP avoids the
    # common two-hit deaths while retaining normal aggression at full health.
    return agent.blstats.hitpoints <= 10


def is_dangerous_monster(monster):
    _, y, x, mon, _ = monster
    is_pet = 'dog' in mon.mname or 'cat' in mon.mname or 'kitten' in mon.mname or 'pony' in mon.mname \
             or 'horse' in mon.mname
    # 'mumak' in mon.mname or 'orc' in mon.mname or 'rothe' in mon.mname \
    # or 'were' in mon.mname or 'unicorn' in mon.mname or 'elf' in mon.mname or 'leocrotta' in mon.mname \
    # or 'mimic' in mon.mname
    return is_pet or mon.mname in INSECTS


def _adjacent_turns(agent, monster):
    """Turns this monster has stayed adjacent to us (a gap of more than 3 turns resets it)."""
    _, y, x, mon, _ = monster
    bl = agent.blstats
    rec = agent.__dict__.setdefault('_adjacent_since', {})
    key = mon.mname
    if max(abs(int(y) - bl.y), abs(int(x) - bl.x)) > 1:
        return 0
    first, last = rec.get(key, (bl.time, bl.time))
    if bl.time - last > 3:
        first = bl.time
    rec[key] = (first, bl.time)
    return bl.time - first


def consider_melee_only_ranged_if_hp_full(agent, monster):
    name = monster[3].mname
    bl = agent.blstats
    # cornered by a gelatinous cube (a tiny Minetown shop, no throwing near the Watch): stepping away
    # failed, and standing still let its paralysing touches kill an XL9 from 90 HP -- hit it instead
    if name == 'gelatinous cube':
        return _adjacent_turns(agent, monster) >= 6 and bl.hitpoints >= 0.4 * bl.max_hitpoints
    # their passive cold does nothing to a cold-resistant Valkyrie
    if name in ('brown mold', 'blue jelly') and \
            (bl.hitpoints == bl.max_hitpoints or agent.character.role == agent.character.VALKYRIE):
        return True
    return False
