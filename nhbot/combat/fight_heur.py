from collections import defaultdict
from itertools import product

import numpy as np
from scipy import signal

from ..glyph import G, MON, Hunger
from .. import jf_config, utils
from ..item import Item
from ..utils import adjacent
from .monster_utils import is_monster_faster, is_dangerous_monster, \
    ONLY_RANGED_SLOW_MONSTERS, EXPLODING_MONSTERS, WEAK_MONSTERS, consider_melee_only_ranged_if_hp_full
from .movement_priority import draw_monster_priority_positive, draw_monster_priority_negative
from .utils import wielding_ranged_weapon, line_dis_from, inside


def spore_blast_hits_friend(agent, y, x):
    """A gas spore killed at (y, x) explodes over its 3x3 square: a pet or peaceful there gets hurt and
    the hero gets the blame (a shopkeeper next to a spore turned hostile and killed an XL8 Valkyrie)."""
    sl = np.s_[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
    if agent.monster_tracker.peaceful_monster_mask[sl].any() or utils.any_in(agent.glyphs[sl], G.PETS):
        return True
    if jf_config.SPORE_SAFE and _spore_blast_hits_people(agent, y, x):
        return True
    # a pet seen here lately but out of view now may be right behind the spore: a thrown dagger's blast
    # killed an unseen kitten ('You kill it!', 'rumble of distant thunder': -15 alignment on a Valkyrie's
    # record that starts at 0, so the first grind prayer failed at T1364)
    seen = agent.global_logic.dive.pet_seen.get(agent.current_level().key())
    return seen is not None and agent.blstats.time - seen < 100 and not utils.any_in(agent.glyphs, G.PETS)


def _spore_blast_hits_people(agent, y, x):
    """SPORE_SAFE: the blast also counts as our attack on whoever we don't know to be peaceful yet: any @ in
    the 3x3 (killing a watchman/shopkeeper/priest angered by it is murder even when he is hostile: Luck -2,
    telepathy lost), and anything at all on the Minetown level (base3-public s13: 'You kill the gas spore!  The
    watch captain is caught in the gas spore's explosion! ... You murderer!', prayers held 1200 turns)."""
    y0, x0 = agent.blstats.y, agent.blstats.x
    gl = agent.global_logic
    minetown = gl.minetown_level is not None and agent.current_level().key() == gl.minetown_level
    # a shop's squares (agent.py marks the shopkeeper's room dilated by one: its walls and door): the shopkeeper
    # stands just inside the door and is out of view from most angles. Killing a spore in or at a shop door angered
    # him in 2 of the 4 murders of 16 recent runs (cmp-main jf41 s5 threw daggers at a spore in the doorway while
    # Wonotobo, one square inside, was out of sight; base10arm jf16 s11 meleed one inside the shop)
    level = agent.current_level()
    if level.shop[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
        return True
    for yy in range(max(y - 1, 0), min(y + 2, agent.glyphs.shape[0])):
        for xx in range(max(x - 1, 0), min(x + 2, agent.glyphs.shape[1])):
            if (yy, xx) in ((y, x), (y0, x0)):
                continue
            g = agent.glyphs[yy, xx]
            if not MON.is_monster(g):
                continue
            if minetown:
                return True
            mlet = MON.permonst(g).mlet
            if (ord(mlet) if isinstance(mlet, str) else mlet) == MON.S_HUMAN:
                return True
    return False


def melee_monster_priority(agent, monsters, monster):
    _, y, x, mon, _ = monster
    ret = 1
    if agent.blstats.hitpoints > 8 or is_monster_faster(agent, monster):
        ret += 15
    if wielding_ranged_weapon(agent) and not is_monster_faster(agent, monster):
        ret -= 6
    if mon.mname in EXPLODING_MONSTERS:
        ret -= 17
    if 'were' in mon.mname:
        ret += 1
    # if not wielding_melee_weapon(agent):
    #     ret -= 5
    if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        if not consider_melee_only_ranged_if_hp_full(agent, monster):
            ret -= 100
            if mon.mname == 'floating eye':
                ret -= 10
            if mon.mname == 'gas spore':
                ret -= 5

    if mon.mname == 'gas spore':
        if spore_blast_hits_friend(agent, y, x):
            return ret - 200
        # handle a specific case when you are trapped by a gas spore
        # SPORE_TRAP_FIX: the HP test was a ratio, true at any HP: a trapped hero hit the spore at 9 HP and its
        # 4d6 blast (explode.c, up to 24) killed it. A spore never attacks (AT_BOOM only when it dies), so waiting
        # for it to float off is safe; hit it only when the blast cannot kill us.
        if len(agent.get_visible_monsters()) == 1 and \
                (agent.blstats.hitpoints > 24 if jf_config.SPORE_TRAP_FIX
                 else agent.blstats.hitpoints / agent.blstats.max_hitpoints):
            dis = agent.bfs()
            for y2, x2 in zip(*np.nonzero(dis != -1)):
                if not adjacent((y, x), (y2, x2)):
                    return ret
            agent.stats_logger.log_event('melee_gas_spore')
            return 1  # a priority higher than random moving around

    return ret


WATCH_GLYPHS = frozenset(MON.from_name(n) for n in ('watchman', 'watch captain'))


def missiles_risk_the_watch(agent):
    """Minetown: a stray missile (a miss, or the rest of a volley past a dying target) that hits a peaceful
    out of sight angers the Watch (a volley killed a Mordor orc and its 2nd dagger hit a hobbit behind it;
    the watchmen killed the XL8). Melee only there."""
    gl = agent.global_logic
    if gl.minetown_level is not None and agent.current_level().key() == gl.minetown_level:
        return True
    return utils.any_in(agent.glyphs, WATCH_GLYPHS)


# role-ran-a: Ranger point-blank archery. With the launcher in hand and matching ammo, a Ranger shoots an
# adjacent monster instead of swapping to its dagger (fight2's 'melee' wields the best melee weapon first).
#  - wield.c ready_weapon (l.162) / dowield (l.306): every wield takes a move, so the bow->dagger swap when
#    a monster steps adjacent, and the dagger->bow swap for the next target at range, each hand the monster
#    a free round (57-86 swap pairs per champion Ranger game; a wererat and a werejackal both bit through
#    such a swap in confirm 203062 / 202550, and the lycanthropy it passed on (mhitu.c AD_WERE l.1312, 1/4 per
#    bite) ended both grinds).
#  - dothrow.c thitmonst (l.1563): to-hit gets +(3 - distance), so +2 at point blank; dothrow.c l.146:
#    Rangers fire rnd(2+) arrows per volley (+1 more for elven/orcish bow+arrows, +1 at Skilled bow);
#    a +2 arrow does d6+2 (d5+2 orcish) against the +1 dagger's d4+1, and nothing requires a gap
#    (zap.c bhit l.3326 hits the first monster on the path, the adjacent square included).
# Only where the stock logic would melee: weak monsters (arrow breakage, dothrow.c l.1713) and the
# ranged-only / exploding kinds keep their old handling. Any exception keeps the old priority.
RANGER_POINT_BLANK = True


def ranger_point_blank(agent, launcher, ammo):
    """A Ranger (not polymorphed) whose best ranged set is a wielded launcher with matching ammo."""
    try:
        return bool(RANGER_POINT_BLANK) and agent.character.role == agent.character.RANGER and \
            not agent.character.prop.polymorph and \
            launcher is not None and ammo is not None and launcher.equipped and \
            ammo.is_fired_projectile(launcher)
    except Exception:
        return False


def ranger_point_blank_priority(agent, monster, default):
    """The shot's priority: one above what melee_monster_priority gives with a melee weapon in hand (16 when
    HP > 8 or the monster is faster), so it wins over melee, which would first swap to the dagger."""
    try:
        _, _, _, mon, _ = monster
        if mon.mname in WEAK_MONSTERS or mon.mname in ONLY_RANGED_SLOW_MONSTERS or \
                mon.mname in EXPLODING_MONSTERS:
            return default
        ret = 2
        if agent.blstats.hitpoints > 8 or is_monster_faster(agent, monster):
            ret += 15
        if 'were' in mon.mname:
            ret += 1
        return ret
    except Exception:
        return default


# hypothesis: a Valkyrie's kitten that steps out of sight into a dark corridor is still there: the dagger
# thrown at a monster behind it kills the pet ("It yowls!  You kill it!  You hear the rumble of distant
# thunder": -15 alignment, -5 Luck), every prayer of the Dlvl 1-3 grind then fails and she starves (judge
# seed 0 died that way at T3395 on Dlvl 1; seed 13 hit an unseen pet twice: "It yelps!  The dagger hits it").
# Only visible floor squares are known to be free of the pet. Every role that throws or fires (Rogue daggers,
# Ranger arrows) faces the same risk, so it is not gated by role.
UNSEEN_PET_TURNS = 20   # (the value in use is jf_config.UNSEEN_PET_TURNS)


def unseen_pet_may_be_at(agent, y, x):
    if not jf_config.UNSEEN_PET_GUARD:
        return False
    where = getattr(agent, '_last_pet_where', None)
    if where is None or agent.glyphs[y, x] in G.VISIBLE_FLOOR or utils.any_in(agent.glyphs, G.PETS):
        return False
    key, turn, positions = where
    elapsed = agent.blstats.time - turn
    if key != (agent.blstats.dungeon_number, agent.blstats.level_number) or elapsed > jf_config.UNSEEN_PET_TURNS:
        return False
    reach = 2 + int(1.5 * elapsed)   # a kitten is speed 18 against our 12
    if any(max(abs(py - y), abs(px - x)) <= reach for py, px in positions):
        agent.log(f'UNSEEN PET may be at {(int(y), int(x))}: last seen {positions} {elapsed} turns ago, no throw')
        return True
    return False


def ranged_priority(agent, dy, dx, monsters):
    if missiles_risk_the_watch(agent):
        return None
    ret = 11

    closest_mon_dis = float('inf')
    for monster in monsters:
        _, my, mx, mon, _ = monster
        assert my != agent.blstats.y or mx != agent.blstats.x
        if mon.mname not in WEAK_MONSTERS + ONLY_RANGED_SLOW_MONSTERS:
            closest_mon_dis = min(closest_mon_dis, line_dis_from(agent, my, mx))

    if closest_mon_dis == 1:
        ret -= 11

    launcher, ammo = agent.inventory.get_best_ranged_set()
    if ammo is None:
        return None

    if launcher is not None and not launcher.equipped:
        ret -= 5

    y, x = agent.blstats.y, agent.blstats.x
    while True:
        y += dy
        x += dx
        if not 0 <= y < agent.glyphs.shape[0] or not 0 <= x < agent.glyphs.shape[1]:
            return None

        if agent.glyphs[y, x] in G.PETS or not agent.current_level().walkable[y, x]:
            return None

        if agent.glyphs[y, x] not in G.MONS and line_dis_from(agent, y, x) > 1 and unseen_pet_may_be_at(agent, y, x):
            return None

        if agent.glyphs[y, x] in G.MONS:
            monster = [m for m in monsters if m[1] == y and m[2] == x]
            if not monster:
                # there is a monster that shouldn't be attacked
                return None
            assert len(monster) == 1
            _, _, _, mon, _ = monster[0]
            dis = line_dis_from(agent, y, x)
            if dis > agent.character.get_range(launcher, ammo):
                return None
            if dis in (1, 2):
                ret -= 5
            if dis == 1:
                ret -= 6
                if mon.mname == 'gas spore':  # only gas spore ?
                    ret -= 100
            # hypothesis: a gas spore's explosion (radius 1) that kills the pet costs -15 alignment
            # ("rumble of distant thunder"), after which every prayer fails and the character
            # starves (DT6A seed 1). Astra: kill spores from range only, away from pets.
            if mon.mname == 'gas spore' and spore_blast_hits_friend(agent, y, x):
                return None
            # a miss, or the rest of a multishot volley, flies on past the target: never with a pet or a
            # peaceful behind it (two unseen games hit Minetown gnomes that way: the Watch killed them)
            by, bx, reach = y, x, agent.character.get_range(launcher, ammo)
            for _ in range(max(reach - dis, 0)):
                by += dy
                bx += dx
                if not 0 <= by < agent.glyphs.shape[0] or not 0 <= bx < agent.glyphs.shape[1] or \
                        not agent.current_level().walkable[by, bx]:
                    break
                if agent.glyphs[by, bx] in G.PETS or \
                        (agent.glyphs[by, bx] in G.MONS and not any(m[1] == by and m[2] == bx for m in monsters)):
                    return None
                if agent.glyphs[by, bx] not in G.MONS and unseen_pet_may_be_at(agent, by, bx):
                    return None
            if dis == 1 and ranger_point_blank(agent, launcher, ammo):
                ret = ranger_point_blank_priority(agent, monster[0], ret)
            return ret, y, x, monster[0]


def get_next_states(agent, wand, y, x, dy, dx):
    if not inside(agent, y, x) or not agent.current_level().walkable[y, x]:
        can_bounce = wand.is_ray_wand()
        if not can_bounce:
            return []
        if dy == 0 or dx == 0:
            return [(y - dy, x - dx, -dy, -dx, 1.0, 1)]
        # TODO: diagonal
        side1 = (y, x - dx)
        side2 = (y - dy, x)
        side1_wall = not inside(agent, *side1) or not agent.current_level().walkable[side1]
        side2_wall = not inside(agent, *side2) or not agent.current_level().walkable[side2]
        dy1, dx1 = side2[0] - side1[0], side2[1] - side1[1]
        dy2, dx2 = side1[0] - side2[0], side1[1] - side2[1]
        if side1_wall and side2_wall:
            return [(y - dy, x - dx, -dy, -dx, 1.0, 1)]
        elif not side1_wall and not side2_wall:
            return [(y - dy, x - dx, -dy, -dx, 1 / 20, 1),
                    (y + dy1, x + dx1, dy1, dx1, 19 / 40, 1),
                    (y + dy2, x + dx2, dy2, dx2, 19 / 40, 1)]
        elif side1_wall:
            return [(y + dy1, x + dx1, dy1, dx1, 1.0, 1)]
        elif side2_wall:
            return [(y + dy2, x + dx2, dy2, dx2, 1.0, 1)]
        else:
            assert 0
    return [(y + dy, x + dx, dy, dx, 1.0, 0)]


def _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, range_left, hit_targets, probability):
    if range_left < 0:
        return

    for y, x, dy, dx, next_prob, range_penalty in get_next_states(agent, wand, y, x, dy, dx):
        range_left -= range_penalty
        monster = [m for m in monsters if m[1] == y and m[2] == x]
        if monster:
            assert len(monster) == 1
            monster = monster[0]
            # For each monster hit, range decreases by 2.
            range_left -= 2
        elif inside(agent, y, x) and agent.glyphs[y, x] in G.PETS:
            monster = 'pet'
            # For each monster hit, range decreases by 2.
            range_left -= 2
        elif inside(agent, y, x) and agent.glyphs[y, x] in G.MONS and (y, x) != (agent.blstats.y, agent.blstats.x):
            # a monster that isn't a known hostile: a peaceful (a lightning bolt at a wraith hit a watch
            # captain and the Watch killed the XL10)
            monster = 'peaceful'
            range_left -= 2
        elif agent.blstats.y == y and agent.blstats.x == x:
            monster = 'self'
            range_left -= 2
        else:
            monster = None

        hit_targets[(y, x, monster)] += probability * next_prob

        _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, range_left - 1, hit_targets, 1.0)


def simulate_wand_path(agent, wand, monsters, dy, dx):
    """ Returns list of tuples (y, x, hit_object, expected_hit_count).
    """
    y, x = agent.blstats.y, agent.blstats.x

    # TODO: random range left from 6 or 7 to 13
    hit_targets = defaultdict(int)
    _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, 13, hit_targets, 1.0)
    for (y, x, hit_object), expected_hit_count in hit_targets.items():
        yield y, x, hit_object, expected_hit_count


_WAN_COLD = None


def get_potential_wand_usages(agent, monsters, dy, dx):
    global _WAN_COLD
    if _WAN_COLD is None:
        from .. import objects as O
        import nle.nethack as nh
        _WAN_COLD = O.from_name('cold', nh.WAND_CLASS)
    ret = []
    if missiles_risk_the_watch(agent):
        return ret
    # FB_OVER_RAYS: in the grind a Wizard's force bolt replaces the ray plans, whose bounce killed three of them
    if jf_config.FB_OVER_RAYS and _fb_castable(agent) and (
            not getattr(getattr(getattr(agent, 'global_logic', None), 'dive', None), 'diving', True)):
        return ret
    player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
    # TODO: also get items recursively from bags
    for item in agent.inventory.items:
        targeted_monsters = set()
        if not item.is_offensive_usable_wand() or agent.inventory.is_known_empty(item):
            continue
        # dive_logic.COLD_RESERVE: a known wand of cold is kept for Medusa's moat (MEDUSA_FREEZE) and the castle's
        dive = getattr(getattr(agent, 'global_logic', None), 'dive', None)
        if dive is not None and dive.cold_reserved(item):
            continue
        # MINO_GUARD (mino_guard.MinoGuard.reserved): a known wand of death is kept for a minotaur
        guard = getattr(dive, 'mino_guard', None) if dive is not None else None
        if guard is not None and guard.reserved(item):
            continue
        priority = 0
        # print('--------------', dy, dx)
        for y, x, monster, p in simulate_wand_path(agent, item, monsters, dy, dx):
            # print(y, x, monster, p)
            if monster == 'pet':
                priority -= p * 20
            elif monster == 'peaceful':
                priority -= p * 200
            elif monster == 'self':
                # SELF_ZAP_FIX: our own bounce costs what the ray does to us, not a flat 30 per expected hit --
                # 6d6 lightning (and blindness), fire, cold, 2d6 magic missile, death. A known wand of lightning at an
                # adjacent giant ant with a wall behind it scored +25 per pass (out and back, 'dangerous') against -30
                # for the return through us: tr0 val-hum-law-fem s210 zapped it 4 times from 68/68 HP ('The bolt of
                # lightning bounces! The bolt of lightning hits you!', blinded, then hit again blind) and died of its
                # own bolts; v2a val-dwa-law-fem s206 the same at a giant ant; b3 wiz-orc-cha-mal s601 and wz0
                # wiz-gno-neu-mal s619 by their own bolts of fire. Only a Valkyrie's cold keeps the old weight.
                if jf_config.SELF_ZAP_FIX and not (item.objs[0] == _WAN_COLD and
                                                   agent.character.role == agent.character.VALKYRIE):
                    priority -= p * jf_config.SELF_ZAP_PENALTY
                else:
                    priority -= p * 30
            elif monster is not None:
                _, y, x, mon, _ = monster
                if mon.mname in WEAK_MONSTERS:
                    priority += min(p, 1) * 1
                elif is_dangerous_monster(monster):
                    priority += p * 25
                else:
                    priority += min(p, 1) * 10
                targeted_monsters.add((y, x, monster))
        if jf_config.SHOP_SAFETY and any(getattr(m[3], 'mname', '') == 'gas spore' and
                                         spore_blast_hits_friend(agent, ty, tx)
                                         for ty, tx, m in targeted_monsters):
            # SHOP_SAFETY: SPORE_SAFE kept fight2's melee and throws off a gas spore whose blast reaches a peaceful, a
            # shop square or the pet, but not its wand plans: wz0 wiz-elf-cha-mal s631 zapped lightning at a spore in a
            # shop -- 'You kill the gas spore! Changdu is caught in the gas spore's explosion! Changdu gets angry!'
            continue
        if targeted_monsters:
            # priority = priority * (1 - player_hp_ratio) - 10
            priority = priority - 15
            if jf_config.AT_ELBERETH_FIX:
                # a zap wipes nothing (zap.c has no u_wipe_engr); only hitting a monster the engraving scares
                # erases it (mon.c setmangry: 'You feel like a hypocrite')
                priority += min((elbereth_attack_penalty(agent, monsters, m) for _, _, m in targeted_monsters),
                                default=0)
            elif on_scaring_elbereth(agent, monsters):
                priority -= 100
            ret.append((priority, ('zap', dy, dx, item, targeted_monsters)))
    return ret


def at_ignorer_adjacent(agent, monsters):
    """AT_ELBERETH_FIX: a monster that melees through Elbereth stands next to us (@ humans and elves, minotaurs;
    an unseen attacker that hurt us on an intact Elbereth). Then waiting on the engraving only hands it free hits:
    base4-jf16 s4 stood 5 turns in its dig pit on Elbereth while two Green-elves took it from 65 to 12 HP -- the
    Elbereth penalty on every attack (-100) left 'wait' as fight2's best action. Attacking it doesn't cost the
    engraving's protection against it (it has none), and kills it."""
    if not jf_config.AT_ELBERETH_FIX:
        return False
    dive = agent.global_logic.dive
    y0, x0 = agent.blstats.y, agent.blstats.x
    return any(adjacent((m[1], m[2]), (y0, x0)) and dive._melee_ignores_elbereth(m[3]) for m in monsters)


def on_scaring_elbereth(agent, monsters):
    """Standing on an Elbereth that actually protects us (see at_ignorer_adjacent)."""
    return agent.inventory.engraving_below_me.lower() == 'elbereth' and not at_ignorer_adjacent(agent, monsters)


def elbereth_attack_penalty(agent, monsters, target):
    """The -100 fight2 puts on attacking from an Elbereth square. Every attack wipes letters off the engraving
    (uhitm.c attack -> u_wipe_engr(3); dothrow.c u_wipe_engr(2)), so it is right against the monsters the
    engraving holds off, but not against one it doesn't (AT_ELBERETH_FIX): that one gets hit, and the Elbereth
    is written again once it is dead (DIG_ESCAPE, fight2's elbereth action)."""
    if agent.inventory.engraving_below_me.lower() != 'elbereth':
        return 0
    if jf_config.AT_ELBERETH_FIX and target is not None and \
            agent.global_logic.dive._melee_ignores_elbereth(target[3]):
        return 0
    return -100


def focus_ignorer(agent, mon):
    """AT_ELBERETH_FIX + AT_FOCUS: hit the monster that ignores Elbereth first while an Elbereth could hold off the
    rest (on one already, or able to write one; not in Gehennom, not blind). The first swing at it smudges the
    engraving, and fight2 then turned on the crowd: harness Big Room seeds 16 and 37 hit a yeti, a jaguar, a hill
    orc and a red naga with the Woodland-/Green-elf still alive, which kept the dig out off (an adjacent ignorer
    stops it) until the XL7 was down to 3-11 HP. With the ignorer dead the dig out writes Elbereth and digs on."""
    if not jf_config.AT_ELBERETH_FIX or not jf_config.AT_FOCUS or in_gehennom(agent) or agent.character.prop.blind:
        return False
    if not agent.global_logic.dive._melee_ignores_elbereth(mon):
        return False
    return agent.inventory.engraving_below_me.lower() == 'elbereth' or agent.can_engrave()


def in_gehennom(agent):
    """monmove.c onscary(): Elbereth scares nothing in Gehennom (Inhell); engraving it there only hands out a
    free hit, and waiting on it is standing still under attack."""
    return jf_config.GEHENNOM_DIVE and agent.current_level().dungeon_number == 1


def elbereth_action(agent, monsters):
    if agent.inventory.engraving_below_me.lower() == 'elbereth':
        return []
    if in_gehennom(agent):
        return []
    if not agent.can_engrave():
        return []
    adj_monsters_count = 0
    dive = agent.global_logic.dive
    for monster in monsters:
        _, my, mx, mon, _ = monster
        if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
            continue
        if not adjacent((my, mx), (agent.blstats.y, agent.blstats.x)):
            continue
        if jf_config.AT_ELBERETH_FIX and dive._melee_ignores_elbereth(mon):
            continue   # the engraving doesn't hold it off: writing it only hands it a free hit
        multiplier = np.clip(20 / agent.blstats.hitpoints, 1.0, 1.5)
        if is_monster_faster(agent, monster):
            multiplier *= 2
        if mon in WEAK_MONSTERS:
            adj_monsters_count += 0.1 * multiplier
            continue
        adj_monsters_count += 1 * multiplier
        if is_dangerous_monster(monster):
            adj_monsters_count += 2 * multiplier

    player_hp_ratio = (agent.blstats.hitpoints / agent.blstats.max_hitpoints) ** 0.5
    if agent.blstats.hitpoints < 30 and adj_monsters_count > 0:
        return [(-15 + 20 * adj_monsters_count * (1 - player_hp_ratio), ('elbereth',))]
    return []


def wait_action(agent, monsters):
    # RANGED_ON_ELB: no waiting on Elbereth while something shoots at us (it only stops melee)
    if agent.global_logic.dive.shot_recently():
        return []
    if on_scaring_elbereth(agent, monsters) and not in_gehennom(agent):
        player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
        priority = 30 - player_hp_ratio * 40
        return [(priority, ('wait',))]
    return []


def get_available_actions(agent, monsters):
    actions = []

    # melee attack actions
    for monster in monsters:
        _, y, x, mon, _ = monster
        if adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, monster)
            priority += elbereth_attack_penalty(agent, monsters, monster)
            if focus_ignorer(agent, mon):
                priority += jf_config.AT_FOCUS
            dy = y - agent.blstats.y
            dx = x - agent.blstats.x
            # hypothesis: refusing all bare contact with cockatrices prevents
            # instant petrification, while leaving ranged attacks and retreat
            # available to both armed and unarmed characters.
            bare_handed = agent.inventory.items.main_hand is None
            bare_hands = agent.inventory.items.gloves is None
            bare_feet = agent.inventory.items.boots is None
            if ord(mon.mlet) == MON.S_COCKATRICE and bare_handed and bare_hands:
                if not bare_feet:
                    actions.append((priority, ('kick', dy, dx)))
            else:
                actions.append((priority, ('melee', dy, dx)))

    # ranged attack actions
    for dy, dx in product([-1, 0, 1], [-1, 0, 1]):
        if dy != 0 or dx != 0:
            ranged_pr = ranged_priority(agent, dy, dx, monsters)
            if ranged_pr is not None:
                pri, y, x, monster = ranged_pr
                pri += elbereth_attack_penalty(agent, monsters, monster)
                if all(monster[3].mname in ONLY_RANGED_SLOW_MONSTERS for monster in monsters):
                    pri += 10
                actions.append((pri, ('ranged', dy, dx)))

            actions.extend(get_potential_wand_usages(agent, monsters, dy, dx))

    actions.extend(force_bolt_actions(agent, monsters))

    to_pickup = decide_what_to_pickup(agent)
    if to_pickup:
        actions.append((15, ('pickup', to_pickup)))

    actions.extend(elbereth_action(agent, monsters))
    actions.extend(wait_action(agent, monsters))

    return actions


FORCE_BOLT_RANGE = 6  # the bolt flies rn1(8, 6) squares, so 6 always reaches


FORCE_BOLT_MAX_RANGE = 13  # rn1(8, 6)


def _force_bolt_tail_safe(agent, level, shop, y0, x0, sy, sx, reach=FORCE_BOLT_MAX_RANGE):
    """The bolt flies on past the monster it hits (bhit: range -= 3) and breaks fragile objects on its way:
    one killed a gnome zombie in a shop doorway and shattered a potion behind it (100 zorkmids, then the
    shopkeeper). Refuse a line that reaches a known shop, or two visible objects (shop stock we have not
    entered yet) before a wall; a lone corpse is no reason to melee instead."""
    objects = 0
    for k in range(1, reach + 1):
        y, x = y0 + sy * k, x0 + sx * k
        if not (0 <= y < level.walkable.shape[0] and 0 <= x < level.walkable.shape[1]):
            return True
        if shop is not None and shop[y, x]:
            return False
        # a bolt that kills the target flies on into whatever stands behind it: a jackal's bolt hit the
        # Wizard's own housecat, which turned on it and killed it
        if k > 1 and (agent.glyphs[y, x] in G.PETS or agent.monster_tracker.peaceful_monster_mask[y, x]):
            return False
        if agent.glyphs[y, x] in G.OBJECTS:
            objects += 1
            if objects >= 2:
                return False
        if not level.walkable[y, x] and level.seen[y, x]:
            return True
    return True


def _fb_castable(agent):
    """FB_OVER_RAYS: force bolt can be cast now (force_bolt_actions' own preconditions, before any target)."""
    character = agent.character
    try:
        return bool(jf_config.FORCE_BOLT and 'force bolt' in getattr(character, 'known_spells', {}) and
                    agent.blstats.energy >= 5 and agent.blstats.hunger_state < Hunger.WEAK and
                    not character.prop.polymorph and agent.blstats.carrying_capacity < 2 and
                    character.spell_fail_chance.get('force bolt', 1) <= 0.3 and
                    not (jf_config.FB_SANITY and _fb_cannot_cast(agent)))
    except Exception:
        return False


def _fb_cannot_cast(agent):
    """FB_SANITY: spell.c rejectcasting(): stunned -> 'You are too impaired to cast a spell.' (no turn passes, so
    fight2 chose the bolt again and again); spelleffects(): confused -> 'You fail to cast the spell correctly.' every
    time, for half the Pw; or a refusal agent.cast saw lately ('Your arms are not free to cast!', 'You lack the
    strength to cast spells.', a forgotten spell's backfire)."""
    prop = agent.character.prop
    if prop.stun or prop.confusion:
        return True
    return agent.blstats.time < getattr(agent, '_cast_refused_until', -1)


def _fb_path_scares(agent, monsters, y0, x0, sy, sx, reach):
    """FB_FOCUS on an Elbereth square: could the bolt hit something the engraving scares, or a peaceful? Every monster
    the bolt passes is hit (zap.c bhit: ZAPPED_WAND goes on with range -= 3) and woken via_attack (bhitm -> wakeup ->
    mon.c setmangry): 'You feel like a hypocrite.' (-5 alignment) and the engraving is deleted. The path ends where the
    bolt stops: a seen wall or closed door."""
    level = agent.current_level()
    by_pos = {(int(m[1]), int(m[2])): m for m in monsters}
    for k in range(1, reach + 1):
        y, x = y0 + sy * k, x0 + sx * k
        if not (0 <= y < level.walkable.shape[0] and 0 <= x < level.walkable.shape[1]):
            return False
        if not level.walkable[y, x] and level.seen[y, x]:
            return False
        g = agent.glyphs[y, x]
        if g in G.MONS or g in G.INVISIBLE_MON or g in G.PETS:
            m = by_pos.get((y, x))
            if m is None or elbereth_attack_penalty(agent, monsters, m) != 0:
                return True
    return False


def _fb_trivial(mon):
    """FB_RESERVE: a monster one or two quarterstaff blows kill, no reason to spend the last bolts on (makemon
    difficulty <= FB_RESERVE_DIFF, not faster than us). The passive and exploding kinds the bolt keeps us off
    (floating eye, acid blob, molds, gas spore) are never trivial."""
    name = getattr(mon, 'mname', '')
    if name in ONLY_RANGED_SLOW_MONSTERS or name in EXPLODING_MONSTERS or name == 'unknown':
        return False
    return getattr(mon, 'difficulty', 99) <= jf_config.FB_RESERVE_DIFF and getattr(mon, 'mmove', 99) <= 12


def _fb_reserve(agent):
    """FB_RESERVE: the Pw kept for real threats (0: none): from XL FB_RESERVE_XL and HP >= FB_RESERVE_HP of max, at most
    a third of max Pw."""
    bl = agent.blstats
    if bl.experience_level < jf_config.FB_RESERVE_XL or bl.hitpoints < jf_config.FB_RESERVE_HP * bl.max_hitpoints:
        return 0
    return min(int(jf_config.FB_RESERVE), int(bl.max_energy) // 3)


def _fb_shopkeepers_home(agent, level):
    """FB_SHOP_KNOWN: every shopkeeper in view stands in (or at the door of) a shop whose interior we know: the tail
    check's dilated shop mask then keeps the bolt off the stock."""
    if not level.shop_interior.any():
        return False
    known = utils.dilate(level.shop_interior, radius=1)
    ys, xs = utils.isin(agent.glyphs, G.SHOPKEEPER).nonzero()
    return all(known[y, x] for y, x in zip(ys, xs))


def force_bolt_actions(agent, monsters):
    """Cast force bolt (2d6, rarely misses) at the nearest hostile on a straight, clear line.

    Wizards start knowing it and it hits about twice as hard as their quarterstaff, yet the fight
    heuristic only ever meleed; rank the bolt just above meleeing the same monster.
    Ported from CleverShovel/nethacker@0d1fb22.
    """
    character = agent.character
    if not jf_config.FORCE_BOLT or 'force bolt' not in getattr(character, 'known_spells', {}) or \
            agent.blstats.energy < 5:
        return []
    if agent.blstats.hunger_state >= Hunger.WEAK or character.prop.polymorph:  # "too hungry to cast"
        return []
    if agent.blstats.carrying_capacity >= 2:  # Stressed: "Your concentration falters"
        return []
    if character.spell_fail_chance.get('force bolt', 1) > 0.3:
        return []
    if jf_config.FB_SANITY and _fb_cannot_cast(agent):
        return []
    on_elbereth = agent.inventory.engraving_below_me.lower() == 'elbereth'
    if on_elbereth and not jf_config.FB_FOCUS:
        return []
    y0, x0 = agent.blstats.y, agent.blstats.x
    level = agent.current_level()
    # the bolt breaks fragile objects on its way: a shop's camera cost 200 zorkmids, then the shopkeeper
    if level.shop_interior[y0, x0] or (utils.isin(agent.glyphs, G.SHOPKEEPER).any() and
                                        not (jf_config.FB_SHOP_KNOWN and _fb_shopkeepers_home(agent, level))):
        return []
    walkable = level.walkable
    peaceful = agent.monster_tracker.peaceful_monster_mask
    # ... and whatever lies under the monster it kills: a bolt from a shop's doorway shattered a potion
    shop = utils.dilate(level.shop_interior, radius=1) if level.shop_interior.any() else None
    reach = int(jf_config.FB_TAIL_REACH) if jf_config.FB_SANITY else FORCE_BOLT_MAX_RANGE
    reserve = _fb_reserve(agent) if jf_config.FB_RESERVE else 0
    best = None
    for monster in monsters:
        _, y, x, mon, _ = monster
        dy, dx = y - y0, x - x0
        dist = max(abs(dy), abs(dx))
        if dist == 0 or dist > FORCE_BOLT_RANGE or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            continue
        if mon.mname in EXPLODING_MONSTERS and dist == 1:
            continue
        sy, sx = int(np.sign(dy)), int(np.sign(dx))
        cy, cx, clear = y0, x0, True
        for _ in range(dist - 1):
            cy, cx = cy + sy, cx + sx
            if not walkable[cy, cx] or agent.glyphs[cy, cx] in G.PETS or peaceful[cy, cx]:
                clear = False
                break
        if not clear or not _force_bolt_tail_safe(agent, level, shop, y0, x0, sy, sx, reach):
            continue
        # FB_FOCUS: from an Elbereth square only at what it doesn't scare, with nothing it scares on the way
        if on_elbereth and (elbereth_attack_penalty(agent, monsters, monster) != 0 or
                            _fb_path_scares(agent, monsters, y0, x0, sy, sx, reach)):
            continue
        # FB_RESERVE: the last bolts are not for trivial monsters
        if reserve and agent.blstats.energy - 5 < reserve and _fb_trivial(mon):
            continue
        priority = melee_monster_priority(agent, monsters, monster) + 2 if dist == 1 else 14
        if jf_config.FB_FOCUS and dist == 1 and focus_ignorer(agent, mon):
            priority += jf_config.AT_FOCUS   # as fight2's melee at the same adjacent ignorer (get_available_actions)
        if best is None or priority > best[0]:
            best = (priority, ('force_bolt', sy, sx))
    return [best] if best is not None else []


def decide_what_to_pickup(agent):
    # never a shop's goods: an unseen game picked up a for-sale dagger (Grimtooth), threw it, owed 2204
    # zorkmids and was killed by the shopkeeper
    projectiles_below_me = [i for i in agent.inventory.items_below_me
                            if (i.is_thrown_projectile() or i.is_fired_projectile()) and
                            i.shop_status == Item.NOT_SHOP]
    my_launcher, ammo = agent.inventory.get_best_ranged_set(additional_ammo=[i for i in projectiles_below_me])
    to_pickup = []
    for item in agent.inventory.items_below_me:
        if item.shop_status != Item.NOT_SHOP:
            continue
        if item.is_thrown_projectile() or (my_launcher is not None and item.is_fired_projectile(launcher=my_launcher)):
            to_pickup.append(item)
    return to_pickup


def goto_action(agent, priority, monsters):
    values = []
    walkable = agent.current_level().walkable
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
        y, x = agent.blstats.y - dy, agent.blstats.x - dx
        if not 0 <= y < walkable.shape[0] or not 0 <= x < walkable.shape[1]:
            continue
        if not np.isnan(priority[y, x]):
            values.append(priority[y, x])
    if len(set(values)) > 1:
        return []

    assert monsters
    for monster in monsters:
        _, my, mx, mon, _ = monster
        if not adjacent((agent.blstats.y, agent.blstats.x), (my, mx)):
            # and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
            return [(1, ('go_to', my, mx))]
    # every monster adjacent and nothing to do (e.g. no weapon known): fight2 falls back to moving/waiting
    return []


def get_corridors_priority_map(walkable):
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wall_count = signal.convolve2d((~walkable).astype(int), k, boundary='symm', mode='same')
    corridor_mask = (wall_count == 6).astype(int)
    corridor_mask[~walkable] = 0
    corridor_dilated = signal.convolve2d(corridor_mask.astype(int), k, boundary='symm', mode='same')
    return corridor_mask + corridor_dilated >= 1


def get_priorities(agent):
    """ Returns a pair (move priority heatmap, other actions (with priorities) list) """
    walkable = agent.current_level().walkable
    priority = np.zeros(walkable.shape, dtype=float)
    # without the monsters a stalled fight let go of (jf_config.FIGHT_STALL_TURNS)
    monsters = agent.fight_monsters()
    for m in monsters:
        draw_monster_priority_positive(agent, m, priority, walkable)
    for m in monsters:
        draw_monster_priority_negative(agent, m, priority, walkable)
    priority[~walkable] = float('nan')

    # TODO: figure out how to use corridors priority so that it improves the score
    # if len([m for m in monsters if m[3].mname not in chain(ONLY_RANGED_SLOW_MONSTERS, WEAK_MONSTERS)]) >= 4:
    #     priority += get_corridors_priority_map(walkable)
    # for _, _, _, mon, _ in monsters:
    #     if ord(mon.mlet) == MON.S_ANT:
    #         priority += get_corridors_priority_map(walkable)
    #         break

    # use relative priority to te current position
    priority -= priority[agent.blstats.y, agent.blstats.x]

    actions = get_available_actions(agent, monsters)
    if not any(a[1][0] in ('melee', 'kick', 'ranged', 'force_bolt') for a in actions):
        actions.extend(goto_action(agent, priority, monsters))
    return priority, actions


def get_move_actions(agent, dis, move_priority_heatmap):
    """ Returns list of tuples (priority, ('move', dy, dx)) """
    ret = []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        y, x = agent.blstats.y + dy, agent.blstats.x + dx
        if not 0 <= y < dis.shape[0] or not 0 <= x < dis.shape[1]:
            continue
        if not dis[y, x] == 1:
            continue
        # a monster left out of the fight (the Valley's graveyard sleepers, VALLEY_GRAVE_FILTER) still occupies its
        # square: moving there panics ('Monster on a next tile') in a loop
        if in_gehennom(agent) and agent.monster_tracker.monster_mask[y, x]:
            continue

        if not np.isnan(move_priority_heatmap[y, x]):
            ret.append((move_priority_heatmap[y, x], ('move', dy, dx)))
    return ret
