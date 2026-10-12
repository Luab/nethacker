from collections import defaultdict
from itertools import product

import numpy as np
from scipy import signal

from ..glyph import G, MON
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
    # a pet seen here lately but out of view now may be right behind the spore: a thrown dagger's blast
    # killed an unseen kitten ('You kill it!', 'rumble of distant thunder': -15 alignment on a Valkyrie's
    # record that starts at 0, so the first grind prayer failed at T1364)
    seen = agent.global_logic.dive.pet_seen.get(agent.current_level().key())
    return seen is not None and agent.blstats.time - seen < 100 and not utils.any_in(agent.glyphs, G.PETS)


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
        if len(agent.get_visible_monsters()) == 1 \
                and agent.blstats.hitpoints / agent.blstats.max_hitpoints:
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


def get_potential_wand_usages(agent, monsters, dy, dx):
    ret = []
    if missiles_risk_the_watch(agent):
        return ret
    player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
    # TODO: also get items recursively from bags
    for item in agent.inventory.items:
        targeted_monsters = set()
        if not item.is_offensive_usable_wand() or agent.inventory.is_known_empty(item):
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
        if targeted_monsters:
            # priority = priority * (1 - player_hp_ratio) - 10
            priority = priority - 15
            if agent.inventory.engraving_below_me.lower() == 'elbereth':
                priority -= 100
            ret.append((priority, ('zap', dy, dx, item, targeted_monsters)))
    return ret


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
    for monster in monsters:
        _, my, mx, mon, _ = monster
        if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
            continue
        if not adjacent((my, mx), (agent.blstats.y, agent.blstats.x)):
            continue
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
    if agent.inventory.engraving_below_me.lower() == 'elbereth' and not in_gehennom(agent):
        player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
        priority = 30 - player_hp_ratio * 40
        return [(priority, ('wait',))]
    return []


def distant_flash_directions(agent, monsters):
    """Unit directions of minotaurs two squares off in a straight line with an open square between, while diving
    with a charged camera (none flashed in the last 3 turns)."""
    # hypothesis: a maze-filler minotaur (speed 15, 3d10/3d10/2d8, ignores Elbereth) killed 5 of 30 dives at
    # Dlvl 25-28, and the camera only fired once it was adjacent -- after its first round (s2: 70 -> 29 HP, dead
    # the next turn). The flash reaches along a line and scares what it blinds within distance^2 < 9 (apply.c
    # use_camera -> flash_hits_mon: monflee 3 in 4), so flash it one square before it closes, and let the dig
    # stop for that turn instead of taking its full round first.
    # sources: https://nethackwiki.com/wiki/Minotaur, https://nethackwiki.com/wiki/Expensive_camera,
    #          https://nethackwiki.com/wiki/Mazes_of_Menace, NetHack 3.6.6 src/apply.c use_camera + src/uhitm.c
    #          flash_hits_mon, /refs/top/1c4099e80253 (_melee_ignores_elbereth)
    if agent.character.prop.blind or agent.character.prop.polymorph or \
            not agent.global_logic.dive.diving or in_gehennom(agent):
        return []
    if not any(item.is_unambiguous() and item.object.name == 'expensive camera' and
               not agent.inventory.is_known_empty(item) for item in agent.inventory.items):
        return []
    if agent.blstats.time - getattr(agent, '_distant_flash_turn', -100) < 3:
        return []
    level = agent.current_level()
    y0, x0 = agent.blstats.y, agent.blstats.x
    dirs = []
    for monster in monsters:
        y, x, mon = monster[1], monster[2], monster[3]
        if getattr(mon, 'mname', '') != 'minotaur':
            continue
        dy, dx = y - y0, x - x0
        if max(abs(dy), abs(dx)) != 2 or dy not in (-2, 0, 2) or dx not in (-2, 0, 2):
            continue
        if not level.walkable[y0 + dy // 2, x0 + dx // 2]:
            continue
        dirs.append((dy // 2, dx // 2))
    return dirs


def camera_actions(agent, monsters):
    """hypothesis: a Tourist's expensive camera (~60-90 charges, unused so far) blinds an adjacent monster and makes
    it flee 3 times in 4 (apply.c use_camera -> flash_hits_mon); flashing attackers at low HP beats trading
    blows at 3/14 HP, which is how most Dlvl 1-3 Tourist games end (sewer rats, hobbits, ants). Only while
    diving: in the levelling grind a fleeing monster is lost XP."""
    if agent.character.prop.blind or agent.character.prop.polymorph or agent.blstats.max_hitpoints <= 0:
        return []
    camera = None
    for item in agent.inventory.items:
        if item.is_unambiguous() and item.object.name == 'expensive camera' and \
                not agent.inventory.is_known_empty(item):
            camera = item
            break
    if camera is None:
        return []
    ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
    if not agent.global_logic.dive.diving:
        return []
    # hypothesis: an adjacent monster that melees through Elbereth (@ humans and elves, minotaurs, the lawful
    # minions: Aleax, couatl) stops every dig step with its attacks, and the dig-diver waited until 50% HP to flash
    # it -- an Aleax took s7's digger 64 -> 23 HP on Dlvl 23 and killed it, a couatl ended s3 on Dlvl 27. Flash
    # such a monster at once, at any HP: blinded, it flees 3 times in 4 and the hole gets dug.
    # sources: https://nethackwiki.com/wiki/Expensive_camera, https://nethackwiki.com/wiki/Elbereth,
    #          https://nethackwiki.com/wiki/Aleax, https://nethackwiki.com/wiki/Tourist, NetHack 3.6.6 src/monmove.c
    #          onscary() (is_lminion), /refs/top/1c4099e80253 (_melee_ignores_elbereth, AT_FOCUS)
    dive = agent.global_logic.dive
    flashed = getattr(agent, '_camera_flashed', {})
    # hypothesis: the flash undoes the Elbereth the dive stands on: a blinded monster no longer respects it
    # (monmove.c onscary), and attacking from the square wipes it ('You feel like a hypocrite. The engraving
    # beneath you fades': fem s5 at Dlvl 12, then a crowd of iguanas, ants and a centaur killed the digger).
    # Leave Elbereth-respecting neighbours alone while it holds; flash only the ones that fight through it.
    # sources: https://nethackwiki.com/wiki/Elbereth, https://nethackwiki.com/wiki/Expensive_camera,
    # https://nethackwiki.com/wiki/Tourist, /refs/top/1c4099e80253 (_melee_ignores_elbereth, AT_ELBERETH_FIX)
    on_elbereth = (agent.inventory.engraving_below_me or '').lower() == 'elbereth' and not in_gehennom(agent)
    # hypothesis: the guard above only looked at an Elbereth already under us, but the dive flashes at low HP
    # and *then* engraves and rests on Elbereth -- and the adjacent flash has blinded the monster for good,
    # so it no longer respects the engraving (a blinded monster that can ordinarily see ignores Elbereth):
    # a plains centaur flashed then rested against killed a Mines camp, s9's scorpion ('turns to flee. The
    # scorpion hits! ... stings!') and s11's coyote / giant ant were flashed right before an Elbereth rest,
    # s7's cobra bit on through its. Where an Elbereth can still be had, leave Elbereth-respecting
    # neighbours unflashed and let the engraving hold them off; flash only the ones that fight through it.
    # sources: https://nethackwiki.com/wiki/Elbereth ('A blinded monster that can ordinarily see will not
    #          respect Elbereth while it is blind'), https://nethackwiki.com/wiki/Expensive_camera,
    #          https://nethackwiki.com/wiki/Tourist, /refs/top/1c4099e80253 (_melee_ignores_elbereth),
    #          /refs/top/47a6c840a4cf (Elbereth-first faint guard)
    # Only once the dive proper has begun (XL 8+): an early fall-dive at XL 1-3 lives on its flashes (s0 flashed a
    # grid bug at 6/14 HP on Dlvl 2 and went on to Dlvl 17; without the flash a bat killed it on Dlvl 3)
    if agent.blstats.experience_level >= 8 and not on_elbereth and not in_gehennom(agent) and \
            dive._elbereth_possible():
        on_elbereth = True
    actions = []
    for dy, dx in distant_flash_directions(agent, monsters):
        actions.append((60, ('camera', dy, dx, camera)))
        agent._distant_flash_turn = agent.blstats.time
    for monster in monsters:
        _, y, x, mon, _ = monster
        if not adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            continue
        if on_elbereth and not dive._melee_ignores_elbereth(mon):
            continue
        if ratio >= 0.5 and not dive._melee_ignores_elbereth(mon):
            continue
        if getattr(mon, 'mflags1', 0) & 0x00001000:  # M1_NOEYES
            continue
        if agent.blstats.time - flashed.get((y, x), -100) < 8:
            continue
        actions.append((25 + 20 * (1 - ratio), ('camera', y - agent.blstats.y, x - agent.blstats.x, camera)))
    return actions


def get_available_actions(agent, monsters):
    actions = []

    # melee attack actions
    for monster in monsters:
        _, y, x, mon, _ = monster
        if adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, monster)
            if agent.inventory.engraving_below_me.lower() == 'elbereth':
                priority -= 100
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
                if agent.inventory.engraving_below_me.lower() == 'elbereth':
                    pri -= 100
                if all(monster[3].mname in ONLY_RANGED_SLOW_MONSTERS for monster in monsters):
                    pri += 10
                actions.append((pri, ('ranged', dy, dx)))

            actions.extend(get_potential_wand_usages(agent, monsters, dy, dx))

    to_pickup = decide_what_to_pickup(agent)
    if to_pickup:
        actions.append((15, ('pickup', to_pickup)))

    actions.extend(camera_actions(agent, monsters))
    actions.extend(elbereth_action(agent, monsters))
    actions.extend(wait_action(agent, monsters))

    return actions


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
    if not any(a[1][0] in ('melee', 'kick', 'ranged') for a in actions):
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

        if not np.isnan(move_priority_heatmap[y, x]):
            ret.append((move_priority_heatmap[y, x], ('move', dy, dx)))
    return ret
