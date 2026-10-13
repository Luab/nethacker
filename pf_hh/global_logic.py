from enum import IntEnum, auto

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from . import objects as O
from . import soko_solver
from . import utils
from . import jf_config
from . import power
from . import castle_power
from .character import Character
from .dive_logic import DiveLogic
from .exceptions import AgentPanic
from .glyph import Hunger, G, MON
from .item import Item, flatten_items
from .item.item_priority_base import ItemPriorityBase
from .level import Level
from .strategy import Strategy


class ItemPriority(ItemPriorityBase):
    MAX_NUMBER_OF_ITEMS = 26 * 2 - 1  # + coin slot, one slot should be left for item arranging
    def __init__(self, agent):
        self.agent = agent
        self._take_sacrificial_corpses = False
        self._drop_gold_till_turn = -float('inf')

    def _split(self, items, forced_items, weight_capacity):
        remaining_weight = weight_capacity
        ret_inv = {}
        for item in forced_items:
            remaining_weight -= item.weight()
            ret_inv[item] = item.count

        ret_bag = {}
        bag = None

        def add_item(item, count=None, to_bag=False):
            nonlocal ret_inv, ret_bag, remaining_weight
            assert isinstance(item, Item)
            if to_bag and bag is not None and item is not bag:
                ret = ret_bag
            else:
                ret = ret_inv

            if remaining_weight < 0 or \
                    (ret is ret_inv and len(ret) >= ItemPriority.MAX_NUMBER_OF_ITEMS):  # TODO: coin slot
                return

            how_many_already_total = ret_inv.get(item, 0) + ret_bag.get(item, 0)
            how_many_already = ret.get(item, 0)
            unit_weight = item.unit_weight(with_content=False)
            # weightless items (e.g. wraith corpses) would make this an infinite count
            max_to_add = item.count if unit_weight <= 0 else int(remaining_weight // unit_weight)
            if count is not None:
                max_to_add = min(max_to_add, count)
            ret[item] = min(item.count, how_many_already_total + max_to_add) - (how_many_already_total - how_many_already)
            remaining_weight -= item.unit_weight(with_content=False) * (ret[item] - how_many_already)

        for item in items:
            if item.is_container() and item.status in [Item.UNCURSED, Item.BLESSED] and item.objs[0].desc == 'bag':
                bag = item  # TODO: select the best
                add_item(bag)

        if self._drop_gold_till_turn < self.agent.blstats.time:
            for item in items:
                if item.category == nh.COIN_CLASS:
                    add_item(item)

        for allow_unknown_status in [False, True]:
            item = self.agent.inventory.get_best_melee_weapon(items=forced_items + items,
                                                              allow_unknown_status=allow_unknown_status)
            if item is not None:
                add_item(item)

            # TOOL_KEEP_FIRST: the digging tool before the armor set (a splint mail pushed a pick-axe out)
            dive_ = getattr(self.agent.global_logic, 'dive', None)
            if jf_config.TOOL_KEEP_FIRST and not allow_unknown_status and dive_ is not None and \
                    dive_.keep_digging_tool():
                tool = dive_.best_digging_tool(forced_items + items)
                if tool is not None:
                    add_item(tool)

            no_shield = dive_ is not None and dive_.mattock_digger()
            for item in self.agent.inventory.get_best_armorset(items=forced_items + items,
                                                               allow_unknown_status=allow_unknown_status):
                if item is not None and not (no_shield and getattr(item.objs[0], 'sub', None) == O.ARM_SHIELD):
                    add_item(item)

        # the dive digs down with a pick-axe: keep one (the tour drops them for lighter loot)
        dive = getattr(self.agent.global_logic, 'dive', None)
        if dive is not None and dive.keep_digging_tool():
            tool = dive.best_digging_tool(forced_items + items)
            if tool is not None:
                add_item(tool)

        # power: boots that may be levitation or water walking boots, for the Castle's moat (never worn before)
        if jf_config.KEEP_MAGIC_BOOTS:
            for item in sorted(items, key=lambda i: i.unit_weight(with_content=False)):
                if power.is_passage_boots(item) and not item.equipped:
                    add_item(item)

        for item in items:
            if item.is_unambiguous():
                if item.object in [
                        O.from_name('healing', nh.POTION_CLASS),
                        O.from_name('extra healing', nh.POTION_CLASS),
                        O.from_name('full healing', nh.POTION_CLASS)]:
                    add_item(item)

                if self.agent.character.role in [Character.RANGER, Character.ROGUE,
                                                 Character.SAMURAI, Character.TOURIST] and \
                        (item.is_launcher() or item.is_fired_projectile()):
                    add_item(item)

        if self.agent.character.alignment == Character.LAWFUL:
            for item in sorted(filter(lambda i: i.objs[0].name == 'long sword', items),
                               key=lambda i: -utils.calc_dps(*self.agent.character.get_melee_bonus(i))):
                add_item(item)
                break

        # power (KEEP_POTIONS): in 23 of 90 base games potion types were dropped for good, mostly for daggers
        # and the dive's pick-axe/mattock (the pack is full: plate mail 450 + tool 100-120). Each unknown potion
        # is levitation 4.6% of the time, the likeliest Castle passage. While diving (digging, few fights):
        # food first, then the passage candidates, then the thrown weapons.
        if jf_config.KEEP_POTIONS and dive is not None and dive.diving:
            for item in sorted(filter(lambda i: i.is_food() and not i.is_corpse(), items),
                               key=lambda x: -x.nutrition_per_weight() - 1000 * (x.objs[0].name == 'sprig of wolfsbane')):
                add_item(item)
            for item in sorted(filter(power.is_passage_candidate, items),
                               key=lambda i: i.unit_weight(with_content=False)):
                add_item(item)

        for item in sorted(filter(lambda i: i.is_thrown_projectile(), items),
                           key=lambda i: -utils.calc_dps(*self.agent.character.get_ranged_bonus(None, i))):
            add_item(item)

        for item in sorted(filter(lambda i: i.is_food() and not i.is_corpse(), items),
                           key=lambda x: -x.nutrition_per_weight() - 1000 * (x.objs[0].name == 'sprig of wolfsbane')):
            add_item(item)

        if jf_config.LICHEN_RESERVE:
            # a never-rotting food reserve (lichen, lizard corpses) for the Weak spells before a safe prayer
            # (agent.reserve_corpse): eaten by eat_from_inventory, like found rations
            left = jf_config.LICHEN_RESERVE
            for item in filter(lambda i: i.is_corpse() and i.monster_id in self.agent.RESERVE_CORPSE_IDS, items):
                if left <= 0:
                    break
                add_item(item, count=left)
                left -= min(item.count, left)

        if self._take_sacrificial_corpses:
            for item in filter(self.agent.global_logic.can_sacrify, items):
                add_item(item)

        # TODO: take nh.COIN_CLASS once shopping is implemented.
        # You have to drop all coins not to be attacked by a vault guard

        for item in sorted(items, key=lambda i: i.unit_weight(with_content=False)):
            if item.category in [nh.POTION_CLASS, nh.RING_CLASS, nh.AMULET_CLASS, nh.WAND_CLASS, nh.SCROLL_CLASS, nh.TOOL_CLASS]:
                if (not isinstance(item.objs[0], O.Container) or not item.is_chest()) and \
                        not item.is_possible_container():
                    to_bag = O.from_name('cancellation', nh.WAND_CLASS) not in item.objs and \
                             not item.is_offensive_usable_wand()
                    to_bag = False
                    if not item.is_container():  # remove condition to pick up bags
                        add_item(item, to_bag=to_bag)

        categories = [nh.WEAPON_CLASS, nh.ARMOR_CLASS, nh.TOOL_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.AMULET_CLASS,
                      nh.RING_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.WAND_CLASS]
        for item in sorted(items, key=lambda i: i.unit_weight(with_content=False)):
            if item.category in categories and not isinstance(item.objs[0], O.Container) and not item.is_corpse():
                if item.status == Item.UNKNOWN:
                    to_bag = O.from_name('cancellation', nh.WAND_CLASS) not in item.objs and \
                             not item.is_offensive_usable_wand()
                    to_bag = False
                    if not item.is_container():  # remove condition to pick up bags
                        add_item(item, to_bag=to_bag)

        r = {None: [ret_inv.get(item, 0) for item in items]}
        if bag is not None:
            r[bag] = [ret_bag.get(item, 0) for item in items]
        for item in items:
            if item.is_container() and item is not bag and ret_inv.get(item, 0) != 0:
                r[item] = [0 for _ in items]
        return r


class Milestone(IntEnum):
    BE_ON_FIRST_LEVEL = auto()
    FIND_GNOMISH_MINES = auto()
    # FIND_LIGHT_GNOMISH_MINES = auto()
    # FARM_LIGHT_GNOMISH_MINES = auto()
    FIND_MINETOWN = auto()
    FIND_SOKOBAN = auto()
    SOLVE_SOKOBAN = auto()
    FIND_MINES_END = auto()
    GO_DOWN = auto() # TODO




class GlobalLogic:
    def __init__(self, agent):
        self.agent = agent
        self.milestone = Milestone(1)
        self.step_completion_log = {}  # Milestone -> (step, turn)

        self.item_priority = ItemPriority(self.agent)

        self.oracle_level = None
        self.minetown_level = None

        self._got_artifact = False
        self._milestone_since = {}   # milestone -> turn it began (jf_config.MINES_SEARCH_TURNS)
        self._stall_anchor = None    # (level key, (y, x), turn, milestone) the tour has kept close to
        self._pick_trip_start = None
        self._pick_trip_done = False
        self.mines_not_found = False

        self.dive = DiveLogic(agent)

    def update(self):
        self.dive.update()

        if not self.agent.character.prop.hallu:
            if utils.isin(self.agent.glyphs, G.ORACLE).any():
                if self.oracle_level is None:
                    self.oracle_level = self.agent.current_level().key()
                else:
                    assert self.oracle_level == self.agent.current_level().key()

            if self.agent.current_level().dungeon_number == Level.GNOMISH_MINES and \
                    utils.isin(self.agent.glyphs, G.SHOPKEEPER).any():
                if self.minetown_level is None:
                    self.minetown_level = self.agent.current_level().key()
                else:
                    assert self.minetown_level == self.agent.current_level().key()

    @utils.debug_log('solving sokoban')
    @Strategy.wrap
    def solve_sokoban_strategy(self):
        # TODO: refactor
        if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
            yield False
        yield True

        def push_bolder(ty, tx, dy, dx):
            while 1:
                if self.agent.bfs()[ty, tx] == -1:
                    return False
                self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                with self.agent.atom_operation():
                    direction = self.agent.calc_direction(ty, tx, ty + dy, tx + dx)
                    self.agent.direction(direction)
                    message = self.agent.message

                if (self.agent.blstats.y, self.agent.blstats.x) == (ty, tx):
                    assert 'You hear a monster behind the boulder.' in message or \
                           'You try to move the boulder, but in vain.' in message, message
                    if self.agent.bfs()[ty + dy, tx + dx] != -1:
                        self.agent.go_to(ty + dy, tx + dx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                        continue
                    else:
                        pickaxe = None
                        for item in flatten_items(self.agent.inventory.items):
                            if item.is_unambiguous() and \
                                    item.object in [O.from_name('pick-axe'), O.from_name('dwarvish mattock')]:
                                pickaxe = item
                                break
                        if pickaxe is not None:
                            with self.agent.atom_operation():
                                pickaxe = self.agent.inventory.move_to_inventory(pickaxe)
                                self.agent.step(A.Command.APPLY)
                                self.agent.type_text(self.agent.inventory.items.get_letter(pickaxe))
                                self.agent.direction(direction)
                            return True
                else:
                    return True

                # TODO: not sure what to do
                self.agent.exploration.explore1(None).run()

        while 1:
            wall_map = utils.isin(self.agent.current_level().objects, G.WALL)
            for smap, answer in soko_solver.maps.items():
                sokomap = soko_solver.convert_map(smap)
                offset = np.array(min(zip(*wall_map.nonzero()))) - \
                         np.array(min(zip(*(sokomap.sokomap == soko_solver.WALL).nonzero())))
                mask = wall_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                offset[1] : offset[1] + sokomap.sokomap.shape[1]]
                if (mask & (sokomap.sokomap == soko_solver.WALL) == mask).all():
                    break
            else:
                assert 0, 'sokomap not found'

            possible_mimics = set()
            last_resort_move = None
            for (y, x), (dy, dx) in answer:
                boulder_map = utils.isin(self.agent.glyphs, G.BOULDER)
                mask = boulder_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                   offset[1] : offset[1] + sokomap.sokomap.shape[1]]
                ty, tx = offset[0] + y - dy, offset[1] + x - dx,
                soko_boulder_mask = sokomap.sokomap == soko_solver.BOULDER
                if self.agent.bfs()[ty, tx] != -1 and \
                        ((soko_boulder_mask | mask) == soko_boulder_mask).all() and \
                        self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER:

                    soko_dis1 = sokomap.bfs()
                    sokomap.move(y, x, dy, dx)
                    soko_dis2 = sokomap.bfs()

                    # see points that will no longer be accessible
                    with self.agent.env.debug_log('checking for monsters'):
                        to_visit_mask = self.agent.bfs() != -1
                        to_visit_mask[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                      offset[1] : offset[1] + sokomap.sokomap.shape[1]] &= \
                                              (soko_dis1 != -1) & (soko_dis2 == -1)

                        with self.agent.env.debug_tiles(to_visit_mask, color=(255, 0, 0, 128)):
                            while to_visit_mask.any():
                                vy, vx = list(zip(*to_visit_mask.nonzero()))[0]
                                to_visit_mask[vy, vx] = 0

                                def clear_neighbors():
                                    to_visit_mask[self.agent.blstats.y, self.agent.blstats.x] = 0
                                    to_visit_mask[utils.isin(self.agent.glyphs, G.VISIBLE_FLOOR)] = 0
                                    return not to_visit_mask[vy, vx]

                                self.agent.go_to(vy, vx, callback=clear_neighbors)

                    if not push_bolder(ty, tx, dy, dx):
                        continue

                    possible_mimics = set()
                    last_resort_move = None

                    if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
                        return

                else:
                    sokomap.move(y, x, dy, dx)

                if (~soko_boulder_mask | mask).all():
                    if self.agent.bfs()[ty, tx] != -1 and \
                            self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER and \
                            last_resort_move is None:
                        last_resort_move = (ty, tx, dy, dx)

                    if not possible_mimics:
                        for mim_y, mim_x in zip(*(~soko_boulder_mask & mask).nonzero()):
                            possible_mimics.add((mim_y + offset[0], mim_x + offset[1]))


            # sokoban configuration is not in the list. Check for mimics
            with self.agent.env.debug_log('mimics'):
                with self.agent.env.debug_tiles(possible_mimics, color=(255, 0, 0, 128)):
                    for mim_y, mim_x in sorted(possible_mimics):
                        if (self.agent.bfs()[max(0, mim_y - 1) : mim_y + 2, max(0, mim_x - 1) : mim_x + 2] != -1).any():
                            self.agent.go_to(mim_y, mim_x, stop_one_before=True,
                                             debug_tiles_args=dict(color=(255, 0, 0), is_path=True))
                            for _ in range(3):
                                self.agent.search()

            # last resort move (sometimes we can see a mimic diagonally but cannot reach it)
            with self.agent.env.debug_log('last_resort'):
                if last_resort_move is not None:
                    ty, tx, dy, dx = last_resort_move
                    push_bolder(ty, tx, dy, dx)
                    continue

            self.agent.stats_logger.log_event('sokoban_dropped')
            self.milestone = Milestone(int(self.milestone) + 1)
            raise AgentPanic('sokomap unsolvable')

    @Strategy.wrap
    def wait_out_unexpected_state_strategy(self):
        yielded = False
        # CASTLE_POLY: on the castle a polymorph is the way over the moat (castle_power): keep acting in the form
        castle_poly = lambda: jf_config.CASTLE_POLY and self.dive.castle.active()
        while (
                self.agent.character.prop.blind or
                self.agent.character.prop.confusion or
                self.agent.character.prop.stun or
                self.agent.character.prop.hallu or
                (self.agent.character.prop.polymorph and not castle_poly())):
            if not yielded:
                yield True
                yielded = True

            self.agent.direction('.')

        if not yielded:
            yield False

    @utils.debug_log('identify_items_on_altar')
    @Strategy.wrap
    def identify_items_on_altar(self):
        mask = utils.isin(self.agent.current_level().objects, G.ALTAR)
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis != -1
        if not mask.any():
            yield False

        yield any((item.status == Item.UNKNOWN for item in flatten_items(self.agent.inventory.items)
                   if item.can_be_dropped_from_inventory()))

        (ty, tx), *_ = zip(*(mask & (dis == dis[mask].min())).nonzero())
        self.agent.go_to(ty, tx)
        items_to_drop = [item for item in flatten_items(self.agent.inventory.items)
                         if item.can_be_dropped_from_inventory() and item.status == Item.UNKNOWN]
        if not items_to_drop:
            raise AgentPanic('items to drop on altar vanished')

        # TODO: move chunking to inventory.drop
        items_to_drop = items_to_drop[:self.agent.inventory.items.free_slots()]

        self.agent.inventory.drop(items_to_drop)

    @utils.debug_log('dip_for_excalibur')
    @Strategy.wrap
    def dip_for_excalibur(self):
        if self.agent.character.alignment != Character.LAWFUL or self.agent.blstats.experience_level < 5:
            yield False
        if self.agent.current_level().dungeon_number == Level.GNOMISH_MINES and \
                (self.minetown_level is None or self.agent.current_level().key() == self.minetown_level):
            yield False

        dis = self.agent.bfs()
        mask = utils.isin(self.agent.current_level().objects, G.FOUNTAIN) & (dis != -1)
        if not mask.any():
            yield False

        def excalibur_candidate():
            candidate = None
            for item in flatten_items(self.agent.inventory.items):
                if item.is_unambiguous() and item.object == O.from_name('long sword'):
                    if item.dmg_bonus is not None: # TODO: better condition for excalibur existance
                        return None
                    candidate = item
            return candidate

        if excalibur_candidate() is None:
            yield False
        yield True

        self.agent.go_to(*list(zip(*mask.nonzero()))[0])

        candidate = excalibur_candidate()
        if candidate is None:
            return

        # TODO: refactor
        with self.agent.atom_operation():
            candidate = self.agent.inventory.move_to_inventory(candidate)
            self.agent.step(A.Command.DIP)
            self.agent.type_text(self.agent.inventory.items.get_letter(candidate))
            if ('What do you want to dip ' in self.agent.message and 'into?' in self.agent.message) or \
                    "You don't have anything to dip " in self.agent.message:
                raise AgentPanic('no fountain here')

    def can_sacrify(self, item):
        if not item.is_corpse() or item.comment == 'old':
            return False
        # picking up a cockatrice-family corpse bare-handed is instant stoning: two unseen-seed games died
        # taking their thrown daggers back from a chickatrice's pile ('Touching a chickatrice corpse is a
        # fatal mistake.'), the corpse selected to carry to an altar
        if self.agent.inventory.items.gloves is None and \
                item.monster_id + nh.GLYPH_MON_OFF in self.agent._petrifying_bodies_mons:
            return False

        mname = MON.permonst(item.monster_id + nh.GLYPH_MON_OFF).mname
        # sacrificing a former pet: "So this is how you repay loyalty?", the god gets angry (an unseen
        # Valkyrie offered her kitten, then again: an Angel of Tyr killed her at T2197). The old check here
        # compared role == [list] (never true) and missed the Valkyrie's kitten; any grown-up pet species
        # may be ours, so none of them is offered
        if mname in ('kitten', 'housecat', 'large cat', 'little dog', 'dog', 'large dog', 'pony', 'horse',
                     'warhorse'):
            return False

        if self.agent.character.alignment != Character.CHAOTIC:
            mapping = {
                Character.HUMAN: MON.M2_HUMAN | MON.M2_WERE,
                Character.DWARF: MON.M2_DWARF,
                Character.ELF: MON.M2_ELF,
                Character.GNOME: MON.M2_GNOME,
                Character.ORC: MON.M2_ORC,
            }
            f2 = MON.permonst(item.monster_id + nh.GLYPH_MON_OFF).mflags2
            if (f2 & mapping[self.agent.character.race]) > 0:
                return False

        return True

    @utils.debug_log('offer_corpses')
    @Strategy.wrap
    def offer_corpses(self):
        self.item_priority._take_sacrificial_corpses = False

        if self._got_artifact:
            yield False

        altars = [p for p, alignment in self.agent.current_level().altars.items()
                  if alignment == self.agent.character.alignment]

        if not altars:
            yield False

        dis = self.agent.bfs()
        altars = [(y, x) for y, x in altars if dis[y, x] != -1]

        if not altars:
            yield False

        self.item_priority._take_sacrificial_corpses = True

        if not any((self.can_sacrify(item) for item in flatten_items(self.agent.inventory.items))):
            yield False

        yield True

        y, x = min(altars, key=lambda p: dis[p])
        self.agent.go_to(y, x)
        with self.agent.panic_if_position_changes():
            while 1:
                for item in flatten_items(self.agent.inventory.items):
                    if self.can_sacrify(item):
                        with self.agent.atom_operation():
                            item = self.agent.inventory.move_to_inventory(item)
                            assert self.can_sacrify(item)
                            self.agent.step(A.Command.OFFER)
                            while ('There is ' in self.agent.message or 'There are ' in self.agent.message) and \
                                    ('sacrifice it?' in self.agent.message or 'sacrifice one?' in self.agent.message):
                                self.agent.type_text('n')
                            assert 'What do you want to sacrifice?' in self.agent.message, self.agent.message
                            self.agent.type_text(self.agent.inventory.items.get_letter(item))
                            if 'Nothing happens.' in self.agent.message:
                                self.agent.inventory.call_item(item, 'old')
                                return
                            if 'Use my gift wisely' in self.agent.message:
                                self._got_artifact = True
                                self.agent.inventory.get_items_below_me()
                                return
                            if 'So this is how you repay loyalty?' in self.agent.message:
                                raise AgentPanic('pet sacrified')
                            assert 'Your sacrifice is consumed in a flash of light' in self.agent.message or \
                                'Your sacrifice is consumed in a burst of flame' in self.agent.message or \
                                ('The blood covers the altar!' in self.agent.message and \
                                 'You have summoned ' in self.agent.message), \
                                self.agent.message
                            break
                else:
                    break

    @utils.debug_log('follow_guard')
    @Strategy.wrap
    def follow_guard(self):
        if not utils.isin(self.agent.glyphs, G.GUARD).any():
            yield False

        if any(item.category == nh.COIN_CLASS for item in flatten_items(self.agent.inventory.items)):
            yield True
            # if 'Please drop that gold and follow me.' in self.message:
            self.agent.stats_logger.log_event('drop_gold')
            self.item_priority._drop_gold_till_turn = self.agent.blstats.time + 100
            self.agent.inventory.arrange_items().run()
            return

        ys, xs = utils.isin(self.agent.glyphs, G.GUARD).nonzero()
        y, x = ys[0], xs[0]

        if utils.adjacent((y, x), (self.agent.blstats.y, self.agent.blstats.x)):
            yield False

        yield True

        self.agent.go_to(y, x, stop_one_before=True)

    def _safe_to_dip(self):
        # SAFE_DIPS (full HP + prayer ready, astra) changes the tour; off until tested on its own
        bl = self.agent.blstats
        # NO_DIP_WITH_TOOL: a fountain dip curses the sword 1 time in 30, silently (fountain.c case 16), and a
        # welded weapon can't be swapped for the pick-axe: eg-glh-public seed 6 carried its pick from T6556, dipped
        # 8 times at XL 7, and dove at XL 8 with 'a cursed thoroughly rusty +1 long sword (weapon in hand)' --
        # digging_tool() None, a tool-less dive (0.206; base 0.466). A digger doesn't need Excalibur.
        if jf_config.NO_DIP_WITH_TOOL and self.dive.digging_tool() is not None:
            return False
        # DEMON_NO_REDIP: not while a released water demon is about -- when it fled out of the vigil's reach the
        # bot walked back to the fountain and dipped twice more next to it, then fought it (DEMON_FIX replay of
        # jf16 s8: killed by the demon 480 turns after the release)
        if jf_config.DEMON_NO_REDIP and bl.time <= self.dive._demon_vigil_until:
            return False
        if not jf_config.SAFE_DIPS:
            # a released water demon (1 dip in ~40) is deadlier to a starving or hurt character; dipping
            # can wait for a healthy moment (SAFE_DIPS' prayer-ready rule halved the Excaliburs)
            return bl.experience_level >= 7 and bl.hunger_state < Hunger.WEAK and \
                bl.hitpoints >= 0.7 * bl.max_hitpoints
        return bl.experience_level >= 7 and bl.hitpoints >= 0.9 * bl.max_hitpoints and \
            self.agent.is_safe_to_pray(800)

    def exploration_strategy(self, search_prio_limit):
        """The tour's exploration (used by the dive phase too)."""
        return (
            Strategy(lambda: self.agent.exploration.explore1(
                search_prio_limit, trap_search_offset=1,
                kick_doors=self.agent.current_level().dungeon_number != Level.GNOMISH_MINES).strategy())
            .preempt(self.agent, [
                self.identify_items_on_altar().every(100),
                self.identify_items_on_altar().condition(
                    lambda: self.agent.current_level().objects[self.agent.blstats.y,
                                                               self.agent.blstats.x] in G.ALTAR),
                # hypothesis: fountain dips summon water demons (killed an XL9 elite game); dip only
                # at (near) full HP with a prayer in hand, as astra did (full HP, retreat ready)
                self.dip_for_excalibur().condition(self._safe_to_dip).every(10),
            ])
        )

    def _pick_trip_active(self):
        agent = self.agent
        if not jf_config.PICK_TRIP_XL or self._pick_trip_done:
            return False
        if self.dive.digging_tool() is not None:
            agent.log('TOUR pick trip: got a digging tool, back to the grind')
            self._pick_trip_done = True
            return False
        if agent.prayer_failed or agent.blstats.experience_level < jf_config.PICK_TRIP_XL:
            return False
        start = self._pick_trip_start
        if start is not None and jf_config.PICK_TRIP_END_XL and \
                agent.blstats.experience_level >= jf_config.PICK_TRIP_END_XL:
            # an XL-7 character on Dlvl 3-6 meets difficulty-5/6 monsters ((depth + XL) / 2): 4 of 60 pt5/pt6
            # games died there to killer bees, soldier ants and orcish arrows, none during the XL 5-6 trips
            agent.log(f'TOUR pick trip: XL {agent.blstats.experience_level} without a tool, back to the grind')
            self._pick_trip_done = True
            return False
        if start is None:
            self._pick_trip_start = agent.blstats.time
            agent.log('TOUR pick trip: off to the Mines for a digging tool')
        elif agent.blstats.time - start > jf_config.PICK_TRIP_TURNS:
            agent.log('TOUR pick trip: out of time, back to the grind')
            self._pick_trip_done = True
            return False
        return True

    def _grind_level(self):
        """jf_config.GRIND_LEVELS {min XL: Dlvl}: the grind's main-dungeon level at this XL (None: the flag is off).

        Random monsters are capped at difficulty (depth + XL) / 2 (makemon.c). On Dlvl 1 that cap is 3 at XL 5-6,
        so XL 5 -> 7 takes ~9,000 turns there (base: ~3,500 + ~5,500) at ~12 hunger prayers per grind, and the
        prayers' rnz(350) failures end 13 of 45 gc-lf1 games in a rescue. Keeping the cap at 4 instead (Dlvl 3 at
        XL 5-6, Dlvl 2 at XL 7) doubles XP per spawn at XL 5-6 (monst.c model: 3.8-4.9 -> 8.3-10.8), so fewer
        turns and prayers, while killer bees, soldier ants and Uruk-hai (difficulty 5-6) stay out: the old
        Dlvl-3 grind at XL 7 (cap 5) lost 4.9-7.8% of games per 1000 turns. Dwarves (difficulty 4) spawn from
        XL 5 there, and the pet kills them for their pick-axes as it does on Dlvl 1 from XL 7."""
        table = jf_config.GRIND_LEVELS
        if not table:
            return None
        xl = self.agent.blstats.experience_level
        keys = [k for k in table if k <= xl]
        return table[max(keys)] if keys else 1

    def _tour_stalled(self):
        """The tour has kept within 8 squares of one spot on one level, same milestone, TOUR_STALL_TURNS turns."""
        bl = self.agent.blstats
        key = self.agent.current_level().key()
        a = self._stall_anchor
        if a is None or a[0] != key or a[3] != self.milestone or \
                max(abs(bl.y - a[1][0]), abs(bl.x - a[1][1])) > 8:
            self._stall_anchor = (key, (bl.y, bl.x), bl.time, self.milestone)
            return False
        return bl.time - a[2] >= jf_config.TOUR_STALL_TURNS

    @staticmethod
    def _stall_goal_met(goal):
        try:
            return goal()
        except Exception:
            return False

    def current_strategy(self):
        # hypothesis: AutoAscend's levelling tour keeps the character alive to XL 10-13 (the elite's
        # recipe); once it is strong, the depth-first dive with the Quest-portal sweep is worth more
        # than further levelling (Home 1 = 0.366, Dlvl 20+ = 0.38+, vs XL 12 = 0.333).
        return self.tour_strategy().until(self.agent, self.dive.should_dive).before(self.dive.strategy())

    @Strategy.wrap
    def tour_strategy(self):
        yield True
        while 1:
            explore_stairs_condition = lambda: False
            restart = lambda: False   # ends the current strategy without finishing the milestone
            if self.milestone == Milestone.BE_ON_FIRST_LEVEL and self._pick_trip_active():
                # PICK_TRIP_XL: a detour from the grind to the Mines for a dwarf's pick-axe, then back to
                # Dlvl 1. With the pick in hand a failed hunger prayer (a quarter of all games) starts a
                # dig-dive instead of a starving stairs rescue, and the XL 8 dive needs no Mines trip.
                self.dive.pick_trip = True
                condition = lambda: False
                level = (Level.GNOMISH_MINES, jf_config.PICK_TRIP_LEVEL)
                restart = lambda: not self._pick_trip_active()
            elif self.milestone == Milestone.BE_ON_FIRST_LEVEL:
                self.dive.pick_trip = False
                condition = self.dive.first_level_done
                # explore_stairs_condition = lambda: self.agent.inventory.items.total_nutrition() == 0 and \
                #                                    self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY
                # GRIND_DEEP_XL: from that XL the grind goes on on Dlvl GRIND_DEEP_LEVEL -- on Dlvl 1 XL 6->8 takes
                # a median 11,500 turns (7 of the grind's ~11 hunger prayers): spawns there are too weak
                deep = bool(jf_config.GRIND_DEEP_XL) and \
                    self.agent.blstats.experience_level >= jf_config.GRIND_DEEP_XL
                level = (Level.DUNGEONS_OF_DOOM, jf_config.GRIND_DEEP_LEVEL if deep else 1)
                if jf_config.GRIND_DEEP_XL and not deep:
                    restart = lambda: self.agent.blstats.experience_level >= jf_config.GRIND_DEEP_XL
                grind_level = self._grind_level()
                if grind_level is not None:
                    # GRIND_LEVELS: the grind's level follows XL, see _grind_level
                    level = (Level.DUNGEONS_OF_DOOM, grind_level)
                    restart = lambda lv=grind_level: self._grind_level() != lv

            elif self.milestone == Milestone.FIND_SOKOBAN:
                condition = lambda: self.agent.current_level().dungeon_number == Level.SOKOBAN
                level = (Level.SOKOBAN, 4)

            elif self.milestone == Milestone.FIND_GNOMISH_MINES:
                since = self._milestone_since.setdefault(self.milestone, self.agent.blstats.time)
                # the branch hides behind unexplored rock on some Dlvl 2-4s: a jf9 game stood searching at
                # one Dlvl 4 spot for 1000+ turns, hunger-praying until a prayer failed (35 of 118 games
                # that left Dlvl 1 never reached the Mines; 14 of 83 that did needed 4000+ turns)
                condition = lambda: self.agent.current_level().dungeon_number == Level.GNOMISH_MINES or \
                    (jf_config.MINES_SEARCH_TURNS and self.agent.blstats.time - since > jf_config.MINES_SEARCH_TURNS)
                level = (Level.GNOMISH_MINES, 1)

            # elif self.milestone == Milestone.FIND_LIGHT_GNOMISH_MINES:
            #     condition = lambda: self.agent.current_level().dungeon_number == Level.GNOMISH_MINES \
            #             and self.agent.current_level().is_light_level()
            #     level = (Level.GNOMISH_MINES, 9)

            # elif self.milestone == Milestone.FARM_LIGHT_GNOMISH_MINES:
            #     condition = lambda: self.agent.blstats.experience_level >= 11
            #     level = (Level.GNOMISH_MINES, self.agent.current_level().level_number)

            elif self.milestone == Milestone.FIND_MINETOWN:
                condition = lambda: self.minetown_level is not None
                level = (Level.GNOMISH_MINES, 4)  # TODO

            elif self.milestone == Milestone.SOLVE_SOKOBAN:
                # TODO: fix the condition, monster can destroy doors
                condition = lambda: self.agent.current_level().key() == (Level.SOKOBAN, 1) and \
                                    not utils.isin(self.agent.current_level().objects, G.DOOR_CLOSED).any()
                level = (Level.SOKOBAN, 1)

            elif self.milestone == Milestone.FIND_MINES_END:
                condition = lambda: self.agent.current_level().key() == (Level.GNOMISH_MINES, 9) or \
                    self.mines_not_found  # TODO
                level = (Level.GNOMISH_MINES, 9)  # TODO

            else:
                # TODO
                condition = lambda: False
                level = (Level.DUNGEONS_OF_DOOM, 100)

            goal = condition
            watch_stall = Milestone.BE_ON_FIRST_LEVEL < self.milestone < Milestone.GO_DOWN and \
                bool(jf_config.TOUR_STALL_TURNS)
            if watch_stall:
                condition = lambda: goal() or self._tour_stalled()

            if condition():
                if watch_stall and not self._stall_goal_met(goal) and self._tour_stalled():
                    # held in one spot for TOUR_STALL_TURNS (a pocket we are too heavy to squeeze out of, a
                    # passage a jelly blocks, a shop door...): the next milestone takes us somewhere else, the
                    # last one (GO_DOWN) is the dive
                    skip = Milestone.FIND_MINES_END if self.milestone in (Milestone.FIND_SOKOBAN,
                                                                           Milestone.SOLVE_SOKOBAN) \
                        else Milestone(int(self.milestone) + 1)
                    self.agent.log(f'TOUR stalled {jf_config.TOUR_STALL_TURNS} turns at {self._stall_anchor[:2]} '
                                   f'in {self.milestone.name}: on to {skip.name}')
                    if self.milestone == Milestone.FIND_GNOMISH_MINES:
                        self.mines_not_found = True
                    self.milestone = skip
                    self._stall_anchor = None
                    continue
                if self.milestone == Milestone.FIND_GNOMISH_MINES and \
                        self.agent.current_level().dungeon_number != Level.GNOMISH_MINES:
                    self.agent.log(f'TOUR no Mines entrance after {jf_config.MINES_SEARCH_TURNS} turns: '
                                   f'skipping Minetown, on to Sokoban')
                    self.mines_not_found = True
                    self.milestone = Milestone.FIND_SOKOBAN
                    continue
                self.milestone = Milestone(int(self.milestone) + 1)
                if jf_config.SKIP_SOKOBAN and self.milestone in (Milestone.FIND_SOKOBAN, Milestone.SOLVE_SOKOBAN) \
                        and self.agent.character.race in (Character.DWARF, Character.GNOME):
                    self.milestone = Milestone.FIND_MINES_END
                continue


            def exploration_strategy(level, **kwargs):
                return (
                    Strategy(lambda: self.agent.exploration.explore1(level, trap_search_offset=1,
                        kick_doors=self.agent.current_level().dungeon_number != Level.GNOMISH_MINES, **kwargs).strategy())
                    .preempt(self.agent, [
                        self.identify_items_on_altar().every(100),
                        self.identify_items_on_altar().condition(
                            lambda: self.agent.current_level().objects[self.agent.blstats.y,
                                                                       self.agent.blstats.x] in G.ALTAR),
                        self.dip_for_excalibur().condition(self._safe_to_dip).every(10),
                    ])
                )

            def go_to_strategy(y, x):
                return (
                    exploration_strategy(None)
                    .preempt(self.agent, [
                        self.agent.exploration.go_to_strategy(y, x).preempt(self.agent, [
                            self.agent.inventory.gather_items(),
                            self.identify_items_on_altar(),
                            self.dip_for_excalibur().condition(self._safe_to_dip),
                        ])
                        .condition(lambda: self._got_artifact or
                                           not any([alignment == self.agent.character.alignment
                                                    for _, alignment in self.agent.current_level().altars.items()]))
                    ])
                    .until(self.agent, lambda: (self.agent.blstats.y, self.agent.blstats.x) == (y, x))
                )

            def homebound(lv=level):
                """In transit to the grind level: don't explore every level passed to exhaustion (the tour's
                first preempt). UPWARD_RETURN: on the way home from a pick trip (eg-trip5a public seed 9 ended
                its trip on Mines 1 at XL 7 and explored that level for 3,500 turns instead of climbing to
                the Dlvl 1 grind). GRIND_LEVELS: moving up to the next grind level (eg-gl-a public seed 11
                spent 4,000 turns at XL 7 on Dlvl 3 before climbing to Dlvl 2), or out of the Mines a
                random unexplored '>' took us into (seed 3 died on Mines 1)."""
                if self.milestone != Milestone.BE_ON_FIRST_LEVEL or self.agent.current_level().key() == lv:
                    return False
                if jf_config.UPWARD_RETURN and self._pick_trip_done:
                    return True
                cur = self.agent.current_level()
                return bool(jf_config.GRIND_LEVELS) and lv[0] == Level.DUNGEONS_OF_DOOM and \
                    (cur.dungeon_number == Level.GNOMISH_MINES or self.agent.blstats.depth > lv[1])
            (
                self.agent.exploration.go_to_level_strategy(*level, go_to_strategy, exploration_strategy(None))
                .before(exploration_strategy(None))#.before(self.agent.exploration.patrol())
                .preempt(self.agent, [
                    exploration_strategy(0).condition(lambda: not homebound()),
                    exploration_strategy(None).until(
                        self.agent, lambda: self.agent.blstats.hitpoints >= 0.8 * self.agent.blstats.max_hitpoints)
                ])
                .preempt(self.agent, [
                    self.agent.exploration.explore_stairs(go_to_strategy, all=True).condition(explore_stairs_condition),
                ])
                .until(self.agent, lambda: condition() or restart())
            ).run()

    def global_strategy(self):
        return (
            self.current_strategy().repeat()
            # lowest priority: a peaceful dwarf's pick-axe while in the Mines (dive_logic.DWARF_HUNT)
            .preempt(self.agent, [
                self.dive.hunt_strategy(),
                self.dive.ditch_pet_strategy(),
                # a tour-mode pick trip looks for the Mines branch itself (dive_logic.FAST_BRANCH)
                self.dive.trip_branch_strategy(),
            ])
            .preempt(self.agent, [
                self.solve_sokoban_strategy()
                .condition(lambda: self.milestone == Milestone.SOLVE_SOKOBAN and
                                   self.agent.current_level().dungeon_number == Level.SOKOBAN)
            ])
            .preempt(self.agent, [
                self.offer_corpses().preempt(self.agent, [
                    self.agent.eat_corpses_from_ground().condition(lambda: self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                ]),
            ])
            .preempt(self.agent, [
                self.wait_out_unexpected_state_strategy(),
            ])
            .preempt(self.agent, [
                self.agent.cure_disease().every(5),
            ])
            .preempt(self.agent, [
                self.agent.eat_corpses_from_ground(only_below_me=True).condition(lambda: self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                # CLAIM_CORPSES: a fresh kill a few steps away is ours before the pet gets it (the pet ate as many
                # jackal corpses as we did in the base grinds; 63% of its meals of our kills came 2+ turns after
                # the kill, while we walked elsewhere)
                self.agent.eat_corpses_from_ground(only_below_me=False, max_dist=jf_config.CLAIM_DIST,
                                                   max_age=jf_config.CLAIM_MAX_AGE)
                .condition(lambda: jf_config.CLAIM_CORPSES and not self.dive.diving and
                           self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                self.agent.eat_corpses_from_ground(only_below_me=not jf_config.EAT_NEARBY_CORPSES).every(5)
                .condition(lambda: self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                # after a failed prayer corpses are the only food left: walk to the ones nearby
                self.agent.eat_corpses_from_ground(only_below_me=False).every(3)
                .condition(lambda: self.agent.prayer_failed and self.agent.blstats.hunger_state >= Hunger.HUNGRY),
                # DIVE_EAT: a hungry dive walks to fresh edible corpses close by (6 of 33 tool dives that died
                # above Dlvl 25 in base/base2 died Weak or Fainting, between hunger prayers)
                self.agent.eat_corpses_from_ground(only_below_me=False).every(3)
                .condition(lambda: jf_config.DIVE_EAT and self.dive.diving and not self.agent.prayer_failed and
                           self.agent.blstats.hunger_state >= Hunger.HUNGRY and
                           self.dive.edible_corpse_within(jf_config.DIVE_EAT_RADIUS)),
                self.agent.eat_from_inventory().every(5),
                self.agent.inventory.buy_food().every(3),
                self.agent.inventory.buy_starter_suit().every(3),
                # power (SELL_PRICE_ID): offer unknown potions/rings/boots to a shopkeeper for their price group
                self.agent.inventory.sell_price_identify().every(3),
            ])
            .preempt(self.agent, [
                # boxed in by diagonal squeezes while carrying > 600 (jf_config.UNSQUEEZE)
                self.agent.unsqueeze(),
            ])
            .preempt(self.agent, [
                self.follow_guard(),
            ])
            .preempt(self.agent, [
                # held by a bear trap: diagonal attempts free us 5x faster (jf_config.BEARTRAP_ESCAPE)
                self.agent.escape_bear_trap(),
            ])
            .preempt(self.agent, [
                self.agent.astra_quiet_recovery(),
                self.agent.fight2(),
            ])
            # the Valley of the Dead only (GEHENNOM_DIVE): walk past the graveyards' sleeping undead
            .preempt(self.agent, [
                self.dive.valley_sneak(),
            ])
            # an Overloaded were form can neither fight nor eat: drop its load first (LYCAN_FIXES)
            .preempt(self.agent, [
                self.agent.were_unload().condition(lambda: jf_config.LYCAN_FIXES),
                # a hold that is over must not leave us standing on its Elbereth (see wipe_hold_elbereth)
                self.dive.wipe_hold_elbereth().condition(lambda: jf_config.HOLD_LOOP),
            ])
            .preempt(self.agent, [
                self.dive.faint_shelter(),
                self.dive.faint_guard().condition(lambda: jf_config.FAINT_GUARD),
            ])
            .preempt(self.agent, [
                self.dive.leave_minetown_hallucinating(),
            ])
            .preempt(self.agent, [
                self.dive.water_demon_vigil(),
            ])
            # a digger with room to dig finishes the hole instead of walking to a fight
            .preempt(self.agent, [
                self.dive.dig_first(),
            ])
            # A quiet, hungry Healer prepares food before committing to another descent.
            .preempt(self.agent, [
                self.agent.astra_boulder_food(),
            ])
            # astra's survival layer, only once diving (the tour keeps the elite's proven behaviour)
            .preempt(self.agent, [
                self.dive.elbereth_rest().condition(lambda: self.dive.diving or jf_config.SURVIVAL_IN_TOUR),
            ])
            .preempt(self.agent, [
                self.dive.retreat_upstairs().condition(lambda: self.dive.diving or jf_config.SURVIVAL_IN_TOUR),
                # the Valley of the Dead only (GEHENNOM_DIVE): up its '<' to heal on the castle level
                self.dive.valley_retreat(),
            ])
            # Gehennom only (GEHENNOM_DIVE): a wand of digging down away from a monster we can't outfight
            .preempt(self.agent, [
                self.dive.gehennom_escape(),
            ])
            # Gehennom only (GEHENNOM_SCARE): drop a scroll of scare monster and hold on it (dig from it / rest on
            # it) -- above the Valley retreat: no castle round trip while a scroll lasts
            .preempt(self.agent, [
                self.dive.gehennom_scare(),
            ])
            # power (CASTLE_POLY): depth 25+ on the main line, losing a fight -> a wand of polymorph at ourselves
            .preempt(self.agent, [
                castle_power.deep_poly_escape_strategy(self.dive),
            ])
            # crossing the castle moat (castle_logic.py): above the survival layer and the fight, which
            # would drag a levitating hero back to land or up the stairs
            .preempt(self.agent, [
                self.dive.castle.crossing_strategy(),
            ])
            .preempt(self.agent, [
                self.agent.engulfed_fight(),
            ])
            .preempt(self.agent, [
                self.agent.emergency_strategy(),
                self.agent.astra_pit_boulder_escape(),
            ])
        )
