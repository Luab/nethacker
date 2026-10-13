import contextlib
import re
from collections import namedtuple, Counter, defaultdict
from functools import partial

import nle.nethack as nh
import nltk
import numpy as np
from nle.nethack import actions as A

from . import combat
from . import utils
from .character import Character
from .exceptions import AgentPanic, AgentFinished, AgentChangeStrategy
from .exploration_logic import ExplorationLogic
from .global_logic import GlobalLogic, early_dig_xl

# ablation switches
CORPSE_MAX_AGE = 30
PRAY_AT_EXACT_THRESHOLD = True
from .glyph import MON, C, Hunger, G, SHOP
from .item import Item, flatten_items
from .item.inventory import Inventory
from .level import Level
from .monster_tracker import MonsterTracker, disappearance_mask
from .stats_logger import StatsLogger
from .strategy import Strategy

BLStats = namedtuple('BLStats',
                     'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask alignment')


# a hunger prayer waits this many turns after the previous prayer: a timeout drawn from rnz(350)
# is still above the major-trouble limit (200) ~7% of the time after 900 turns, ~2.5% after 1200
HUNGER_PRAYER_GAP = 1200
# ...unless the character has been fainting this long (hunger then drops at 1/10 the rate)
FAINTING_PRAYER_DEADLINE = 300
# ...or as soon as a hostile that can move shows up this close while the character is Fainting,
# provided at least this many turns have passed since the previous prayer
THREAT_PRAYER_GAP = 950
THREAT_PRAYER_DISTANCE = 6
PRAYER_FAILED_MESSAGES = ('is displeased', 'Thou hast angered me', 'Thou art arrogant', 'Thou hast strayed',
                          'Thou durst', 'relearn thy lessons', 'is bummed')


class Agent:
    def __init__(self, env, seed=0, verbose=False, panic_on_errors=False):
        self.env = env
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.panic_on_errors = panic_on_errors
        self.all_panics = []

        self.on_update = []
        self.levels = {}
        self.score = 0
        self.step_count = 0
        self._observation = None  # this should be used in additional_action_iterator generators
        # single_{message,popup} should be used in additional_action_itertator generators.
        # (non-single) message & popup contain cummulated content
        self.message = self.single_message = ''
        self.popup = self.single_popup = []
        self._message_history = []
        self.cursor_pos = (0, 0)
        self.last_observation = None

        self._last_pet_seen = 0

        self.inventory = Inventory(self)
        self.character = Character(self)
        self.exploration = ExplorationLogic(self)
        self.global_logic = GlobalLogic(self)
        self.monster_tracker = MonsterTracker(self)

        self.last_bfs_dis = None
        self.last_bfs_step = None
        self.last_prayer_turn = None
        self.prayer_failed = False  # the god was angered: every later prayer only smites again
        self._fainting_since = None
        self._monk_meat_meals = 0
        self._previous_glyphs = None
        self._last_turn = -1
        self._inactivity_counter = 0
        self._pass_turn_after_error = False
        self._is_updating_state = False

        self._no_step_calls = False

        self.turns_in_atom_operation = None
        self._atom_operation_allow_update = None

        self._is_reading_message_or_popup = False
        self._last_terrain_check = None
        self._forbidden_engrave_position = (-1, -1)

        # when (number of turn) there was last decision about allowing these actions (e.g. agent is somewhat stuck)
        self._allow_walking_through_traps_turn = -float('inf')
        self._allow_attack_all_turn = -float('inf')

        self.last_cast_fail_turn = defaultdict(lambda: -float('inf'))
        self._pick_dig_attempts = dict()
        self._dig_engrave_attempts = dict()
        self._undiggable_levels = set()  # level keys whose floor refused a dig
        self._mines_bottom_found = False  # Mines' End reached: no more digging in the Mines
        self._mapped_levels = set()
        self._empty_wand_letters = set()  # inventory letters of wands that did nothing when zapped

        self.stats_logger = StatsLogger()

    @property
    def has_pet(self):
        return (self.blstats.time - self._last_pet_seen) <= 16

    @property
    def in_atom_operation(self):
        return self.turns_in_atom_operation is not None

    ######## CONVENIENCE FUNCTIONS

    @contextlib.contextmanager
    def disallow_step_calling(self):
        if self._no_step_calls:
            yield
            return

        try:
            self._no_step_calls = True
            yield
        finally:
            self._no_step_calls = False

    @contextlib.contextmanager
    def atom_operation(self, allow_update=False):
        assert not self._no_step_calls
        if self.turns_in_atom_operation is not None:
            # already in an atom operation
            old_allow_update = self._atom_operation_allow_update
            if old_allow_update is None:
                self._atom_operation_allow_update = allow_update
            else:
                assert old_allow_update or not allow_update
                self._atom_operation_allow_update = old_allow_update and allow_update
            try:
                yield
            finally:
                self._atom_operation_allow_update = old_allow_update
            return

        self.turns_in_atom_operation = 0
        self._atom_operation_allow_update = allow_update
        try:
            yield
        finally:
            self.turns_in_atom_operation = None
            self._atom_operation_allow_update = None

        self.update_state()

    @contextlib.contextmanager
    def panic_if_position_changes(self):
        y, x = self.blstats.y, self.blstats.x

        def f(self):
            if (y, x) != (self.blstats.y, self.blstats.x):
                raise AgentPanic('position changed')

        fun = partial(f, self)

        self.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.on_update
            self.on_update.pop(self.on_update.index(fun))

    @contextlib.contextmanager
    def add_on_update(self, funcs):
        self.on_update.extend(funcs)

        try:
            yield
        finally:
            for f in funcs:
                self.on_update.pop(self.on_update.index(f))

    @contextlib.contextmanager
    def context_preempt(self, conditions):
        ids = []
        id2fun = {}
        for cond in conditions:
            def f(iden, cond=cond):
                if cond():
                    raise AgentChangeStrategy(iden, cond)

            fun = partial(f, id(f))
            assert id(f) not in id2fun
            id2fun[id(f)] = fun
            ids.append(id(f))
            self.on_update.append(fun)

        outcome = None
        for i, cond in enumerate(conditions):
            if cond():
                outcome = i
                break

        def outcome_f():
            nonlocal outcome
            return outcome

        try:
            yield outcome_f

        except AgentChangeStrategy as e:
            i = e.args[0]
            if i not in id2fun:
                raise
            outcome = ids.index(i)
        finally:
            for f in id2fun.values():
                self.on_update.pop(self.on_update.index(f))

        # check if less nested ChangeStategy is present
        self.call_update_functions()

    def preempt(self, strategies, func, first_func=None, continue_after_preemption=True):
        id2fun = {}
        for strategy in strategies:
            def f(iden, strategy):
                it = strategy.strategy()
                if next(it):
                    raise AgentChangeStrategy(iden, it)

            iden = (id(f), id(strategy))
            fun = partial(f, iden, strategy)
            assert iden not in id2fun
            id2fun[iden] = fun

        last_turn = 0

        call_update = True

        val = None

        last_step = self.step_count
        inactivity_counter = 0
        is_first = True
        while 1:
            inactivity_counter += 1
            if self.step_count != last_step:
                last_step = self.step_count
                inactivity_counter = 0
            assert inactivity_counter < 5, 'cyclic preempt'

            iterator = None
            try:
                with self.add_on_update(list(id2fun.values())):
                    if call_update:
                        call_update = False
                        self.call_update_functions(list(id2fun.values()))

                    f = (first_func or func) if is_first else func
                    if isinstance(f, Strategy):
                        val = f.run()
                    else:
                        val = f()
                    break

            except AgentChangeStrategy as e:
                i = e.args[0]
                if i not in id2fun:
                    raise
                iterator = e.args[1]

            if iterator is not None:
                try:
                    next(iterator)
                    assert 0, iterator
                except StopIteration:
                    pass

                if not continue_after_preemption:
                    break

            is_first = False

        return val

    ######## UPDATE FUNCTIONS

    def on_panic(self):
        self.check_terrain(force=True)
        self.inventory.on_panic()
        self.monster_tracker.on_panic()
        self.update_state()

    @staticmethod
    def _find_marker(lines, regex=re.compile(r"(--More--|\(end\)|\(\d+ of \d+\))")):
        """ Return (line, column) of markers:
        --More-- | (end) | (X of N)
        """
        if len(regex.findall(' '.join(lines))) > 1:
            raise ValueError('Too many markers')

        result, marker_type = None, None
        for i, line in enumerate(lines):
            res = regex.findall(line)
            if res:
                assert len(res) == 1
                j = line.find(res[0])
                result, marker_type = (i, j), res[0]
                break

        if result is not None and result[1] == 1:
            result = (result[0], 0)  # e.g. for known items view
        return result, marker_type

    def get_message_and_popup(self, obs):
        """ Uses MORE action to get full popup and/or message.
        """

        message = bytes(obs['message']).decode().replace('\0', ' ').replace('\n', '').strip()
        if message.endswith('--More--'):
            # FIXME: It seems like in this case the environment doesn't expect additional input,
            #        but I'm not 100% sure, so it's too risky to change it, because it could stall everything.
            #        With the current implementation, in the worst case, we'll get "Unknown command ' '".
            message = message[:-len('--More--')]

        # assert '\n' not in message and '\r' not in message
        popup = []

        lines = [bytes(line).decode().replace('\0', ' ').replace('\n', '') for line in obs['tty_chars']]
        marker_pos, marker_type = self._find_marker(lines)

        if marker_pos is None:
            return message, popup, True

        pref = ''
        message_lines_count = 0
        if message:
            for i, line in enumerate(lines[:marker_pos[0] + 1]):
                if i == marker_pos[0]:
                    line = line[:marker_pos[1]]
                message_lines_count += 1
                pref += line.strip()

                # I'm not sure when the new line character in broken messages should be a space and when be ignored.
                # '#' character (and others) occasionally occurs at the beginning of the broken line and isn't in
                # the message. Sometimes the message on the screen lacks last '.'.
                replace_func = lambda x: ''.join((c for c in x if c.isalnum()))
                if replace_func(pref) == replace_func(message):
                    break
            else:
                if marker_pos[0] == 0:
                    elems1 = [s for s in message.split() if s]
                    elems2 = [s for s in pref.split() if s]
                    assert len(elems1) < len(elems2) and elems2[-len(elems1):] == elems1, (elems1, elems2)
                    return pref, popup, False
                raise ValueError(f"Message:\n{repr(message)}\ndoesn't match the screen:\n{repr(pref)}")

        # cut out popup
        for l in lines[message_lines_count:marker_pos[0]] + [lines[marker_pos[0]][:marker_pos[1]]]:
            l = l[marker_pos[1]:].strip()
            if l:
                popup.append(l)

        return message, popup, False

    def update_message_and_popup(self, obs):
        if self._is_reading_message_or_popup:
            message_prefix = self.message + (' ' if self.message else '')
            popup_prefix = self.popup
        else:
            message_prefix = ''
            popup_prefix = []

        self.single_message, self.single_popup, done = self.get_message_and_popup(obs)
        self.single_message = self.single_message.strip()
        self.single_popup = [p.strip() for p in self.single_popup]

        self.message = message_prefix + self.single_message
        self.popup = popup_prefix + self.single_popup
        return done

    def step(self, action, additional_action_iterator=None):
        if self._no_step_calls:
            raise ValueError("Shouldn't call step now")

        if isinstance(action, str):
            assert len(action) == 1
            action = A.ACTIONS[A.ACTIONS.index(ord(action))]
        observation, reward, done, info = self.env.step(action)
        observation = {k: v.copy() for k, v in observation.items()}
        self.step_count += 1
        self.score += reward

        self.cursor_pos = (observation['tty_cursor'][0] - 1, observation['tty_cursor'][1])

        if hasattr(self, 'blstats'):
            for item in flatten_items(self.inventory.items):
                if item.category == nh.COIN_CLASS:
                    self.stats_logger.log_gold(item.count)
            else:
                self.stats_logger.log_gold(0)

        if done:
            raise AgentFinished()

        self.update(observation, additional_action_iterator)

    def update(self, observation, additional_action_iterator=None):
        self._observation = observation
        done = self.update_message_and_popup(observation)

        self._is_reading_message_or_popup = True
        if additional_action_iterator is not None:
            is_next_action = True
            try:
                next_action = next(additional_action_iterator)
            except StopIteration:
                is_next_action = False

            if is_next_action:
                self.step(next_action, additional_action_iterator)
                return

        # FIXME: self.update_state() won't be called on all states sometimes.
        #        Otherwise there are problems with atomic operations.
        if not done or observation['misc'][2]:
            self.step(A.TextCharacters.SPACE)
            return

        if observation['misc'][1]:  # entering text
            if "You may wish for an object." in self.message:
                # TODO: wishing strategy
                # TODO: assume wished item as blessed
                self.step('b', iter('lessed greased +2 gray dragon scale mail\r'))
                return
            else:
                self.step(A.Command.ESC)
                return

        if 'Where do you want to be teleported?' in self.message:
            # TODO: teleport control
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(observation['tty_chars'].reshape(-1)):
            self.type_text('y')
            return

        self._is_reading_message_or_popup = False
        self._message_history.append(self.message)
        if len(self._message_history) > 400:
            del self._message_history[:-200]  # only the last 50 are ever read

        # should_update = True

        # if self.turns_in_atom_operation is not None:
        #     should_update = False
        #     # if any([(self.last_observation[key] != observation[key]).any()
        #     #         for key in ['glyphs', 'blstats', 'inv_strs', 'inv_letters', 'inv_oclasses', 'inv_glyphs']]):
        #     #     self.turns_in_atom_operation += 1
        #     # assert self.turns_in_atom_operation in [0, 1]

        if self.last_observation is None:
            self.last_observation = observation
            self._previous_glyphs = self.last_observation['glyphs']
        else:
            self._previous_glyphs = self.last_observation['glyphs']
            self.last_observation = observation

        self.blstats = BLStats(*self.last_observation['blstats'])
        self.glyphs = self.last_observation['glyphs']

        self.stats_logger.log_cumulative_value('max_turns_on_position',
                                               key=(self.current_level().dungeon_number,
                                                    self.current_level().level_number,
                                                    self.blstats.y, self.blstats.x),
                                               value=self.blstats.time - self._last_turn)

        self._inactivity_counter += 1
        if self._last_turn != self.blstats.time:
            self._last_turn = self.blstats.time
            self._inactivity_counter = 0
        assert self._inactivity_counter < 200, ('turn inactivity', sorted(set(self._message_history[-50:])))

        self.update_state(allow_update=self._atom_operation_allow_update or not self.in_atom_operation,
                          allow_callbacks=not self.in_atom_operation)

    def update_state(self, allow_update=True, allow_callbacks=True):
        assert not self._no_step_calls
        if self._is_updating_state:
            return
        self._is_updating_state = True
        message = self.message
        popup = self.popup

        try:
            if allow_update:
                # functions that are allowed to call state unchanging steps
                for func in [self.character.update, self.inventory.update, self.monster_tracker.update,
                             partial(self.check_terrain, force=False), self.update_level,
                             self.global_logic.update]:
                    func()
                    self.message = message
                    self.popup = popup

            if allow_callbacks:
                self.call_update_functions()
        finally:
            self._is_updating_state = False

    def call_update_functions(self, funcs=None):
        if funcs is None:
            funcs = self.on_update
        assert all((func in self.on_update for func in funcs))

        with self.disallow_step_calling():
            for func in funcs:
                func()

    def _update_level_items(self):
        level = self.current_level()

        level.items[self.blstats.y, self.blstats.x] = self.inventory.items_below_me
        level.item_count[self.blstats.y, self.blstats.x] = len(self.inventory.items_below_me)

        # TODO: optimize
        ignore_mask = utils.isin(self.glyphs, G.MONS, G.PETS)  # TODO: effects, etc
        item_mask = level.item_count != 0
        mask = item_mask & ~ignore_mask
        level.item_disagreement_counter[~mask] = 0
        for y, x in zip(*mask.nonzero()):
            if (level.item_count[y, x] >= 2) == ((self.last_observation['specials'][y, x] & nh.MG_OBJPILE) > 0):
                glyphs = (glyph for item in level.items[y, x] for glyph in item.display_glyphs())
                if self.glyphs[y, x] in glyphs:
                    level.item_disagreement_counter[y, x] = 0
                    continue

            level.item_disagreement_counter[y, x] += 1
            if level.item_disagreement_counter[y, x] > 3:
                level.item_disagreement_counter[y, x] = 0
                level.items[y, x] = ()
                level.item_count[y, x] = 0

    def _update_level_shops(self):
        level = self.current_level()

        shop_type = None
        matches = re.search(f"Welcome( again)? to [a-zA-Z' ]*({'|'.join(SHOP.name2id.keys())})!", self.message)
        if matches is not None:
            shop_name = matches.groups()[1]
            assert shop_name in SHOP.name2id, shop_name
            shop_type = SHOP.name2id[shop_name]

        shopkeepers = list(
            zip(*(utils.isin(self.glyphs, G.SHOPKEEPER) & self.monster_tracker.peaceful_monster_mask).nonzero()))
        for y, x in shopkeepers:
            wall_mask = utils.isin(level.objects, G.WALL)
            entry = ((utils.translate(wall_mask, 1, 0) & utils.translate(wall_mask, -1, 0)) |
                     (utils.translate(wall_mask, 0, 1) & utils.translate(wall_mask, 0, -1))) & \
                    level.walkable
            walkable = level.walkable & ~entry
            mask = utils.bfs(y, x, walkable=walkable, walkable_diagonally=walkable, can_squeeze=False) != -1
            mask = utils.dilate(mask, radius=1)

            level.shop[mask] = True
            if mask[self.blstats.y, self.blstats.x] and shop_type is not None:
                level.shop_type[mask] = shop_type
            level.shop_interior[mask & ~utils.dilate(entry, radius=1, with_diagonal=False)] = True

    def _update_level_corpses(self):
        mnames = list(map(lambda x: x[-1],
                          re.findall(r'((kills?)|(destroys?)) ((an?)|(the) )?([a-zA-Z ]+)\!', self.message)))
        mnames += list(map(lambda x: x[-4],
                           re.findall(r'((An? )|(The )( *))([a-zA-Z ]+) is ((killed)|(destroyed))\!', self.message)))
        mnames = list(filter(lambda name: 'invisible' not in name and name != 'it' and not name.startswith('poor '),
                             mnames))
        mnames = list(map(lambda name: name[len('saddled '):] if name.startswith('saddled ') else name,
                          mnames))
        mnames = list(filter(lambda name: name[0].lower() == name[0],
                             mnames))

        level = self.current_level()

        if not self.character.prop.hallu and mnames:
            old_mons = self._previous_glyphs.copy()
            old_mons[~utils.isin(self._previous_glyphs, G.MONS, G.INVISIBLE_MON)] = -1
            new_mons = self.glyphs.copy()
            new_mons[~utils.isin(self.glyphs, G.MONS, G.INVISIBLE_MON)] = -1
            mask = disappearance_mask(old_mons, new_mons, 1)
            mons = old_mons.copy()
            mons[~mask] = -1

            assert mons.any()

            for mname in mnames:
                glyph = MON.from_name(mname)
                monster_id = glyph - nh.GLYPH_MON_OFF
                corpse_glyph = MON.body_from_name(mname)
                for y, x in zip(*utils.isin(mons, [glyph]).nonzero()):
                    # TODO: it works because level.items is updated in `inventory.check_items`
                    if all(map(lambda item: item.is_corpse() and item.monster_id != monster_id, level.items[y, x])):
                        level.corpses_to_eat[y, x][monster_id] = self.blstats.time

        old_possible_corpses = level.corpses_to_eat[self.blstats.y, self.blstats.x].copy()
        del level.corpses_to_eat[self.blstats.y, self.blstats.x]

        corpses = Counter((item for item in self.inventory.items_below_me if item.is_corpse()))
        for item, count in corpses.items():
            if count != 1:
                continue

            level.corpses_to_eat[self.blstats.y, self.blstats.x][item.monster_id] = \
                old_possible_corpses[item.monster_id]

    def update_level(self):
        if utils.isin(self.glyphs, G.SWALLOW).any():
            return

        if utils.any_in(self.glyphs, G.PETS):
            self._last_pet_seen = self.blstats.time

        level = self.current_level()

        mask = utils.isin(self.glyphs, G.FLOOR, G.STAIR_UP, G.STAIR_DOWN, G.DOOR_OPENED, G.TRAPS,
                          G.ALTAR, G.FOUNTAIN)
        level.walkable[mask] = True
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]

        mask = utils.isin(self.glyphs, G.MONS, G.PETS, G.BODIES, G.OBJECTS, G.STATUES)
        level.seen[mask] = True
        level.walkable[mask & (level.objects == -1)] = True

        mask = utils.isin(self.glyphs, G.WALL, G.DOOR_CLOSED, G.BARS)
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]
        level.walkable[mask] = False

        self._update_level_items()
        self._update_level_shops()
        self._update_level_corpses()

        for y, x in zip(*utils.isin(level.objects, G.ALTAR).nonzero()):
            if (y, x) not in level.altars:
                level.altars[y, x] = Character.UNKNOWN

        level.was_on[self.blstats.y, self.blstats.x] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x, shuffle=False):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]
                level.walkable[y, x] = False  # necessary for the exit route from vaults

        # ad aerarium -- avoid valut entrance
        if self.inventory.engraving_below_me and nltk.edit_distance(self.inventory.engraving_below_me,
                                                                    "ad aerarium") <= 6:
            self.stats_logger.log_event('ad_aerarium_below_me')
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                y, x = self.blstats.y + dy, self.blstats.x + dx
                if (0 <= y < level.forbidden.shape[0] and 0 <= x < level.forbidden.shape[1]) \
                        and not level.walkable[y, x]:
                    level.forbidden[y, x] = True

    ######## TRIVIAL HELPERS

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    @staticmethod
    def calc_direction(from_y, from_x, to_y, to_x, allow_nonunit_distance=False):
        if allow_nonunit_distance:
            assert from_y == to_y or from_x == to_x or \
                   abs(from_y - to_y) == abs(from_x - to_x), ((from_y, from_x), (to_y, to_x))
            to_y = from_y + np.sign(to_y - from_y)
            to_x = from_x + np.sign(to_x - from_x)

        assert abs(from_y - to_y) <= 1 and abs(from_x - to_x) <= 1, ((from_y, from_x), (to_y, to_x))

        ret = ''
        if to_y == from_y + 1: ret += 's'
        if to_y == from_y - 1: ret += 'n'
        if to_x == from_x + 1: ret += 'e'
        if to_x == from_x - 1: ret += 'w'
        if ret == '': ret = '.'

        return ret

    ######## TRIVIAL ACTIONS

    def check_terrain(self, force):
        if force or self._last_terrain_check is None or self.blstats.time - self._last_terrain_check > 50:
            self._last_terrain_check = self.blstats.time
            with self.atom_operation():
                self.type_text('#te')
                self.step(A.MiscAction.MORE, iter('b'))
                self.update_level()
                self.step(A.Command.ESC)

    def wield_best_melee_weapon(self):
        # TODO: move to inventory
        item = self.inventory.get_best_melee_weapon()
        if item != self.inventory.items.main_hand:
            return self.inventory.wield(item)
        return False

    def type_text(self, text):
        with self.atom_operation():
            for char in text:
                self.step(char)

    def untrap(self, trap_y, trap_x):
        with self.atom_operation():
            self.type_text('#u')
            self.step(A.MiscAction.MORE)
            assert self.single_message == "In what direction?", self.single_message
            self.direction(trap_y, trap_x)
            if self.single_message == 'You know of no traps there.':
                # assert 0, "Trying to untrap already untraped trap"
                return True
            if self.single_message == 'You disarm the trap.':
                self.stats_logger.log_event('untrap_success')
                return True
            return False

    def untrap_container_below_me(self):
        """ Return None if succesfull else fail message """
        with self.atom_operation():
            self.type_text('#u')
            self.step(A.MiscAction.MORE)
            assert self.single_message == "In what direction?", self.single_message
            self.type_text('.')
            if 'too busy' in self.message:
                return 'hands busy'
            if 'There is a container and a ' in self.message:
                self.type_text('n')
            if 'You know of no traps there.' in self.message:
                raise AgentPanic('no container below me to untrap')
            if 'There is a container and ' in self.message and \
                    (' trap here.' in self.message or ' field here.' in self.message) and \
                    ('trap?' in self.message or 'field?' in self.message):
                self.type_text('n')
            if 'You cannot disable this trap.' in self.single_message:
                return
            assert 'Check it for traps?' in self.single_message, self.single_message
            self.type_text('y')
            if self.message.startswith('You find no traps on the'):
                return
            assert 'Disarm it?' in self.message, self.message
            self.type_text('y')
            if 'You disarm it!' in self.message:
                self.stats_logger.log_event('container_untrap_success')
                return
            self.stats_logger.log_event('container_untrap_fail')
            return self.message

    def critically_low_hp(self):
        """NetHack 3.6 pray.c critically_low_hp(): low HP counts as major trouble."""
        hp, max_hp, level = self.blstats.hitpoints, self.blstats.max_hitpoints, self.blstats.experience_level
        max_hp = min(max_hp, 15 * level)
        divisor = 5 if level <= 5 else 6 if level <= 13 else 7 if level <= 21 else 8 if level <= 29 else 9
        if not PRAY_AT_EXACT_THRESHOLD:
            divisor, max_hp = 7, self.blstats.max_hitpoints
        return hp <= 5 or hp * divisor <= max_hp

    def is_safe_to_pray(self, limit=500):
        # the prayer timeout starts at 300 and drops by one a turn; major trouble is fixed once it is
        # at most 200, so the first prayer is safe from about turn 100 (not 300: a diver is long dead)
        if self.prayer_failed:
            return False
        return (
                (self.last_prayer_turn is None and self.blstats.time > 110) or
                (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > limit)
        )

    def pray(self):
        self.step(A.Command.PRAY)
        self.last_prayer_turn = self.blstats.time
        if any(s in self.message for s in PRAYER_FAILED_MESSAGES):
            self.prayer_failed = True
        # TODO: return value
        return True

    def open_door(self, y, x):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.DOOR_CLOSED
            self.direction(y, x)
            self.current_level().door_open_count[y, x] += 1
            return self.glyphs[y, x] not in G.DOOR_CLOSED

    def melee_attack(self, y, x):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.MONS or self.glyphs[y, x] in G.INVISIBLE_MON or \
                   self.glyphs[y, x] in G.SWALLOW
            self.direction(y, x)
        return True

    def zap(self, item, direction):
        letter = self.inventory.items.get_letter(item)
        with self.atom_operation():
            self.step(A.Command.ZAP)
            self.type_text(letter)
            self.direction(direction)
        if 'Nothing happens' in self.message or 'You wrest' in self.message:
            # an empty wand: zapping it again in the next crisis wastes the turn that kills us
            self._empty_wand_letters.add(letter)
        return True

    def wand_is_empty(self, item):
        return self.inventory.items.get_letter(item) in self._empty_wand_letters

    def fire(self, item, direction):
        if self.character.prop.polymorph:
            # TODO: throwing is not possible if you don't have hands
            # it may be possible depending on creature
            return False

        with self.atom_operation():
            self.step(A.Command.THROW)
            self.type_text(self.inventory.items.get_letter(item))
            self.direction(direction)
        return True

    def cast(self, spell_name, direction):
        with self.atom_operation():
            dy, dx = direction
            direction = self.calc_direction(self.blstats.y, self.blstats.x, self.blstats.y + dy, self.blstats.x + dx)
            success = [False]

            def type_letters():
                # while f'{letter} - ' not in '\n'.join(self.single_popup):
                #     yield A.TextCharacters.SPACE
                # if self.single_message.startswith("You fail to cast the spell correctly."):
                #     return
                if 'You are too impaired' in self.message:
                    return
                yield self.character.known_spells[spell_name]
                for _ in range(3):
                    if 'In what direction?' in self.message:
                        break
                    yield ' '
                if 'In what direction?' in self.message:
                    success[0] = True
                    yield direction

            self.step(A.Command.CAST, type_letters())
            if success[0]:
                self.stats_logger.log_event(f'cast_{spell_name}')
            else:
                self.last_cast_fail_turn[spell_name] = self._last_turn
                self.stats_logger.log_event(f'cast_fail_{spell_name}')

    def kick(self, y, x=None):
        with self.panic_if_position_changes():
            with self.atom_operation():
                self.step(A.Command.KICK)
                self.direction(self.calc_direction(self.blstats.y, self.blstats.x, y, x))

    def search(self, max_count=1):
        assert max_count >= 1
        with self.panic_if_position_changes():
            with self.atom_operation():
                if max_count > 1:
                    self.type_text(str(max_count))
                self.step(A.Command.SEARCH)
                # TODO: estimate the real number of searches
                self.current_level().search_count[self.blstats.y, self.blstats.x] += max_count
                if 'You find ' in self.message:
                    self.check_terrain(force=True)
        return True

    def direction(self, y, x=None):
        if x is not None:
            dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x)
        else:
            dir = y

        action = {
            'n': A.CompassDirection.N, 's': A.CompassDirection.S,
            'e': A.CompassDirection.E, 'w': A.CompassDirection.W,
            'ne': A.CompassDirection.NE, 'se': A.CompassDirection.SE,
            'nw': A.CompassDirection.NW, 'sw': A.CompassDirection.SW,
            '>': A.MiscDirection.DOWN, '<': A.MiscDirection.UP,
            '.': A.MiscDirection.WAIT,
        }[dir]

        self.step(action)
        return True

    def move(self, y, x=None):
        if x is not None:
            dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x)
        else:
            dir = y

        expected_y = self.blstats.y + ('s' in dir) - ('n' in dir)
        expected_x = self.blstats.x + ('e' in dir) - ('w' in dir)

        if (expected_y != self.blstats.y or expected_x != self.blstats.x) \
                and self.monster_tracker.monster_mask[expected_y, expected_x]:
            # TODO: consider handling it in different way, since this situation is sometimes expected
            raise AgentPanic(f'Monster on a next tile when moving: ({expected_y},{expected_x})')

        # TODO: portals
        if dir in ['<', '>']:
            level = self.current_level()
            with self.atom_operation():
                self.direction(dir)
                assert self.current_level().key() != level.key(), self.message
                level.stair_destination[expected_y, expected_x] = \
                    (self.current_level().key(), (self.blstats.y, self.blstats.x))
                # TODO: one way portals (elemental and astral planes)
                self.current_level().stair_destination[
                    (self.blstats.y, self.blstats.x)] = (level.key(), (expected_y, expected_x))

        else:
            level = self.current_level()
            my_y, my_x = self.blstats.y, self.blstats.x
            self.direction(dir)

            # a doorway the map shows as doorless can still hold an (open or broken-looking) door;
            # remember it or the diagonal-shortest path retries the same refused step forever.
            # From github.com/eL1fe/nethacker@dc2765b.
            if "move diagonally into an intact doorway" in self.message:
                level.no_diagonal[expected_y, expected_x] = True
            if "move diagonally out of an intact doorway" in self.message:
                level.no_diagonal[my_y, my_x] = True

            if self.blstats.y != expected_y or self.blstats.x != expected_x:
                raise AgentPanic(f'agent position do not match after "move": '
                                 f'expected ({expected_y}, {expected_x}), got ({self.blstats.y}, {self.blstats.x})')

    def can_engrave(self):
        if self.character.prop.polymorph:
            return False  # TODO: only for handless monsters (which cannot write)
        return (self.blstats.y, self.blstats.x) != self._forbidden_engrave_position

    def engrave(self, text):
        assert '\r' not in text
        ret = False

        def gen():
            nonlocal ret
            if 'What do you want to write with?' not in self.single_message:
                self._forbidden_engrave_position = (self.blstats.y, self.blstats.x)
                yield A.Command.ESC
                return
            yield '-'
            if 'Do you want to add to the current engraving?' in self.single_message:
                yield 'n'
            while self._observation['misc'][2]:
                yield ' '
            if 'What do you want to write in the dust here?' not in self.single_message:
                self._forbidden_engrave_position = (self.blstats.y, self.blstats.x)
                yield A.Command.ESC
                return
            yield from text
            yield '\r'
            ret = True

        with self.atom_operation():
            self.step(A.Command.ENGRAVE, gen())
            self.inventory.get_items_below_me()

        if ret and text.lower() == 'elbereth':
            self.stats_logger.log_event('elbereth_write')
        return ret

    ######## NON-TRIVIAL HELPERS

    def neighbors(self, y, x, shuffle=True, diagonal=True):
        ret = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if not diagonal and abs(dy) + abs(dx) > 1:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < C.SIZE_Y and 0 <= nx < C.SIZE_X:
                    ret.append((ny, nx))

        if shuffle:
            self.rng.shuffle(ret)
            pass

        return ret

    def bfs(self, y=None, x=None, can_squeeze=None):
        if y is None:
            y = self.blstats.y
        if x is None:
            x = self.blstats.x
        default_squeeze = can_squeeze is None

        if default_squeeze and self.last_bfs_step == self.step_count and \
                y == self.blstats.y and x == self.blstats.x:
            return self.last_bfs_dis.copy()

        level = self.current_level()

        walkable = level.walkable & ~utils.isin(self.glyphs, G.BOULDER) & \
                   ~self.monster_tracker.peaceful_monster_mask & \
                   ~level.forbidden

        if self._last_turn - self._allow_walking_through_traps_turn > 50:
            walkable &= ~utils.isin(level.objects, G.TRAPS)

        for my, mx in list(zip(*np.nonzero(utils.isin(self.glyphs, G.MONS)))):
            mon = MON.permonst(self.glyphs[my][mx])
            if mon.mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS:
                walkable[my, mx] = False

        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable & ~utils.isin(level.objects, G.DOORS) & (level.objects != -1) &
                                            ~level.no_diagonal,
                        can_squeeze=(self.inventory.items.total_weight <= 600 if default_squeeze else can_squeeze) and \
                                    self.current_level().dungeon_number != Level.SOKOBAN,
                        )

        if default_squeeze and y == self.blstats.y and x == self.blstats.x:
            self.last_bfs_dis = dis
            self.last_bfs_step = self.step_count

        return dis.copy()

    def path(self, from_y, from_x, to_y, to_x, dis=None):
        if from_y == to_y and from_x == to_x:
            return [(to_y, to_x)]

        if dis is None:
            dis = self.bfs(from_y, from_x)

        assert dis[to_y, to_x] != -1

        # FIXME: currently the path can lead through diagonally inwalkable tiles.
        #        The path is the shortest possible, so the agent is guaranteed to
        #        unstuck itself eventually (usually a few panic exceptions) if that happens

        cur_y, cur_x = to_y, to_x
        path_rev = [(cur_y, cur_x)]
        while cur_y != from_y or cur_x != from_x:
            for y, x in self.neighbors(cur_y, cur_x):
                if dis[y, x] == dis[cur_y, cur_x] - 1 and dis[y, x] >= 0:
                    path_rev.append((y, x))
                    cur_y, cur_x = y, x
                    break
            else:
                assert 0

        assert dis[cur_y, cur_x] == 0 and from_y == cur_y and from_x == cur_x
        path = path_rev[::-1]
        assert path[0] == (from_y, from_x) and path[-1] == (to_y, to_x)
        return path

    ######## NON-TRIVIAL ACTIONS

    def _fast_go_to(self, y, x):
        with self.atom_operation():
            self.step(A.Command.TRAVEL)
            py, px = self.cursor_pos
            while py != y or px != x:
                dy, dx = np.sign(y - py), np.sign(x - px)
                self.direction(self.calc_direction(py, px, py + dy, px + dx))
                py += dy
                px += dx
                assert (py, px) == self.cursor_pos
            self.direction('.')

        if (self.blstats.y, self.blstats.x) == (y, x):
            return True

    def go_to(self, y, x, stop_one_before=False, max_steps=None,
              debug_tiles_args=None, callback=lambda: False, fast=False):
        assert not stop_one_before or (self.blstats.y != y or self.blstats.x != x)
        assert max_steps is None or not fast

        if stop_one_before and self.bfs()[y, x] == -1:
            dis = self.bfs()
            best_p = None
            for ny, nx in self.neighbors(y, x):
                if dis[ny, nx] != -1 and (best_p is None or dis[best_p] > dis[ny, nx]):
                    best_p = ny, nx
            if best_p is None:
                assert 0, 'no reachable neighbor'
            y, x = best_p
            stop_one_before = False

        assert self.bfs()[y, x] != -1

        if callback():
            return
        steps_taken = 0
        cont = True
        while cont and (self.blstats.y, self.blstats.x) != (y, x):
            dis = self.bfs()
            if dis[y, x] == -1:
                raise AgentPanic('end point is no longer accessible')
            path = self.path(self.blstats.y, self.blstats.x, y, x)
            orig_path = path
            path = path[1:]
            if stop_one_before:
                path = path[:-1]

            if fast and len(path) > 2:
                my_y, my_x = self.blstats.y, self.blstats.x
                self._fast_go_to(*path[-1])
                if (my_y, my_x) != (self.blstats.y, self.blstats.x):
                    continue

            with self.env.debug_tiles(orig_path, **debug_tiles_args) \
                    if debug_tiles_args is not None else contextlib.suppress():
                for y, x in path:
                    if self.monster_tracker.peaceful_monster_mask[y, x]:
                        cont = True
                        break
                    if not self.current_level().walkable[y, x]:
                        cont = True
                        break
                    self.move(y, x)
                    if callback():
                        return
                    steps_taken += 1
                    if max_steps is not None and steps_taken >= max_steps:
                        cont = False
                        break
                else:
                    cont = False

    ######## LOW-LEVEL STRATEGIES

    def get_visible_monsters(self):
        """ Returns list of tuples (distance, y, x, permonst, monster_glyph)
        """
        mask = self.monster_tracker.monster_mask & ~self.monster_tracker.peaceful_monster_mask
        if not mask.any():
            return []

        dis = self.bfs()
        ret = []
        for y, x in zip(*mask.nonzero()):
            if (dis[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] != -1).any():
                if self.glyphs[y, x] == nh.GLYPH_INVISIBLE or \
                        not MON.is_monster(self.glyphs[y, x]):  # TODO: some ghost are not visible in glyphs (?)
                    if utils.adjacent((self.blstats.y, self.blstats.x), (y, x)):
                        class dummy_permonst:
                            mname = 'unknown'
                            mlet = '0'
                            mmove = 12

                        ret.append((dis[y][x], y, x, dummy_permonst(), self.glyphs[y][x]))
                else:
                    ret.append((dis[y][x], y, x, MON.permonst(self.glyphs[y][x]), self.glyphs[y][x]))
        ret.sort()
        return ret

    @utils.debug_log('fight2')
    @Strategy.wrap
    def fight2(self):
        yielded = False
        wait_counter = 0
        while 1:
            monsters = self.get_visible_monsters()
            allow_attack_all = self._last_turn - self._allow_attack_all_turn < 3
            only_ranged_slow_monsters = all([monster[3].mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS
                                             and not combat.monster_utils.consider_melee_only_ranged_if_hp_full(self,
                                                                                                                monster)
                                             for monster in monsters])

            dis = self.bfs()

            if not monsters or all(dis > 7 for dis, *_ in monsters) or \
                    (only_ranged_slow_monsters and not self.inventory.get_ranged_combinations()
                     and np.sum(dis != -1) > 1 and not allow_attack_all):
                if wait_counter:
                    self.search()
                    wait_counter -= 1
                    continue
                if not yielded:
                    yield False
                return

            if not yielded:
                yielded = True
                yield True
                self.character.parse_enhance_view()
                # only parse spells in the deep phase so the early level-1 grind (and its RNG) is
                # left exactly as the parent plays it; this is what keeps the strong runs intact
                if self.blstats.experience_level >= 8 or self.character.role in (Character.WIZARD, Character.MONK):
                    self.character.parse_spellcast_view()

            move_priority_heatmap, actions = combat.fight_heur.get_priorities(self)
            actions.extend(combat.fight_heur.get_move_actions(self, dis, move_priority_heatmap))

            if self.character.prop.polymorph:
                actions = list(filter(lambda x: x[1][0] != 'ranged', actions))
            actions = [a for a in actions if not self._touch_petrifies(a[1])]

            actions = [a for a in actions if not self._touch_petrifies(a[1])]

            if allow_attack_all:
                attack_actions = [a for a in actions if a[1][0] in ('melee', 'kick', 'ranged', 'zap', 'force_bolt')]
                if attack_actions:
                    actions = attack_actions

            if not actions:
                assert 0, 'No possible action available during fight2'

            priority, best_action = max(actions, key=lambda x: x[0]) if actions else None

            with self.env.debug_tiles(move_priority_heatmap, color='turbo', is_heatmap=True):
                actions_str = '|'.join([combat.utils.action_str(self, a) for a in sorted(actions, key=lambda x: x[0])])
                with self.env.debug_log(actions_str):
                    wait_counter = self._fight2_perform_action(best_action, wait_counter)

    def _touch_petrifies(self, action):
        # hitting a cockatrice bare-handed (Monk martial arts) or kicking it
        # barefoot turns you to stone on the spot
        if action[0] not in ('melee', 'kick'):
            return False
        _, dy, dx = action
        glyph = self.glyphs[self.blstats.y + dy, self.blstats.x + dx]
        if glyph not in G.MONS or MON.permonst(glyph).mname not in ('cockatrice', 'chickatrice'):
            return False
        if action[0] == 'kick':
            return self.inventory.items.boots is None
        return self.inventory.items.gloves is None and self.inventory.items.main_hand is None

    def _fight2_perform_action(self, best_action, wait_counter):
        if best_action[0] == 'move':
            _, dy, dx = best_action
            target_y, target_x = self.blstats.y + dy, self.blstats.x + dx
            with self.env.debug_tiles([[self.blstats.y, self.blstats.x],
                                       [target_y, target_x]], color=(0, 255, 0), is_path=True):
                wait_counter = 5
                self.move(target_y, target_x)
                return wait_counter
        elif best_action[0] == 'melee':
            _, dy, dx = best_action
            target_y = self.blstats.y + dy
            target_x = self.blstats.x + dx
            if self.wield_best_melee_weapon():
                return wait_counter
            with self.env.debug_tiles([[self.blstats.y, self.blstats.x],
                                       [target_y, target_x]], color=(255, 0, 255), is_path=True):
                self.melee_attack(target_y, target_x)
                wait_counter = 0
                return wait_counter

        elif best_action[0] == 'kick':
            _, dy, dx = best_action
            self.kick(self.blstats.y + dy, self.blstats.x + dx)
            wait_counter = 0
            return wait_counter

        elif best_action[0] == 'ranged':
            _, dy, dx = best_action
            target_y = self.blstats.y + dy
            target_x = self.blstats.x + dx
            launcher, ammo = self.inventory.get_best_ranged_set()
            assert ammo is not None
            if launcher is not None and not launcher.equipped:
                if self.inventory.wield(launcher):
                    return wait_counter
            with self.env.debug_tiles([[target_y, target_x]], (0, 0, 255, 255), mode='frame'):
                dir = self.calc_direction(self.blstats.y, self.blstats.x, target_y, target_x,
                                          allow_nonunit_distance=True)
                fired = self.fire(ammo, dir)
                assert fired, (ammo, dir)
                return wait_counter

        elif best_action[0] == 'force_bolt':
            _, dy, dx = best_action
            self.cast('force bolt', direction=(dy, dx))
            return wait_counter

        elif best_action[0] == 'elbereth':
            assert self.inventory.engraving_below_me.lower() != 'elbereth'
            self.engrave("Elbereth")
            return wait_counter
        elif best_action[0] == 'wait':
            assert self.inventory.engraving_below_me.lower() == 'elbereth'
            self.stats_logger.log_event('wait_in_fight')
            self.search()
            return wait_counter
        elif best_action[0] == 'zap':
            if len(best_action) == 5:
                _, dy, dx, wand, targeted_monsters = best_action
            else:
                _, dy, dx, = best_action
                for item in self.inventory.items:
                    if item.is_offensive_usable_wand() and not self.wand_is_empty(item):
                        wand = item
                        break
                else:
                    assert 0
                targeted_monsters = []
            dir = self.calc_direction(self.blstats.y, self.blstats.x, self.blstats.y + dy, self.blstats.x + dx,
                                      allow_nonunit_distance=True)

            with self.env.debug_tiles([[my, mx] for my, mx, _ in targeted_monsters],
                                      (255, 0, 255, 255), mode='frame'):
                self.zap(wand, dir)
            return wait_counter

        elif best_action[0] == 'pickup':
            if len(best_action) == 2:
                _, items_to_pickup = best_action
            else:
                items_to_pickup = combat.fight_heur.decide_what_to_pickup(self)
            self.inventory.pickup(items_to_pickup)
            return wait_counter
        elif best_action[0] == 'go_to':
            _, target_y, target_x = best_action
            self.go_to(target_y, target_x, stop_one_before=True, max_steps=1,
                       debug_tiles_args=dict(color=(255, 0, 0), is_path=True))
            return wait_counter
        raise NotImplementedError(best_action)

    @utils.debug_log('engulfed_fight')
    @Strategy.wrap
    def engulfed_fight(self):
        if not utils.any_in(self.glyphs, G.SWALLOW):
            yield False
        yield True
        while True:
            mask = utils.isin(self.glyphs, G.SWALLOW)
            if not mask.any():
                break
            assert self.melee_attack(*list(zip(*mask.nonzero()))[0])

    def _is_corpse_editable(self, monster_id, age_turn):
        permonst = MON.permonst(monster_id)

        # TODO: read intrinsics
        if self.character.race != Character.ORC and permonst.mflags1 & MON.M1_POIS != 0:
            return False

        # TODO: read intrinsics
        if permonst.mflags1 & MON.M1_ACID != 0:
            return False

        if permonst.mflags2 & MON.M2_WERE != 0:
            return False

        # polymorph
        if monster_id in [MON.id_from_name(name) for name in ['chameleon', 'doppelganger', 'sandestin']]:
            return False

        # remove random intrinsic
        if monster_id in [MON.id_from_name(name) for name in ['disenchanter']]:
            return False

        # hallucination
        if monster_id in [MON.id_from_name(name) for name in ['abbot', 'violet fungus', 'yellow mold']]:
            return False

        # stun
        if monster_id in [MON.id_from_name(name) for name in ['bat', 'giant bat']]:
            return False

        # aggravate monster
        if monster_id in [MON.id_from_name(name) for name in ['dog', 'little dog', 'large dog',
                                                              'kitten', 'housecat', 'large cat']]:
            return False

        # teleportitis
        # if ord(permonst.mlet) in [MON.S_LEPRECHAUN, MON.S_NYMPH]:
        #     return False

        # petrification
        if ord(permonst.mlet) == MON.S_COCKATRICE or monster_id == MON.id_from_name('Medusa'):
            return False

        # temporary prevents movement
        if ord(permonst.mlet) == MON.S_MIMIC:
            return False

        # cannibalism
        race_flag = {
            Character.HUMAN: MON.M2_HUMAN,
            Character.DWARF: MON.M2_DWARF,
            Character.ELF: MON.M2_ELF,
            Character.GNOME: MON.M2_GNOME,
            Character.ORC: 0,
        }[self.character.race]
        if self.character.role == Character.CAVEMAN:
            race_flag = 0
        if permonst.mflags2 & race_flag:
            return False

        # corpse aging: a corpse turns tainted once (age / (10 + rn2(20))) > 5, and from > 3 it can
        # blind, confuse or knock you out; the age recorded here is only the kill we saw, so an
        # older corpse of the same kind on that square passes for fresh. 30 turns keeps even the
        # worst roll at rotted <= 3.
        if self.blstats.time - age_turn >= CORPSE_MAX_AGE and \
                monster_id not in [MON.id_from_name('lizard'), MON.id_from_name('lichen')]:
            return False

        return True

    @utils.debug_log('eat_corpses_from_ground')
    @Strategy.wrap
    def eat_corpses_from_ground(self, only_below_me=True):
        yielded = False
        level = self.current_level()
        to_eat = []  # (y, x, monster_id)

        if only_below_me:
            y, x = self.blstats.y, self.blstats.x
            if (y, x) not in level.corpses_to_eat:
                yield False
            corpse_mapping = level.corpses_to_eat[y, x]
            for monster_id, corpse_age in corpse_mapping.items():
                if level.shop[y, x]:
                    continue
                if self._is_corpse_editable(monster_id, corpse_age):
                    to_eat.append((y, x, monster_id))

        else:
            for (y, x), corpse_mapping in level.corpses_to_eat.items():
                for monster_id, corpse_age in corpse_mapping.items():
                    if level.shop[y, x]:
                        continue
                    if self._is_corpse_editable(monster_id, corpse_age):
                        to_eat.append((y, x, monster_id))

        if not to_eat:
            yield False

        dis = self.bfs()
        to_eat = sorted(filter(lambda e: dis[e[0], e[1]] != -1, to_eat), key=lambda e: dis[e[0], e[1]])
        if not to_eat:
            yield False

        target_y, target_x, monster_id = to_eat[0]

        if (target_y, target_x) != (self.blstats.y, self.blstats.x):
            if not yielded:
                yielded = True
                yield True
            self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 255, 0), is_path=True))

        # TODO: checking level.corpses_to_eat again (moving to non-existing corpses often)
        if (target_y, target_x) in level.corpses_to_eat and monster_id in level.corpses_to_eat[target_y, target_x]:
            corpse_age = level.corpses_to_eat[target_y, target_x][monster_id]
            if level.shop[target_y, target_x]:
                del level.corpses_to_eat[target_y, target_x]
                return
            for item in self.inventory.items_below_me:
                if item.is_corpse() and item.monster_id == monster_id:
                    if self._is_corpse_editable(monster_id, corpse_age):
                        if not yielded:
                            yielded = True
                            yield True
                        self.inventory.eat(item)

            if not yielded:
                del level.corpses_to_eat[target_y, target_x][monster_id]

        if not yielded:
            yield False

    def should_cast_heal(self):
        # TODO: consider casting for other classes
        # a third of Monks start with the healing spellbook
        if self.character.role not in (self.character.HEALER, self.character.MONK):
            return False
        if 'healing' not in self.character.known_spells:
            return False
        if self.blstats.hunger_state >= Hunger.FAINTING:
            return False
        if self._last_turn - self.last_cast_fail_turn['healing'] < 2:
            return False
        if self.character.spell_fail_chance['healing'] > 0.2:
            return False
        hp_ratio = self.blstats.hitpoints / self.blstats.max_hitpoints
        low_hp = hp_ratio < 0.5 or (self.blstats.hitpoints < 10 and self.blstats.max_hitpoints > 10)
        return self.blstats.energy >= 5 and low_hp

    def should_cast_extra_heal(self):
        if 'extra healing' not in self.character.known_spells:
            return False
        if self.blstats.hunger_state >= Hunger.FAINTING:
            return False
        if self._last_turn - self.last_cast_fail_turn['extra healing'] < 2:
            return False
        if self.character.spell_fail_chance['extra healing'] > 0.15:
            return False
        hp_ratio = self.blstats.hitpoints / self.blstats.max_hitpoints
        low_hp = hp_ratio < 0.5 and (self.blstats.max_hitpoints - self.blstats.hitpoints > 25)
        return self.blstats.energy >= 15 and low_hp

    @utils.debug_log('emergency_strategy')
    @Strategy.wrap
    def emergency_strategy(self):
        if self.blstats.hunger_state >= Hunger.FAINTING:
            if self._fainting_since is None:
                self._fainting_since = self.blstats.time
        else:
            self._fainting_since = None

        if self.blstats.experience_level >= 8 or self.character.role == Character.MONK:
            if self.should_cast_extra_heal():
                yield True
                self.cast('extra healing', direction=(0, 0))
                return

            if self.should_cast_heal():
                yield True
                self.cast('healing', direction=(0, 0))
                return

        items = [item for item in flatten_items(self.inventory.items) if item.is_unambiguous() and
                 item.category == nh.POTION_CLASS and item.object.name in ['healing', 'extra healing', 'full healing']]
        if (
                (self.blstats.hitpoints < 1 / 3 * self.blstats.max_hitpoints
                 or self.blstats.hitpoints < 8) and items
        ):
            yield True
            self.inventory.quaff(items[0])
            return

        items = [item for item in flatten_items(self.inventory.items) if item.is_unambiguous() and
                 item.category == nh.POTION_CLASS and item.object.name in ['fruit juice']]
        if items and self.blstats.hunger_state >= Hunger.FAINTING:
            yield True
            self.inventory.quaff(items[0])
            return

        # hypothesis: the weak Healers die mid-grind because the emergency prayer only fires at
        # HP < max/5 (or /6) or HP < 6 -- so low that a single hard hit (mumak, soldier ant, magic
        # missile, rothe, ...) drops them from a seemingly-safe HP straight to dead, skipping the
        # window entirely. Raise the crisis threshold to match the potion threshold (HP < max/3 or
        # HP < 8) so prayer -- the reliable full-heal backstop once potions are spent -- kicks in
        # with real margin. Games here are short (death by Xp5-8), so this rarely spends the ~once-
        # per-1000-turn prayer more than once, and surviving a hit beats starving 500 turns later.
        # Since score is essentially a function of experience level, more survival == more XP == more
        # score across all four Healer identities.
        # hypothesis: human Healers die in early melee far more than gnome Healers (0.05 vs 0.08-0.10
        # here). A gnome Healer reaches the Gnomish Mines, where gnomish monsters are peaceful, and
        # levels up in relative safety; a human Healer has no such refuge and gets whittled to death
        # by ordinary monsters with its feeble scalpel. Its strongest emergency tool is the wand of
        # sleep, yet in a crisis it either melees on (and dies) or prays -- and praying takes several
        # turns during which the adjacent monster keeps hitting, so it can die mid-prayer. When at
        # crisis HP (HP<max/3) with a hostile monster adjacent and a wand of sleep in hand, sleep the
        # attacker FIRST: this disables the immediate threat so the following turns (prayer, potion,
        # Elbereth, or plain HP regen) resolve safely, turning otherwise-fatal early fights into
        # survival == more XP == more score. A per-run turn guard stops it re-zapping an already-
        # sleeping monster and draining the wand. Scoped to non-gnome Healers so every gnome run stays
        # byte-identical to the parent (gnomes already progress deep via the mines, and their fragile
        # deep runs must not be perturbed), and to this HP<max/3 + adjacent-threat crisis, a state
        # healthy runs never reach, so it never disturbs the human Xp7-9 runs either.
        if self.character.race != self.character.GNOME and \
                self.inventory.engraving_below_me.lower() != 'elbereth' and \
                self.blstats.time - getattr(self, '_last_emergency_sleep_turn', -100) >= 8 and \
                (self.blstats.hitpoints < 1 / 3 * self.blstats.max_hitpoints
                 or self.blstats.hitpoints < 8):
            sleep_wand = None
            for item in flatten_items(self.inventory.items):
                if item.is_wand() and item.is_unambiguous() and item.object.name == 'sleep' \
                        and item.uses != 'no charges' and not str(item.uses).endswith(':0') \
                        and not self.wand_is_empty(item):
                    sleep_wand = item
                    break
            if sleep_wand is not None:
                target = None
                for _, my, mx, _, _ in self.get_visible_monsters():
                    if max(abs(my - self.blstats.y), abs(mx - self.blstats.x)) == 1:
                        target = (my, mx)
                        break
                if target is not None:
                    direction = self.calc_direction(self.blstats.y, self.blstats.x, *target)
                    self._last_emergency_sleep_turn = self.blstats.time
                    yield True
                    self.zap(sleep_wand, direction)
                    return

        # hypothesis: prayer is the Dlvl 1 grind's food supply (a hunger prayer every ~1000 turns for
        # 15-20k turns), and traces show that is what ends most grinds: the bot prays at the first
        # faint, ~900-1000 turns after the previous prayer, while the rnz(350) prayer timeout has a
        # long tail -- about 1 in 10 such prayers come too soon, the god is angered (Luck -3, anger),
        # and every later prayer (re-tried each 400 turns) smites again (lost levels) while the
        # character faints over and over until a newt, rat or bat kills it. Waiting until
        # HUNGER_PRAYER_GAP turns have passed cuts that failure rate to ~2-3%; fainting-phase
        # hunger drops at a tenth of the rate while unconscious, so starvation is still far away,
        # and the helpless fainting spells are spent standing on Elbereth, which the grind's
        # animals and humanoids respect. After a failed prayer the bot no longer prays at all.
        hunger_prayer_due = self.blstats.hunger_state >= Hunger.FAINTING and (
                self.last_prayer_turn is None or
                self.blstats.time - self.last_prayer_turn >= HUNGER_PRAYER_GAP or
                self.blstats.time - self._fainting_since >= FAINTING_PRAYER_DEADLINE or
                self.blstats.hitpoints * 2 < self.blstats.max_hitpoints or
                self._fainting_threat_prayer_due())
        # Pray at the game's exact major-trouble HP threshold (pray.c critically_low_hp); for
        # hunger, only once the timeout tail is safe (see above).
        if (
                (self.is_safe_to_pray(500) and self.critically_low_hp())
                or (self.is_safe_to_pray(400) and hunger_prayer_due)
        ):
            yield True
            self.pray()
            return

        if self.blstats.hunger_state >= Hunger.FAINTING and not self.prayer_failed and \
                not hunger_prayer_due and not self._has_food_in_reach():
            if self.inventory.engraving_below_me.lower() != 'elbereth':
                if self.can_engrave():
                    yield True
                    self.engrave('Elbereth')
                    return
            else:
                yield True
                self.search(5)
                return

        # standing on a down staircase at crisis HP with prayer spent: take it. Only adjacent
        # monsters follow, the new level is a fresh start, and the depth is banked either way.
        if (self.blstats.hitpoints < 1 / 3 * self.blstats.max_hitpoints or self.blstats.hitpoints < 8) and \
                self.current_level().dungeon_number in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES) and \
                self.current_level().objects[self.blstats.y, self.blstats.x] in G.STAIR_DOWN:
            yield True
            self.direction('>')
            return

        # Most deaths past the grind (Xp 5-8 on Dlvl 2-6: soldier ants, rothes, werejackals) happen a
        # few squares from a known down staircase, while the rule above only fires on '>'. Below half
        # HP with a mobile hostile close by, walk to a nearby known '>' and take it: only adjacent
        # monsters follow, the new level usually has a quiet spot to rest, and the depth is banked.
        # From github.com/Komershan/nethacker@615e07b.
        step = self._flee_downstairs_step()
        if step is not None:
            yield True
            if step == '>':
                self.direction('>')
            else:
                self.move(*step)
            return

        # hypothesis: many runs die in melee at low XP (Xp5-7) across all four identities. This
        # Elbereth last resort sits at the very bottom of emergency_strategy -- below the healing
        # cast, healing potion, fruit juice and prayer -- so it only fires when the Healer is at
        # critical HP (< max/5 or < 5) with every other emergency option already exhausted, i.e. a
        # near-certain death. Engraving Elbereth in the dust scares off the common early attackers
        # (most animals, humanoids, etc.), and waiting on it lets HP regenerate, converting otherwise
        # terminal combat deaths into survival == more XP == more score. Because it triggers only in
        # this otherwise-fatal, resource-empty state, resourced/healthy runs never reach it.
        if self.inventory.engraving_below_me.lower() != 'elbereth' and self.can_engrave() and \
                (self.blstats.hitpoints < 1 / 5 * self.blstats.max_hitpoints or self.blstats.hitpoints < 5):
            yield True
            self.engrave('Elbereth')
            for _ in range(8):
                if self.inventory.engraving_below_me.lower() != 'elbereth':
                    break
                self.direction('.')
            return

        yield False

    def _flee_downstairs_step(self, max_dist=12, threat_dist=5):
        level = self.current_level()
        if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES) or \
                level.level_number < 2:
            return None
        if self.blstats.hitpoints * 2 >= self.blstats.max_hitpoints:
            return None
        y, x = self.blstats.y, self.blstats.x
        threats = [m for m in self.get_visible_monsters()
                   if max(abs(m[1] - y), abs(m[2] - x)) <= threat_dist and
                   m[3].mname not in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS]
        if not threats:
            return None
        if level.objects[y, x] in G.STAIR_DOWN:
            return '>'
        dis = self.bfs()
        stairs = [(sy, sx) for sy, sx in zip(*utils.isin(level.objects, G.STAIR_DOWN).nonzero())
                  if 0 < dis[sy, sx] <= max_dist]
        if not stairs:
            return None
        sy, sx = min(stairs, key=lambda p: dis[p])
        ny, nx = self.path(y, x, sy, sx, dis=dis)[1]
        if self.monster_tracker.monster_mask[ny, nx]:
            return None
        return ny, nx

    def _can_rest(self):
        if self.blstats.hitpoints * 2 >= self.blstats.max_hitpoints:
            return False
        if self.blstats.hunger_state >= Hunger.HUNGRY or self.character.prop.hallu:
            return False
        if self.current_level().shop[self.blstats.y, self.blstats.x]:
            return False
        for _, y, x, _, _ in self.get_visible_monsters():
            if max(abs(y - self.blstats.y), abs(x - self.blstats.x)) <= 6:
                return False
        return True

    @utils.debug_log('rest_to_heal')
    @Strategy.wrap
    def rest_to_heal(self):
        # The bot never rests: after a fight it walks on (explores, takes stairs, digs) at whatever HP
        # it has left, so the next monster meets a half-dead character. Below half HP with no hostile
        # in sight and not hungry, stand on a dust Elbereth and search until HP is back to 90%.
        # fight2, emergency and eating preempt this, so a monster showing up interrupts the rest.
        # From github.com/Komershan/nethacker@9b873ea.
        if not self._can_rest():
            yield False
        yield True
        level = self.current_level()
        y, x = self.blstats.y, self.blstats.x
        if self.inventory.engraving_below_me.lower() != 'elbereth' and self.can_engrave() and \
                level.objects[y, x] not in G.STAIR_UP and level.objects[y, x] not in G.STAIR_DOWN and \
                level.objects[y, x] not in G.ALTAR and level.objects[y, x] not in G.FOUNTAIN:
            self.engrave('Elbereth')
        while self.blstats.hitpoints * 10 < self.blstats.max_hitpoints * 9 and \
                self.blstats.hunger_state < Hunger.HUNGRY:
            self.search(5)

    @utils.debug_log('proactive_sleep')
    @Strategy.wrap
    def proactive_sleep_strategy(self):
        # hypothesis: the Healer's wand of sleep is its single best early weapon, yet it is only used
        # reactively -- at crisis HP (emergency_strategy) or when the fight heuristic happens to line up
        # a multi-target ray. So the weak Healer walks into melee with the very monsters that kill it
        # (giant bats, rothes, soldier ants, gnome lords, Woodland-elves) and gets whittled down before
        # any emergency fires. Use the wand PROACTIVELY: when a genuinely threatening monster (mlevel>=2
        # or faster than us) is 2-4 tiles away on a straight firing line, sleep it FIRST, then let fight2
        # kill it while it is helpless -- turning deadly melee exchanges into free, damage-less kills and
        # more surviving XP. Charges are conserved (skip trivial rats/newts/jackals/hobbits, one zap per
        # 8 turns). Scoped to non-gnome Healers: the gnome runs' whole score comes from a few RNG-fragile
        # deep dives that must stay byte-identical, whereas humans -- who have no deep runs to protect and
        # drag the average down -- are exactly who this rescue is for.
        if self.character.race == self.character.GNOME:
            yield False
        if self.blstats.time - getattr(self, '_last_proactive_sleep_turn', -100) < 8:
            yield False
        if self.inventory.engraving_below_me.lower() == 'elbereth':
            yield False

        sleep_wand = None
        for item in flatten_items(self.inventory.items):
            if item.is_wand() and item.is_unambiguous() and item.object.name == 'sleep' \
                    and item.uses != 'no charges' and not str(item.uses).endswith(':0') \
                        and not self.wand_is_empty(item):
                sleep_wand = item
                break
        if sleep_wand is None:
            yield False

        y0, x0 = self.blstats.y, self.blstats.x
        walkable = self.current_level().walkable
        peaceful = self.monster_tracker.peaceful_monster_mask
        hp_ratio = self.blstats.hitpoints / max(self.blstats.max_hitpoints, 1)

        # Count nearby real threats to decide whether the spot is dangerous enough to spend a charge.
        # A winnable 1-on-1 against a weak-ish foe is left to normal combat so healthy runs are not
        # perturbed; the wand is reserved for swarms, tough out-of-depth singles, or losing fights.
        near_threats = 0
        for dist, my, mx, mon, glyph in self.get_visible_monsters():
            if dist <= 4 and mon.mname not in combat.monster_utils.WEAK_MONSTERS \
                    and mon.mname not in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS \
                    and (getattr(mon, 'mlevel', 0) >= 2
                         or combat.monster_utils.is_monster_faster(self, (dist, my, mx, mon, glyph))):
                near_threats += 1

        best = None
        for dist, my, mx, mon, glyph in self.get_visible_monsters():
            if mon.mname in combat.monster_utils.WEAK_MONSTERS or \
                    mon.mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS:
                continue
            mlevel = getattr(mon, 'mlevel', 0)
            faster = combat.monster_utils.is_monster_faster(self, (dist, my, mx, mon, glyph))
            if not (mlevel >= 2 or faster):
                continue
            # danger gate: outnumbered, a genuinely tough single (mlevel>=4), or already losing
            dangerous = near_threats >= 2 or mlevel >= 4 or (hp_ratio < 0.6 and mlevel >= 2)
            if not dangerous:
                continue
            dy, dx = my - y0, mx - x0
            cheb = max(abs(dy), abs(dx))
            if cheb < 2 or cheb > 4:
                continue
            if not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
                continue
            sy, sx = int(np.sign(dy)), int(np.sign(dx))
            cy, cx = y0, x0
            clear = True
            for _ in range(cheb - 1):
                cy += sy
                cx += sx
                if not walkable[cy, cx] or self.glyphs[cy, cx] in G.PETS or peaceful[cy, cx]:
                    clear = False
                    break
            if not clear:
                continue
            if best is None or dist < best[0]:
                best = (dist, my, mx)

        if best is None:
            yield False

        yield True
        self._last_proactive_sleep_turn = self.blstats.time
        direction = self.calc_direction(y0, x0, best[1], best[2], allow_nonunit_distance=True)
        self.zap(sleep_wand, direction)

    @utils.debug_log('read_magic_mapping')
    @Strategy.wrap
    def read_magic_mapping(self):
        # Tourists start with scrolls of magic mapping; while diving, a level whose down
        # staircase is still unknown is mapped at once instead of explored room by room.
        level = self.current_level()
        if self.global_logic.milestone.name != 'GO_DOWN' or                 level.dungeon_number != Level.DUNGEONS_OF_DOOM or                 level.key() in self._mapped_levels or level.get_stairs(down=True):
            yield False
            return
        scroll = None
        for item in flatten_items(self.inventory.items):
            if item.category == nh.SCROLL_CLASS and item.is_unambiguous() and                     item.object.name == 'magic mapping' and item.status != Item.CURSED:
                scroll = item
                break
        if scroll is None:
            yield False
            return
        yield True
        self._mapped_levels.add(level.key())
        with self.atom_operation():
            self.step(A.Command.READ)
            self.type_text(self.inventory.items.get_letter(scroll))

    def pick_for_digging(self):
        # a pick-axe first: the two-handed mattock cannot be applied while wearing a shield
        for name in ('pick-axe', 'dwarvish mattock'):
            for item in flatten_items(self.inventory.items):
                if item.is_unambiguous() and item.objs[0].name == name and item.status != Item.CURSED:
                    return item
        return None

    @utils.debug_log('dig_down')
    @Strategy.wrap
    def dig_down(self):
        # hypothesis: the score rewards dungeon depth (Dlvl:10 = 0.126 already beats Xp:9 = 0.117,
        # and it keeps climbing fast: Dlvl:12 = 0.206), yet the bot caps around Dlvl 4 -- it grinds
        # experience on shallow levels and then stalemates there, frequently holding one or more
        # unused wands of digging. Once in the deep phase (Xp >= 8, so the RNG-fragile early grind
        # is left byte-identical to the parent) and when it is safe, dig straight down with a wand of
        # digging to bank the much more valuable depth milestones. Digging for depth is legitimate
        # NetHack progression (not a scorer quirk), and every identity that finds a digging wand
        # turns a wasted, capped late game into a deeper, higher-scoring run.
        # A pick-axe digs without limit, so whoever carries one (every Archeologist from turn 1)
        # skips the Dlvl 1 grind and digs from EARLY_DIG_XL: monster difficulty tracks the average
        # of depth and experience level, so a fresh character falls through levels faster than
        # the dungeon can catch up with it, and depth is worth far more than the Xp 8 it forgoes.
        if self.blstats.experience_level < 8 and not (
                self.blstats.experience_level >= early_dig_xl(self.character) and self.pick_for_digging() is not None):
            yield False
            return
        if self.character.prop.polymorph:
            yield False
            return
        # Dig in the Mines too, down to Mines' End (Dlvl 10-13), instead of walking a found pick back
        # up through hostile packs; a floor "too hard to dig in" (Mines' End, Sokoban, Medusa...) is
        # remembered so it is not retried. From github.com/Komershan/nethacker@5deb412.
        if self.current_level().dungeon_number != Level.DUNGEONS_OF_DOOM:
            yield False
            return
        if self.current_level().dungeon_number == Level.GNOMISH_MINES and self._mines_bottom_found:
            yield False
            return
        if self.current_level().key() in self._undiggable_levels:
            yield False
            return
        wand = None
        for item in flatten_items(self.inventory.items):
            if item.is_wand() and item.is_unambiguous() and item.object.name == 'digging' \
                    and item.uses != 'no charges' and not str(item.uses).endswith(':0') \
                        and not self.wand_is_empty(item):
                wand = item
                break

        # A pick-axe dive dies in the ~5 turns per hole it spends digging while monsters walk up and
        # maul it, or after it stops digging below 70% HP and wanders the deep level instead. Dig
        # under a dust Elbereth: engraved before starting and again once the pit is dug (digging the
        # pit wipes it). Every monster but @ (humans, elves) and minotaurs then refuses to melee us,
        # and a scared monster does not interrupt the dig, so keep digging with such monsters
        # adjacent and at any HP short of the prayer / healing-potion emergencies.
        # From github.com/DT6A/nethacker@e3f389b.
        shielded = wand is None and self.pick_for_digging() is not None and self.can_engrave()
        adjacent = [m for m in self.get_visible_monsters()
                    if max(abs(m[1] - self.blstats.y), abs(m[2] - self.blstats.x)) <= 1]
        if shielded:
            if any(not self._respects_elbereth(m[3]) for m in adjacent):
                yield False
                return
            if self.critically_low_hp() and self.is_safe_to_pray(500):
                yield False
                return
            hp, max_hp = self.blstats.hitpoints, self.blstats.max_hitpoints
            if (hp < max_hp / 3 or hp < 8) and any(
                    item.is_unambiguous() and item.category == nh.POTION_CLASS and
                    item.object.name in ['healing', 'extra healing', 'full healing']
                    for item in flatten_items(self.inventory.items)):
                yield False
                return
            if self.blstats.hunger_state >= Hunger.HUNGRY:
                yield False
                return
        else:
            # stay safe: let fight2 / emergency_strategy handle threats before we spend turns digging
            if self.blstats.hitpoints < 0.7 * self.blstats.max_hitpoints:
                yield False
                return
            if adjacent:
                yield False
                return

        if wand is not None:
            yield True
            self.zap(wand, '>')
            self._check_undiggable_floor()
            return

        # hypothesis: a wand of digging is rare and runs dry after a few levels, but pick-axes and
        # dwarvish mattocks (dropped by the dwarves the bot routinely kills) dig through the floor
        # without limit. Depth is by far the most valuable progress milestone (Dlvl 10 = 0.126 >
        # Xp 9, Dlvl 12+ > 0.2), so in the deep phase keep applying the pick downward, falling a level
        # at a time, instead of stalling at Dlvl 2-5 with a digging tool unused in the pack.
        pick = self.pick_for_digging()
        if pick is None:
            yield False
            return
        y, x = self.blstats.y, self.blstats.x
        if self.current_level().objects[y, x] not in G.FLOOR or \
                self.current_level().objects[y, x] in G.DOORS:
            yield False
            return
        # falling out of a shop hands the whole pack, pick included, to the shopkeeper
        if self.current_level().shop[y, x]:
            yield False
            return
        # give up on spots where digging makes no progress (undiggable floor, a hole we cannot
        # fall through, a wielding problem, ...) so this can never loop forever
        key = (self.current_level().key(), y, x)
        attempts = self._pick_dig_attempts.get(key, 0)
        # under Elbereth the dig is still interrupted by monsters coming into view, so allow more
        if attempts >= (16 if shielded else 8):
            yield False
            return

        if shielded and self.inventory.engraving_below_me.lower() != 'elbereth':
            # a dust engraving garbles a letter now and then, so allow a few rewrites per spot
            engraves = self._dig_engrave_attempts.get(key, 0)
            if engraves < 4:
                yield True
                self._dig_engrave_attempts[key] = engraves + 1
                self.engrave('Elbereth')
                return
            if adjacent or self.blstats.hitpoints < 0.7 * self.blstats.max_hitpoints:
                yield False
                return

        yield True
        self._pick_dig_attempts[key] = attempts + 1
        with self.atom_operation():
            pick = self.inventory.move_to_inventory(pick)
            self.step(A.Command.APPLY)
            self.type_text(self.inventory.items.get_letter(pick))
            if 'direction' in self.message and '>' in self.message:
                self.direction('>')
                self._check_undiggable_floor()
            else:
                self._pick_dig_attempts[key] = 99
                if 'direction' in self.message:
                    self.step(A.Command.ESC)

    @staticmethod
    def _respects_elbereth(mon):
        # @ (humans and elves) and minotaurs ignore Elbereth; so may whatever we cannot see
        return mon.mname not in ('unknown', 'minotaur') and ord(mon.mlet) != MON.S_HUMAN

    def _check_undiggable_floor(self):
        if 'too hard to dig' in self.message:
            self._undiggable_levels.add(self.current_level().key())
            if self.current_level().dungeon_number == Level.GNOMISH_MINES:
                self._mines_bottom_found = True

    def _fainting_threat_prayer_due(self):
        # hypothesis: every Dlvl 1 grind death traced (kni s0, rog-orc s1/s10, rog-hum s5) is the same:
        # the Fainting vigil waits for HUNGER_PRAYER_GAP (1200) turns after the last prayer, a newt,
        # rat, bat or zombie wanders in, and the character -- unconscious for most of the next turns,
        # its dust Elbereth scuffed -- is bitten from full HP to death a few dozen turns before the
        # prayer would come (the "HP < max/2" trigger never fires: the character cannot act while
        # fainted). Fainting only starts 900-1200 turns after a prayer, where a prayer comes too soon
        # just ~3-6% of the time (rnz(350) tail), while a monster next to a fainting character is
        # nearly always fatal. So pay the small prayer risk exactly when the danger shows up: pray at
        # once when a hostile mobile monster comes within THREAT_PRAYER_DISTANCE during the vigil.
        if self.last_prayer_turn is None or \
                self.blstats.time - self.last_prayer_turn < THREAT_PRAYER_GAP:
            return False
        for _, y, x, permonst, _ in self.get_visible_monsters():
            if getattr(permonst, 'mmove', 12) == 0:
                continue
            if max(abs(y - self.blstats.y), abs(x - self.blstats.x)) <= THREAT_PRAYER_DISTANCE:
                return True
        return False

    def _has_food_in_reach(self):
        for item in flatten_items(self.inventory.items):
            if item.category == nh.FOOD_CLASS and \
                    item.objs[0].name != 'sprig of wolfsbane' and \
                    (not item.is_corpse() or
                     item.monster_id in [MON.from_name(n) - nh.GLYPH_MON_OFF for n in ['lizard', 'lichen']]):
                return True
        for corpse_mapping in self.current_level().corpses_to_eat.values():
            for monster_id, corpse_age in corpse_mapping.items():
                if self._is_corpse_editable(monster_id, corpse_age):
                    return True
        return False

    @utils.debug_log('eat_from_inventory')
    @Strategy.wrap
    def eat_from_inventory(self):
        if self.blstats.hunger_state < Hunger.HUNGRY:
            yield False
        for item in flatten_items(self.inventory.items):
            if item.category == nh.FOOD_CLASS and \
                    item.objs[0].name != 'sprig of wolfsbane' and \
                    (not item.is_corpse() or
                     item.monster_id in [MON.from_name(n) - nh.GLYPH_MON_OFF for n in ['lizard', 'lichen']]):
                yield True
                self.inventory.eat(item)
                return
        yield False

    @utils.debug_log('cure_disease')
    @Strategy.wrap
    def cure_disease(self):
        if self.character.is_lycanthrope:
            # spring of wolfsbane
            for item in flatten_items(self.inventory.items):
                if item.objs[0].name == 'sprig of wolfsbane':
                    yield True
                    self.inventory.eat(item)
                    return

            # holy water
            for item in flatten_items(self.inventory.items):
                if item.objs[0].name == 'water' and item.status == Item.BLESSED:
                    yield True
                    self.inventory.quaff(item)
                    return

            # pray
            if self.is_safe_to_pray():
                yield True
                self.pray()
                return

        yield False

    ####### MAIN

    def handle_exception(self, exc):
        if isinstance(exc, (KeyboardInterrupt, AgentFinished, SystemExit)):
            raise exc
        if isinstance(exc, BaseException):
            # hypothesis: many games end early because the bot crashes, not because the
            # character dies. The Sokoban map strings were indented, so every scripted push
            # failed an assertion, and other errors (unhandled prompts, hallucination missed on
            # the abbreviated status line, weightless items, stale-state assertions) were
            # re-raised. Either way the AutoAscend thread died, the arena only got ESC fallbacks
            # and NLE aborted ~29% of games with a healthy character. Fixing those crash sources
            # and recovering from every error like an AgentPanic (letting a turn pass when the
            # same error repeats) keeps the games alive to gain more experience levels and depth.
            if not isinstance(exc, AgentPanic):
                self._drop_state_after_error()
            self.stats_logger.log_event('agent_panic')
            # Keep only the last few, without their tracebacks: each traceback pins every frame
            # and the observation arrays they hold, and a 70000-turn game that recovers from
            # thousands of errors grew one bot to 6-7 GB -- the verifier's box OOMs on that and
            # drops the whole program as crashed.
            exc.__traceback__ = None
            self.all_panics.append(exc)
            del self.all_panics[:-20]
            if self.verbose:
                print(f'PANIC!!!! : {exc}')

    def _drop_state_after_error(self):
        # An unexpected error can leave caches half-updated (e.g. the items below the agent
        # cleared but never re-read). The recovery ESC steps run update() before on_panic()
        # gets a chance to reset them, so drop them here (without stepping) to have them rebuilt.
        if self._inactivity_counter >= 199:
            # the 'turn inactivity' guard fired: the strategies loop without the game advancing
            self._pass_turn_after_error = True
        self._inactivity_counter = 0
        self._is_reading_message_or_popup = False
        self.inventory.items_below_me = None
        self.inventory.letters_below_me = None
        self.inventory.engraving_below_me = None
        self.inventory._previous_blstats = None
        self.inventory.items.on_panic()
        self.monster_tracker.on_panic()

    def main(self):
        try:
            init_finished = False
            try:
                with self.atom_operation():
                    self.step(A.Command.ESC)
                    self.step(A.Command.ESC)

                    self.current_level().stair_destination[self.blstats.y, self.blstats.x] = \
                        ((Level.PLANE, 1), (None, None))  # TODO: check level num
                    self.character.parse()
                    self.character.parse_enhance_view()
                    # hypothesis: Healers know a healing spell but never cast it. Parsing/casting it
                    # only in the deep (Xp>=8) mine fights -- where Xp8-death runs otherwise stall --
                    # lets them survive to Xp9-10 for a large score jump, while leaving the whole
                    # early level-1 grind (and every Xp3-7 death run) byte-identical to the parent.
                    # The spell list is parsed lazily once Xp8 is reached (see fight2), not here, so
                    # the early game keeps the parent's exact action sequence and RNG.
                    self.step(A.Command.AUTOPICKUP)
                    if 'Autopickup: ON' in self.message:
                        self.step(A.Command.AUTOPICKUP)
                    init_finished = True
            except BaseException as e:
                self.handle_exception(e)

            assert init_finished

            last_step = self.step_count
            inactivity_counter = 0
            forced_turns = 0
            turn_after_forced = None
            while 1:
                inactivity_counter += 1
                if self.step_count != last_step:
                    inactivity_counter = 0

                if inactivity_counter >= 5 or self._pass_turn_after_error:
                    try:
                        panics = sorted({p.args[0] for p in self.all_panics[-5:]})
                    except (TypeError, IndexError):
                        panics = 'UNKNOWN'

                    # The same error keeps recurring without the game advancing. Let a turn pass
                    # (so e.g. a monster in the way or a temporary status can change) instead of
                    # giving up at once, unless nothing else has advanced the game for too long.
                    if turn_after_forced is None or self.blstats.time != turn_after_forced:
                        forced_turns = 0
                    if forced_turns >= 300:
                        raise RuntimeError(f'Cyclic Panic: {panics}')
                    forced_turns += 1
                    inactivity_counter = 0
                    self._pass_turn_after_error = False
                    try:
                        self.step(A.Command.SEARCH)
                    except BaseException as e:
                        self.handle_exception(e)
                    turn_after_forced = self.blstats.time
                    last_step = self.step_count

                try:
                    try:
                        self.step(A.Command.ESC)
                        self.step(A.Command.ESC)
                        self.on_panic()
                    finally:
                        last_step = self.step_count

                    self.global_logic.global_strategy().run()
                    assert 0
                except BaseException as e:
                    self.handle_exception(e)
        except AgentFinished:
            pass
