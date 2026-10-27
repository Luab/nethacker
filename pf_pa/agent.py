import contextlib
import re
import sys
import traceback
from collections import namedtuple, Counter, defaultdict
from functools import partial

import nle.nethack as nh
import nltk
import numpy as np
from nle.nethack import actions as A

from . import combat
from . import jf_config, jf_log, jf_scenario
from . import power
from . import utils
from .character import Character
from .exceptions import AgentPanic, AgentFinished, AgentChangeStrategy
from .exploration_logic import ExplorationLogic
from .global_logic import GlobalLogic
from .glyph import MON, C, Hunger, G, SHOP, SS
from .item import Item, flatten_items
from .item.inventory import Inventory
from .level import Level
from .monster_tracker import MonsterTracker, disappearance_mask
from .nhmodel.prayer import PrayerModel, rnz_cdf
from .recovery import Recovery, SearchResult, SearchStalled
from .spell_healing import SpellHealing
from .stats_logger import StatsLogger
from .strategy import Strategy

BLStats = namedtuple('BLStats',
                     'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask alignment')


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
        self._corpse_debug_pos = None
        self._faint_msg_turn = None    # FAINT_MEASURE_FIX: turn of the screen that first showed a faint
        self._paralysis_end_turn = -10 ** 9   # STARVE_UNMEASURED_GAP: turn of the last 'You can move again'
        self._attack_ctx = None       # (turn, melee target, throw direction, origin, glyphs before) CORPSE_TRACK

        self.inventory = Inventory(self)
        self.character = Character(self)
        self.exploration = ExplorationLogic(self)
        self.global_logic = GlobalLogic(self)
        self.monster_tracker = MonsterTracker(self)

        self.last_bfs_dis = None
        self.last_bfs_step = None
        self.last_prayer_turn = None
        self.prayer_hold_until = -1
        self._fainting_since = None   # turn Fainting was first seen (jf_config.STARVE_CLOCK)
        self._weak_since = None       # turn Weak was first seen (jf_config.THREAT_PRAYER_GAP)
        self._faint_measure = None    # (turn, uhunger estimate) from the last faint's length
        self._pray_reason = None      # which rule asked for the next prayer (logged)
        self._no_kick_until = -1      # wounded legs: no kicking until this turn
        self._last_resort_stairs_turn = -10 ** 9
        self._last_resort_dig_turn = -1   # LAST_RESORT_DIG: at most one zap per turn
        self._last_resort_zapped = set()  # LR_WAND_ONCE: glyphs of unknown wands the last resort already zapped
        self.prayer_failed = False
        self._monk_meat_meals = 0
        self._previous_glyphs = None
        self._last_turn = -1
        self._inactivity_counter = 0
        self._pass_turn_after_error = False
        self._update_failures = {}
        self._text_prompt_escapes = 0
        self._teleport_prompt_escapes = 0
        self._suppressed_updates = {}
        self._last_panic_signature = None
        self._panic_repeats = 0
        self._random_walk_steps = 0
        self.resumed_game = False  # set by the driver when a fresh agent takes over a running game
        self._petrifying_bodies = frozenset(MON.body_from_name(n) for n in ('cockatrice', 'chickatrice'))
        self._petrifying_bodies_mons = frozenset(MON.from_name(n) for n in ('cockatrice', 'chickatrice'))
        self._wet_glyphs = frozenset({SS.S_pool, SS.S_water, SS.S_lava})
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
        self._hb_actions = Counter()   # actions and go_to targets since the last STATUS heartbeat
        self._hb_gotos = Counter()
        self._fight_stall = None       # (turn a contactless fight began, XP then, last fight2 turn) FIGHT_STALL_TURNS
        self._fight_ignored = {}       # (level key, glyph) -> [y, x, until turn, anywhere]: monsters fight2 let go
        self._fight_moves = []         # (turn, level key, position) of fight2's consecutive moves (FIGHT_STALL_MOVES)

        self.last_cast_fail_turn = defaultdict(lambda: -float('inf'))
        self.spell_healing = SpellHealing()
        self.recovery = Recovery(self)
        self._recovery_response_steps = None

        self.stats_logger = StatsLogger()

        # PRAYER_MODEL (nhmodel/prayer.py): pray.c's own odds replace the fixed prayer gaps where they differ
        try:
            self.prayer_model = PrayerModel(self) if jf_config.PRAYER_MODEL else None
        except Exception:
            self.prayer_model = None

    def log(self, msg):
        if jf_log.enabled():
            bl = getattr(self, 'blstats', None)
            where = f't{bl.time} d{bl.depth} {bl.dungeon_number}:{bl.level_number} xl{bl.experience_level} ' \
                    f'hp{bl.hitpoints}/{bl.max_hitpoints}' if bl is not None else 't?'
            jf_log.log(f'[s{self.step_count} {where}] {msg}')

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
        if 'You can move again' in self.single_message:
            try:
                self._paralysis_end_turn = int(obs['blstats'][nh.NLE_BL_TIME])
            except Exception:
                pass
        # FAINT_MEASURE_FIX: the turn of the screen that first shows a faint (its --More-- comes at the faint's start)
        if 'You faint from lack of food' in self.single_message and 'You regain consciousness' not in self.single_message:
            try:
                self._faint_msg_turn = int(obs['blstats'][nh.NLE_BL_TIME])
            except Exception:
                self._faint_msg_turn = None
        return done

    def step(self, action, additional_action_iterator=None):
        if self._no_step_calls:
            raise ValueError("Shouldn't call step now")

        if self._recovery_response_steps is not None:
            if self._recovery_response_steps <= 0:
                raise SearchStalled("recovery search prompt did not finish")
            self._recovery_response_steps -= 1

        if isinstance(action, str):
            assert len(action) == 1
            action = A.ACTIONS[A.ACTIONS.index(ord(action))]
        observation, reward, done, info = self.env.step(action)
        observation = {k: v.copy() for k, v in observation.items()}
        self.step_count += 1
        self._hb_actions[getattr(action, 'name', str(action))] += 1
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
        # Preserve the direct response before recursive prompt/housekeeping updates.
        return observation

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

        # an unanswered naming prompt ('Your orange potion boils and explodes!  Call an orange potion:')
        # isn't always flagged as text entry: the bot's keys went into it ('lll...', ':/M', '#te') until a
        # potion type was called '#te', the item parser choked and the game stalled to the timeout
        if re.search(r'Call an? [^:]*:\s*$', self.single_message) and self._text_prompt_escapes < 5:
            # a scroll that turned to dust on pickup names this game's scare monster label (power.py)
            power.note_dust_prompt(self)
            self._text_prompt_escapes += 1
            self.step(A.Command.ESC)
            return

        if observation['misc'][1] and self._text_prompt_escapes < 5:  # entering text
            if "You may wish for an object." in self.message:
                # TODO: assume wished item as blessed
                # SPARE_WISHES / castle_logic (agent.wish_purpose): GDSM first, then the Castle passage
                # castle_logic may set agent.wish_text (an explicit wish) and/or agent.wish_purpose
                purpose = getattr(self, 'wish_purpose', None)
                text = getattr(self, 'wish_text', None) or \
                    (power.wish_text(self, purpose) if (jf_config.SPARE_WISHES or purpose) else power.WISH_GDSM)
                self.log(f'POWER wishing for {text!r}')
                power.note_wish(self, text)
                self.step(text[0], iter(text[1:] + '\r'))
                return
            else:
                # a text-entry flag that survives ESC after ESC recursed update->step->update until
                # RecursionError (a b4 dive); after 5 escapes treat the flag as stale
                self._text_prompt_escapes += 1
                self.step(A.Command.ESC)
                return
        if not observation['misc'][1]:
            self._text_prompt_escapes = 0

        if 'Where do you want to be teleported?' in self.single_message and self._teleport_prompt_escapes < 3:
            # TODO: teleport control. Checking the accumulated self.message (it keeps the prompt text
            # after ESC answered it) recursed update->step->update into a RecursionError mid-dive.
            self._teleport_prompt_escapes += 1
            self.step(A.Command.ESC)
            return
        self._teleport_prompt_escapes = 0

        if b'[yn]' in bytes(observation['tty_chars'].reshape(-1)):
            # a foocubus: don't let it put a (maybe cursed: levitation strands a dive) ring on us
            if 'Would you wear it for me?' in self.single_message:
                self.type_text('n')
                return
            # Dlvl 1's up stairs leave the dungeon and end the game (a jf11 agent that took over mid-game
            # hadn't recorded them, explored them as unknown stairs and said 'y': score 0.018 at T875)
            if 'Still climb?' in self.single_message:
                self.type_text('n')
                if hasattr(self, 'blstats'):
                    self.current_level().stair_destination[self.blstats.y, self.blstats.x] = \
                        ((Level.PLANE, 1), (None, None))
                return
            # an opened tin: 'It smells like dwarves. Eat it?' was answered 'y' -- 'You consume pureed
            # dwarf. You cannibal! You will regret this!' (Luck -2..-5, and prayers fail with Luck < 0)
            if 'Eat it?' in self.single_message and self._bad_tin(self.message):
                self.type_text('n')
                return
            self.type_text('y')
            return

        self._is_reading_message_or_popup = False
        self._message_history.append(self.message)

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
        self.recovery.observe(self.blstats)
        self.glyphs = self.last_observation['glyphs']

        if self._prayer_model_active():
            try:
                self.prayer_model.observe()
            except Exception:
                self._prayer_model_error()

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

        if getattr(self, '_wish_pending', None) is not None:
            power.learn_wished(self)   # WISH_LEARN: 'p - a granite ring.' names the wished ring

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
                for name, func in [('character', self.character.update), ('inventory', self.inventory.update),
                                   ('monsters', self.monster_tracker.update),
                                   ('terrain', partial(self.check_terrain, force=False)),
                                   ('level', self.update_level), ('global', self.global_logic.update)]:
                    if self._suppressed_updates.get(name, -1) >= self.step_count:
                        if name == 'inventory':
                            self.inventory.set_unknown_below_me()
                        continue
                    try:
                        func()
                        self._update_failures[name] = 0
                    except (AgentFinished, KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as e:
                        # an updater that fails on every step (e.g. an unparseable item under us)
                        # would freeze the agent forever: after 3 failures in a row, skip it a while
                        self._update_failures[name] = self._update_failures.get(name, 0) + 1
                        if self._update_failures[name] >= 3:
                            self._update_failures[name] = 0
                            self._suppressed_updates[name] = self.step_count + 50
                            self.log(f'suppressing updater {name} for 50 steps: {type(e).__name__} {str(e)[:150]}')
                        raise
                    finally:
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

    _DIG_TOOL_REFUSED = re.compile(r'leave (?:your|the) (?:pick-axe|mattock|digging tool)s? outside|'
                                   r'to let you in with your (?:pick-axe|mattock|digging tool)')

    def _carries_digging_tool(self):
        return any(o.name in ('pick-axe', 'dwarvish mattock')
                   for item in flatten_items(self.inventory.items) for o in item.objs)

    def _update_level_shops(self):
        level = self.current_level()

        if self._DIG_TOOL_REFUSED.search(self.message):
            # the refusal comes as we step into the doorway, the shopkeeper standing just inside it
            y0, x0 = self.blstats.y, self.blstats.x
            inside = [(y, x) for y, x in self.neighbors(y0, x0, shuffle=False)
                      if self.glyphs[y, x] in G.SHOPKEEPER]
            if inside:
                sy, sx = inside[0]
                level.refused_doors[(y0, x0)] = (sy - y0, sx - x0)
                level.dig_tool_refused = self.blstats.time

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
                recorded = []
                for y, x in zip(*utils.isin(mons, [glyph]).nonzero()):
                    # TODO: it works because level.items is updated in `inventory.check_items`
                    if all(map(lambda item: item.is_corpse() and item.monster_id != monster_id, level.items[y, x])):
                        level.corpses_to_eat[y, x][monster_id] = self.blstats.time
                        recorded.append((int(y), int(x)))
                # tour only: the dive's eating is dive-safety's (the first food arm's dives did worse)
                if jf_config.CORPSE_TRACK and not self.global_logic.dive.diving:
                    recorded += self._track_kill_positions(mname, level, recorded)
                if jf_config.CORPSE_DEBUG:
                    was = [(int(y), int(x)) for y, x in zip(*utils.isin(self._previous_glyphs, [glyph]).nonzero())]
                    body = [(int(y), int(x)) for y, x in zip(*utils.isin(self.glyphs, [corpse_glyph]).nonzero())]
                    self.log(f'CORPSE kill {mname} recorded={recorded} was_at={was} bodies_now={body} '
                             f'me={(self.blstats.y, self.blstats.x)}')

        old_possible_corpses = level.corpses_to_eat[self.blstats.y, self.blstats.x].copy()
        del level.corpses_to_eat[self.blstats.y, self.blstats.x]

        corpses = Counter((item for item in self.inventory.items_below_me if item.is_corpse()))
        for item, count in corpses.items():
            if count != 1:
                continue

            level.corpses_to_eat[self.blstats.y, self.blstats.x][item.monster_id] = \
                old_possible_corpses[item.monster_id]
        if jf_config.CORPSE_DEBUG and corpses and self._corpse_debug_pos != (self.blstats.y, self.blstats.x):
            self._corpse_debug_pos = (self.blstats.y, self.blstats.x)
            self.log(f'CORPSE below {[(MON.permonst(it.monster_id + nh.GLYPH_MON_OFF).mname, n, self.blstats.time - old_possible_corpses[it.monster_id]) for it, n in corpses.items()]} '
                     f'hunger={self.blstats.hunger_state}')
        elif not corpses:
            self._corpse_debug_pos = None

    _DIRS = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1),
             'ne': (-1, 1), 'nw': (-1, -1), 'se': (1, 1), 'sw': (1, -1)}
    CORPSE_ROT_AGE = 250   # corpses rot away by then (mkobj.c start_corpse_timeout): older records are stale

    def _track_kill_positions(self, mname, level, recorded_now):
        """CORPSE_TRACK: kill squares the glyph-disappearance test above misses. In two jf14 grinds 27% of the
        kills went unrecorded (thrown-dagger kills parsed a few observations later, after the bot's bag check;
        pack monsters with a same-kind neighbour; a remembered non-corpse item on the square), and a corpse
        of unknown age is never eaten: 47 of the 125 edible corpses we stood on had no age, and the pet ate
        most of them. Sources here: our last attack's target square / the first square of that kind along
        the throw, and a square whose victim glyph turned into its corpse. An existing record younger than
        CORPSE_ROT_AGE is kept, so a corpse never looks younger than it may be."""
        glyph = MON.from_name(mname)
        monster_id = glyph - nh.GLYPH_MON_OFF
        corpse_glyph = MON.body_from_name(mname)
        cands = []
        ctx = self._attack_ctx
        if ctx is not None and self.blstats.time - ctx[0] <= 2 and \
                re.search(r'You (?:kill|destroy) ', self.message) is not None:
            _, target, direction, origin, before = ctx
            if target is not None:
                if before[target] == glyph:
                    cands.append((int(target[0]), int(target[1])))
            elif direction in self._DIRS:
                dy, dx = self._DIRS[direction]
                y, x = origin
                for _ in range(13):
                    y, x = y + dy, x + dx
                    if not (0 <= y < before.shape[0] and 0 <= x < before.shape[1]):
                        break
                    if before[y, x] == glyph:
                        cands.append((int(y), int(x)))
                        break
        if self._previous_glyphs is not None:
            for y, x in zip(*((self._previous_glyphs == glyph) & (self.glyphs == corpse_glyph)).nonzero()):
                cands.append((int(y), int(x)))
        added = []
        for y, x in cands:
            if (y, x) in recorded_now or (y, x) in added:
                continue
            if any(item.is_corpse() and item.monster_id == monster_id for item in level.items[y, x]):
                continue
            old = level.corpses_to_eat.get((y, x))
            if old is not None and monster_id in old and self.blstats.time - old[monster_id] < self.CORPSE_ROT_AGE:
                continue
            level.corpses_to_eat[y, x][monster_id] = self.blstats.time
            added.append((y, x))
        return added

    _PIT_IN = ('You are still in a pit', 'You dig a pit in the', 'You fall into a pit', 'You land on a set of sharp iron')

    def _update_pit_state(self):
        """PIT_AWARE_FIGHT: where we last were in a pit (trap.c climb_pit: only move attempts get us out)."""
        here = (*self.current_level().key(), self.blstats.y, self.blstats.x)
        msg = self.message
        if 'You crawl to the edge of the pit' in msg:
            self._in_pit_at = None
        elif any(s in msg for s in self._PIT_IN):
            self._in_pit_at = here
        elif getattr(self, '_in_pit_at', None) is not None and self._in_pit_at != here:
            self._in_pit_at = None

    def in_pit(self):
        return getattr(self, '_in_pit_at', None) == (*self.current_level().key(), self.blstats.y, self.blstats.x)

    def _update_beartrap_state(self):
        """BEARTRAP_ESCAPE: where we are held by a bear trap (trap.c: utrap = rn1(4, 4))."""
        here = (*self.current_level().key(), self.blstats.y, self.blstats.x)
        msg = self.message
        if 'You finally wriggle free' in msg or 'You escape a bear trap' in msg:
            self._beartrap_at = None
        elif 'A bear trap closes on your foot' in msg or 'You are caught in a bear trap' in msg:
            self._beartrap_at = here
        elif getattr(self, '_beartrap_at', None) is not None and self._beartrap_at != here:
            self._beartrap_at = None

    @utils.debug_log('escape_bear_trap')
    @Strategy.wrap
    def escape_bear_trap(self):
        """BEARTRAP_ESCAPE: hack.c trapmove() frees us from a bear trap one step per move attempt -- every diagonal
        attempt, but only 1 in 5 orthogonal ones ('[why does diagonal movement give quickest escape?]'). The
        strategies' go_to kept trying the same orthogonal step: 747 failed attempts, 731 trapped turns in 32
        episodes over 90 base2 games (43 turns once), each a panic; 25 in a row made the panic-loop breaker forbid
        the target square for good (base2-jf25 s11 was then boxed in for 5000+ turns). Try diagonals instead:
        4-7 turns. Below fight2, so an adjacent attacker is fought from the trap first."""
        here = (*self.current_level().key(), self.blstats.y, self.blstats.x)
        # (the dive only: in the spawn-limited grind trapped turns cost little, and the b3 guard's jf14 s0 reshuffled
        # from a Dlvl-1 bear trap at T9)
        if not jf_config.BEARTRAP_ESCAPE or not self.global_logic.dive.diving or \
                getattr(self, '_beartrap_at', None) != here:
            yield False
        level = self.current_level()
        y0, x0 = self.blstats.y, self.blstats.x
        options = [(y, x) for y, x in self.neighbors(y0, x0, shuffle=False)
                   if y != y0 and x != x0 and level.walkable[y, x] and not self.monster_tracker.monster_mask[y, x]
                   and not utils.isin(level.objects[y:y + 1, x:x + 1], G.TRAPS).any()
                   and not level.intact_doors[y0, x0] and level.objects[y, x] not in G.DOORS]
        if not options:
            yield False
        yield True
        y, x = options[0]
        if getattr(self, '_beartrap_logged', None) != here:
            self._beartrap_logged = here
            self.log(f'BEARTRAP escape: diagonal attempts from {here[2:]}')
        with self.atom_operation():
            self.direction(self.calc_direction(y0, x0, y, x))

    def _update_level_sessile(self, level):
        """SESSILE_MEMORY: remember where the monsters fight2 never melees sit (molds, jellies, floating eyes, gas
        spores: ONLY_RANGED_SLOW_MONSTERS). Out of sight, such a square shows the item under the monster, so
        check_items walked back to 'check' it, found the mold in the way ('Monster on a next tile'), panicked and
        came back again: a red mold on an item pile cost jf26 s0 219 panics over 3700 grind turns, jf16 s4 247 over
        3700; a floating eye blocking a Dlvl 6 corridor cost jf26 s1 235 over 3900 dive turns. The BFS keeps such
        a square closed until we stand next to it and see it empty."""
        mons = utils.isin(self.glyphs, G.MONS) & ~self.monster_tracker.peaceful_monster_mask
        t = self.blstats.time
        if not self.character.prop.hallu:
            for y, x in zip(*mons.nonzero()):
                mon = MON.permonst(self.glyphs[y, x])
                if mon.mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS:
                    # molds and jellies never move; a floating eye or gas spore drifts (forget it sooner)
                    if (int(y), int(x)) not in level.sessile:
                        self.log(f'SESSILE {mon.mname} at {(int(y), int(x))} remembered')
                    level.sessile[int(y), int(x)] = (t, 1000 if mon.mmove == 0 else 150)
        if level.sessile and not self.character.prop.blind:
            y0, x0 = self.blstats.y, self.blstats.x
            for (y, x), (seen, keep) in list(level.sessile.items()):
                # gone once seen empty from next to it (a monster there would show), or long unseen
                if (max(abs(y - y0), abs(x - x0)) <= 1 and not mons[y, x]) or t - seen > keep:
                    del level.sessile[(y, x)]

    def update_level(self):
        if utils.isin(self.glyphs, G.SWALLOW).any():
            return

        # hypothesis: stepping onto a cockatrice-family corpse to fetch thrown daggers (or feeling it
        # while blind) petrified an XL9 elite game; astra: never touch the corpse. Without gloves,
        # squares showing such a corpse are off-limits for pathing and item gathering.
        bodies = utils.isin(self.glyphs, self._petrifying_bodies)
        if bodies.any():
            self.current_level().petrify_until[bodies] = self.blstats.time + 300  # corpses rot in ~250

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

        # water and lava are never walkable once seen (a dive on Medusa's level kept walking into the
        # water: a monster or item glyph first shown there had made the square 'walkable'; a jf10 grind
        # fell into a Dlvl 1 pool 30 times in 500 turns, diluting its potions and fading its scrolls)
        mask = utils.isin(self.glyphs, self._wet_glyphs)
        if mask.any():
            level.seen[mask] = True
            level.objects[mask] = self.glyphs[mask]
            level.walkable[mask] = False

        self._update_level_items()
        self._update_level_shops()
        self._update_level_corpses()
        # (the dive only: in the spawn-limited grind the loop costs steps, not score, and a change there reshuffles
        # every game)
        if jf_config.SESSILE_MEMORY and self.global_logic.dive.diving:
            self._update_level_sessile(level)
        if jf_config.PIT_AWARE_FIGHT:
            self._update_pit_state()
        if jf_config.BEARTRAP_ESCAPE:
            self._update_beartrap_state()

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
            if 'seem to be too busy' in self.single_message:  # held, or a welded two-hander
                return self.single_message
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

    def is_safe_to_pray(self, limit=500, certain_death=False):
        # pray.c: 'Since you are in Gehennom, Tyr can't help you' -- nothing is fixed, and unless the alignment
        # record is high the god gets angry (angrygods) -- so no prayer at all there, not even for certain death
        if jf_config.GEHENNOM_DIVE and self.current_level().dungeon_number == 1:
            return False
        if getattr(self, '_sim_no_prayers', False):   # dev only: SIM_RESCUE_PRAYER
            return False
        # certain_death (stoning, sliming, ...): a prayer that may fail beats dying, so the holds below
        # don't apply (an s10 dive was petrified during its dwarf-hunt hold; half the time the kill
        # hadn't cost any Luck and the prayer would have worked)
        # the dive's dwarf hunt: a peaceful kill may cost Luck -1, and prayers fail while Luck < 0
        if not certain_death and self.blstats.time < self.prayer_hold_until:
            return False
        # after a failed prayer the god stays angry (pray.c: a too-soon prayer sets ugangr, Luck -3):
        # 45 of 46 prayers made within 500 turns of a failure failed again, some summoning a minion
        # ('Thou durst call upon me? Then die, mortal!'); half of those 2000+ turns later worked
        model_gate = self._prayer_model_gate(certain_death)
        if model_gate is False:
            return False
        if not certain_death and self.prayer_failed and self.last_prayer_turn is not None and \
                self.blstats.time - self.last_prayer_turn < self.PRAYER_FAILURE_WAIT and model_gate is None:
            return False
        return (
                (self.last_prayer_turn is None and self.blstats.time > (100 if jf_config.EXACT_PRAYER else 300)) or
                (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > limit)
        )

    # angrygods() messages -- after one of them the god stays angry, so waiting for a safe prayer is pointless
    PRAYER_FAILURE_MESSAGES = ('is displeased', 'is bummed', 'Thou hast angered me', 'Thou must relearn thy lessons',
                               'Thou art arrogant', 'Thou hast strayed', 'Thou durst')

    # prayer timeout is rnz(350) after a successful prayer and hunger is fixed only if it is below 200,
    # so a hunger prayer fails in ~7% of cases after 900 turns but only in ~2% after 1200 turns
    def _tour_gaps_by_xl(self):
        """TOUR_GAPS_BY_XL: [[min_xl, weak_gap, faint_gap], ...] -- the entry with the highest min_xl <= XL (the
        late grind is food-poor: at XL 7 STARVE_DEADLINE fired at gaps 1142-1352 in the 1500/1400 arm)."""
        best = None
        for entry in jf_config.TOUR_GAPS_BY_XL or ():
            if self.blstats.experience_level >= entry[0] and (best is None or entry[0] >= best[0]):
                best = entry
        return best

    @property
    def SAFE_HUNGER_PRAYER_GAP(self):
        # TOUR_WEAK_PRAYER_GAP: a longer Weak gap in the tour only (the dive has no faint guard);
        # DIVE_WEAK_PRAYER_GAP: a shorter one deep in the dive (castle: a faint beside a troll is worse)
        if not self.global_logic.dive.diving:
            by_xl = self._tour_gaps_by_xl()
            if by_xl is not None:
                return by_xl[1]
            if jf_config.TOUR_WEAK_PRAYER_GAP:
                return jf_config.TOUR_WEAK_PRAYER_GAP
        elif self._dive_tool_hunger_gap():
            return self._dive_tool_hunger_gap()
        elif jf_config.DIVE_WEAK_PRAYER_GAP and self.blstats.depth >= jf_config.DIVE_GAP_MIN_DEPTH:
            return jf_config.DIVE_WEAK_PRAYER_GAP
        return jf_config.WEAK_PRAYER_GAP

    def _dive_tool_hunger_gap(self):
        """DIVE_TOOL_HUNGER_GAP: the Weak and Fainting prayer gap while digging down (a digging tool carried).
        DIVE_PRAYER_GAP starts the dive 800 turns after the last prayer -- usually a hunger prayer that reset
        nutrition to 900 -- so a food-poor dive is Hungry within ~100 turns and Weak/Fainting at gaps 850-1000,
        where the dive's 1200/1100 rules make it faint for another 100-250 turns among Dlvl 10-20's monsters
        (base5+base5arm: 9 of 73 tool dives Weak before gap 1100, 3 of them killed fainted at Dlvl 10-17).
        A prayer then fails 5.5-7.8% of the time (rnz(350)), and the emergency HP prayer it uses up rarely buys
        depth: 1 of 23 HP prayers in those dives led deeper; the rest died on the same level. Stairs dives
        (no tool) last thousands of turns and keep the long gaps."""
        dive = self.global_logic.dive
        if not jf_config.DIVE_TOOL_HUNGER_GAP or not dive.diving:
            return 0
        if dive.digging_tool() is None:
            return 0
        # the dive's first hunger cycle only (no prayer since the dive began): later cycles -- long camps at the
        # castle pray every ~900-1100 turns for thousands of turns -- keep the long gaps and their lower risk
        start = dive.dive_start_turn
        if self.last_prayer_turn is not None and start is not None and self.last_prayer_turn > start:
            return 0
        return jf_config.DIVE_TOOL_HUNGER_GAP

    def _faint_prayer_gap(self):
        if not self.global_logic.dive.diving:
            by_xl = self._tour_gaps_by_xl()
            if by_xl is not None:
                return by_xl[2]
            if jf_config.TOUR_FAINT_PRAYER_GAP:
                return jf_config.TOUR_FAINT_PRAYER_GAP
        elif self._dive_tool_hunger_gap():
            return self._dive_tool_hunger_gap()
        elif jf_config.DIVE_FAINT_PRAYER_GAP and self.blstats.depth >= jf_config.DIVE_GAP_MIN_DEPTH:
            return jf_config.DIVE_FAINT_PRAYER_GAP
        return jf_config.FAINT_PRAYER_GAP
    PRAYER_FAILURE_WAIT = 2000
    PRAYER_SUCCESS_MESSAGES = ('is well-pleased', 'is pleased.', 'is satisfied', 'hopeful feeling',
                               'You feel much better', 'stomach feels content')

    def _hunger_threat(self):
        """A hostile close enough to reach us during a faint (THREAT_PRAYER_GAP): within THREAT_RADIUS, able to
        walk up to us (not the stationary / passive-only ones: molds, lichens, floating eyes), and serious:
        difficulty >= THREAT_MIN_DIFFICULTY (hill orc, rothe, giant ant, werejackal: 4), two or more of them,
        one that ignores Elbereth, or our HP below half. A lone iguana (3 tries) at a 1045 gap made a public-7
        prayer fail that the base game made safely at 1101 -- a faint next to one costs little."""
        bl = self.blstats
        near = []
        near_dist = []
        for _, y, x, mon, _ in self.get_visible_monsters():
            d = max(abs(int(y) - bl.y), abs(int(x) - bl.x))
            if d > jf_config.THREAT_RADIUS:
                continue
            if mon.mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS or getattr(mon, 'mmove', 12) <= 3:
                continue
            near.append(mon)
            near_dist.append((mon, d))
        if not near:
            return None
        dive = self.global_logic.dive
        # THREAT_DIFF_GAP: the difficulty trigger only from its own (longer) gap -- a rothe circled a Fainting XL5
        # on Dlvl 3 for 90 turns and hit it during a faint once the dust Elbereth had worn (t31/t32-jf16 s14, gap
        # 1607); at 1300+ a prayer fails ~2%, below that the trigger cost more than it saved (gaps 990-1045)
        diff_ok = not jf_config.THREAT_DIFF_GAP or self.is_safe_to_pray(jf_config.THREAT_DIFF_GAP)
        for mon in near:
            if (diff_ok and getattr(mon, 'difficulty', 0) >= jf_config.THREAT_MIN_DIFFICULTY) or \
                    dive._ignores_elbereth(mon):
                return mon.mname
        # THREAT_NO_ELBERETH: Fainting with no intact Elbereth under us (the guard's dust Elbereth wears off --
        # random wipes ~1/85 per turn over a 600-turn hold, 28% typos per write) and a hostile close: g17h2-jf14
        # s13 (sewer rats on 'F|bereth', HP 31->0) and g15h2-jf14 s13 (a rothe, no engraving left) died fainted
        if jf_config.THREAT_NO_ELBERETH and bl.hunger_state >= Hunger.FAINTING and \
                (self.inventory.engraving_below_me or '').lower() != 'elbereth':
            close = [mon for mon, d in near_dist if d <= jf_config.THREAT_NE_RADIUS]
            # not a lone newt/gecko/grid bug/sewer rat (the tsmoke-jf14 version prayed at those): something that
            # kills a fainted character -- difficulty >= THREAT_NE_MIN_DIFFICULTY, faster than us, or two of them
            # (g17r-jf14: a hobgoblin, a little dog, a bat each killed from 58-75 HP during 20-25-turn faints)
            serious = [mon for mon in close if getattr(mon, 'difficulty', 0) >= jf_config.THREAT_NE_MIN_DIFFICULTY or
                       getattr(mon, 'mmove', 12) > 12]
            if serious or len(close) >= 2:
                return f'no-elbereth:{(serious or close)[0].mname}'
        if len(near) >= jf_config.THREAT_MIN_COUNT or bl.hitpoints < jf_config.THREAT_HP_FRAC * bl.max_hitpoints:
            return f'{len(near)}x{near[0].mname}'
        return None

    def uhunger_weak_estimate(self):
        """Nutrition left while Weak: eat.c turns Weak when uhunger drops below 50, then -1 per turn (plus one
        per melee attack). None when not Weak."""
        if self._weak_since is None or self.blstats.hunger_state != Hunger.WEAK:
            return None
        return 49 - (self.blstats.time - self._weak_since)

    def threat_prayer_due(self):
        """THREAT_PRAYER_GAP: the hunger prayer comes early (from that gap instead of 1100/1200) when a hostile is
        close while Fainting, or while Weak within THREAT_WEAK_MARGIN nutrition of Fainting.
        Evidence (base-* 37ad93a, 90 games): 9 of the 10 Dlvl-1 deaths were fainted next to ordinary melee
        monsters at gaps 950-1110, waiting for the 1100-turn Fainting rule; 114 of 268 fainting cycles had a
        hostile attack during the faints and 9 of those died (~8%), while rnz(350) fails a prayer 6.2% of the
        time at a 950 gap, 5.5% at 1000 and 3.9% at 1100 -- and the Weak->Fainting transition always faints
        at once (eat.c newuhs), so an approaching monster gets 10+ free turns."""
        if not jf_config.THREAT_PRAYER_GAP or self.prayer_failed or self.global_logic.dive.diving:
            return False
        bl = self.blstats
        if bl.hunger_state < Hunger.WEAK:
            return False
        if bl.hunger_state == Hunger.WEAK:
            est = self.uhunger_weak_estimate()
            if est is None or est > jf_config.THREAT_WEAK_MARGIN:
                return False
        if not self.is_safe_to_pray(jf_config.THREAT_PRAYER_GAP):
            return False
        threat = self._hunger_threat()
        if threat is None:
            return False
        self._pray_reason = f'hunger-threat {threat} est={self.uhunger_weak_estimate()}'
        return True

    def fainting_prayer_due(self):
        """A hunger prayer while Fainting: DT6A/s13 rule (a fixed gap) or the starvation clock."""
        bl = self.blstats
        if bl.hunger_state < Hunger.FAINTING:
            return False
        if self._food_first():
            return False
        if self.threat_prayer_due():
            return True
        if self._paralysis_prayer_due():
            return True
        if not jf_config.STARVE_CLOCK:
            if self.is_safe_to_pray(self._faint_prayer_gap()):
                return True
            # STARVE_DEADLINE: with long Fainting gaps (1400-1600) a cycle that began with a prayer that fixed no
            # hunger (an HP prayer while Not Hungry) could starve first: eat.c kills below -(100 + 10 * Con)
            if jf_config.STARVE_DEADLINE and not self.prayer_failed and self._starvation_near():
                self._pray_reason = f'starve-deadline est={self.uhunger_estimate()}'
                return self.is_safe_to_pray(100, certain_death=True)
            return False
        if self.is_safe_to_pray(jf_config.FAINT_PRAYER_GAP_LONG):
            self._pray_reason = 'faint-gap'
            return True
        # waiting out the clock is for quiet moments: a Fainting character wakes up to free hits (a jf8
        # grind fainted on a worn Elbereth next to a pony and a rothe at a 1021-turn gap and died 25
        # turns before its deadline prayer). When actually hurt, or with a monster that Elbereth doesn't
        # stop close by, the old 1000-turn rule applies. (Any hostile in view was too broad: jf9 prayed
        # at 1002-1019-turn gaps three times, where rnz(350) fails ~5.5% of the time.)
        # A hostile within 3 counts too: faints are helpless and a dust Elbereth wears (a jf8 grind
        # fainted next to giant ants on a smudged one).
        # Only real threats count: with any hostile within 3 this rule made 122 of ~480 prayers in jf14/jf16
        # at 1000-1399-turn gaps (3.5-5.4% failure); a newt or jackal can't kill a sheltering character.
        dive = self.global_logic.dive

        def threat(m):
            if m[0] <= 5 and dive._ignores_elbereth(m[3]):
                return True
            # anything adjacent: a dust Elbereth can vanish during a long faint, and then even a bat or a
            # jackal has 15-30 free turns (all 10 tour deaths while fainted in six runs came at 961-1396-turn
            # gaps with a weak monster next to us)
            if m[0] <= 1:
                return True
            return m[0] <= 2 and (combat.monster_utils.is_dangerous_monster(m) or getattr(m[3], 'mlevel', 0) >= 4)

        if self.is_safe_to_pray(jf_config.FAINT_PRAYER_GAP) and \
                (bl.hitpoints < 0.5 * bl.max_hitpoints or any(threat(m) for m in self.get_visible_monsters())):
            self._pray_reason = 'faint-danger'
            return True
        since = self._fainting_since if self._fainting_since is not None else bl.time
        death_line = -(100 + 10 * bl.constitution)
        est = self.uhunger_estimate()
        if est is not None:
            near = est <= death_line + jf_config.FAINT_ESTIMATE_MARGIN
        else:  # no faint measured yet this spell: the worst case, 1 nutrition per turn from 0
            near = bl.time >= since + (-death_line) - jf_config.STARVE_MARGIN
        if near and not self.prayer_failed:
            self._pray_reason = f'faint-deadline est={est}'
            # starving for certain otherwise: a prayer that may fail is the only way out. Once, though:
            # after a failure the god is angry and more prayers only bring his wrath (a clock-jf6 game
            # prayed at gaps of 534, 4, 22, 2 turns and was 'killed by the wrath of Tyr')
            return self.is_safe_to_pray(100, certain_death=True)
        return False

    def _starvation_near(self):
        """The uhunger estimate from the last faint's length (or 1/turn since Fainting began, if no faint was
        measured) is within FAINT_ESTIMATE_MARGIN of eat.c's starvation line -(100 + 10 * Con)."""
        bl = self.blstats
        since = self._fainting_since if self._fainting_since is not None else bl.time
        death_line = -(100 + 10 * bl.constitution)
        est = self.uhunger_estimate()
        if est is not None:
            return est <= death_line + jf_config.FAINT_ESTIMATE_MARGIN
        return bl.time >= since + (-death_line) - jf_config.STARVE_MARGIN

    def _paralysis_prayer_due(self):
        """STARVE_UNMEASURED_GAP: Fainting right after a paralysis ended ('You can move again'): the hunger spent
        while helpless is unknown (no faints measured: eat.c newuhs faints only if multi >= 0) and the next freeze
        may come at once. t31-jf16 s12 was frozen by a floating eye three times running (the stall breaker's melee)
        and starved in the third freeze at gap 1167; the default Fainting rule would have prayed at 1136, between
        freezes. (A first version prayed whenever no faint was measured -- that is also every spell whose first
        faint came during an atom operation -- and cut the long gaps back to 1100.)"""
        if not jf_config.STARVE_UNMEASURED_GAP or self.prayer_failed:
            return False
        bl = self.blstats
        if bl.hunger_state < Hunger.FAINTING or bl.time - self._paralysis_end_turn > 3:
            return False
        if not self.is_safe_to_pray(jf_config.STARVE_UNMEASURED_GAP):
            return False
        self._pray_reason = 'paralysis-faint'
        return True

    def uhunger_estimate(self):
        """From the last measured faint (eat.c: a faint lasts 10 - uhunger/10 moves), decayed by one per
        turn since (the worst case). None if no faint was measured during this Fainting spell."""
        m = self._faint_measure
        if m is None or self._fainting_since is None or m[0] < self._fainting_since:
            return None
        turn, est = m
        return est - (self.blstats.time - turn)

    def _critically_low_hp(self):
        """pray.c critically_low_hp(): the only HP level at which prayer fixes anything. DT6A's
        'HP < 12' made an XL1 Valkyrie (18 max HP) pray at 11 HP -- no trouble, so once the timeout
        wasn't zero: 'Tyr is displeased', Luck -3, and every later prayer failed."""
        bl = self.blstats
        xl = bl.experience_level
        maxhp = min(bl.max_hitpoints, 15 * xl)
        divisor = 5 if xl <= 5 else 6 if xl <= 13 else 7 if xl <= 21 else 8 if xl <= 29 else 9
        return bl.hitpoints <= 5 or bl.hitpoints * divisor <= maxhp

    def _hunger_prayer_gap(self):
        # A prayer resets nutrition to 900 and Weak comes ~850 turns later, so DT6A's 1200-turn gap
        # left every Dlvl-1 grind Fainting for hundreds of turns and praying at Fainting on a 400-turn
        # gap (failures, level drain, starvation). With EARLY_FIXES pray at Weak from a 900-turn gap
        # (rnz(350) failure risk ~6%) instead of fainting first.
        return 900 if jf_config.EARLY_FIXES else self.SAFE_HUNGER_PRAYER_GAP

    def _food_first(self):
        """Plenty of food carried (e.g. bought): eat it rather than risk a hunger prayer below FOOD_FIRST_GAP
        (1200-1399 gaps fail ~2.7%, Fainting ones at 1000-1099 ~3-5%), keeping the prayer for HP emergencies."""
        return bool(jf_config.FOOD_FIRST_MIN) and \
            self.inventory.carried_nutrition() >= jf_config.FOOD_FIRST_MIN and \
            not self.is_safe_to_pray(jf_config.FOOD_FIRST_GAP)

    def _eat_before_praying(self):
        if self._food_first():
            return True
        # hypothesis: at XL < 5 the emergency prayer is the only answer to a bad fight (an XL2 elite
        # game spent it on hunger at T1350 and died to a goblin at T1660 with nothing left); eat the
        # food we carry instead of praying for hunger while that weak.
        if self.blstats.experience_level >= 5 or not jf_config.EARLY_FIXES:
            return False
        return any(item.category == nh.FOOD_CLASS and item.objs[0].name != 'sprig of wolfsbane' and
                   not item.is_corpse() for item in flatten_items(self.inventory.items))

    def pray(self):
        gap = None if self.last_prayer_turn is None else self.blstats.time - self.last_prayer_turn
        model_note, limit = '', None
        if self._prayer_model_active():
            try:
                model = self.prayer_model
                limit = model.trouble_limit()
                model_note = f' model[{model.summary()} p_answered={model.p_answered(limit):.2f}]'
            except Exception:
                self._prayer_model_error()
        self.log(f'PRAY hp={self.blstats.hitpoints}/{self.blstats.max_hitpoints} hunger={self.blstats.hunger_state} '
                 f'gap={gap} reason={self._pray_reason}{model_note}')
        self._pray_reason = None
        if jf_config.SIM_RESCUE_PRAYER and self.blstats.hunger_state >= Hunger.WEAK and \
                not self.global_logic.dive.diving:
            # dev only (the arena never sets it; ported from rescue's 78a30e1): the Nth tour hunger prayer 'fails'
            # -- it is skipped and no prayer is made again, so every game that gets that far becomes a rescue dive
            self._sim_hunger_prayers = getattr(self, '_sim_hunger_prayers', 0) + 1
            if self._sim_hunger_prayers >= jf_config.SIM_RESCUE_PRAYER:
                self.log('SIM prayer failure: no prayers from here on')
                self._sim_no_prayers = True
                self.prayer_failed = True
                self.last_prayer_turn = self.blstats.time
                self.search()
                return True
        turn_before = self.blstats.time
        history_len = len(self._message_history)
        self.step(A.Command.PRAY)
        self.last_prayer_turn = self.blstats.time
        messages = ' '.join(self._message_history[history_len:] + [self.message])
        failed = answered = False
        if any(msg in messages for msg in self.PRAYER_FAILURE_MESSAGES):
            self.prayer_failed = failed = True
        elif any(msg in messages for msg in self.PRAYER_SUCCESS_MESSAGES):
            self.prayer_failed = False  # pleased() only runs with the god appeased and Luck >= 0
            answered = True
        if limit is not None and self._prayer_model_active():
            try:
                self.prayer_model.on_prayer(turn_before, messages, answered, failed, limit)
                self.log(f'PRAY result {"FAILED" if failed else "ok"} model[{self.prayer_model.summary()}] '
                         f'prayers={self.prayer_model.prayers} failures={self.prayer_model.failures}')
            except Exception:
                self._prayer_model_error()
        # TODO: return value
        return True


    ######## PRAYER MODEL (nhmodel/prayer.py: pray.c's odds; 3 errors -> the old rules for the rest of the game)

    def _prayer_model_active(self):
        model = getattr(self, 'prayer_model', None)
        return model is not None and not model.disabled

    def _prayer_model_error(self):
        model = getattr(self, 'prayer_model', None)
        if model is None:
            return
        model.errors += 1
        self.log(f'PRAYER MODEL error #{model.errors}: {traceback.format_exc(limit=3)!r}')
        if model.errors >= 3:
            model.disabled = True

    def _prayer_model_gate(self, certain_death):
        """is_safe_to_pray's model check: False when pray.c says the prayer must fail whatever the gap --
        p_type 1 (can_pray, pray.c:1823): an angry god (u.ugangr, which only a sacrifice lowers), Luck < 0
        (own pet killed, murder, a guilty unicorn, cannibalism, a broken mirror, Friday 13th) or a negative
        alignment record. True when the model has a verdict of its own on a past failure (the fixed
        PRAYER_FAILURE_WAIT is then replaced: a failure the timeout explains means ugangr > 0 and no prayer
        ever works again; one that Luck explains is over once Luck has timed out). None: no opinion."""
        if not self._prayer_model_active():
            return None
        try:
            model = self.prayer_model
            naughty = model.naughty()
            if certain_death:
                # a prayer that cannot be answered doesn't beat dying either, but costs nothing: old rule
                return None
            if naughty >= 0.5:
                return False
            if self.prayer_failed:
                return True  # p_ugangr < 0.5 here: the failure came from Luck/record trouble that is over
            return None
        except Exception:
            self._prayer_model_error()
            return None

    # the old HP rule prayed > 500 turns after the last prayer: P(rnz(350) <= 200 + 501), pray.c:1220, 1819
    SAFE_PRAYER_P = rnz_cdf(350, 200 + 501)

    def _prayer_holds_ok(self):
        """is_safe_to_pray's non-timeout vetoes: Gehennom (the god can't help, pray.c:1901-1906) and the
        dive's holds (murder, alignment budget)."""
        if jf_config.GEHENNOM_DIVE and self.current_level().dungeon_number == 1:
            return False
        return self.blstats.time >= self.prayer_hold_until

    def _hp_prayer_due(self, low_hp_old, poly_buffer):
        """The low-HP prayer. With the model: only in pray.c's TROUBLE_HIT window (critically_low_hp; the
        DT6A 'HP < 12' rule prayed outside it, where only ublesscnt == 0 answers: 1 in 3 of those prayers
        fail at a 500 gap and anger the god for good); 'hp' when the timeout posterior is as safe as the old
        500-turn rule (from turn ~103 for the first prayer: u_init.c:644 starts it at 300, major trouble
        needs <= 200), 'hp-doom' at any gap when the prayer beats the fight (PrayerModel.hp_decision)."""
        if self._prayer_model_active() and not poly_buffer:
            try:
                if not self._prayer_holds_ok():
                    return False
                model = self.prayer_model
                decision = model.hp_decision(self.SAFE_PRAYER_P, jf_config.DOOM_MARGIN, jf_config.DOOM_MIN_P)
                if decision == 'hp-doom' and not self._doom_prayer_beats_exits(model):
                    decision = None
                if decision is not None:
                    gap = None if self.last_prayer_turn is None else self.blstats.time - self.last_prayer_turn
                    self._pray_reason = f'{decision} p_hp={model.last_p_hp:.2f} p_die3={model.last_p_die:.2f}' \
                        if decision == 'hp-doom' else 'hp'
                    if decision == 'hp-doom':
                        self.log(f'DOOM prayer: gap {gap} p_hp={model.last_p_hp:.2f} p_die3={model.last_p_die:.2f}')
                    return True
                return False
            except Exception:
                self._prayer_model_error()
        return self.is_safe_to_pray(500) and low_hp_old

    def _emergency_downstairs_available(self):
        """The exit both the doom comparison and emergency executor can use."""
        level = self.current_level()
        return (jf_config.LAST_RESORT and
                level.objects[self.blstats.y, self.blstats.x] in G.STAIR_DOWN and
                level.dungeon_number != Level.SOKOBAN and
                self.blstats.time - self._last_resort_stairs_turn > 20 and
                self.blstats.carrying_capacity < 4)

    def _doom_prayer_beats_exits(self, model):
        """Exits that beat a long-shot prayer: the down stairs underfoot (the last resort takes them: they
        bank a level and shed every non-follower, dog.c:keepdogs), and a fresh dust Elbereth when everything
        in reach respects it (monmove.c:onscary; the engraving is legible 0.96^8 = 72% of the time,
        engrave.c:1052-1058, and the engrave turn itself is one more round of attacks)."""
        y, x = self.blstats.y, self.blstats.x
        near = [m for m in self.get_visible_monsters() if max(abs(m[1] - y), abs(m[2] - x)) <= 2]
        unseen_harm = model.unseen_attack_dps() > 0
        if not near and not unseen_harm:
            return False
        if self._emergency_downstairs_available():
            return False
        dive = self.global_logic.dive
        engraving = (self.inventory.engraving_below_me or '').lower()
        if not unseen_harm and engraving != 'elbereth' and not self.character.prop.blind and self.can_engrave() and \
                not any(dive._ignores_elbereth(m[3]) for m in near):
            p_elbereth = 0.72 * (1.0 - model.death_probability(turns=1))
            if model.last_p_hp < p_elbereth:
                return False
        return True

    def _faint_doom_prayer_due(self):
        """Fainting beside hostiles (PrayerModel.faint_decision): the fixed Fainting gap (1100) waited through
        faints next to a goblin at a 979-turn gap and a little dog at 1087 (both died helpless, prayers
        answered 94-96% of the time there). Only where the timeout is as safe as the old HP rule."""
        if not self._prayer_model_active() or self.blstats.hunger_state < Hunger.FAINTING or self.prayer_failed:
            return False
        try:
            if not self._prayer_holds_ok():
                return False
            ok, p_ok, p_die = self.prayer_model.faint_decision(self.SAFE_PRAYER_P, jf_config.DOOM_MARGIN)
            if ok:
                gap = None if self.last_prayer_turn is None else self.blstats.time - self.last_prayer_turn
                self._pray_reason = f'faint-doom p_ok={p_ok:.2f} p_die8={p_die:.2f}'
                self.log(f'DOOM faint prayer: gap {gap} p_ok={p_ok:.2f} p_die8={p_die:.2f}')
                return True
        except Exception:
            self._prayer_model_error()
        return False

    def _status_prayer_due(self):
        """Stoned, slimed, strangled or sick (pray.c TROUBLE_STONED..TROUBLE_SICK, each fatal within 5-20
        turns): the old rule waited for a 100-turn gap; the model prays at any gap where the timeout
        posterior leaves a real chance (22% right after a prayer, rnz(350) <= 200)."""
        if not self._prayer_model_active():
            return False
        try:
            p = self.prayer_model.status_decision()
            if p >= 0.05:
                self._pray_reason = f'status p={p:.2f}'
                return True
        except Exception:
            self._prayer_model_error()
        return False

    def open_door(self, y, x):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.DOOR_CLOSED
            self.direction(y, x)
            self.current_level().door_open_count[y, x] += 1
            return self.glyphs[y, x] not in G.DOOR_CLOSED

    def _note_attack(self, target=None, direction=None):
        """CORPSE_TRACK: remember where our attack went (and the map before it), so that a kill's corpse gets
        its age even when the kill message is parsed a few observations later or the victim had a
        same-kind neighbour. See _track_kill_positions."""
        if jf_config.CORPSE_TRACK and getattr(self, 'glyphs', None) is not None:
            self._attack_ctx = (self.blstats.time, target, direction, (self.blstats.y, self.blstats.x),
                                self.glyphs.copy())

    def melee_attack(self, y, x):
        if jf_config.FEYE_BLIND and not self.character.prop.blind and self._is_floating_eye_at(y, x) and \
                self._feye_safe_attack(y, x):
            return True
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.MONS or self.glyphs[y, x] in G.INVISIBLE_MON or \
                   self.glyphs[y, x] in G.SWALLOW
            self._note_attack(target=(y, x))
            self.direction(y, x)
        return True

    _FEYE_TOOLS = ('blindfold', 'towel')

    def _feye_safe_attack(self, y, x):
        """FEYE_BLIND: hit a floating eye without meeting its gaze. uhitm.c passive(): the AD_PLYS freeze
        (d(lvl+1, 70) turns, 127 at Wis <= 12 three times in four) needs canseemon(eye), so a blind hero hits it
        safely. 121 freezes in 180 base5/base6 games (45 games); base6-jf16 s12 froze at T9224 and a rock mole
        killed the 75-HP XL7 before it could move again.
        1) a blindfold/towel carried (58 of 180 games had one; both are generated uncursed, mkobj.c): put it
           on, attack the unseen eye with F+direction (attack_checks lets a forcefight through), take it off;
        2) else throw at it (only where fight2's own ranged rules allow a throw: no pet/peaceful behind it,
           no Watch about);
        3) else the plain melee, as before (FEYE_FIX's refusal confined bots behind eyes for 55k-185k turns).
        Returns True if the attack was handled here."""
        tool = next((i for i in self.inventory.items if i.objs[0].name in self._FEYE_TOOLS and not i.equipped and
                     i.status != Item.CURSED), None)
        if tool is not None and not getattr(self, '_feye_blindfold_stuck', False):
            return self._feye_blind_melee(tool, y, x)
        dy, dx = y - self.blstats.y, x - self.blstats.x
        if not self.character.prop.polymorph and self.inventory.get_best_ranged_set()[1] is not None:
            monsters = self.get_visible_monsters()
            if combat.fight_heur.ranged_priority(self, dy, dx, monsters) is not None:
                _, ammo = self.inventory.get_best_ranged_set()
                self.log(f'FEYE throwing {ammo.text!r} at the floating eye instead of hitting it')
                return self.fire(ammo, self.calc_direction(self.blstats.y, self.blstats.x, y, x))
        return False

    def _feye_blind_melee(self, tool, y, x):
        # one atomic operation: no strategy may take over (and nothing re-reads the map) while we can't see
        with self.atom_operation():
            return self._feye_blind_melee_steps(tool, y, x)

    def _feye_blind_melee_steps(self, tool, y, x):
        letter = self.inventory.items.get_letter(tool)
        hp0 = self.blstats.hitpoints

        def put_on():
            if 'What do you want to put on?' in self.single_message:
                yield letter

        self.step(A.Command.PUTON, put_on())
        if not self.character.prop.blind:
            self.log(f'FEYE could not put on {tool.text!r}: {self.message[:80]!r}')
            self._feye_blindfold_stuck = True   # never retry this (e.g. a face already covered)
            return False
        self.log(f'FEYE blinded with {tool.text!r} to hit the floating eye at {(y, x)}')
        d = self.calc_direction(self.blstats.y, self.blstats.x, y, x)
        killed = False
        for _ in range(8):
            self.step(A.Command.FIGHT)
            self.direction(d)
            msg = self.message
            if 'You kill' in msg or 'You destroy' in msg:
                killed = True
                break
            if 'thin air' in msg:
                break
            # something else is chewing on us while we can't see: stop and look
            if self.blstats.hitpoints < min(hp0 - 10, 0.6 * self.blstats.max_hitpoints):
                break
        self.inventory.items.update(force=True)
        worn = next((i for i in self.inventory.items if i.objs[0].name in self._FEYE_TOOLS and i.equipped), None)
        wletter = self.inventory.items.get_letter(worn) if worn is not None else letter

        def remove():
            if 'What do you want to remove?' in self.single_message:
                yield wletter

        self.step(A.Command.REMOVE, remove())
        if self.character.prop.blind:
            # a cursed one can't come off: pray.c counts it as major trouble (TROUBLE_CURSED_BLINDFOLD)
            self.log(f'FEYE blindfold stuck: {self.message[:80]!r}')
            self._feye_blindfold_stuck = True
        self.inventory.items.update(force=True)
        self.log(f'FEYE blind attack done: killed={killed}')
        return True

    def zap(self, item, direction):
        # astra: an empty wand prints "Nothing happens" without asking for a direction, and a blindly
        # queued direction key then becomes a move or a melee attack. Only answer the prompt if it's
        # there, and remember wands that turned out empty.
        with self.atom_operation():
            self.step(A.Command.ZAP)
            if 'carrying so much stuff' in self.message:
                raise AgentPanic('zap: too heavily loaded')
            self.type_text(self.inventory.items.get_letter(item))
            if 'What do you want to zap?' in self.single_message or "You don't have that object" in self.message:
                self.step(A.Command.ESC)
                raise AgentPanic('zap: no such wand in the inventory')
            if not jf_config.LATE_FIXES or 'In what direction?' in self.message:
                self.direction(direction)
            elif 'Nothing happens' in self.message or 'You wrest' in self.message:
                self.inventory.empty_wands.add(item.text)
        return True

    def fire(self, item, direction):
        if self.character.prop.polymorph:
            # TODO: throwing is not possible if you don't have hands
            # it may be possible depending on creature
            return False

        self._note_attack(direction=direction)
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
        if self.blstats.time < self._no_kick_until:
            raise AgentPanic('legs too wounded to kick')
        with self.panic_if_position_changes():
            with self.atom_operation():
                self.step(A.Command.KICK)
                if 'no shape for kicking' in self.message or 'cannot kick effectively' in self.message:
                    # wounded legs: the door-kicking loop retried until the turn-inactivity guard fired
                    self._no_kick_until = self.blstats.time + 100
                    raise AgentPanic('legs too wounded to kick')
                self.direction(self.calc_direction(self.blstats.y, self.blstats.x, y, x))

    def search(self, max_count=1, *, return_result=False):
        """Search normally, or report a single recovery command's completion.

        Completion means a new response was parsed without a prompt or refusal;
        it does not require the displayed game turn to advance.
        """
        assert max_count >= 1
        assert not return_result or max_count == 1
        if return_result and (self._is_reading_message_or_popup or
                              self._observation['misc'].any()):
            return SearchResult('prompt', sent=False)
        with self.panic_if_position_changes():
            with self.atom_operation():
                if max_count > 1:
                    self.type_text(str(max_count))
                previous_observation = self.last_observation
                previous_step = self.step_count
                previous_turn = self.blstats.time
                previous_budget = self._recovery_response_steps
                if return_result:
                    # Includes the search and automatic prompt responses only;
                    # subsequent inventory/terrain housekeeping is outside it.
                    self._recovery_response_steps = 8
                try:
                    response = self.step(A.Command.SEARCH)
                finally:
                    self._recovery_response_steps = previous_budget
                if return_result:
                    result = SearchResult.from_response(
                        response, previous_observation, self.last_observation,
                        self.step_count > previous_step, previous_turn, self.blstats.time,
                        self.message, self.popup, self._is_reading_message_or_popup)
                # TODO: estimate the real number of searches
                self.current_level().search_count[self.blstats.y, self.blstats.x] += max_count
                if 'You find ' in self.message:
                    self.check_terrain(force=True)
        return result if return_result else True

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
            from_y, from_x = self.blstats.y, self.blstats.x
            self.direction(dir)

            if self.blstats.y != expected_y or self.blstats.x != expected_x:
                # the map thought this was a doorless doorway, but the door is intact: remember it as a
                # door so BFS stops planning diagonal steps through it (a b5 dive looped on this)
                if 'diagonally out of an intact doorway' in self.message:
                    self.current_level().intact_doors[from_y, from_x] = True
                elif 'diagonally into an intact doorway' in self.message:
                    self.current_level().intact_doors[expected_y, expected_x] = True
                raise AgentPanic(f'agent position do not match after "move": '
                                 f'expected ({expected_y}, {expected_x}), got ({self.blstats.y}, {self.blstats.x})')

    def hands_welded(self):
        """engrave.c/do_wear.c freehand(): a cursed (welded) two-hander, or a cursed weapon with a shield,
        leaves no free hand -- engraving, taking armor off, #untrap and #loot then fail without a turn
        (a welded dwarvish mattock: 2703 'cannot release your weapon' and 270 'no free hand' asserts)."""
        main = self.inventory.items.main_hand
        if main is None or main.status != Item.CURSED:
            return False
        return bool(getattr(main.objs[0], 'bi', False)) or self.inventory.items.off_hand is not None

    def _weld_prayer_due(self):
        """WELD_PRAY: diving with the hands welded to a two-hander, or to a weapon beside a cursed shield -- exactly
        pray.c's TROUBLE_UNUSEABLE_HANDS (welded(uwep) && !freehand()), a major trouble the prayer fixes; a welded
        one-hander beside an uncursed shield is only minor trouble, which a Luck-0 prayer leaves alone."""
        if self.prayer_failed or not self.global_logic.dive.diving or self.character.prop.polymorph:
            return False
        main = self.inventory.items.main_hand
        if main is None or main.status != Item.CURSED:
            return False
        shield = self.inventory.items.off_hand
        if not (getattr(main.objs[0], 'bi', False) or (shield is not None and shield.status == Item.CURSED)):
            return False
        return self.is_safe_to_pray(jf_config.WELD_PRAY_GAP)

    def can_engrave(self):
        if self.character.prop.polymorph:
            return False  # TODO: only for handless monsters (which cannot write)
        if self.hands_welded():
            return False
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

    def bfs(self, y=None, x=None, force_squeeze=False):
        if y is None:
            y = self.blstats.y
        if x is None:
            x = self.blstats.x

        if not force_squeeze and self.last_bfs_step == self.step_count and y == self.blstats.y and \
                x == self.blstats.x:
            return self.last_bfs_dis.copy()

        level = self.current_level()

        walkable = level.walkable & ~utils.isin(self.glyphs, G.BOULDER) & \
                   ~self.monster_tracker.peaceful_monster_mask & \
                   ~level.forbidden
        if jf_config.TEMP_FORBID:
            walkable &= ~(level.forbidden_until > self.blstats.time)
        if jf_config.HAZARD_FIXES and self.inventory.items.gloves is None:
            walkable &= ~(level.petrify_until > self.blstats.time)

        if self._last_turn - self._allow_walking_through_traps_turn > 50:
            walkable &= ~utils.isin(level.objects, G.TRAPS)
        elif jf_config.SAFE_TRAP_WALK:
            # walking through known traps is a last resort: never into the ones that can wreck the game
            # (polymorph, fire, sleeping gas, magic, anti-magic, rust), nor off the level during the tour or while
            # climbing out of a branch (CLIMB_NO_FALL)
            dive = self.global_logic.dive
            exits_ok = dive.diving and not (jf_config.CLIMB_NO_FALL and dive.climbing())
            walkable &= ~utils.isin(level.objects, self.UNSAFE_TRAPS if exits_ok
                                    else self.UNSAFE_TRAPS | self.LEVEL_EXIT_TRAPS)

        # a shopkeeper blocks the door to anyone carrying a pick-axe or mattock ('Will you please leave
        # your pick-axe outside?'): walking to the goods (check_items) looped at shop doors for 3000-16000
        # turns in 25 of ~330 games, fainting through hunger prayers. While we carry one, the refusing door
        # is closed to us -- when standing in it, the squares on the shopkeeper's side of it.
        # (Only the door: AutoAscend's shop mask can leak through a door into the corridors.)
        if level.dig_tool_refused is not None and self.blstats.time - level.dig_tool_refused < 5000 and \
                self._carries_digging_tool():
            for (dy, dx), (iy, ix) in level.refused_doors.items():
                if (dy, dx) != (y, x):
                    walkable[dy, dx] = False
                    continue
                # the inward direction, and its two neighbours along the wall side
                for ny, nx in {(dy + iy, dx + ix), (dy + iy, dx + (ix or 1)), (dy + iy, dx - (ix or 1)),
                               (dy + (iy or 1), dx + ix), (dy - (iy or 1), dx + ix)}:
                    if (ny, nx) != (dy, dx) and 0 <= ny < walkable.shape[0] and 0 <= nx < walkable.shape[1] and \
                            max(abs(ny - dy - iy), abs(nx - dx - ix)) <= 1 and \
                            (ny - dy) * iy + (nx - dx) * ix > 0:
                        walkable[ny, nx] = False

        for my, mx in list(zip(*np.nonzero(utils.isin(self.glyphs, G.MONS)))):
            mon = MON.permonst(self.glyphs[my][mx])
            if mon.mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS:
                walkable[my, mx] = False
        if jf_config.SESSILE_MEMORY and level.sessile:
            # ...also while out of sight (see _update_level_sessile)
            for (my, mx) in level.sessile:
                if (my, mx) != (y, x):
                    walkable[my, mx] = False

        walkable_diagonally = walkable & ~utils.isin(level.objects, G.DOORS) & (level.objects != -1) \
            & ~level.intact_doors
        can_squeeze = (force_squeeze or self.inventory.items.total_weight <= 600) and \
            self.current_level().dungeon_number != Level.SOKOBAN
        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable_diagonally,
                        can_squeeze=can_squeeze,
                        )

        if y == self.blstats.y and x == self.blstats.x and not force_squeeze:
            self.last_bfs_dis = dis
            self.last_bfs_step = self.step_count

        return dis.copy()

    UNSAFE_TRAPS = frozenset({SS.S_polymorph_trap, SS.S_fire_trap, SS.S_sleeping_gas_trap, SS.S_magic_trap,
                              SS.S_anti_magic_trap, SS.S_rust_trap})
    LEVEL_EXIT_TRAPS = frozenset({SS.S_trap_door, SS.S_hole, SS.S_level_teleporter})

    def _diagonal_step_ok(self, masks, y, x, ty, tx):
        """The BFS rule for a diagonal step (y, x) -> (ty, tx) (utils.bfs)."""
        walkable, walkable_diagonally, can_squeeze = masks
        return bool(walkable_diagonally[ty, tx] and walkable_diagonally[y, x] and
                    (can_squeeze or walkable[ty, x] or walkable[y, tx]))

    def path(self, from_y, from_x, to_y, to_x, dis=None):
        if from_y == to_y and from_x == to_x:
            return [(to_y, to_x)]

        if dis is None:
            dis = self.bfs(from_y, from_x)

        assert dis[to_y, to_x] != -1

        # FIXME: currently the path can lead through diagonally inwalkable tiles.
        #        The path is the shortest possible, so the agent is guaranteed to
        #        unstuck itself eventually (usually a few panic exceptions) if that happens
        # PATH_DIAG_FIX: it didn't unstick. The BFS reaches a door orthogonally, but walking the distances
        # back picked any square one step closer -- also the one diagonal to the door. A jf26 s0 grind
        # tried the same diagonal step into an open door 90 times ('You can't move diagonally into an
        # intact doorway', each a panic and a restart of the strategy). Among the (identically shuffled)
        # candidates the first legal step is taken; the old choice only when none is legal. Only doorways
        # the game has already refused (level.intact_doors) are checked: a first try costs no game time, and
        # leaving it alone keeps every other path (and game) exactly as before.
        masks = None
        if jf_config.PATH_DIAG_FIX and self.current_level().intact_doors.any():
            # (walkable, walkable diagonally, can squeeze): a diagonal step is refused only at those doorways
            refused = self.current_level().intact_doors
            masks = (~refused, ~refused, True)

        cur_y, cur_x = to_y, to_x
        path_rev = [(cur_y, cur_x)]
        while cur_y != from_y or cur_x != from_x:
            chosen = None
            for y, x in self.neighbors(cur_y, cur_x):
                if dis[y, x] == dis[cur_y, cur_x] - 1 and dis[y, x] >= 0:
                    if masks is None or y == cur_y or x == cur_x or \
                            self._diagonal_step_ok(masks, y, x, cur_y, cur_x):
                        chosen = (y, x)
                        break
                    if chosen is None:
                        chosen = (y, x)   # an illegal diagonal: kept only if no legal step turns up
            # (one shuffle per step as before, so the RNG stream -- and every later choice -- is unchanged)
            assert chosen is not None
            path_rev.append(chosen)
            cur_y, cur_x = chosen

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
        # STATUS heartbeat: which strategy keeps walking where (stall diagnoses)
        self._hb_gotos[(y, x, sys._getframe(1).f_code.co_name)] += 1
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
        stuck_replans = 0   # GOTO_TARGET_FIX re-plans that took no step (see below)
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
                # GOTO_TARGET_FIX: the loop variables were (y, x), overwriting the target, so a path blocked
                # mid-way (a peaceful stepping in, a square found unwalkable) made the next round head for the
                # blocked square itself: 'end point is no longer accessible', a panic and a restart of the
                # strategy instead of a new route to the real target (Mines peacefuls; jf26 s9 looped on it)
                ty, tx = y, x
                steps_before = steps_taken
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
                if jf_config.GOTO_TARGET_FIX and self.global_logic.dive.diving:   # (dive only: keeps the grind)
                    if cont and (y, x) != (ty, tx):
                        self.log(f'GOTO path blocked at {(int(y), int(x))}: re-planning to {(int(ty), int(tx))}')
                        # blocked at the very first step (the BFS and the walkable/peaceful masks disagree about
                        # that square), the same path came back every round without a step taken: base7-public
                        # s7 livelocked on Dlvl 25 until the driver's hang guard fired. After 3 such rounds, the
                        # panic the loop raised before GOTO_TARGET_FIX restarts the strategy instead.
                        stuck_replans = stuck_replans + 1 if steps_taken == steps_before else 0
                        if stuck_replans >= 3:
                            raise AgentPanic('go_to: path blocked at its first step')
                    y, x = ty, tx

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

    def _hurt_recently(self, turns=3):
        hist = self.global_logic.dive._hp_history
        now = self.blstats.time
        recent = [hp for t, hp in hist if t >= now - turns]
        return bool(recent) and max(recent) > self.blstats.hitpoints

    def _fight_stall_turns(self):
        """jf_config.FIGHT_STALL_TURNS; in the Valley of the Dead (GEHENNOM_DIVE) VALLEY_FIGHT_STALL_TURNS when that
        is 0: its graveyards' sleeping undead, seen across their boundaries, held fight2 dancing between two
        squares for 1300-2500 turns in 3 of 15 valley-x10 scenario games (they starved)."""
        if jf_config.FIGHT_STALL_TURNS:
            return jf_config.FIGHT_STALL_TURNS
        return jf_config.VALLEY_FIGHT_STALL_TURNS if self.global_logic.dive.in_valley() else 0

    # monsters fight2 never melees and that don't come at us: once let go, they stay let go even when adjacent
    # (rescue's public/6 run: fight2 stepped away from an adjacent gas spore, the explorer walked back beside it,
    # and the stall breaker re-fired every ~30 turns, 54 times)
    _FIGHT_PASSIVE = frozenset(combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS) - {'gelatinous cube', 'Oracle'}

    def fight_monsters(self):
        """get_visible_monsters() without those a stalled fight let go of (jf_config.FIGHT_STALL_TURNS). They count
        again once adjacent (unless passive), after FIGHT_IGNORE_TURNS, or when something hurts us (only those
        within 3 when any is: a sleeping zoo further off stays let go)."""
        monsters = self.get_visible_monsters()
        # the Valley's graveyards: their sleepers are left alone unless they attack (dive.VALLEY_GRAVE_FILTER)
        monsters = self.global_logic.dive.valley_fight_filter(monsters)
        if self._fight_stall_turns() <= 0 or not self._fight_ignored:
            return monsters
        bl = self.blstats
        near = lambda m: max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 3
        hurt = self._hurt_recently(3)
        if hurt and not any(near(m) for m in monsters):
            # hurt by something unseen or from afar: it may be one we let go
            self._fight_ignored.clear()
            return monsters
        key = self.current_level().key()
        kept = []
        for m in monsters:
            _, y, x, mon, glyph = m
            ent = self._fight_ignored.get((key, glyph))
            if ent is not None and ent[2] >= bl.time and (ent[3] or max(abs(y - ent[0]), abs(x - ent[1])) <= 2) and \
                    (mon.mname in self._FIGHT_PASSIVE or not utils.adjacent((bl.y, bl.x), (y, x))) and \
                    not (hurt and near(m)):
                ent[0], ent[1] = int(y), int(x)   # a slow drifter (a floating eye) stays let go
                continue
            kept.append(m)
        return kept

    def _note_fight_stall(self, best_action, monsters):
        """FIGHT_STALL_TURNS: fight2's 'strike first' heatmap parks us two squares from a monster and waits for
        it to step in; one that never comes (asleep, immobile, trapped, out of throwing line with the missiles
        spent) kept fight2 -- which preempts the dive plan, exploration and eating -- dancing between two
        squares: public s9 (a sleeping nymph held the XL8 dive on Dlvl 4 for 3300 turns and three hunger
        prayers), jf26 s3 (4000 turns on Dlvl 3, died there), public s11 (a starving rescue stepped W/E beside
        zombies for ~1500 turns until it starved). In a 45-game profile, 67 fight2 stalls lasted 150+ turns.
        Two triggers (merged with rescue's FIGHT_STALL_BREAK):
        - the dance itself: FIGHT_STALL_MOVES consecutive moves among at most 3 squares (over >= n/2 turns);
        - FIGHT_STALL_TURNS turns with no contact (melee/kick/zap, or holding out on Elbereth), kill or damage
          while above half HP (legitimate fights resolve sooner: most 40-149-turn ones ended in a kill).
        Then the monsters fight2 faces are let go (FIGHT_IGNORE_TURNS, see fight_monsters)."""
        bl = self.blstats
        st = self._fight_stall
        key = self.current_level().key()
        # contact, or deliberately holding out on Elbereth / hurt: not a stall to break
        contact = best_action[0] in ('melee', 'kick', 'zap', 'wait', 'elbereth') or \
            bl.hitpoints < 0.5 * bl.max_hitpoints
        names = frozenset(m[3].mname for m in monsters if not utils.adjacent((bl.y, bl.x), (m[1], m[2])))
        ended = None if st is None else 'gap' if bl.time - st[2] > 10 else 'contact' if contact else \
            'kill' if bl.experience_points != st[1] else 'hurt' if self._hurt_recently(3) else None
        if st is None or ended:
            if st is not None and st[2] - st[0] >= 20:
                # diagnostics (also with FIGHT_STALL_TURNS < 0: log only, no behaviour change)
                self.log(f'FIGHT stall of {st[2] - st[0]} turns ended ({ended}): {sorted(st[3])}')
            self._fight_stall = (bl.time, bl.experience_points, bl.time, names)
            self._fight_moves = []
            return
        self._fight_stall = (st[0], st[1], bl.time, st[3] | names)
        n = jf_config.FIGHT_STALL_MOVES
        if best_action[0] in ('move', 'go_to'):
            self._fight_moves = (self._fight_moves + [(bl.time, key, (bl.y, bl.x))])[-max(n, 1):]
        else:
            self._fight_moves = []   # a throw or a pickup: not (yet) a dance
        dance = bool(n) and len(self._fight_moves) >= n and len({m[1:] for m in self._fight_moves}) <= 3 and \
            bl.time - self._fight_moves[0][0] >= n // 2
        if self._fight_stall_turns() < 0:
            mark = (bl.time - st[0]) // 500
            if mark and mark != getattr(self, '_fight_stall_mark', None):
                self._fight_stall_mark = mark
                self.log(f'FIGHT stall ongoing {bl.time - st[0]} turns: {sorted(st[3] | names)}')
            if dance:
                self.log(f'FIGHT dance (log only): {n} moves among {len({m[1:] for m in self._fight_moves})} squares')
                self._fight_moves = []
            return
        # the tour's grind (spawn-limited: its turns cost little) only lets go of the long stalls: the dance trigger
        # there fired on molds and floating eyes within ~40 turns and just reshuffled games (6 of 15 public games)
        diving = self.global_logic.dive.diving
        if not diving:
            dance = False
        if not dance and bl.time - st[0] < (self._fight_stall_turns() if diving else jf_config.FIGHT_STALL_TOUR_TURNS):
            return
        until = bl.time + jf_config.FIGHT_IGNORE_TURNS
        let_go = []
        for _, y, x, mon, glyph in monsters:
            if mon.mname in self._FIGHT_PASSIVE or not utils.adjacent((bl.y, bl.x), (y, x)):
                # let go twice on this level: anywhere on it from now on (a leprechaun that keeps its distance and
                # hops around re-engaged fight2 every 40 turns: 649 let-go events in 45 games)
                again = (key, glyph) in self._fight_ignored
                self._fight_ignored[(key, glyph)] = [int(y), int(x), until, again]
                let_go.append((mon.mname, int(y), int(x)))
        self.log(f'FIGHT stalled ({"dance" if dance else f"{bl.time - st[0]} turns"}, no contact, kill or damage): '
                 f'letting go of {let_go}')
        self._fight_stall = None
        self._fight_moves = []

    @utils.debug_log('fight2')
    @Strategy.wrap
    def fight2(self):
        yielded = False
        wait_counter = 0
        while 1:
            monsters = self.fight_monsters()
            # hallucinating, every monster looks hostile: in the Mines (peaceful to a dwarf; Minetown's Watch)
            # the bot attacked peacefuls and was arrested (~half of 13 angry-Watch games after a yellow light).
            # Only fight back when something is actually hurting us; otherwise wait it out.
            # (everywhere, not only the Mines: a hallucinating XL9 angered a peaceful on the Oracle's level,
            # then hit the Oracle and died to her passive magic missiles)
            if monsters and self.character.prop.hallu and \
                    (not self._hurt_recently() or
                     self.current_level().key() == self.global_logic.minetown_level):
                # in Minetown not even when hurt: a hallucinating XL10 hit a peaceful there, killed four
                # angry watchmen ('You murderer!' x4 = Luck -8, every later prayer failed) and starved;
                # dive_logic.leave_minetown_hallucinating takes the stairs out instead
                monsters = []
            allow_attack_all = self._last_turn - self._allow_attack_all_turn < 3
            # a gelatinous cube close by keeps the fight on, to step out of its paralysing reach
            only_ranged_slow_monsters = all([monster[3].mname in combat.monster_utils.ONLY_RANGED_SLOW_MONSTERS
                                             and not combat.monster_utils.consider_melee_only_ranged_if_hp_full(self,
                                                                                                                monster)
                                             and not (monster[3].mname == 'gelatinous cube' and monster[0] <= 3)
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
                # self.character.parse_spellcast_view()

            move_priority_heatmap, actions = combat.fight_heur.get_priorities(self)
            actions.extend(combat.fight_heur.get_move_actions(self, dis, move_priority_heatmap))

            if self.character.prop.polymorph:
                actions = list(filter(lambda x: x[1][0] != 'ranged', actions))

            if jf_config.PIT_AWARE_FIGHT and self.global_logic.dive.diving and self.in_pit() and \
                    any(utils.adjacent((self.blstats.y, self.blstats.x), (m[1], m[2])) for m in monsters):
                # B001: climbing out of a pit takes several tries ('You are still in a pit'), each a turn the
                # attacker gets for free (jf16 s11: in our dig pit beside a Grey-elf and a werewolf, 90 -> 38 HP
                # while fight2 kept stepping). With an attacker next to us, fight from the pit.
                stay = [a for a in actions if a[1][0] not in ('move', 'go_to')]
                if stay:
                    if len(stay) < len(actions) and getattr(self, '_pit_fight_logged', None) != self.blstats.time // 50:
                        self._pit_fight_logged = self.blstats.time // 50
                        self.log('PIT fight: in a pit beside an attacker, no stepping out')
                    actions = stay

            if jf_config.FEYE_FIX and not self.character.prop.blind:
                # never melee a floating eye we can see: its passive gaze freezes us for up to 127 turns. The
                # exploration's stall breaker (allow_attack_all, below) keeps only attacks, and the eye's -110
                # melee priority was then the best: 401 freezes in 223 dev games, 35 of them died frozen (a
                # pony killed an XL7 grinder at 75/75 HP, gc-lf2 jf16 s4)
                # ... except as the stall breaker's last resort when it is safe-ish: an eye blocking a dead-end
                # corridor with no missiles left held a grind for 110k turns (grind-food g15h-public s10: stuck at
                # (18,31)-(18,35) behind a floating eye, 97 prayers; the ledger's B003). A freeze averages ~100
                # turns (d(lvl+1, 70), cap 127): only with nothing else hostile in view, near-full HP and fed
                bl = self.blstats
                others = [m for m in monsters if not self._is_floating_eye_at(m[1], m[2])]
                feye_ok = allow_attack_all and not others and bl.hitpoints >= 0.9 * bl.max_hitpoints and \
                    bl.hunger_state < Hunger.WEAK
                if not feye_ok:
                    actions = [a for a in actions if not (a[1][0] == 'melee' and self._is_floating_eye_at(
                        self.blstats.y + a[1][1], self.blstats.x + a[1][2]))]
            if jf_config.SPORE_SAFE:
                # a gas spore whose blast would reach a pet, a peaceful or any @ is never hit: its -200 melee
                # priority was still picked when the stall breaker below kept only attacks (see B006)
                actions = [a for a in actions if not (
                    a[1][0] in ('melee', 'kick') and self._spore_unsafe_at(self.blstats.y + a[1][1],
                                                                          self.blstats.x + a[1][2]))]
            if allow_attack_all:
                attack_actions = [a for a in actions if a[1][0] in ('melee', 'kick', 'ranged', 'zap')]
                if attack_actions:
                    actions = attack_actions

            if not actions:
                # nothing possible (cornered, inventory unknown): let a turn pass instead of a panic loop
                # that freezes the game clock until the no-progress timeout (an s6 dive, T38441)
                self.search()
                continue

            priority, best_action = max(actions, key=lambda x: x[0]) if actions else None

            with self.env.debug_tiles(move_priority_heatmap, color='turbo', is_heatmap=True):
                actions_str = '|'.join([combat.utils.action_str(self, a) for a in sorted(actions, key=lambda x: x[0])])
                with self.env.debug_log(actions_str):
                    wait_counter = self._fight2_perform_action(best_action, wait_counter)
            if self._fight_stall_turns():
                self._note_fight_stall(best_action, monsters)

    _GAS_SPORE = None

    def _spore_unsafe_at(self, y, x):
        if Agent._GAS_SPORE is None:
            Agent._GAS_SPORE = MON.from_name('gas spore')
        return 0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1] and \
            self.glyphs[y, x] == Agent._GAS_SPORE and combat.fight_heur.spore_blast_hits_friend(self, y, x)

    _FLOATING_EYE = None

    def _is_floating_eye_at(self, y, x):
        if Agent._FLOATING_EYE is None:
            Agent._FLOATING_EYE = MON.from_name('floating eye')
        return 0 <= y < self.glyphs.shape[0] and 0 <= x < self.glyphs.shape[1] and \
            self.glyphs[y, x] == Agent._FLOATING_EYE

    @Strategy.wrap
    def unsqueeze(self):
        """UNSQUEEZE: a diagonal step between two rock corners needs <= 600 carried (hack.c cant_squeeze_thru), and the
        item logic fills up to ~950. Picking up loot in a corridor bend left a base2-public s9 Valkyrie with no
        way out of its square for 8000 turns (the whole XL7-8 grind, then 4700 turns of its dive; STALL
        reachable=1, both exits diagonal squeezes); base-public s10, base-jf16 s0 and base-jf26 s12 (Dlvl 1) and
        base2-public s1 (Mines 1) were boxed in the same way. When a squeeze would open up the level, drop to 550
        for a while and leave the dropped pile alone."""
        # the dive only (the b3 guard fired it in 3 public grinds on boxes that resolve themselves, reshuffling them),
        # and only once boxed in on one square for UNSQUEEZE_TURNS
        if not jf_config.UNSQUEEZE or self.inventory.items.total_weight <= 600 or \
                not self.global_logic.dive.diving or \
                self.blstats.time < getattr(self, '_squeeze_cap_until', -1) or \
                self.current_level().dungeon_number == Level.SOKOBAN:
            yield False
        here = (*self.current_level().key(), self.blstats.y, self.blstats.x)
        reach = int((self.bfs() != -1).sum())
        if reach > 5:
            self._boxed_since = None
            yield False
        boxed = getattr(self, '_boxed_since', None)
        if boxed is None or boxed[0] != here:
            self._boxed_since = (here, self.blstats.time)
            yield False
        if self.blstats.time - boxed[1] < jf_config.UNSQUEEZE_TURNS:
            yield False
        wide = int((self.bfs(force_squeeze=True) != -1).sum())
        if wide < reach + 10:
            yield False
        # never at the price of the essentials: what we wear and wield and the digging tool must fit under the cap
        # (the b3 guard's grind version dropped a pick-axe in jf16 s6 -- the split keeps it only while diving -- and
        # the dive, 0.602 in base3, had no tool: 0.075)
        tool = self.global_logic.dive.digging_tool()
        essential = sum(i.weight() for i in self.inventory.items.all_items if i.equipped or i is tool)
        if essential > 540:
            yield False
        yield True
        self._boxed_since = None
        self.log(f'UNSQUEEZE: boxed in ({reach} squares) carrying {self.inventory.items.total_weight}; a squeeze '
                 f'reaches {wide}: dropping to 550')
        self._squeeze_cap_until = self.blstats.time + 300
        self.inventory.arrange_items().run()
        self.inventory.unreachable_items_until[self.inventory._here()] = self.blstats.time + 5000

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
                    if item.is_offensive_usable_wand() and not self.inventory.is_known_empty(item):
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
        if jf_config.ENGULF_WIELD:
            # a dig-diver is swallowed with the pick-axe in hand (base-jf25 s13: 'You begin bashing monsters
            # with your pick-axe' inside a fire vortex, dead 2 turns later): every blow from inside hits, so
            # one turn to take up the real weapon pays for itself
            best = self.inventory.get_best_melee_weapon()
            main = self.inventory.items.main_hand
            if best is not None and main is not best and (main is None or main.status != Item.CURSED):
                self.log(f'ENGULFED: wielding {best.text!r} instead of {main.text if main else None!r}')
                self.inventory.wield(best)
        while True:
            mask = utils.isin(self.glyphs, G.SWALLOW)
            if not mask.any():
                break
            assert self.melee_attack(*list(zip(*mask.nonzero()))[0])

    def _is_corpse_editable(self, monster_id, age_turn):
        permonst = MON.permonst(monster_id)

        # hypothesis: starving (Weak or worse) with HP to spare, poison (-1d4 Str, -1d15 HP) or acid
        # (-1d15 HP) beats fainting next to a monster -- two XL11 dives died that way in the Mines
        starving = jf_config.STARVING_EATS and self.blstats.hunger_state >= Hunger.WEAK and \
            self.blstats.hitpoints > 40 and (jf_config.LATE_FIXES or self.global_logic.dive.diving)

        # POISON_EATS: kobolds are ~15% of the grind's corpse nutrition that nobody eats (a pet won't touch them
        # either); eat.c: 4 times in 5 -1d4 Str (18/xx points: barely matters) and -1d15 HP -- fine at high HP
        poison_ok = jf_config.POISON_EATS and not self.global_logic.dive.diving and \
            self.blstats.hitpoints >= max(jf_config.POISON_EATS_MIN_HP, 0.6 * self.blstats.max_hitpoints) and \
            getattr(permonst, 'cnutrit', 0) >= 50

        # TODO: read intrinsics
        if self.character.race != Character.ORC and permonst.mflags1 & MON.M1_POIS != 0 and not starving and \
                not poison_ok:
            return False

        # TODO: read intrinsics
        if permonst.mflags1 & MON.M1_ACID != 0 and not starving:
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

        # a lycanthrope eating its own were family is a cannibal too (eat.c maybe_cannibal: were_beastie(pm)
        # == u.ulycn): Luck -2..-5 and permanent aggravate monster. Public s13 ate three jackal corpses right
        # after a werejackal bite and its next prayer failed ('Thou art arrogant'), so did jf25 s10's
        if jf_config.LYCAN_FIXES and permonst.mname in self.character.were_family():
            return False

        # corpse aging: eat.c taints at rotted = age / (10 + rn2(20)) > 5 (a cursed corpse gets +2) and
        # "rotten" (vomiting, passing out) from rotted > 3 -- 30 turns keeps clear of both
        if self.blstats.time - age_turn >= jf_config.CORPSE_MAX_AGE and \
                monster_id not in [MON.id_from_name('lizard'), MON.id_from_name('lichen')]:
            return False

        return True

    RESERVE_CORPSE_IDS = frozenset(MON.id_from_name(n) for n in ('lichen', 'lizard'))

    def carried_reserve_corpses(self):
        return sum(item.count for item in flatten_items(self.inventory.items)
                   if item.is_corpse() and item.monster_id in self.RESERVE_CORPSE_IDS)

    def reserve_corpse(self, monster_id):
        """LICHEN_RESERVE: keep a lichen/lizard corpse (they never rot) instead of eating it off the floor while
        not Weak. Eaten at once it only lengthens whatever cycle we are in; carried, eat_from_inventory spends it
        when Weak before a safe prayer gap -- the starved cycles (31% of the base grinds' prayer cycles turned
        Weak 800-899 turns after the last prayer, 26% fainted, and all 9 fainting deaths came in cycles that
        had eaten 0-100 nutrition). We ate ~9 lichen corpses per grind (~1800 nutrition)."""
        if not (jf_config.LICHEN_RESERVE and monster_id in self.RESERVE_CORPSE_IDS and
                self.blstats.hunger_state < Hunger.WEAK and not self.global_logic.dive.diving and
                self.carried_reserve_corpses() < jf_config.LICHEN_RESERVE):
            return False
        # only if the item priority can keep it (it keeps items in order within character.carrying_capacity):
        # a heavy pack left a jf14 lichen corpse neither picked up nor eaten while we walked over it for 400 turns
        try:
            weight = sum(item.weight() for item in flatten_items(self.inventory.items))
        except Exception:
            return False
        return weight + 2 * MON.permonst(monster_id + nh.GLYPH_MON_OFF).cwt <= self.character.carrying_capacity

    @utils.debug_log('eat_corpses_from_ground')
    @Strategy.wrap
    def eat_corpses_from_ground(self, only_below_me=True, max_dist=None, max_age=None):
        # max_dist / max_age (CLAIM_CORPSES): only fresh corpses a few steps away
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
                if self.reserve_corpse(monster_id):
                    continue
                if self._is_corpse_editable(monster_id, corpse_age):
                    to_eat.append((y, x, monster_id))

        else:
            for (y, x), corpse_mapping in level.corpses_to_eat.items():
                if max_dist is not None and max(abs(y - self.blstats.y), abs(x - self.blstats.x)) > max_dist:
                    continue
                for monster_id, corpse_age in corpse_mapping.items():
                    if level.shop[y, x]:
                        continue
                    if max_age is not None and self.blstats.time - corpse_age > max_age:
                        continue
                    if self.reserve_corpse(monster_id):
                        continue
                    if self._is_corpse_editable(monster_id, corpse_age):
                        to_eat.append((y, x, monster_id))

        if not to_eat:
            yield False

        dis = self.bfs()
        to_eat = sorted(filter(lambda e: dis[e[0], e[1]] != -1 and (max_dist is None or dis[e[0], e[1]] <= max_dist),
                               to_eat), key=lambda e: dis[e[0], e[1]])
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
                # a stack of same-kind corpses has one merged (averaged) age we can't know: 'There are 2
                # raven corpses here; eat one?' -> 'Ulch - that meat was tainted!', dead at Dlvl 22
                if item.is_corpse() and item.monster_id == monster_id and item.count == 1:
                    if self._is_corpse_editable(monster_id, corpse_age):
                        if not yielded:
                            yielded = True
                            yield True
                        self.inventory.eat(item)

            if not yielded:
                del level.corpses_to_eat[target_y, target_x][monster_id]

        if not yielded:
            yield False

    def should_try_spell_healing(self):
        # Healers begin with renewable healing. Query current spell rows only
        # when injured, rather than trusting startup letters or armor penalties.
        if self.character.role != Character.HEALER:
            return False
        prop = self.character.prop
        if (prop.confusion or prop.stun or prop.hallu or prop.polymorph or
                self.blstats.hunger_state >= Hunger.WEAK or
                self.blstats.carrying_capacity >= 2):
            return False
        return self.spell_healing.ready(self.blstats.hitpoints, self.blstats.max_hitpoints,
                                        self.blstats.energy, self.blstats.time)

    @utils.debug_log('emergency_strategy')
    @Strategy.wrap
    def emergency_strategy(self):

        # hypothesis (astra guard.py stop list): stoning, sliming, strangling and food poisoning /
        # terminal illness kill within a few turns; prayer fixes all of them, so a riskier-than-usual
        # prayer beats certain death. Stoning: a carried lizard corpse cures it without prayer.
        deadly = int(self.last_observation['blstats'][nh.NLE_BL_CONDITION]) & (
            nh.BL_MASK_STONE | nh.BL_MASK_SLIME | nh.BL_MASK_STRNGL | nh.BL_MASK_FOODPOIS | nh.BL_MASK_TERMILL)
        # Only fires when death is otherwise certain within a few turns, so it can never lower a
        # max-progress score: on in every configuration (the elite's early game is untouched).
        if deadly:
            if deadly & nh.BL_MASK_STONE:
                lizards = [item for item in flatten_items(self.inventory.items) if item.is_corpse() and
                           item.monster_id == MON.from_name('lizard') - nh.GLYPH_MON_OFF]
                if lizards:
                    yield True
                    self.log('EMERGENCY stoning: eating a lizard corpse')
                    self.inventory.eat(lizards[0])
                    return
            if self.current_level().dungeon_number != 1 and \
                    (self.is_safe_to_pray(100, certain_death=True) or self._status_prayer_due()):
                yield True
                self.log(f'EMERGENCY deadly status {deadly:#x}: praying')
                self.pray()
                return

        # a were form's HP is only a buffer: at 0 we rehumanize with the HP we had before (polyself.c), so
        # low form HP is no reason to spend healing, a prayer or the last resort (jf16 s11 and jf25 s9 prayed
        # at 1/11 and 2/13 in rat form: the prayer fixed only the rat's HP, reset the prayer timeout without
        # fixing hunger, and both starved before the next safe prayer; jf25 s10 zapped its wands and drank
        # a full healing as a 6-HP jackal)
        poly_buffer = jf_config.LYCAN_FIXES and self.character.poly_hp_is_buffer()

        items = [item for item in flatten_items(self.inventory.items) if item.is_unambiguous() and
                 item.category == nh.POTION_CLASS and item.object.name in ['healing', 'extra healing', 'full healing']]
        if (
                (self.blstats.hitpoints < 1 / 3 * self.blstats.max_hitpoints
                 or self.blstats.hitpoints < 8) and items and not poly_buffer
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

        # LOWHP_EXACT: only where pray.c sees TROUBLE_HIT -- the DT6A 'HP < 12' rule prayed at 10/49 HP (no HP
        # trouble: base2-jf26 s8 got its lycanthropy cured, stayed at 10 HP with the timeout reset, and died)
        if jf_config.EARLY_FIXES or jf_config.EXACT_PRAYER or jf_config.LOWHP_EXACT:
            low_hp = self._critically_low_hp()
        else:
            # DT6A's absolute 'HP < 12' never at full HP or polymorphed: turned into a wererat (8 max HP), an
            # XL6 prayed at 8/8 -- no trouble per pray.c, 538 turns after its last prayer: failed, god angry
            low_hp = (self.blstats.hitpoints < 1 / (5 if self.blstats.experience_level < 6 else 6)
                      * self.blstats.max_hitpoints or
                      (self.blstats.hitpoints < (12 if self.character.role != Character.MONK or
                                                 self._monk_meat_meals == 0 else 8) and
                       self.blstats.hitpoints < self.blstats.max_hitpoints and
                       not self.character.prop.polymorph))
        if poly_buffer:
            low_hp = False
        hp_prayer = self._hp_prayer_due(low_hp, poly_buffer)
        # Make the escape preferred by the doom comparison executable even
        # without a visible target. Unknown-source attacks cannot be assumed
        # to respect Elbereth; retain the existing concrete downstairs escape.
        if low_hp and not hp_prayer and self._prayer_model_active() and \
                self.prayer_model.unseen_attack_dps() > 0 and self._emergency_downstairs_available():
            yield True
            self.log('EMERGENCY unseen attack: down the stairs')
            self._last_resort_stairs_turn = self.blstats.time
            self.global_logic.dive._retreat_blocked_until = self.blstats.time + 20
            self.move('>')
            return
        if (
                hp_prayer
                or self.fainting_prayer_due()
                or self._faint_doom_prayer_due()
                or self.threat_prayer_due()
                or (not self.prayer_failed and self.blstats.hunger_state >= Hunger.WEAK and
                    self.is_safe_to_pray(self._hunger_prayer_gap()) and not self._eat_before_praying())
        ):
            yield True
            self.pray()
            return

        if jf_config.WELD_PRAY and self._weld_prayer_due():
            yield True
            self._pray_reason = 'welded hands'
            self.pray()
            return

        # Renewable healing follows immediate status cures, potions and prayer;
        # it can replenish moderate wounds before they become an emergency.
        if self.should_try_spell_healing():
            yield True
            self.spell_healing.cast(self, A.Command.CAST)
            return

        # Last resort (LAST_RESORT): about to die, no safe prayer, a hostile adjacent. The game is
        # usually lost here, so gambles have positive value for a max-progress score: stairs (down
        # also banks depth), unknown wands at the attacker, unknown potions, unknown scrolls.
        # Overtaxed or worse: NetHack refuses zapping, reading and quaffing without using a turn, and the last
        # resort looped on it (330 turn-inactivity asserts in one jf24 game)
        if jf_config.LAST_RESORT and self._critically_low_hp() and self.blstats.carrying_capacity < 4 and \
                not poly_buffer:
            y, x = self.blstats.y, self.blstats.x
            adjacent = [m for m in self.get_visible_monsters() if utils.adjacent((m[1], m[2]), (y, x))]
            # LR_ELBERETH: with every monster close by respecting Elbereth, the Elbereth rest (below us) is the
            # safer answer: a scared monster doesn't melee, while a zap from the square erases it ('You feel
            # like a hypocrite') and an unknown ray can bounce back (base2-jf25 s1: a wand of cold at an adjacent
            # jackal at 2 HP; base2-public s3: zapped from a fresh Elbereth, then a potion of sickness killed
            # at 3 HP)
            if adjacent and jf_config.LR_ELBERETH and self.current_level().dungeon_number != 1:  # not Gehennom
                dive = self.global_logic.dive
                close = dive._near_hostiles(radius=3)
                engraving = (self.inventory.engraving_below_me or '').lower()
                if not any(dive._ignores_elbereth(m[3]) for m in close) and not self.character.prop.blind and \
                        (engraving == 'elbereth' or self.can_engrave()):
                    adjacent = []
            if adjacent:
                level = self.current_level()
                here = level.objects[y, x]
                # the stairs once: after that the same crowd followed us (an s9 rescue dive ping-ponged
                # down/retreat-up at 15 HP until it died, never trying its wands or potions)
                stairs_ok = self.blstats.time - self._last_resort_stairs_turn > 20
                dive = self.global_logic.dive
                # LAST_RESORT_DIG: a known wand of digging zapped down is a way out that also banks a level
                # (dig.c zap_dig: a hole at once where the floor can be dug; not on stairs, where the beam
                # bounces). ~12% of castle arrivals carried one, engrave-identified.
                if jf_config.LAST_RESORT_DIG and here not in G.STAIR_UP and here not in G.STAIR_DOWN and \
                        level.dungeon_number != Level.SOKOBAN and level.key() not in dive.undiggable:
                    wand = next((i for i in self.inventory.items if i.category == nh.WAND_CLASS and
                                 i.is_unambiguous() and i.object.name == 'digging' and
                                 not self.inventory.is_known_empty(i) and i.comment != 'EMPT'), None)
                    if wand is not None and self._last_resort_dig_turn != self.blstats.time:
                        yield True
                        self._last_resort_dig_turn = self.blstats.time
                        self.log(f'LAST RESORT: zapping {wand.text!r} down')
                        self.zap(wand, '>')
                        return
                if stairs_ok and here in G.STAIR_DOWN and level.dungeon_number != Level.SOKOBAN:
                    yield True
                    self.log('LAST RESORT: down the stairs')
                    self._last_resort_stairs_turn = self.blstats.time
                    dive._retreat_blocked_until = self.blstats.time + 20
                    self.move('>')
                    return
                # (not the Valley's '<': it leads to the castle's east edge, with no way back down to Gehennom)
                castle_key = dive.castle.castle_key
                if stairs_ok and here in G.STAIR_UP and self.blstats.depth > 1 and \
                        level.dungeon_number != Level.SOKOBAN and not dive.in_valley() and \
                        not (jf_config.CASTLE_NO_RETREAT and castle_key is not None and level.key() == castle_key):
                    yield True
                    self.log('LAST RESORT: up the stairs')
                    self._last_resort_stairs_turn = self.blstats.time
                    self.move('<')
                    return
                # DESPERATE_PRAYER_GAP: a prayer that may come too soon beats dying. pray.c fixes critically low HP
                # while the timeout is <= 200; after a successful prayer it is rnz(350), which leaves ~62% of
                # prayers working 250 turns later and ~77% after 400 (the usual rule waits 500: ~87%). A failure
                # costs Luck -3 and an angry god -- about what dying here costs. In base/base2, 14 deaths came
                # 250-499 turns after a prayer that had worked, 7 of them with this last resort firing.
                # (only in the dive below Dlvl 4: the grind's jackals at low HP are usually beaten, and a failed
                # prayer there ruins the game; and only when Elbereth can't help: against trolls a 12-HP XL8 sat
                # it out on Elbereth in 8 of 10 harness games -- dive-mid-desperate, off arm)
                # (hurt while standing on Elbereth counts too: wands and breath ignore it -- base2-jf14 s12 read
                # five unknown scrolls at 4 HP while a bugbear zapped magic missiles, 364 turns after a prayer)
                elbereth_futile = any(dive._ignores_elbereth(m[3]) for m in adjacent) or \
                    self.character.prop.blind or not self.can_engrave() or \
                    ((self.inventory.engraving_below_me or '').lower() == 'elbereth' and self._hurt_recently(2))
                if jf_config.DESPERATE_PRAYER_GAP and not self.prayer_failed and level.dungeon_number != 1 and \
                        dive.diving and self.blstats.depth >= 5 and elbereth_futile and \
                        self.is_safe_to_pray(jf_config.DESPERATE_PRAYER_GAP):
                    yield True
                    gap = None if self.last_prayer_turn is None else self.blstats.time - self.last_prayer_turn
                    self.log(f'LAST RESORT: desperate prayer (gap {gap})')
                    self.pray()
                    return
                # top-level items only: a wand inside a bag has no inventory letter, and zapping one left the
                # 'What do you want to zap?' prompt looping at 9 HP until a goblin finished the XL6
                items = list(self.inventory.items)
                _, my, mx, _, _ = adjacent[0]
                # in Minetown (or with the Watch in view) a ray or an area scroll can hit the Watch (a
                # scroll of earth dropped a boulder on a watch captain): only the potions, which touch us
                watch = combat.fight_heur.missiles_risk_the_watch(self)
                for item in items:
                    if item.category == nh.WAND_CLASS and not item.is_unambiguous() and not watch and \
                            not self.inventory.is_known_empty(item) and item.comment != 'EMPT':
                        # LR_WAND_ONCE: an attack wand names itself when zapped at a monster (zap.c learn_it),
                        # so one still unknown after a zap does nothing useful here -- a castle guard game zapped
                        # the same uranium wand 7 times at 2-3 HP and never reached its potions
                        if jf_config.LR_WAND_ONCE and item.glyphs[0] in self._last_resort_zapped:
                            continue
                        yield True
                        self._last_resort_zapped.add(item.glyphs[0])
                        self.log(f'LAST RESORT: zapping unknown {item.text!r}')
                        self.zap(item, self.calc_direction(y, x, my, mx))
                        return
                for item in items:
                    if item.category == nh.POTION_CLASS and not item.is_unambiguous():
                        yield True
                        self.log(f'LAST RESORT: quaffing unknown {item.text!r}')
                        self.inventory.quaff(item)
                        return
                for item in items:
                    if item.category == nh.SCROLL_CLASS and not item.is_unambiguous() and not watch:
                        yield True
                        self.log(f'LAST RESORT: reading unknown {item.text!r}')
                        with self.atom_operation():
                            self.step(A.Command.READ)
                            self.type_text(self.inventory.items.get_letter(item))
                        return

        # if self.inventory.engraving_below_me.lower() != 'elbereth' and self.can_engrave() and \
        #         (self.blstats.hitpoints < 1 / 5 * self.blstats.max_hitpoints or self.blstats.hitpoints < 5):
        #     yield True
        #     self.engrave('Elbereth')
        #     for _ in range(8):
        #         if self.inventory.engraving_below_me.lower() != 'elbereth':
        #             break
        #         self.direction('.')
        #     return

        yield False

    @utils.debug_log('eat_from_inventory')
    @Strategy.wrap
    def eat_from_inventory(self):
        if self.blstats.hunger_state < Hunger.HUNGRY:
            yield False
        # hypothesis: prayer is the main food source, but a hunger prayer made on the bare ~900-1100 turn
        # starvation cycle comes too soon in ~4-7% of cases -- the hunger is not fixed and the god gets angry,
        # so the character usually starves or dies while fainting (a common cause of early deaths); keeping
        # the stored food as a reserve that is eaten only when a prayer would be risky (and praying already
        # when Weak if it is safe) should make those failures rarer and raise progression for every character
        # the dive eats what it carries as soon as it is Hungry: its prayers are for HP emergencies
        # (a dive fainted at Dlvl 6 and died fighting); the tour keeps DT6A's hoard-and-pray policy
        diving = self.global_logic.dive.diving
        if not diving and not self.prayer_failed and self.blstats.hunger_state < Hunger.FAINTING and \
                (self.blstats.hunger_state == Hunger.HUNGRY or self.is_safe_to_pray(self.SAFE_HUNGER_PRAYER_GAP)) \
                and not (self.blstats.hunger_state >= Hunger.WEAK and self._eat_before_praying()):
            yield False
        for item in self.edible_carried_food():
            yield True
            self.inventory.eat(item)
            return
        yield False

    @utils.debug_log('were_unload')
    @Strategy.wrap
    def were_unload(self):
        """In a were form, drop what the form can't carry so we can eat (LYCAN_FIXES).

        polyself.c polymon() keeps the form for rn1(500, 500) turns unless its HP runs out, and weight_cap()
        scales with the form's corpse weight: a wererat (cwt 40) carries ~27, so the whole pack Overloads
        it ('You can't even move a handspan with this load!') and eat.c refuses to eat above Strained ('You
        can't do that while carrying so much stuff'). Public s4 (5 food items) and jf16 s11 starved that
        way. arrange_items() is off while polymorphed, so nothing is picked up again until we change back;
        then the usual pickup logic collects the pile."""
        # LYCAN_UNLOAD_ALWAYS: Overtaxed/Overloaded in a were form whatever the hunger: an Overloaded rat can't
        # move at all, and the bot's move attempts ('You collapse under your load') panicked for ~1100 turns
        # at one square until it died on Dlvl 2 (robustness, base3arm-jf25 s1: only Hungry, so no unload)
        always = jf_config.LYCAN_UNLOAD_ALWAYS
        if not jf_config.LYCAN_FIXES or not self.character.prop.polymorph or \
                self.blstats.carrying_capacity < 4 or (self.blstats.hunger_state < Hunger.WEAK and not always):
            yield False
        food = self.edible_carried_food()
        if not food and not always:
            yield False
        keep = set(id(i) for i in food)
        to_drop = [item for item in self.inventory.items
                   if id(item) not in keep and item.can_be_dropped_from_inventory() and
                   item.category != nh.COIN_CLASS and
                   not (item.is_container() and any(id(i) in keep for i in flatten_items([item]))) and
                   # CASTLE_POLY: the wand of polymorph is our way over the castle moat (castle_power)
                   not (jf_config.CASTLE_POLY and item.is_wand() and item.is_unambiguous() and
                        item.object.name == 'polymorph')]
        if not to_drop:
            yield False
        yield True
        self.log(f'LYCAN were form Overloaded (hunger {self.blstats.hunger_state}): dropping {len(to_drop)} items')
        self.inventory.drop(to_drop, smart=False)

    _TIN_SMELL = re.compile(r'It smells like (?:the )?([A-Za-z -]+?)\.')
    _BAD_TIN_WORDS = ('cockatrice', 'chickatrice', 'Medusa', 'green slime', 'were', 'little dog', 'large dog',
                      'dogs', 'kitten', 'housecat', 'large cat', 'chameleon', 'doppelganger', 'sandestin',
                      'genetic engineer')
    _CANNIBAL_WORDS = {Character.HUMAN: ('human',),
                       Character.DWARF: ('dwar',), Character.ELF: ('elf', 'elves', 'Woodland-el', 'Green-el',
                                                                    'Grey-el', 'Elvenking'),
                       Character.GNOME: ('gnom',), Character.ORC: ()}

    def _bad_tin(self, message):
        m = self._TIN_SMELL.search(message)
        if m is None:
            return False
        what = m.group(1)
        if any(w in what for w in self._BAD_TIN_WORDS):
            return True
        # a lycanthrope's own were family is cannibalism in a tin as well (eat.c maybe_cannibal)
        if jf_config.LYCAN_FIXES and \
                any(name in what for name in ('jackal', 'fox', 'coyote', 'rat', 'wolf', 'warg')
                    if any(name in f for f in self.character.were_family())):
            return True
        return any(w in what for w in self._CANNIBAL_WORDS.get(self.character.race, ()))

    def carried_food_nutrition(self):
        """Nutrition of edible_carried_food (corpses by their monster, unidentified items as 0)."""
        total = 0
        for item in self.edible_carried_food():
            if item.is_corpse():
                total += getattr(MON.permonst(item.monster_id + nh.GLYPH_MON_OFF), 'cnutrit', 0) * item.count
            elif item.is_unambiguous():
                total += getattr(item.object, 'nutrition', 0) * item.count
        return total

    def edible_carried_food(self):
        """What eat_from_inventory eats: food, but not wolfsbane or corpses other than lizard/lichen."""
        return [item for item in flatten_items(self.inventory.items)
                if item.category == nh.FOOD_CLASS and item.objs[0].name != 'sprig of wolfsbane' and
                (not item.is_corpse() or
                 item.monster_id in [MON.from_name(n) - nh.GLYPH_MON_OFF for n in ['lizard', 'lichen']])]

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

            # pray: lycanthropy is a nuisance, not an emergency -- wait for the gap where prayers fail ~1%
            # (at the old 500-turn gap rnz(350) leaves ~12% of these too soon)
            # pray.c fixes the worst trouble first: in a were form at <= 5 HP (or 1/7) that is TROUBLE_HIT
            # (7), above TROUBLE_LYCANTHROPE (6), and with Luck 0 only half the prayers fix more than one
            # trouble -- wait until the form's HP is back up, or it dies and we rehumanize
            if jf_config.LYCAN_FIXES and self.character.prop.polymorph and \
                    (self.blstats.hitpoints <= 5 or self.blstats.hitpoints * 7 <= self.blstats.max_hitpoints):
                yield False
            # Hungry (minor trouble, not fixed at Luck 0): a cure prayer now restarts the prayer timeout just
            # before the hunger runs out, so the next ~1000 turns are spent Weak/Fainting with no prayer (gc-lf2
            # public s4 cured at T10201 while Hungry, fainted at gap 445 among werejackals and died). Wait for
            # Weak: that hunger prayer fixes starvation and, half the time, the lycanthropy too (pray.c pleased)
            # (food or not: a lichen corpse let a Hungry cure prayer through at gap 1204 with ~120 nutrition
            # left; Weak 75 turns later, a Fainting hold until the 1400 gap, dead at XL 6 -- grind-food's
            # t2smoke public s0). The dive eats when Hungry, so there the cure may come early.
            # Not Hungry is no better: that prayer fixes no hunger either and restarts the timer, so with ~500
            # nutrition left the next hunger prayer comes after ~600+ turns of Fainting (grind-food). Only with
            # a food ration's worth carried is the early cure free.
            if jf_config.LYCAN_CURE_WAIT and self.blstats.hunger_state < Hunger.WEAK and \
                    not self.global_logic.dive.diving and self.inventory.carried_nutrition() < 800:
                yield False
            if self.is_safe_to_pray(jf_config.WEAK_PRAYER_GAP):
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
            self.all_panics.append(exc)
            self.all_panics = self.all_panics[-50:]
            if jf_log.enabled():
                import traceback
                deep = type(exc).__name__ == 'AgentHang'
                tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__,
                                                        limit=None if deep else -6))
                self.log(f'PANIC {type(exc).__name__}: {str(exc)[:300]}\n{tb[-(12000 if deep else 1500):]}')
            self._note_repeated_panic(exc)
            if self.verbose:
                print(f'PANIC!!!! : {exc}')

    def _note_repeated_panic(self, exc):
        """The same panic over and over (a move that never lands, a target it can't reach) can
        burn the rest of the game while turns trickle by. Break the loop: forbid the square the
        failing move aims at, or walk randomly for a few steps."""
        signature = f'{type(exc).__name__}:{str(exc)[:120]}'
        if signature == self._last_panic_signature:
            self._panic_repeats += 1
        else:
            self._last_panic_signature = signature
            self._panic_repeats = 1
        if self._panic_repeats < 25:
            return
        self._panic_repeats = 0
        m = re.search(r'expected \((\d+), (\d+)\)', str(exc))
        if m is not None:
            y, x = int(m.group(1)), int(m.group(2))
            try:
                if jf_config.TEMP_FORBID:
                    # TEMP_FORBID: the moves usually fail because of our own state -- caught in a bear trap, stuck in
                    # a web, a were form overloaded -- not because of the square: a permanent forbid boxed a
                    # base2-jf25 s11 dive in on Mines 6 for 5000+ turns after it wriggled out of a bear trap
                    self.current_level().forbidden_until[y, x] = self.blstats.time + 100
                else:
                    self.current_level().forbidden[y, x] = True
            except Exception:
                pass
            self.log(f'panic loop: forbidding {(y, x)} ({signature})')
        else:
            self.log(f'panic loop: random walk ({signature})')
        self._random_walk_steps = 8

    def _random_walk(self):
        y, x = self.blstats.y, self.blstats.x
        level = self.current_level()
        options = [(ny, nx) for ny, nx in self.neighbors(y, x)
                   if level.walkable[ny, nx] and not self.monster_tracker.monster_mask[ny, nx]]
        if options:
            ny, nx = options[self.rng.randint(len(options))]
            self.direction(self.calc_direction(y, x, ny, nx))
        else:
            self.step(A.Command.SEARCH)

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

                    # a new game starts on Dlvl 1's up stairs (they leave the dungeon); a dev scenario
                    # (dev/scenario.py, JF_SCENARIO) starts wherever the harness put us
                    if not self.resumed_game and not jf_scenario.active():
                        self.current_level().stair_destination[self.blstats.y, self.blstats.x] = \
                            ((Level.PLANE, 1), (None, None))  # TODO: check level num
                    try:
                        self.character.parse()
                    except Exception:
                        # a fresh agent taking over mid-game: the identity can't change, reuse it
                        prev = getattr(self, 'previous_character', None)
                        if prev is None:
                            raise
                        for field in ('role', 'race', 'alignment', 'gender', 'self_glyph'):
                            setattr(self.character, field, getattr(prev, field))
                    self.character.parse_enhance_view()
                    # self.character.parse_spellcast_view()
                    self.step(A.Command.AUTOPICKUP)
                    if 'Autopickup: ON' in self.message:
                        self.step(A.Command.AUTOPICKUP)
                    init_finished = True
            except BaseException as e:
                self.handle_exception(e)

            assert init_finished
            jf_scenario.apply(self)

            last_step = self.step_count
            inactivity_counter = 0
            forced_turns = 0
            turn_after_forced = None
            while 1:
                inactivity_counter += 1
                if self.step_count != last_step:
                    inactivity_counter = 0

                if self._random_walk_steps > 0:
                    self._random_walk_steps -= 1
                    try:
                        self.step(A.Command.ESC)
                        self._random_walk()
                    except BaseException as e:
                        self.handle_exception(e)
                    last_step = self.step_count
                    continue

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
