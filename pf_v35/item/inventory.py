import contextlib
import re
from functools import partial
from itertools import chain

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from pf_v35 import objects as O, utils
from pf_v35 import power
from pf_v35.character import Character
from pf_v35.exceptions import AgentPanic
from pf_v35.glyph import G, MON, Hunger
from pf_v35 import jf_config
from pf_v35.item import ItemManager, Item, ContainerContent, check_if_triggered_container_trap, \
    find_equivalent_item, flatten_items
from pf_v35.item.inventory_items import InventoryItems
from pf_v35.strategy import Strategy


class Inventory:
    _name_to_category = {
        'Amulets': nh.AMULET_CLASS,
        'Armor': nh.ARMOR_CLASS,
        'Comestibles': nh.FOOD_CLASS,
        'Coins': nh.COIN_CLASS,
        'Gems/Stones': nh.GEM_CLASS,
        'Potions': nh.POTION_CLASS,
        'Rings': nh.RING_CLASS,
        'Scrolls': nh.SCROLL_CLASS,
        'Spellbooks': nh.SPBOOK_CLASS,
        'Tools': nh.TOOL_CLASS,
        'Weapons': nh.WEAPON_CLASS,
        'Wands': nh.WAND_CLASS,
        'Boulders/Statues': nh.ROCK_CLASS,
        'Chains': nh.CHAIN_CLASS,
        'Iron balls': nh.BALL_CLASS,
    }

    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self.agent)
        self.items = InventoryItems(self.agent)

        self._previous_blstats = None
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None

        self.skip_engrave_counter = 0
        self.scare_labels = set()     # scroll labels known to be scare monster in this game (power.note_dust_prompt)
        self.dropped_scrolls = set()  # (level key, (y, x), appearance) of scrolls we dropped: never picked up again
        self._sell_tested = set()     # (level key, glyph) already offered to this level's shopkeeper (SELL_PRICE_ID)
        self.empty_wands = set()  # inventory texts of wands that answered "Nothing happens"
        self.multi_container_squares = set()  # (dungeon, level, y, x) where #loot asks 'Loot which containers?'
        self._container_failures = {}         # (dungeon, level, y, x) -> failed use_container attempts
        self.unreachable_items_until = {}     # (dungeon, level, y, x) -> turn: items at a pit bottom out of reach
        self._buy_food_blocked = {}           # (dungeon, level, y, x) -> (failed walks, skip until turn): BUY_FOOD_GIVEUP

    def is_known_empty(self, item):
        return item.text in self.empty_wands

    # ---- scare monster (jf_config.SCARE_KEEP, power.py): picked up a second time it turns to dust

    @staticmethod
    def _scroll_key(item):
        """A scroll's appearance from its text ('labeled FOO', 'unlabeled', 'of teleportation')."""
        text = item.text or ''
        m = re.search(r'labeled ([A-Z0-9 ]+)', text)
        if m:
            return 'labeled ' + m.group(1).strip()
        if 'unlabeled' in text:
            return 'unlabeled'
        m = re.search(r'scrolls? of ([a-z ]+)', text)
        return 'of ' + m.group(1).strip() if m else text

    def _droppable(self, item):
        """arrange_items may drop it. With SCARE_KEEP a carried scroll that may be scare monster stays: a heavy
        armour swap dropped all light loot and picked it up again (base-jf26 s14: 'The scroll turns to dust')."""
        if not item.can_be_dropped_from_inventory():
            return False
        return not (power.keep_scroll(item) and item in self.items.all_items)

    def dropped_here(self, item, pos=None):
        """A scroll that may be scare monster which we dropped on this square: never pick it up again."""
        # (the set is filled only by SCARE_KEEP drops and by castle_power's arrival drill)
        if not self.dropped_scrolls or not power.is_scare_candidate(item):
            return False
        pos = pos if pos is not None else (self.agent.blstats.y, self.agent.blstats.x)
        return (self.agent.current_level().key(), (int(pos[0]), int(pos[1])), self._scroll_key(item)) in \
            self.dropped_scrolls

    def _note_dropped(self, items, counts, force=False):
        if not (jf_config.SCARE_KEEP or force):
            return
        here = (self.agent.current_level().key(), (int(self.agent.blstats.y), int(self.agent.blstats.x)))
        for item, count in zip(items, counts):
            if count and power.is_scare_candidate(item):
                self.dropped_scrolls.add(here + (self._scroll_key(item),))

    def set_unknown_below_me(self):
        """Stand-in when the square can't be parsed: pretend nothing useful is here."""
        if self.items_below_me is None:
            self.items_below_me = []
            self.letters_below_me = []
        if self.engraving_below_me is None:
            self.engraving_below_me = ''

    def on_panic(self):
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None
        self._previous_blstats = None

        self.item_manager.on_panic()
        self.items.on_panic()

    def update(self):
        self.item_manager.update()
        self.items.update()

        if self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number) or \
                (self.engraving_below_me is None or self.engraving_below_me.lower() == 'elbereth'):
            assume_appropriate_message = self._previous_blstats is not None and not self.engraving_below_me

            self._previous_blstats = self.agent.blstats
            self.items_below_me = None
            self.letters_below_me = None
            self.engraving_below_me = None

            self.get_items_below_me(assume_appropriate_message=assume_appropriate_message)

        assert self.items_below_me is not None and self.letters_below_me is not None and self.engraving_below_me is not None

    @contextlib.contextmanager
    def panic_if_items_below_me_change(self):
        old_items_below_me = self.items_below_me
        old_letters_below_me = self.letters_below_me

        def f(self):
            if (
                    [(l, i.text) for i, l in zip(old_items_below_me, old_letters_below_me)] !=
                    [(l, i.text) for i, l in zip(self.items_below_me, self.letters_below_me)]
            ):
                raise AgentPanic('items below me changed')

        fun = partial(f, self)

        self.agent.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.agent.on_update
            self.agent.on_update.pop(self.agent.on_update.index(fun))

    ####### ACTIONS

    def wield(self, item, smart=True):
        if smart:
            if item is not None:
                item = self.move_to_inventory(item)

        if item is None:  # fists
            letter = '-'
        else:
            letter = self.items.get_letter(item)

        if item is not None and item.equipped:
            return True

        if self.agent.character.prop.polymorph:
            # TODO: depends on kind of a monster
            return False

        if (self.items.main_hand is not None and self.items.main_hand.status == Item.CURSED) or \
                (item is not None and item.objs[0].bi and self.items.off_hand is not None):
            return False

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            if "Don't be ridiculous" in self.agent.message:
                return False
            assert 'What do you want to wield' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            if 'You cannot wield a two-handed sword while wearing a shield.' in self.agent.message or \
                    'You cannot wield a two-handed weapon while wearing a shield.' in self.agent.message or \
                    ' welded to your hand' in self.agent.message:
                return False
            assert re.search(r'(You secure the tether\.  )?([a-zA-z] - |welds?( itself| themselves| ) to|'
                             r'You are already wielding that|You are empty handed|You are already empty handed)', \
                             self.agent.message), (self.agent.message, self.agent.popup)

        return True

    def wear(self, item, smart=True):
        assert item is not None

        if smart:
            item = self.move_to_inventory(item)
            # TODO: smart should be more than that (taking off the armor for shirts, etc)
        letter = self.items.get_letter(item)

        if item.equipped:
            return True

        for i in self.items:
            assert not isinstance(i, O.Armor) or i.sub != item.sub or not i.equipped, (i, item)

        with self.agent.atom_operation():
            self.agent.step(A.Command.WEAR)
            if "Don't even bother." in self.agent.message:
                return False
            assert 'What do you want to wear?' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            assert 'You finish your dressing maneuver.' in self.agent.message or \
                   'You are now wearing ' in self.agent.message or \
                   'Your foot is trapped!' in self.agent.message, self.agent.message

        return True

    def takeoff(self, item):
        # TODO: smart

        assert item is not None and item.equipped, item
        letter = self.items.get_letter(item)
        assert item.status != Item.CURSED, item

        equipped_armors = [i for i in self.items if i.is_armor() and i.equipped]
        assert item in equipped_armors

        with self.agent.atom_operation():
            self.agent.step(A.Command.TAKEOFF)

            is_take_off_message = lambda: \
                'You finish taking off ' in self.agent.message or \
                'You were wearing ' in self.agent.message or \
                'You feel that monsters no longer have difficulty pinpointing your location.' in self.agent.message

            if len(equipped_armors) > 1:
                if is_take_off_message():
                    raise AgentPanic('env did not ask for the item to takeoff')
                assert 'What do you want to take off?' in self.agent.message, self.agent.message
                self.agent.type_text(letter)
            if 'It is cursed.' in self.agent.message or 'They are cursed.' in self.agent.message:
                return False
            assert is_take_off_message(), self.agent.message

        return True

    def use_container(self, container, items_to_put, items_to_take, items_to_put_counts=None,
                      items_to_take_counts=None):
        assert container in self.items.all_items or container in self.items_below_me
        assert all((item in self.items.all_items for item in items_to_put))
        assert all((item in container.content.items for item in items_to_take))
        assert container.is_container()
        assert len(items_to_take) - len(items_to_put) <= self.items.free_slots()  # TODO: take counts into consideration
        assert not container.content.locked, container

        def gen():
            if ' vanished!' in self.agent.message:
                self.item_manager.container_contents.pop(container.container_id)
                raise AgentPanic('some items from the container vanished')
            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '
            assert 'You have no free hand.' not in self.agent.single_message, 'TODO: handle it'
            assert 'Do what with ' in self.agent.single_popup[0]
            if items_to_take and any(' is empty' in line for line in self.agent.single_popup[:1] +
                                     [self.agent.single_message]):
                # our record of its contents is stale (the menu then has no 'take out' entry and the
                # prompt loop asserted ~275 times in one run)
                self.item_manager.container_contents.pop(container.container_id, None)
                yield A.Command.ESC
                raise AgentPanic('container is empty')
            if items_to_put and items_to_take:
                yield 'r'
            elif items_to_put and not items_to_take:
                yield 'i'
            elif not items_to_put and items_to_take:
                yield 'o'
            else:
                assert 0
            if items_to_put:
                if 'Put in what type of objects?' in self.agent.single_popup[0]:
                    yield from 'a\r'
                assert 'Put in what?' in self.agent.single_popup[0], (
                    self.agent.single_message, self.agent.single_popup)
                yield from self._select_items_in_popup(items_to_put, items_to_put_counts)
            if items_to_take:
                while not self.agent.single_popup or self.agent.single_popup[0] not in [
                    'Take out what type of objects?', 'Take out what?']:
                    assert ' inside, you are blasted by a ' not in self.agent.message, self.agent.message
                    assert self.agent.single_message or self.agent.single_popup, (self.agent.message, self.agent.popup)
                    yield ' '
                if self.agent.single_popup[0] == 'Take out what type of objects?':
                    yield from 'a\r'
                assert 'Take out what?' in self.agent.single_popup[0]
                yield from self._select_items_in_popup(items_to_take, items_to_take_counts)

                if self.agent._observation['misc'][2]:
                    yield ' '
                while 'You have ' in self.agent.single_message and ' removing ' in self.agent.single_message and \
                        'Continue? [ynq] (q)' in self.agent.single_message:
                    yield 'y'

        below = container in self.items_below_me and container not in self.items.all_items
        try:
            self._use_container_steps(container, gen)
        except AgentPanic:
            # CONTAINER_LOOP_FIX: a floor container whose take-out menu never matches our record ('no popup, but
            # some items were not selected yet') was retried 45,453 times in one robustness guard game (457k steps,
            # then the driver's hang guard). After 3 failures on a square, leave its containers alone.
            if jf_config.CONTAINER_LOOP_FIX and below:
                here = self._here()
                self._container_failures[here] = self._container_failures.get(here, 0) + 1
                if self._container_failures[here] >= 3:
                    self.multi_container_squares.add(here)
            raise

        for item in chain(self.items.all_items, self.items_below_me):
            if item.is_container() and item.container_id == container.container_id:
                self.check_container_content(item)

    def _use_container_steps(self, container, gen):
        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in check_container_content
            if container in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                assert "You can't do that while carrying so much stuff." not in self.agent.message, self.agent.message
                self.agent.step(self.items.get_letter(container), gen())
            elif container in self.items_below_me:
                self.agent.step(A.Command.LOOT)
                while True:
                    self._escape_multi_container_menu()
                    assert 'Loot in what direction?' not in self.agent.message
                    if "You don't find anything here to loot." in self.agent.message:
                        raise AgentPanic('no container to loot')
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                                                              position=(
                                                                  *self.agent.current_level().key(),
                                                                  self.agent.blstats.y,
                                                                  self.agent.blstats.x))
                    if it.container_id == container.container_id:
                        break
                    self.agent.step('n')

                self.agent.step('y', gen())
            else:
                assert 0

    def _here(self):
        return (*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x)

    def _escape_multi_container_menu(self):
        """With several containers on the square #loot shows a 'Loot which containers?' menu that the
        prompt loop doesn't handle (one game hit it ~2300 times, 100k wasted steps): leave it alone."""
        if 'Loot which containers?' in self.agent.popup:
            self.multi_container_squares.add(self._here())
            self.agent.step(A.Command.ESC)
            raise AgentPanic('several containers here')

    def check_container_content(self, item):
        assert item.is_possible_container() or item.is_container()
        assert item in self.items.all_items or item in self.items_below_me

        is_bag_of_tricks = False
        if item.content is not None:
            content = item.content
            content.reset()
        else:
            content = ContainerContent()

        def gen():
            nonlocal content, is_bag_of_tricks

            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '

            if 'It develops a huge set of teeth and bites you!' in self.agent.single_message:
                is_bag_of_tricks = True
                return

            if 'Hmmm, it turns out to be locked.' in self.agent.single_message or 'It is locked.' in self.agent.single_message:
                content.locked = True
                yield A.Command.ESC
                return

            if check_if_triggered_container_trap(self.agent.single_message):
                self.agent.stats_logger.log_event('triggered_undetected_trap')
                raise AgentPanic('triggered trap while looting')

            if 'You have no hands!' in self.agent.single_message or \
                    'You have no free hand.' in self.agent.single_message:
                return

            if ' vanished!' in self.agent.message:
                raise AgentPanic('some items from the container vanished')

            if 'cat' in self.agent.message and ' inside the box is ' in self.agent.message:
                raise AgentPanic('encountered a cat in a box')

            assert self.agent.single_popup, (self.agent.single_message)
            if '\no - ' not in '\n'.join(self.agent.single_popup):
                # ':' sometimes doesn't display items correctly if there's >= 22 items (the first page isn't shown)
                yield ':'
                if ' is empty' in self.agent.single_message:
                    return
                # if self.agent.single_popup and 'Contents of ' in self.agent.single_popup[0]:
                #     for text in self.agent.single_popup[1:]:
                #         if not text:
                #             continue
                #         content.items.append(self.item_manager.get_item_from_text(text, position=None))
                #     return
                assert 0, (self.agent.single_message, self.agent.single_popup)

            yield from 'o'
            if ' is empty' in self.agent.single_message and not self.agent.single_popup:
                return
            if self.agent.single_popup and self.agent.single_popup[0] == 'Take out what type of objects?':
                yield from 'a\r'
            if self.agent.single_popup and 'Take out what?' in self.agent.single_popup[0]:
                category = None
                while self.agent._observation['misc'][2]:
                    yield ' '
                assert self.agent.popup.count('Take out what?') == 1, self.agent.popup
                for text in self.agent.popup[self.agent.popup.index('Take out what?') + 1:]:
                    if not text:
                        continue
                    if text in self._name_to_category:
                        category = self._name_to_category[text]
                        continue
                    assert category is not None
                    assert text[1:4] == ' - '
                    text = text[4:]
                    content.items.append(self.item_manager.get_item_from_text(text, category=category, position=None))
                return

            assert 0, (self.agent.single_message, self.agent.single_popup)

        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in use_container
            if item in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                if "You can't do that while carrying so much stuff." in self.agent.message:
                    return  # TODO: is not changing the content in this case a good way to handle this?
                self.agent.step(self.items.get_letter(item), gen())
                if 'You have no hands!' in self.agent.message:
                    return
            else:
                self.agent.step(A.Command.LOOT)
                while True:
                    if "You don't find anything here to loot." in self.agent.message:
                        # a 'possible container' that isn't one: skip the square (a jf21 game retried the
                        # #loot here until the turn-inactivity guard fired, 31 times)
                        self.multi_container_squares.add(self._here())
                        raise AgentPanic('no container below me')
                    self._escape_multi_container_menu()
                    assert 'There is ' in self.agent.message and ', loot it?' in self.agent.message, self.agent.message
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                                                              position=(
                                                                  *self.agent.current_level().key(),
                                                                  self.agent.blstats.y,
                                                                  self.agent.blstats.x))
                    if (item.container_id is not None and it.container_id == item.container_id) or \
                            (item.container_id is None and item.text == it.text):
                        break
                    self.agent.step('n')
                self.agent.step('y', gen())

            if is_bag_of_tricks:
                assert item.content is None
                raise AgentPanic('bag of tricks bites')

            if item in self.items.all_items and item.comment != item.container_id:
                self.call_item(item, item.container_id)

            if item.content is None:
                assert item.container_id is not None
                assert item.container_id not in self.item_manager.container_contents
                self.item_manager.container_contents[item.container_id] = content
                item.content = content

            # TODO: make it more elegant
            if len(item.glyphs) == 1 and item.glyphs[0] not in self.item_manager._is_not_bag_of_tricks:
                self.item_manager._is_not_bag_of_tricks.add(item.glyphs[0])
                self.item_manager.update_possible_objects(item)

    def _select_items_in_popup(self, items, counts=None):
        assert counts is None or len(counts) == len(items)
        items = list(items)
        while 1:
            if not self.agent.single_popup:
                raise AgentPanic('no popup, but some items were not selected yet')
            for line_i in range(len(self.agent.single_popup)):
                line = self.agent.single_popup[line_i]
                if line[1:4] != ' - ':
                    continue

                for item in items:
                    if item.text != line[4:]:
                        continue

                    i = items.index(item)
                    letter = line[0]

                    if counts is not None and counts[i] != item.count:
                        yield from str(counts[i])
                    yield letter

                    items.pop(i)
                    if counts is not None:
                        count = counts.pop(i)
                    else:
                        count = None
                    break

                if not items:
                    yield '\r'
                    return

            yield ' '
        assert not items

    def get_items_below_me(self, assume_appropriate_message=False):
        with self.agent.panic_if_position_changes():
            with self.agent.atom_operation():
                if not assume_appropriate_message:
                    self.agent.step(A.Command.LOOK)
                elif 'Things that are here:' in self.agent.popup or \
                        re.search('There are (several|many) objects here\.', self.agent.message):
                    # LOOK is necessary even when 'Things that are here' popup is present for some very rare cases
                    self.agent.step(A.Command.LOOK)

                if 'Something is ' in self.agent.message and 'You read: "' in self.agent.message:
                    index = self.agent.message.index('You read: "') + len('You read: "')
                    assert '"' in self.agent.message[index:]
                    engraving = self.agent.message[index: index + self.agent.message[index:].index('"')]
                    self.engraving_below_me = engraving
                else:
                    self.engraving_below_me = ''

                if 'Things that are here:' not in self.agent.popup and 'There is ' not in '\n'.join(self.agent.popup):
                    if 'You see no objects here.' in self.agent.message:
                        items = []
                        letters = []
                    elif 'You see here ' in self.agent.message:
                        item_str = self.agent.message[self.agent.message.index('You see here ') + len('You see here '):]
                        item_str = item_str[:item_str.index('.')]
                        items = [self.item_manager.get_item_from_text(item_str,
                                                                      position=(*self.agent.current_level().key(),
                                                                                self.agent.blstats.y,
                                                                                self.agent.blstats.x))]
                        letters = [None]
                    else:
                        items = []
                        letters = []
                else:
                    self.agent.step(A.Command.PICKUP)  # FIXME: parse LOOK output, add this fragment to pickup method
                    if 'Pick up what?' not in self.agent.popup:
                        if 'You cannot reach the bottom of the pit.' in self.agent.message or \
                                'You cannot reach the bottom of the abyss.' in self.agent.message or \
                                'You cannot reach the floor.' in self.agent.message or \
                                'There is nothing here to pick up.' in self.agent.message or \
                                ' solidly fixed to the floor.' in self.agent.message or \
                                'You read:' in self.agent.message or \
                                "You don't see anything in here to pick up." in self.agent.message or \
                                'You cannot reach the ground.' in self.agent.message or \
                                "You don't feel anything in here to pick up." in self.agent.message:
                            items = []
                            letters = []
                        elif re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                                'Continue?' in self.agent.message:
                            # only one object here can be picked up (e.g. the iron chain attached to the
                            # ball is skipped), so PICKUP tries to lift it at once. It is too heavy anyway.
                            self.agent.step(A.Command.ESC)
                            items = []
                            letters = []
                        else:
                            assert 0, (self.agent.message, self.agent.popup)
                    else:
                        lines = self.agent.popup[self.agent.popup.index('Pick up what?') + 1:]
                        category = None
                        items = []
                        letters = []
                        for line in lines:
                            if line in self._name_to_category:
                                category = self._name_to_category[line]
                                continue
                            assert line[1:4] == ' - ', line
                            letter, line = line[0], line[4:]
                            letters.append(letter)
                            items.append(self.item_manager.get_item_from_text(line, category,
                                                                              position=(
                                                                                  *self.agent.current_level().key(),
                                                                                  self.agent.blstats.y,
                                                                                  self.agent.blstats.x)))

                self.items_below_me = items
                self.letters_below_me = letters
                return items

    def pickup(self, items, counts=None):
        # TODO: if polyphormed, sometimes 'You are physically incapable of picking anything up.'
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert len(items) > 0
        assert all(map(lambda item: item in self.items_below_me, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.letters_below_me[self.items_below_me.index(item)] for item in items]
        screens = [max(self.letters_below_me[:self.items_below_me.index(item) + 1].count('a') - 1, 0) for item in items]

        with self.panic_if_items_below_me_change():
            self.get_items_below_me()

        one_item = len(self.items_below_me) == 1
        with self.agent.atom_operation():
            if one_item:
                assert all((s in [0, None] for s in screens))
                self.agent.step(A.Command.PICKUP)
                drop_count = items[0].count - counts[0]
            else:
                text = ' '.join((
                    ''.join([(str(count) if item.count != count else '') + letter
                             for letter, item, count, screen in zip(letters, items, counts, screens)
                             if count != 0 and screen == current_screen])
                    for current_screen in range(max(screens) + 1)))
                self.agent.step(A.Command.PICKUP, iter(list(text) + [A.MiscAction.MORE]))

            while re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                    'Continue?' in self.agent.message:
                self.agent.type_text('y')
            if 'You cannot reach the bottom of the pit' in self.agent.message:
                # standing at a pit's edge: its items are out of reach until we are in it; the gatherer
                # retried without the clock moving until the 'turn inactivity' guard fired (54 times)
                self.unreachable_items_until[self._here()] = self.agent.blstats.time + 300
                raise AgentPanic('items at the bottom of a pit are out of reach')
            if one_item and drop_count:
                letter = re.search(r'([a-zA-Z$]) - ', self.agent.message)
                assert letter is not None, self.agent.message
                letter = letter[1]

        if one_item and drop_count:
            self.drop(self.items.all_items[self.items.all_letters.index(letter)], drop_count, smart=False)

        self.get_items_below_me()

        return True

    def drop(self, items, counts=None, smart=True):
        if smart:
            items = self.move_to_inventory(items)

        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert all(map(lambda x: isinstance(x, (int, np.int32, np.int64)), counts)), list(map(type, counts))
        assert len(items) > 0
        assert all(map(lambda item: item in self.items.all_items, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.items.all_letters[self.items.all_items.index(item)] for item in items]
        texts_to_type = [(str(count) if item.count != count else '') + letter
                         for letter, item, count in zip(letters, items, counts) if count != 0]

        if all((not i.can_be_dropped_from_inventory() for i in items)):
            return False

        def key_gen():
            if 'Drop what type of items?' in '\n'.join(self.agent.single_popup):
                yield 'a'
                yield A.MiscAction.MORE
            assert 'What would you like to drop?' in '\n'.join(self.agent.single_popup), \
                (self.agent.single_message, self.agent.single_popup)
            i = 0
            while texts_to_type:
                for text in list(texts_to_type):
                    letter = text[-1]
                    if f'{letter} - ' in '\n'.join(self.agent.single_popup):
                        yield from text
                        texts_to_type.remove(text)

                if texts_to_type:
                    yield A.TextCharacters.SPACE
                    i += 1

                assert i < 100, ('infinite loop', texts_to_type, self.agent.message)
            yield A.MiscAction.MORE

        with self.agent.atom_operation():
            self.agent.step(A.Command.DROPTYPE, key_gen())
        self._note_dropped(items, counts)
        self.get_items_below_me()

        return True

    def move_to_inventory(self, items):
        # all items in self.items will be updated!

        if not isinstance(items, list):
            is_list = False
            items = [items]
        else:
            is_list = True

        moved_items = {item for item in items if item in self.items.all_items}

        if len(moved_items) != len(items):
            with self.agent.atom_operation():
                its = list(filter(lambda i: i in self.items_below_me, items))
                if its:
                    moved_items = moved_items.union(its)
                    self.pickup(its)
                for container in chain(self.items_below_me, self.items):
                    if container.is_container():
                        its = list(filter(lambda i: i in container.content.items, items))
                        if its:
                            moved_items = moved_items.union(its)
                            self.use_container(container, items_to_take=its, items_to_put=[])

                assert moved_items == set(items), ('TODO: nested containers', moved_items, items)

            # TODO: HACK
            self.agent.last_observation = self.agent.last_observation.copy()
            for key in ['inv_strs', 'inv_oclasses', 'inv_glyphs', 'inv_letters']:
                self.agent.last_observation[key] = self.agent._observation[key].copy()
            self.items.update(force=True)

            ret = []
            for item in items:
                ret.append(find_equivalent_item(item, filter(lambda i: i not in ret, self.items.all_items)))
        else:
            ret = items

        if not is_list:
            assert len(ret) == 1
            return ret[0]
        return ret

    def call_item(self, item, name):
        assert item in self.items.all_items, item
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            self.agent.step(A.Command.CALL, iter(f'i{letter}#{name}\r'))
        return True

    def quaff(self, item, smart=True):
        return self.eat(item, quaff=True, smart=smart)

    def eat(self, item, quaff=False, smart=True):
        if not quaff and item.is_corpse() and self.agent.character.role == Character.MONK and \
                ord(MON.permonst(item.monster_id).mlet) not in \
                [MON.S_BLOB, MON.S_JELLY, MON.S_FUNGUS]:
            self.agent._monk_meat_meals += 1
        if smart:
            if not quaff and item in self.items_below_me:
                with self.agent.atom_operation():
                    self.agent.step(A.Command.EAT)
                    while '; eat it? [ynq]' in self.agent.message or \
                            '; eat one? [ynq]' in self.agent.message:
                        if f'{item.text} here; eat it? [ynq]' in self.agent.message or \
                                f'{item.text} here; eat one? [ynq]' in self.agent.message:
                            self.agent.type_text('y')
                            return True
                        self.agent.type_text('n')
                    # if "What do you want to eat?" in self.agent.message or \
                    #         "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no such food is lying here')
                    assert 0, self.agent.message

            # TODO: eat directly from ground if possible
            item = self.move_to_inventory(item)

        assert item in self.items.all_items, item or item in self.items_below_me
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            if quaff:
                def text_gen():
                    if self.agent.message.startswith('Drink from the fountain?'):
                        yield 'n'

                self.agent.step(A.Command.QUAFF, text_gen())
            else:
                self.agent.step(A.Command.EAT)
            if item in self.items.all_items:
                while re.search('There (is|are)[a-zA-Z0-9- ]* here; eat (it|one)\?', self.agent.message):
                    self.agent.type_text('n')
                self.agent.type_text(letter)
                return True

            elif item in self.items_below_me:
                while ' eat it? [ynq]' in self.agent.message or \
                        ' eat one? [ynq]' in self.agent.message:
                    if item.text in self.agent.message:
                        self.type_text('y')
                        return True
                if "What do you want to eat?" in self.agent.message or \
                        "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no food is lying here')

                assert 0, self.agent.message

        assert 0

    ######## STRATEGIES helpers

    def get_best_melee_weapon(self, items=None, *, return_dps=False, allow_unknown_status=False):
        if self.agent.character.role == Character.MONK:
            return None

        if items is None:
            items = self.items
        # select the best
        best_item = None
        best_dps = utils.calc_dps(*self.agent.character.get_melee_bonus(None, large_monster=False))
        for item in flatten_items(items):
            if item.is_weapon() and \
                    (item.status in [Item.UNCURSED, Item.BLESSED] or
                     (allow_unknown_status and item.status == Item.UNKNOWN)):
                to_hit, dmg = self.agent.character.get_melee_bonus(item, large_monster=False)
                dps = utils.calc_dps(to_hit, dmg)
                # dps = item.get_dps(large_monster=False)  # TODO: what about monster size
                if best_dps < dps:
                    best_dps = dps
                    best_item = item
        if return_dps:
            return best_item, best_dps
        return best_item

    def get_ranged_combinations(self, items=None, throwing=True, allow_best_melee=False, allow_wielded_melee=False,
                                allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        items = flatten_items(items)
        launchers = [i for i in items if i.is_launcher()]
        ammo_list = [i for i in items if i.is_fired_projectile()]
        valid_combinations = []

        # TODO: should this condition be used here
        if any(l.equipped and l.status == Item.CURSED for l in launchers):
            launchers = [l for l in launchers if l.equipped]

        for launcher in launchers:
            for ammo in ammo_list + additional_ammo:
                if ammo.is_fired_projectile(launcher):
                    if launcher.status in [Item.UNCURSED, Item.BLESSED] or \
                            (allow_unknown_status and launcher.status == Item.UNKNOWN):
                        valid_combinations.append((launcher, ammo))

        if throwing:
            best_melee_weapon = None
            if not allow_best_melee:
                best_melee_weapon = self.get_best_melee_weapon()
            wielded_melee_weapon = None
            if not allow_wielded_melee:
                wielded_melee_weapon = self.items.main_hand
            valid_combinations.extend([(None, i) for i in items
                                       if i.is_thrown_projectile()
                                       and i != best_melee_weapon and i != wielded_melee_weapon])

        return valid_combinations

    def get_best_ranged_set(self, items=None, *, throwing=True, allow_best_melee=False,
                            allow_wielded_melee=False,
                            return_dps=False, allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        # never throw unpaid goods (you owe for them, and a shopkeeper kills a thief)
        items = [i for i in flatten_items(items) if i.shop_status != Item.UNPAID]

        best_launcher, best_ammo = None, None
        best_dps = -float('inf')
        for launcher, ammo in self.get_ranged_combinations(items, throwing, allow_best_melee, allow_wielded_melee,
                                                           allow_unknown_status, additional_ammo):
            to_hit, dmg = self.agent.character.get_ranged_bonus(launcher, ammo)
            dps = utils.calc_dps(to_hit, dmg)
            if dps > best_dps:
                best_launcher, best_ammo, best_dps = launcher, ammo, dps
        if return_dps:
            return best_launcher, best_ammo, best_dps
        return best_launcher, best_ammo

    def get_best_armorset(self, items=None, *, return_ac=False, allow_unknown_status=False):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_items = [None] * O.ARM_NUM
        best_ac = [None] * O.ARM_NUM
        for item in items:
            if not item.is_armor() or not item.is_unambiguous():
                continue
            if jf_config.KEEP_MAGIC_BOOTS and power.never_wear(item):
                continue  # kept for the Castle; cursed levitation boots would end the dig-dive

            # TODO: consider other always allowed items than dragon hide
            is_dragonscale_armor = item.object.metal == O.DRAGON_HIDE

            allowed_statuses = [Item.UNCURSED, Item.BLESSED] + ([Item.UNKNOWN] if allow_unknown_status else [])
            if item.status not in allowed_statuses and not is_dragonscale_armor:
                continue

            slot = item.object.sub
            ac = item.get_ac()

            if self.agent.character.role == Character.MONK and slot == O.ARM_SUIT:
                continue

            if best_ac[slot] is None or best_ac[slot] > ac:
                best_ac[slot] = ac
                best_items[slot] = item

        if return_ac:
            return best_items, best_ac
        return best_items

    ######## LOW-LEVEL STRATEGIES

    def gather_items(self):
        return (
            self.pickup_and_drop_items()
                .before(self.check_containers())
                .before(self.wear_best_stuff())
                .before(self.wand_engrave_identify())
                .before(self.use_spare_wishes())
                .before(self.wear_life_saving())
                .before(self.go_to_unchecked_containers())
                .before(self.check_items()
                        .before(self.go_to_item_to_pickup()).repeat().every(5)
                        .preempt(self.agent, [
                self.pickup_and_drop_items(),
                self.check_containers(),
            ])).repeat()
        )

    @utils.debug_log('inventory.arrange_items')
    @Strategy.wrap
    def arrange_items(self):
        yielded = False

        if self.agent.character.prop.polymorph:
            # TODO: only handless
            yield False

        while 1:
            if jf_config.CONTAINER_LOOP_FIX and self._here() in self.multi_container_squares:
                # containers here are left alone (their contents aren't ours to plan with)
                below = list(self.items_below_me)
            else:
                below = flatten_items(self.items_below_me)
            items_below_me = list(filter(lambda i: i.shop_status == Item.NOT_SHOP and not self.dropped_here(i),
                                         below))
            forced_items = list(filter(lambda i: not self._droppable(i), flatten_items(self.items)))
            assert all((item in self.items.all_items for item in forced_items))
            free_items = list(filter(lambda i: self._droppable(i),
                                     flatten_items(sorted(self.items, key=lambda x: x.text))))
            all_items = free_items + items_below_me

            item_split = self.agent.global_logic.item_priority.split(
                all_items, forced_items, self.agent.character.carrying_capacity)

            assert all((container is None or container in self.items_below_me or container in self.items.all_items or \
                        (sum(item_split[container]) == 0 and not container.content.items)
                        for container in item_split)), 'TODO: nested containers'

            cont = False

            # put into containers
            for container in item_split:
                if container is not None:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if item in self.items.all_items and counts[i] > 0]
                    if not indices:
                        continue
                    if not yielded:
                        yielded = True
                        yield True

                    self.use_container(container, [all_items[i] for i in indices], [],
                                       items_to_put_counts=[counts[i] for i in indices])
                    cont = True
                    break
            if cont:
                continue

            # drop on ground
            counts = item_split[None]
            indices = [i for i, item in enumerate(free_items) if
                       item in self.items.all_items and counts[i] != item.count]
            if indices:
                if not yielded:
                    yielded = True
                    yield True
                assert self.drop([free_items[i] for i in indices], [free_items[i].count - counts[i] for i in indices],
                                 smart=False)
                continue

            # take from container
            for container in all_items:
                if not container.is_container():
                    continue

                if container in item_split:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if
                               item in container.content.items and counts[i] != item.count]
                    items_to_take_counts = [all_items[i].count - counts[i] for i in indices]
                else:
                    counts = np.array(list(item_split.values())).sum(0)
                    indices = [i for i, item in enumerate(all_items) if
                               item in container.content.items and counts[i] != 0]
                    items_to_take_counts = [counts[i] for i in indices]

                if not indices:
                    continue
                if not yielded:
                    yielded = True
                    yield True

                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]

                self.use_container(container, [], [all_items[i] for i in indices],
                                   items_to_take_counts=items_to_take_counts)
                cont = True
                break
            if cont:
                continue

            # pick up from ground
            to_pickup = np.array([counts[len(free_items):] for counts in item_split.values()]).sum(0)
            assert len(to_pickup) == len(items_below_me)
            indices = [i for i, item in enumerate(items_below_me) if to_pickup[i] > 0 and item in self.items_below_me]
            if len(indices) > 0:
                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]
                if not yielded:
                    yielded = True
                    yield True
                assert self.pickup([items_below_me[i] for i in indices], [to_pickup[i] for i in indices])
                continue

            break

        for container in item_split:
            for item, count in zip(all_items, item_split[container]):
                assert count == 0 or count == item.count
                assert count == 0 or item in (
                    container.content.items if container is not None else self.items.all_items)

        if not yielded:
            yield False

    def _determine_possible_wands(self, message, item):

        wand_regex = '[a-zA-Z ]+'
        floor_regex = '[a-zA-Z]+'
        mapping = {
            f"The engraving on the {floor_regex} vanishes!": ['cancellation', 'teleportation', 'make invisible'],
            # TODO?: cold,  # (if the existing engraving is a burned one)

            "A few ice cubes drop from the wand.": ['cold'],
            f"The bugs on the {floor_regex} stop moving": ['death', 'sleep'],
            f"This {wand_regex} is a wand of digging!": ['digging'],
            "Gravel flies up from the floor!": ['digging'],
            f"This {wand_regex} is a wand of fire!": ['fire'],
            "Lightning arcs from the wand. You are blinded by the flash!": ['lightning'],
            f"This {wand_regex} is a wand of lightning!": ['lightning'],
            f"The {floor_regex} is riddled by bullet holes!": ['magic missile'],
            f'The engraving now reads:': ['polymorph'],
            f"The bugs on the {floor_regex} slow down!": ['slow monster'],
            f"The bugs on the {floor_regex} speed up!": ['speed monster'],
            "The wand unsuccessfully fights your attempt to write!": ['striking'],

            # activated effects:
            "A lit field surrounds you!": ['light'],
            "You may wish for an object.": ['wishing'],
            "You feel self-knowledgeable...": ['enlightenment']  # TODO: parse the effect
            # TODO: "The wand is too worn out to engrave.": [None],  # wand is exhausted
        }

        for msg, wand_types in mapping.items():
            res = re.findall(msg, message)
            if len(res) > 0:
                assert len(res) == 1
                return [O.from_name(w, nh.WAND_CLASS) for w in wand_types]

        # TODO: "wand is cancelled (x:-1)" ?
        # TODO: "secret door detection self-identifies if secrets are detected" ?

        res = re.findall(f'Your {wand_regex} suddenly explodes!', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return None

        res = re.findall('The wand is too worn out to engrave.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            self.agent.inventory.call_item(item, 'EMPT')
            return None

        res = re.findall(f'{wand_regex} glows, then fades.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return [p for p in O.possibilities_from_glyph(item.glyphs[0])
                    if p.name not in ['light', 'wishing']]
            # TODO: wiki says this:
            # return [O.from_name('opening', nh.WAND_CLASS),
            #         O.from_name('probing', nh.WAND_CLASS),
            #         O.from_name('undead turning', nh.WAND_CLASS),
            #         O.from_name('nothing', nh.WAND_CLASS),
            #         O.from_name('secret door detection', nh.WAND_CLASS),
            #         ]

        assert 0, message

    @utils.debug_log('inventory.wand_engrave_identify')
    @Strategy.wrap
    def wand_engrave_identify(self):
        if self.agent.character.prop.polymorph:
            yield False  # TODO: only for handless monsters (which cannot write)

        self.skip_engrave_counter -= 1
        if self.agent.character.prop.blind or self.skip_engrave_counter > 0 or self.agent.hands_welded():
            yield False
            return
        yielded = False
        for item in self.agent.inventory.items:
            if not isinstance(item.objs[0], O.Wand):
                continue
            if item.is_unambiguous():
                continue
            if self.agent.current_level().objects[self.agent.blstats.y, self.agent.blstats.x] not in G.FLOOR:
                continue
            if item.glyphs[0] in self.item_manager._already_engraved_glyphs:
                continue
            if len(item.glyphs) > 1:
                continue
            if item.comment == 'EMPT':
                continue

            if not yielded:
                yield True
            yielded = True
            self.skip_engrave_counter = 8

            with self.agent.atom_operation():
                wand_types = self._engrave_single_wand(item)

                if wand_types is None:
                    # there is a problem with engraving on this tile
                    continue

                self.item_manager._glyph_to_possible_wand_types[item.glyphs[0]] = wand_types
                self.item_manager._already_engraved_glyphs.add(item.glyphs[0])
                self.item_manager.possible_objects_from_glyph(item.glyphs[0])

            # uncomment for debugging (stopping when there is a new wand being identified)
            # print(len(self.item_manager.possible_objects_from_glyph(item.glyphs[0])))
            # print(self.item_manager._glyph_to_possible_wand_types)
            # input('==================3')

        if yielded:
            self.agent.inventory.items.update(force=True)

        if not yielded:
            yield False

    @utils.debug_log('inventory.use_spare_wishes')
    @Strategy.wrap
    def use_spare_wishes(self):
        """SPARE_WISHES: a wand of wishing keeps rnd(3) - 1 charges after the engrave-test wish (51 of 3158 games
        in our runs had one). Zap it until it's empty; power.wish_text picks the wish (the Castle passage ring,
        then speed boots)."""
        if not jf_config.SPARE_WISHES or self.agent.character.prop.polymorph or self.agent.hands_welded():
            yield False
            return
        wand = next((i for i in self.items if i.is_unambiguous() and i.object == power.WISH_WAND and
                     not power._empty(self.agent, i)), None)
        if wand is None:
            yield False
            return
        yield True
        self.agent.log(f'POWER zapping {wand.text!r} for {power.wish_text(self.agent)!r}')
        self.agent.zap(wand, None)
        if 'Nothing happens' in self.agent.message:
            self.empty_wands.add(wand.text)
        self.items.update(force=True)

    @utils.debug_log('inventory.wear_life_saving')
    @Strategy.wrap
    def wear_life_saving(self):
        """SPARE_WISHES: put on a known amulet of life saving (a wish) when no amulet is worn."""
        if not jf_config.SPARE_WISHES or self.agent.character.prop.polymorph or \
                any(i.category == nh.AMULET_CLASS and i.equipped for i in self.items):
            yield False
            return
        amulet = next((i for i in self.items if i.is_unambiguous() and i.object == power.LS_AMULET), None)
        if amulet is None:
            yield False
            return
        yield True
        letter = self.items.get_letter(amulet)

        def gen():
            if 'What do you want to put on?' in self.agent.single_message:
                yield letter

        self.agent.log(f'POWER putting on {amulet.text!r}')
        with self.agent.atom_operation():
            self.agent.step(A.Command.PUTON, gen())
        self.items.update(force=True)

    def _engrave_single_wand(self, item):
        """ Returns possible objects or None if current tile not suitable for identification."""

        def msg():
            return self.agent.message

        def smsg():
            return self.agent.single_message

        self.agent.step(A.Command.LOOK)
        if msg() != 'You see no objects here.':
            return None
        # if 'written' in msg() or 'engraved' in msg() or 'see' not in msg() or 'read' in msg():
        #     return None

        skip_engraving = [False]

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield '-'
            # if 'Do you want to add to the current engraving' in smsg():
            #     yield 'q'
            #     assert smsg().strip() == 'Never mind.', smsg()
            #     skip_engraving[0] = True
            #     return
            if smsg().startswith('You wipe out the message that was written'):
                yield ' '
                skip_engraving[0] = True
                return
            if smsg().startswith('You cannot wipe out the message that is burned into the floor here.'):
                skip_engraving[0] = True
                return
            assert smsg().startswith('You write in the dust with your fingertip.'), smsg()
            yield ' '
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield 'x'
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield '\r'

        for _ in range(5):
            # write 'x' with finger in the dust
            self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

            if skip_engraving[0]:
                assert msg().strip().endswith('Never mind.') \
                       or 'You cannot wipe out the message that is burned into the floor here.' in msg(), msg()
                return None

            # this is usually true, but something unrelated like: "You hear crashing rock." may happen
            # assert msg().strip() in '', msg()

            # check if the written 'x' is visible when looking
            self.agent.step(A.Command.LOOK)
            if 'Something is written here in the dust.' in msg() \
                    and 'You read: "x"' in msg():
                break
            else:
                # this is usually true, but something unrelated like:
                #   "There is a doorway here.  Something is written here in the dust. You read: "4".
                #    You see here a giant rat corpse."
                # may happen
                # assert "You see no objects here" in msg(), msg()
                return None
        else:
            assert 0, msg()

        # try engraving with the wand
        letter = self.agent.inventory.items.get_letter(item)
        possible_wand_types = []

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield letter
            if 'Do you want to add to the current engraving' in smsg():
                self.agent.type_text('y')
                # assert 'You add to the writing in the dust with' in smsg(), smsg()
                # self.agent.type_text(' ')
            r = self._determine_possible_wands(smsg(), item)
            if r is not None:
                possible_wand_types.extend(r)
            else:
                # wand exploded
                skip_engraving[0] = True

        self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

        if skip_engraving[0]:
            return None

        if 'Do you want to add to the current engraving' in smsg():
            self.agent.type_text('q')
            assert smsg().strip() == 'Never mind.', smsg()

        return possible_wand_types

    @utils.debug_log('inventory.wear_best_stuff')
    @Strategy.wrap
    def wear_best_stuff(self):
        if self.agent.hands_welded():
            yield False   # armor can't come off (or go on over it) with the hands welded
            return
        yielded = False
        while 1:
            best_armorset = self.get_best_armorset()

            # TODO: twoweapon
            for slot, name in [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'),
                               (O.ARM_BOOTS, 'boots'), (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'),
                               (O.ARM_CLOAK, 'cloak')]:
                if best_armorset[slot] == getattr(self.items, name) or \
                        (getattr(self.items, name) is not None and getattr(self.items, name).status == Item.CURSED):
                    continue
                additional_cond = True
                if slot == O.ARM_SHIELD:
                    additional_cond &= self.items.main_hand is None or not self.items.main_hand.objs[0].bi
                if slot == O.ARM_GLOVES:
                    additional_cond &= self.items.main_hand is None or self.items.main_hand.status != Item.CURSED
                if slot == O.ARM_SHIRT or slot == O.ARM_SUIT:
                    additional_cond &= self.items.cloak is None or self.items.cloak.status != Item.CURSED
                if slot == O.ARM_SHIRT:
                    additional_cond &= self.items.suit is None or self.items.suit.status != Item.CURSED

                if additional_cond:
                    if not yielded:
                        yielded = True
                        yield True
                    if (slot == O.ARM_SHIRT or slot == O.ARM_SUIT) and self.items.cloak is not None:
                        self.takeoff(self.items.cloak)
                        break
                    if slot == O.ARM_SHIRT and self.items.suit is not None:
                        self.takeoff(self.items.suit)
                        break
                    if getattr(self.items, name) is not None:
                        self.takeoff(getattr(self.items, name))
                        break
                    assert best_armorset[slot] is not None
                    self.wear(best_armorset[slot])
                    break
            else:
                break

        if not yielded:
            yield False

    @utils.debug_log('inventory.check_items')
    @Strategy.wrap
    def check_items(self):
        mask = utils.isin(self.agent.glyphs, G.OBJECTS, G.BODIES, G.STATUES)
        if not mask.any():
            yield False

        dis = self.agent.bfs()

        mask &= self.agent.current_level().item_count == 0
        if not mask.any():
            yield False

        mask &= dis > 0
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.go_to_unchecked_containers')
    @Strategy.wrap
    def go_to_unchecked_containers(self):
        mask = self.agent.current_level().item_count != 0
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis > 0
        if not mask.any():
            yield False

        key = self.agent.current_level().key()
        for y, x in zip(*mask.nonzero()):
            # squares whose containers are left alone (several containers; CONTAINER_LOOP_FIX failures): check_containers
            # skips them, so walking there only ping-pongs with the exploration
            if jf_config.CONTAINER_LOOP_FIX and (*key, int(y), int(x)) in self.multi_container_squares:
                mask[y, x] = False
                continue
            for item in self.agent.current_level().items[y, x]:
                if not item.is_possible_container():
                    mask[y, x] = False

        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.check_containers')
    @Strategy.wrap
    def check_containers(self):
        yielded = False
        # a welded two-hander (a cursed dwarvish mattock the dive dug with) leaves no free hand for
        # #untrap or #loot: 450 'Your hands seem to be too busy' panics in the s10-s13 runs
        main = self.items.main_hand
        if main is not None and main.status == Item.CURSED and getattr(main.objs[0], 'bi', False):
            yield False
        if self._here() in self.multi_container_squares:
            yield False
        bl = self.agent.blstats
        hurt = bl.hitpoints < max(15, bl.max_hitpoints // 2)
        for item in self.agent.inventory.items_below_me:
            if item.is_possible_container():
                # an unidentified bag may be a bag of tricks: 'It develops a huge set of teeth and bites
                # you!' (d10) killed an XL1 that had just prayed out of a bear trap
                if hurt and any(o.name == 'bag of tricks' for o in item.objs):
                    continue
                if not yielded:
                    yielded = True
                    yield True
                if item.is_chest() and not (item.is_unambiguous() and item.object.name == 'ice box'):
                    fail_msg = self.agent.untrap_container_below_me()
                    if fail_msg is not None and check_if_triggered_container_trap(fail_msg):
                        raise AgentPanic('triggered trap while looting')
                self.check_container_content(item)
        if not yielded:
            yield False

    # jf: buying food. AutoAscend never shopped, but a hunger prayer costs the prayer an HP emergency needs
    # (33 of 37 tour deaths came under 1000 turns after the last prayer; 14 of them fainting) and fails
    # 1.5-2.6% of the time. The tour reaches Minetown with ~240 gold (median): 3-4 food rations.
    # Nutrition by name (objects.c); only foods safe for anyone (no tripe, eggs, tins or corpses).
    BUY_FOOD_NUTRITION = {'food ration': 800, 'cram ration': 600, 'lembas wafer': 800, 'K-ration': 400,
                          'C-ration': 300, 'pancake': 200, 'candy bar': 100, 'cream pie': 100,
                          'fortune cookie': 40, 'apple': 50, 'orange': 80, 'pear': 50, 'melon': 100,
                          'banana': 80, 'carrot': 50, 'slime mold': 80, 'kelp frond': 30}

    def carried_nutrition(self):
        return sum(self.BUY_FOOD_NUTRITION.get(item.object.name, 0) * item.count
                   for item in self.agent.edible_carried_food() if item.is_unambiguous())

    def _food_for_sale(self, dis):
        level = self.agent.current_level()
        gold = self.agent.blstats.gold
        best = None
        for y, x in zip(*(level.shop_interior & (level.item_count > 0)).nonzero()):
            if dis[y, x] == -1:
                continue
            for item in level.items[y, x]:
                if item.shop_status != Item.FOR_SALE or not item.is_unambiguous():
                    continue
                nutrition = self.BUY_FOOD_NUTRITION.get(item.object.name)
                if nutrition is None or not item.price or item.price > gold:
                    continue
                score = nutrition / item.price - dis[y, x] / 1000
                if best is None or score > best[0]:
                    best = (score, int(y), int(x), item.object.name, item.price)
        return best

    def pay_or_drop_unpaid(self):
        """Never walk off with unpaid goods: pay (one item on the bill: '... for N zorkmids.  Pay? [yn]',
        answered 'y'), and drop whatever is still unpaid (not enough gold)."""
        if not any(i.shop_status == Item.UNPAID for i in flatten_items(self.items)):
            return
        self.agent.step(A.Command.PAY)
        unpaid = [i for i in flatten_items(self.items) if i.shop_status == Item.UNPAID]
        if unpaid:
            self.agent.log(f'SHOP could not pay for {[i.text for i in unpaid]}: dropping')
            self.drop(unpaid)

    # ---- sell-offer price identification (jf_config.SELL_PRICE_ID)
    # Dropping an item in a shop that buys its class makes the shopkeeper offer its base price / 2, or 3/8 of it
    # from a quarter of the shopkeepers (shk.c set_cost, unidentified items). Declining leaves it ours
    # ('no charge') to pick up again. A potion's price narrows levitation to the 200 zm group (speed,
    # levitation, enlightenment, full healing, polymorph): castle_logic then quaffs one or two potions instead
    # of six, most of them paralysis/sleeping/blindness risks next to the moat. 21 of 90 base games entered a
    # general store or liquor emporium after the dive started, carrying 6-9 potion types.
    _SELL_BUYERS = {nh.POTION_CLASS: (1, 4), nh.RING_CLASS: (1, 7), nh.ARMOR_CLASS: (1, 2), nh.AMULET_CLASS: (1, 7)}
    _SELL_OFFER = re.compile(r'offers( only)? (\d+) gold pieces? for (?:your|the) ')

    @staticmethod
    def _sell_offers(cost):
        """Possible per-unit offers for an unidentified item of this base price."""
        normal = (cost * 10 // 2 + 5) // 10
        reduced = (cost * 3 * 10 // 8 + 5) // 10
        return {max(normal, 1), max(reduced, 1)}

    def _sell_candidates(self, shop_type):
        key = self.agent.current_level().key()
        out = []
        for item in self.items:
            if item.equipped or item.is_unambiguous() or item.category not in self._SELL_BUYERS or \
                    shop_type not in self._SELL_BUYERS[item.category] or not power.is_passage_candidate(item):
                continue
            if (key, item.glyphs[0]) in self._sell_tested or len({o.cost for o in item.objs}) <= 1:
                continue
            out.append(item)
        # the levitation carriers first: potions, rings, boots
        out.sort(key=lambda i: (i.category != nh.POTION_CLASS, i.category != nh.RING_CLASS))
        return out

    def _sell_test(self, item):
        """Drop one unit, decline the shopkeeper's offer. Returns the offer prompts seen."""
        letter = self.items.get_letter(item)
        prompts = []

        def gen():
            if item.count > 1:
                yield '1'
            yield letter
            for _ in range(8):
                obs = self.agent._observation
                if obs['misc'][0]:            # a y/n question: the sale offer, or credit instead of gold
                    prompts.append(self.agent.single_message)
                    yield 'n'
                elif obs['misc'][2]:          # --More--
                    yield A.MiscAction.MORE
                else:
                    return

        with self.agent.atom_operation():
            self.agent.step(A.Command.DROP, gen())
        return prompts

    def _sell_record(self, item, prompts):
        g = item.glyphs[0]
        offer = None
        for p in prompts:
            m = self._SELL_OFFER.search(p)
            if m and not m.group(1):
                offer = int(m.group(2))
        if offer is None:
            self.agent.log(f'POWER sell-test {item.text!r}: no clean offer ({prompts!r})')
            return
        # armour prices include 10 per point of enchantment (getprice): allow +0..+2
        spes = (0, 1, 2) if item.category == nh.ARMOR_CLASS else (0,)
        fits = [o for o in item.objs if any(offer in self._sell_offers(o.cost + 10 * s) for s in spes)]
        if not fits:
            self.agent.log(f'POWER sell-test {item.text!r}: offer {offer} fits nothing in {[o.name for o in item.objs]}')
            return
        lo, hi = min(o.cost for o in fits), max(o.cost for o in fits)
        old = self.item_manager._glyph_to_price_range.get(g)
        if old is not None:
            lo, hi = max(lo, old[0]), min(hi, old[1])
        if lo > hi or not any(lo <= o.cost <= hi for o in item.objs):
            return
        self.item_manager._glyph_to_price_range[g] = (lo, hi)
        self.item_manager.possible_objects_from_glyph(g)
        self.agent.log(f'POWER sell-test {item.text!r}: offer {offer} -> base {lo}-{hi}: '
                       f'{sorted({o.name for o in fits})}')

    @utils.debug_log('inventory.sell_price_identify')
    @Strategy.wrap
    def sell_price_identify(self):
        agent = self.agent
        if not jf_config.SELL_PRICE_ID or agent.character.prop.hallu or agent.character.prop.blind or \
                agent.character.prop.polymorph or agent.hands_welded():
            yield False
            return
        level = agent.current_level()
        bl = agent.blstats
        if not level.shop_interior[bl.y, bl.x] or bl.hunger_state >= Hunger.WEAK or \
                bl.hitpoints < 0.5 * bl.max_hitpoints or agent.get_visible_monsters() or \
                not utils.isin(agent.glyphs, G.SHOPKEEPER).any():
            yield False
            return
        # a dunce cap or a bare shirt changes the offers (divisor 3)
        items = self.items
        if (items.helm is not None and any(o.name == 'dunce cap' for o in items.helm.objs)) or \
                (items.shirt is not None and items.suit is None and items.cloak is None):
            yield False
            return
        candidates = self._sell_candidates(int(level.shop_type[bl.y, bl.x]))
        if not candidates:
            yield False
            return
        # an empty square of the shop floor, so the pickup can only take our own item back
        dis = agent.bfs()
        free = level.shop_interior & (level.item_count == 0) & (dis != -1)
        if not free.any():
            yield False
            return
        yield True
        if not free[bl.y, bl.x]:
            ty, tx = min(zip(*free.nonzero()), key=lambda p: dis[p])
            agent.go_to(ty, tx)
            return
        item = candidates[0]
        self._sell_tested.add((level.key(), item.glyphs[0]))
        prompts = self._sell_test(item)
        self._sell_record(item, prompts)
        self.get_items_below_me()
        mine = [i for i in self.items_below_me if i.shop_status == Item.NOT_SHOP]
        if mine:
            self.pickup(mine)
        self.items.update(force=True)

    @utils.debug_log('inventory.buy_food')
    @Strategy.wrap
    def buy_food(self):
        agent = self.agent
        bl = agent.blstats
        if not jf_config.BUY_FOOD or bl.gold < 20 or bl.hunger_state >= Hunger.FAINTING:
            yield False
        level = agent.current_level()
        if not level.shop_interior.any():
            yield False
        if any(i.shop_status == Item.UNPAID for i in flatten_items(self.items)):
            yield True
            self.pay_or_drop_unpaid()
            return
        if self.carried_nutrition() >= jf_config.BUY_FOOD_UNTIL or agent._carries_digging_tool() or \
                agent.get_visible_monsters():
            yield False
        if jf_config.SHOP_GUARD and agent.character.teleportitis and not agent.character.teleport_control:
            # a random teleport between the pickup and the payment takes the goods out unpaid: Kops and an
            # angry shopkeeper (base4-jf14 s10, dead)
            yield False
        dis = agent.bfs()
        target = self._food_for_sale(dis)
        if target is None:
            yield False
        _, y, x, name, price = target
        key = (level.dungeon_number, level.level_number, y, x)
        if jf_config.BUY_FOOD_GIVEUP and self._buy_food_blocked.get(key, (0, -1))[1] > bl.time:
            yield False
        yield True
        if (bl.y, bl.x) != (y, x):
            # walk there and buy in one go (between every(3) turns check_items walked us off the square again)
            try:
                agent.go_to(y, x)
            finally:
                # BUY_FOOD_GIVEUP: a shopkeeper standing in the path panicked every go_to ('Monster on a next
                # tile'), and the tour walked back between two tries: base4-jf14 s6 went N/S ~660 times per
                # 500 turns for 1500 turns on its Dlvl-2 grind. After 3 failed walks, leave that item alone.
                if jf_config.BUY_FOOD_GIVEUP and (agent.blstats.y, agent.blstats.x) != (y, x):
                    fails = self._buy_food_blocked.get(key, (0, -1))[0] + 1
                    self._buy_food_blocked[key] = (fails, agent.blstats.time + 2000 if fails >= 3 else -1)
            if (agent.blstats.y, agent.blstats.x) != (y, x):
                return
        items = [i for i in self.items_below_me
                 if i.shop_status == Item.FOR_SALE and i.is_unambiguous() and i.object.name == name]
        if not items:
            return
        agent.log(f'SHOP buying {name} for {price} (gold {bl.gold})')
        self.pickup(items[0], 1)
        self.pay_or_drop_unpaid()

    @utils.debug_log('inventory.go_to_item_to_pickup')
    @Strategy.wrap
    def go_to_item_to_pickup(self):
        level = self.agent.current_level()
        dis = self.agent.bfs()

        # TODO: free (no charge) items
        mask = ~level.shop_interior & (dis > 0)
        if not mask.any():
            yield False

        mask[mask] = self.agent.current_level().item_count[mask] != 0
        for (dn, ln, uy, ux), until in self.unreachable_items_until.items():
            if (dn, ln) == level.key() and until > self.agent.blstats.time:
                mask[uy, ux] = False

        items = {}
        for y, x in sorted(zip(*mask.nonzero()), key=lambda p: dis[p]):
            for i in level.items[y, x]:
                assert i not in items
                if self.dropped_here(i, (y, x)):
                    continue
                items[i] = (y, x)

        if not items:
            yield False

        def expand(item, pos):
            # a container left alone (CONTAINER_LOOP_FIX) offers only itself: its contents can't be taken out, and
            # walking to them ping-ponged with the exploration for 3000 turns (base3-jf14 s0, dead there at 0.075)
            if jf_config.CONTAINER_LOOP_FIX and item.is_container() and \
                    (*level.key(), int(pos[0]), int(pos[1])) in self.multi_container_squares:
                return [item]
            return flatten_items([item])

        items = {i: pos for item, pos in items.items() for i in expand(item, pos)}

        free_items = list(filter(lambda i: self._droppable(i), flatten_items(self.items)))
        forced_items = list(filter(lambda i: not self._droppable(i), flatten_items(self.items)))
        item_split = self.agent.global_logic.item_priority.split(
            free_items + list(items.keys()), forced_items,
            self.agent.character.carrying_capacity)
        counts = np.array(list(item_split.values())).sum(0)

        counts = counts[len(free_items):]
        assert len(counts) == len(items)
        if sum(counts) == 0:
            yield False
        yield True

        for (i, _), c in sorted(zip(items.items(), counts), key=lambda x: dis[x[0][1]]):
            if c != 0:
                target_y, target_x = items[i]
                break
        else:
            assert 0

        with self.agent.env.debug_tiles([(y, x) for _, (y, x) in items.items()], color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.pickup_and_drop_items')
    @Strategy.wrap
    def pickup_and_drop_items(self):
        # TODO: free (no charge) items
        self.item_manager.price_identification()
        if self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            yield False
        if self.unreachable_items_until.get(self._here(), -1) > self.agent.blstats.time:
            yield False
        if len(self.items_below_me) == 0:
            yield False

        yield from self.arrange_items().strategy()
