import contextlib
import re
from functools import partial
from itertools import chain

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from nhbot import objects as O, utils
from nhbot import power
from nhbot.character import Character
from nhbot.exceptions import AgentPanic
from nhbot.glyph import G, MON, Hunger
from nhbot import jf_config
from nhbot.item import ItemManager, Item, ContainerContent, check_if_triggered_container_trap, \
    find_equivalent_item, flatten_items
from nhbot.item.inventory_items import InventoryItems
from nhbot.item.utils import public_buc, public_noncursed
from nhbot.strategy import Strategy
from nhbot.shop_transactions import ShopTransaction, quantity_price, bill_cleared, current_shop_credit


MELEE_BASHING = frozenset({'dart', 'shuriken', 'boomerang', 'arrow', 'elven arrow', 'orcish arrow',
                           'silver arrow', 'ya', 'crossbow bolt'})


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
        self._blind_look_skipped = False   # WR_BLIND_LOOK: a floor look was skipped while blind (redo on sight)

        self.skip_engrave_counter = 0
        self.scare_labels = set()     # scroll labels known to be scare monster in this game (power.note_dust_prompt)
        self.dropped_scrolls = set()  # (level key, (y, x), appearance) of scrolls we dropped: never picked up again
        self._sell_tested = set()     # (level key, glyph) already offered to this level's shopkeeper (SELL_PRICE_ID)
        self.empty_wands = set()  # inventory texts of wands that answered "Nothing happens"
        self.multi_container_squares = set()  # (dungeon, level, y, x) where #loot asks 'Loot which containers?'
        self._container_failures = {}         # (dungeon, level, y, x) -> failed use_container attempts
        self.unreachable_items_until = {}     # (dungeon, level, y, x) -> turn: items at a pit bottom out of reach
        self._item_inspection_failures = {}   # optional approach only; never suppress pickup or required travel
        self._buy_food_blocked = {}           # (dungeon, level, y, x) -> (failed walks, skip until turn): BUY_FOOD_GIVEUP
        self._shop_transaction = None  # survives strategy preemption and on_panic()
        self._shop_settling = False
        self._shop_credits = {}  # (level, shop id) -> (exact ShopRecord, observed amount)
        self._equipment_preparation = None  # intent/retry survives READ or equipment preemption
        self._preparation_protection_rings = set()  # rings this policy, not combat-only wearing, manages

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
        self.invalidate_shop_credit()
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None
        self._previous_blstats = None

        self.item_manager.on_panic()
        self.items.on_panic()

    def update(self):
        self.item_manager.update()
        self.items.update()

        # WR_BLIND_LOOK: sight is back on a square whose floor was never looked at -- look now (the engraving under us
        # may be a good Elbereth that a blind '' would have us wipe and rewrite)
        blind_redo = self._blind_look_skipped and not self.agent.character.prop.blind
        if blind_redo or self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number) or \
                (self.engraving_below_me is None or self.engraving_below_me.lower() == 'elbereth'):
            assume_appropriate_message = not blind_redo and self._previous_blstats is not None and \
                not self.engraving_below_me

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
            if jf_config.ROBUST_FIXES and self._wear_refused(item):
                return False
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
        # NetHack counts only the outermost cloak/suit/shirt for automatic
        # selection. Never issue T for a covered target: it might remove the
        # sole outer layer without asking which item we meant.
        if ((item is self.items.suit and self.items.cloak is not None) or
                (item is self.items.shirt and (self.items.cloak is not None or self.items.suit is not None))):
            return False

        with self.agent.atom_operation():
            self.agent.step(A.Command.TAKEOFF)

            is_take_off_message = lambda: \
                'You finish taking off ' in self.agent.message or \
                'You were wearing ' in self.agent.message or \
                'You feel that monsters no longer have difficulty pinpointing your location.' in self.agent.message

            # Observe the prompt instead of guessing from total armor count;
            # covered inner layers do not count, and paranoid removal can
            # request a letter even when just one accessible piece is worn.
            if 'What do you want to take off?' in self.agent.message:
                self.agent.type_text(letter)
            if 'It is cursed.' in self.agent.message or 'They are cursed.' in self.agent.message:
                return False
            if jf_config.ROBUST_FIXES and self._wear_refused(item):
                return False
            assert is_take_off_message(), self.agent.message

        return True

    # ROBUST_FIXES (eL1fe, extended): do_wear.c refusals that take no game time -- canwearobj() for W, select_off()
    # for T. The welded ones don't mark the weapon cursed (no set_bknown in canwearobj), so hands_welded() never
    # learns why and wear_best_stuff would retry every step; the trap and Glib ones pass after a while.
    _WEAR_REFUSED_WELDED = re.compile(r'You cannot do that while holding your |You cannot wear gloves over your |'
                                      r'You are unable to take off your gloves while wielding |'
                                      r'You cannot release your \w+ to take off ')
    _WEAR_REFUSED_BRIEF = re.compile(r'Your foot is trapped!|Your feet are stuck in the |is attached to the buried ball!|'
                                     r'bear trap prevents you from pulling your |and cannot pull your |'
                                     r'are too slippery to pull on |are too slippery to take off')

    def _wear_refused(self, item):
        """A wear/take-off refusal: the item's armor slot waits (agent._wear_blocked_until, read by
        wear_best_stuff) 1000 turns for a welded weapon, 20 for a trapped foot or slippery fingers."""
        msg = self.agent.message or ''
        if self._WEAR_REFUSED_WELDED.search(msg):
            wait = 1000
        elif self._WEAR_REFUSED_BRIEF.search(msg):
            wait = 20
        else:
            return False
        slot = getattr(item.objs[0], 'sub', None)
        blocked = getattr(self.agent, '_wear_blocked_until', {})
        blocked[slot] = self.agent.blstats.time + wait
        self.agent._wear_blocked_until = blocked
        self.agent.log(f'ROBUST wear/takeoff refused, slot {slot} waits {wait} turns: {msg[:80]!r}')
        return True

    def use_container(self, container, items_to_put, items_to_take, items_to_put_counts=None,
                      items_to_take_counts=None):
        if container in self._quarantined_floor_items():
            raise AgentPanic('unrecovered shop gold container requires explicit recovery')
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

    def _blind_look_skip(self):
        """WR_BLIND_LOOK (weak-role lane; DT6A's Tourist and vlomshakov's Healer leaders both skip it): blind, ':' is a
        real move -- invent.c look_here() returns !!Blind ('You try to feel what is lying here') -- and it reads no dust
        engraving anyway (engrave.c read_engr_at: DUST only when !Blind). It ran after every step and every Elbereth
        (engrave() reads its work back), so a blinded digger gave the ravens / snakes a free turn per action: in 62 of
        E2's 270 fresh games the last 30 turns held blind looks, most of them on turns something hit us. Only while
        diving (the tour's pickup/container bookkeeping needs the real floor): the floor counts as empty and unengraved
        until sight returns, when update() looks again."""
        if not self.agent.character.prop.blind:
            return False
        # DEEP_BLIND_LOOK (deep-arrivals, F095): the same skip at depth >= DEEP_BLIND_LOOK_DEPTH only (Medusa's ravens
        # and snakes), so a game is unchanged until its first blind moment down there
        if not jf_config.WR_BLIND_LOOK and \
                not (jf_config.DEEP_BLIND_LOOK and self.agent.blstats.depth >= jf_config.DEEP_BLIND_LOOK_DEPTH):
            return False
        dive = getattr(getattr(self.agent, 'global_logic', None), 'dive', None)
        return dive is not None and dive.diving

    def get_items_below_me(self, assume_appropriate_message=False):
        if self._blind_look_skip():
            self._blind_look_skipped = True
            self.items_below_me = []
            self.letters_below_me = []
            self.engraving_below_me = ''
            return []
        self._blind_look_skipped = False
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
                                "You don't feel anything in here to pick up." in self.agent.message or \
                                (jf_config.ROBUST_FIXES2 and
                                 'You are physically incapable of picking anything up.' in self.agent.message):
                            # (ROBUST_FIXES2: pickup.c notake() -- a polymorph form that cannot pick up, e.g. the
                            # garter snake of a self-zapped wand of polymorph; the assert below repeated on every look)
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

    def pickup(self, items, counts=None, *, shop_purchase=False):
        # TODO: if polyphormed, sometimes 'You are physically incapable of picking anything up.'
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        quarantined = self._quarantined_floor_items()
        if any(item in quarantined for item, count in zip(items, counts) if count):
            raise AgentPanic('unrecovered shop gold container requires explicit recovery')
        assert len(items) > 0
        assert all(map(lambda item: item in self.items_below_me, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))
        if any(i.shop_status != Item.NOT_SHOP for i, c in zip(items, counts) if c):
            if not shop_purchase or self._shop_transaction is None:
                raise AgentPanic('shop goods require an explicit purchase transaction')

        letters = [self.letters_below_me[self.items_below_me.index(item)] for item in items]
        screens = [max(self.letters_below_me[:self.items_below_me.index(item) + 1].count('a') - 1, 0) for item in items]

        with self.panic_if_items_below_me_change():
            self.get_items_below_me()

        one_item = len(self.items_below_me) == 1
        with self.agent.atom_operation():
            if one_item:
                assert all((s in [0, None] for s in screens))
                if counts[0] != items[0].count:
                    # A count prefix picks exactly N, without first billing the
                    # entire stack and hoping to return the excess afterward.
                    self.agent.type_text(str(counts[0]))
                self.agent.step(A.Command.PICKUP)
            else:
                text = ' '.join((
                    ''.join([(str(count) if item.count != count else '') + letter
                             for letter, item, count, screen in zip(letters, items, counts, screens)
                             if count != 0 and screen == current_screen])
                    for current_screen in range(max(screens) + 1)))
                self.agent.step(A.Command.PICKUP, iter(list(text) + [A.MiscAction.MORE]))

            refused_heavy = False
            while re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                    'Continue?' in self.agent.message:
                if jf_config.HEAVY_LIFT_GUARD and \
                        re.search('You have (much trouble|extreme difficulty) lifting ', self.agent.message) and \
                        not re.search(r'pick-axe|mattock|broad pick|gold piece', self.agent.message):
                    # pickup.c lift_object: 'much trouble' means Stressed once lifted, 'extreme difficulty'
                    # Strained or worse ('You cannot fight while so heavily loaded'); the item split never plans
                    # that, so its weight was misread (a lycanthrope's corpse, see item_manager) -- leave it
                    # (not a digging tool, which the dive needs whatever it costs, nor gold)
                    self.agent.type_text('n')
                    refused_heavy = True
                    continue
                self.agent.type_text('y')
            if refused_heavy:
                self.unreachable_items_until[self._here()] = self.agent.blstats.time + 300
                raise AgentPanic('too heavy to lift without becoming Stressed')
            if 'You cannot reach the bottom of the pit' in self.agent.message:
                # standing at a pit's edge: its items are out of reach until we are in it; the gatherer
                # retried without the clock moving until the 'turn inactivity' guard fired (54 times)
                self.unreachable_items_until[self._here()] = self.agent.blstats.time + 300
                raise AgentPanic('items at the bottom of a pit are out of reach')
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
        if item.shop_status != Item.NOT_SHOP:
            # Consuming unpaid goods leaves a used-up bill after the inventory
            # entry disappears. Settle first, including after an interruption.
            return False
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

    @staticmethod
    def is_magicbane(item):
        """The observed artifact name, not an athame glyph or a gift message."""
        if not (item.is_weapon() and item.is_unambiguous() and item.object.name == 'athame'):
            return False
        if item.naming:
            return item.naming == 'Magicbane'
        name = (item.text or '').split(' (', 1)[0]
        return ' named ' not in name and ' called ' not in name and \
            re.search(r'(?:^| )Magicbane$', name) is not None

    def owned_magicbane(self, items=None, *, require_noncursed=True):
        """A Gnome Wizard's accessible utility weapon; no fictional DPS bonus.

        Unknown/cursed artifacts may be retained, but a new wield needs public
        noncurse evidence or the altar controller's one-time, observed gift
        transaction. Already holding it does not require another wield action.
        """
        character = self.agent.character
        if character.role != Character.WIZARD or getattr(character, 'race', None) != Character.GNOME:
            return None
        if items is None:
            items = self.items
        for item in items:
            if not self.is_magicbane(item) or item.count <= 0 or item.shop_status != Item.NOT_SHOP:
                continue
            if not require_noncursed:
                return item
            if item.status == Item.CURSED or re.search(r'\bcursed\b', item.text or ''):
                continue
            if self._public_noncursed(item) or (item is self.items.main_hand and item.equipped):
                return item
            altar = getattr(getattr(self.agent, 'global_logic', None), 'altar_preparation', None)
            if altar is not None and altar.confirmed_gift(item):
                return item
        return None

    def get_best_melee_weapon(self, items=None, *, return_dps=False, allow_unknown_status=False):
        if self.agent.character.role == Character.MONK:
            return None

        if items is None:
            items = self.items
        magicbane = self.owned_magicbane(items)
        if magicbane is not None:
            # Prefer its magic resistance/debuff utility explicitly, without
            # inflating the damage estimate used by combat and tool decisions.
            dps = utils.calc_dps(*self.agent.character.get_melee_bonus(magicbane, large_monster=False))
            return (magicbane, dps) if return_dps else magicbane
        magicbane_policy = self.agent.character.role == Character.WIZARD and \
            getattr(self.agent.character, 'race', None) == Character.GNOME
        # select the best
        best_item = None
        best_dps = utils.calc_dps(*self.agent.character.get_melee_bonus(None, large_monster=False))
        for item in flatten_items(items):
            if magicbane_policy and self.is_magicbane(item):
                continue  # do not bypass public BUC/ownership via the legacy DPS fallback
            # darts, shuriken, boomerangs, arrows and bolts only "bash" in melee; wielding them also
            # kept them out of the throwing set (eL1fe f68e24e)
            if item.is_weapon() and item.is_unambiguous() and item.object.name in MELEE_BASHING:
                continue
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

        if jf_config.SHOP_SAFETY or any(ammo.status == Item.CURSED for _, ammo in valid_combinations):
            # dothrow.c throwit: a cursed (or greased) missile slips / misfires 1 time in 7 in a random direction
            # (u.dx, u.dy = rn2(3) - 1). tr0 ran-gno-neu-mal s203 threw its known 'cursed crude dagger' at a newt in a
            # shop -- 'The crude dagger slips as you throw it! Annootok gets angry!' -- and b3 val-dwa-law-fem s604 an
            # orcish dagger of unknown BUC in an armor shop ('Yildizeli gets angry!'); both shopkeepers' wands of
            # striking killed the XL7 heroes. In a shop or with a shopkeeper in view only missiles known not to be
            # cursed fly; elsewhere, with a peaceful or the pet in view, no known-cursed ones.
            risk = self._missile_slip_risk()
            # A known cursed missile can veer into a protected target. Keep
            # this guarantee even when the broader unknown-BUC policy is off.
            if risk is not None:
                valid_combinations = [(l, a) for l, a in valid_combinations if a.status != Item.CURSED]
            if jf_config.SHOP_SAFETY and risk == 'shop':
                valid_combinations = [(l, a) for l, a in valid_combinations
                                      if a.status in (Item.UNCURSED, Item.BLESSED)]

        return valid_combinations

    def _missile_slip_risk(self):
        """SHOP_SAFETY: 'shop' (standing in a shop, or a shopkeeper in view), 'peaceful' (a peaceful or our pet in
        view) or None."""
        agent = self.agent
        try:
            bl = agent.blstats
            level = agent.current_level()
            if level.shop[bl.y, bl.x] or level.shop_interior[bl.y, bl.x] or \
                    utils.isin(agent.glyphs, G.SHOPKEEPER).any():
                return 'shop'
            if agent.monster_tracker.peaceful_monster_mask.any() or utils.isin(agent.glyphs, G.PETS).any():
                return 'peaceful'
        except Exception:
            return 'shop'
        return None

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

    def get_best_armorset(self, items=None, *, return_ac=False, allow_unknown_status=False, armor_up=False):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_items = [None] * O.ARM_NUM
        best_ac = [None] * O.ARM_NUM
        for item in items:
            if not item.is_armor() or not item.is_unambiguous():
                continue
            if (jf_config.KEEP_MAGIC_BOOTS or jf_config.BOOTS_KEEP) and power.never_wear(item):
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

            # spell.c percent_success: metallic body armor adds urole.spelarmr (10), metallic gloves 6, and a shield
            # heavier than a small one quarters the chance. A found chain mail took a Healer's healing from 0% to
            # 67% failure and it died on Dlvl 3-4 with full Pw; mithril took an orcish Wizard's force bolt to 79%.
            # Healers and Wizards skip those pieces (a metal helm or boots still leave the spells castable).
            # (found by our `nethackers evolve` run night1-r2)
            if self.agent.character.role in (Character.HEALER, Character.WIZARD) and (
                    (slot in (O.ARM_SUIT, O.ARM_GLOVES) and O.IRON <= item.object.metal <= O.MITHRIL) or
                    (slot == O.ARM_SHIELD and item.object.wt > 30 and item.object.name != 'shield of reflection')):
                continue

            # (ARMOR_UP: a tie keeps what we wear -- no swap for nothing)
            if best_ac[slot] is None or best_ac[slot] > ac or (armor_up and best_ac[slot] == ac and item.equipped):
                best_ac[slot] = ac
                best_items[slot] = item

        if return_ac:
            return best_items, best_ac
        return best_items

    ######## LOW-LEVEL STRATEGIES

    def _credit_clone_reserved(self):
        state = getattr(self.agent, '_credit_clone', None)
        return state is not None and state.phase not in ('done', 'abandoned')

    def _preparation_shop_reserved(self):
        from .. import shop_preparation
        return shop_preparation.reserves_inventory(self.agent)

    def _quarantined_floor_items(self, position=None, items=None):
        """An abandoned no-charge sack can still contain billable shop gold.

        Keep its whole contents out of generic collection while inside any
        shop or entrance.  Outside recovery and carried owned bags remain
        available; the reservation survives the original strategy's expiry.
        """
        level = self.agent.current_level()
        position = position or (self.agent.blstats.y, self.agent.blstats.x)
        items = self.items_below_me if items is None else items
        from .. import shop_preparation
        protected = shop_preparation.quarantined_items(self.agent, items, position)
        unresolved = getattr(self.agent, '_credit_clone_unrecovered', {})
        if not unresolved:
            return protected
        if not level.shop_interior[position] and not any(
                position in shop.entrances for shop in level.shops.records.values()):
            return protected
        return protected | set(flatten_items(
            item for item in (items or []) if item not in self.items.all_items and
            (level.key(), getattr(item, 'container_id', None)) in unresolved))

    def gather_items(self):
        return (
            self.pickup_and_drop_items()
                .before(self.check_containers())
                .before(self.prepare_owned_equipment())
                .before(self.wear_best_stuff())
                .before(self.read_scrolls_to_identify())
                .before(self.wear_identified_beneficial())
                .before(self.wear_starting_rings())
                .before(self.identify_amulet_by_wear())
                .before(self.wear_combat_only_rings_amulets())
                .before(self.shed_rings_amulets_when_hungry())
                .before(self.wand_engrave_identify())
                .before(self.use_spare_wishes())
                .before(self.wear_life_saving())
                .before(self.go_to_unchecked_containers())
                .before(self.check_items()
                        .before(self.go_to_item_to_pickup()).repeat().every(5)
                        .preempt(self.agent, [
                self.pickup_and_drop_items(),
                self.check_containers(),
            ])).repeat().condition(lambda: not self._credit_clone_reserved() and
                                   not self._preparation_shop_reserved())
        )

    @utils.debug_log('inventory.arrange_items')
    @Strategy.wrap
    def arrange_items(self):
        if self._credit_clone_reserved() or self._preparation_shop_reserved():
            yield False
            return
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
            quarantined = self._quarantined_floor_items()
            items_below_me = list(filter(lambda i: i.shop_status == Item.NOT_SHOP and not self.dropped_here(i),
                                         (i for i in below if i not in quarantined)))
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

    # WAND_ENGRAVE_TEXT: the DUST wands that still say nothing once the 'x' is written (engrave.c: opening, locking,
    # probing, undead turning and nothing have no engrave effect; secret door detection and create monster name
    # themselves only when zapnodir finds or shows something)
    _SILENT_ENGRAVE_WANDS = ('opening', 'locking', 'probing', 'undead turning', 'nothing', 'secret door detection',
                             'create monster')

    def _wand_types_after_text(self, message, item):
        """WAND_ENGRAVE_TEXT: possible types of a wand that wrote an 'x' onto the dust (the post_engr_text message is
        in `message`, or no message at all); None if the message says nothing about it (then the old rule)."""
        types = self._wand_types_from_message(message)
        if types is not None:
            return types
        silent = [p for p in O.possibilities_from_glyph(item.glyphs[0]) if p.name in self._SILENT_ENGRAVE_WANDS]
        return silent or None

    def _wand_types_from_message(self, message):
        """The engrave-test message table: the wand types a message names, or None."""
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
        }
        for msg, wand_types in mapping.items():
            if re.search(msg, message):
                return [O.from_name(w, nh.WAND_CLASS) for w in wand_types]
        return None

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

    def text_engrave_on(self):
        """WAND_ENGRAVE_TEXT, once the dive has started (coordinator/strong-dive: a text engrave in the grind reshuffled
        nearly every game from T~700 on and made paired comparisons unpaired; the grind keeps the old empty answer
        and the dive re-tests those wands once -- Medusa, the mazes and the castle are where the names matter)."""
        if not jf_config.WAND_ENGRAVE_TEXT:
            return False
        try:
            return bool(self.agent.global_logic.dive.diving)
        except AttributeError:
            return False

    def wand_retest_wanted(self, item):
        """WAND_ENGRAVE_TEXT: a wand whose grind engrave test said only 'glows, then fades' is tested once more with
        text once the dive has started."""
        if not item.glyphs or item.is_unambiguous() or not self.text_engrave_on():
            return False
        g = item.glyphs[0]
        im = self.item_manager
        if jf_config.DEEP_WAND_TEST and len(item.glyphs) == 1 and g not in im._already_engraved_glyphs and \
                item.comment != 'EMPT' and not self.is_known_empty(item) and 'unpaid' not in (item.text or ''):
            # DEEP_WAND_TEST (deep-arrivals): a wand never engrave-tested at all (picked up where the grind's test
            # didn't run -- cand-g jf53 s1, jf55 s7, jf47 s4 reached Medusa carrying one) is tested at the same quiet
            # dive moment: digging names itself, cold says 'ice cubes' (WAND_RESERVE / MEDUSA_FREEZE use them)
            return True
        return g in im._engraved_textless and g not in im._engraved_text and item.comment != 'EMPT' and \
            not self.is_known_empty(item)

    @utils.debug_log('inventory.wand_engrave_identify')
    @Strategy.wrap
    def wand_engrave_identify(self):
        if self.agent.character.prop.polymorph:
            yield False  # TODO: only for handless monsters (which cannot write)

        self.skip_engrave_counter -= 1
        if self.agent.character.prop.blind or self.skip_engrave_counter > 0 or self.agent.no_free_hand():
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
            if item.glyphs[0] in self.item_manager._already_engraved_glyphs and not self.wand_retest_wanted(item):
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

    @utils.debug_log('inventory.wand_text_retest')
    @Strategy.wrap
    def wand_text_retest(self):
        """WAND_ENGRAVE_TEXT, once the dive has started: at a quiet moment (nothing in view within 6, HP >= 60%, a bare
        floor square) engrave-test once more, with text, each wand whose grind test said only 'glows, then fades' --
        cold (the castle's moat), sleep/death, striking, magic missile (the mazes' minotaurs) are then named before
        Medusa's level. The dive's own exploration rarely runs wand_engrave_identify (it digs)."""
        if not self.text_engrave_on():
            yield False
            return
        agent = self.agent
        wands = [i for i in self.items if i.is_wand() and len(i.glyphs) == 1 and self.wand_retest_wanted(i)]
        if not wands:
            yield False
            return
        bl = agent.blstats
        prop = agent.character.prop
        level = agent.current_level()
        here = (level.key(), int(bl.y), int(bl.x))
        if prop.blind or prop.confusion or prop.stun or prop.hallu or prop.polymorph or agent.no_free_hand() or \
                bl.hitpoints < 0.6 * bl.max_hitpoints or getattr(self, '_retest_bad_square', None) == here or \
                int(agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & nh.BL_MASK_LEV or \
                level.objects[bl.y, bl.x] not in G.FLOOR or level.shop_interior[bl.y, bl.x]:
            yield False
            return
        if any(max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 6 for m in agent.get_visible_monsters()):
            yield False
            return
        yield True
        item = wands[0]
        agent.log(f'ENGRAVE re-test with text (the grind test said nothing): {item.text!r}')
        with agent.atom_operation():
            types = self._engrave_single_wand(item)
        if types is None:
            self._retest_bad_square = here   # (not a bare square after all: try elsewhere)
        else:
            self.item_manager._engraved_text.add(item.glyphs[0])   # (one re-test per wand, whatever it said)
            self.item_manager._glyph_to_possible_wand_types[item.glyphs[0]] = types
            self.item_manager._already_engraved_glyphs.add(item.glyphs[0])
            self.item_manager.possible_objects_from_glyph(item.glyphs[0])
        self.items.update(force=True)

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
        # WISH_TELEPORT_ROUTE: a wish prompt during the engrave-test comes from a wand of wishing (>= 2 wishes)
        self.agent._last_wand_use_step = self.agent.step_count

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
        wrote = [False]   # WAND_ENGRAVE_TEXT: text went onto the dust (the post-text message is in agent.message)
        text_path = [False]
        text_on = self.text_engrave_on()   # (dive only: the grind keeps the old empty answer)

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield letter
            if text_on and 'Do you want to add to the current engraving' in smsg():
                # a DUST wand: write text so that engrave.c prints its post_engr_text (ledger B016). 'n' wipes our
                # finger's 'x' and the wand writes a fresh 'Elbereth' in the same action (DUST: len/10 = 0 extra
                # turns), so the test square is left with a working Elbereth (coordinator's suggestion); a wand
                # already named before the prompt (zapnodir: light, enlightenment, a wish) gets the old empty answer
                text_path[0] = True
                yield 'n'
                for _ in range(4):
                    # 'You wipe out the message...--More--' / 'You write in the dust with a X wand.--More--' come
                    # before the text prompt (NLE misc: getlin and xwaitforspace are both set then)
                    if not self.agent._observation['misc'][2]:
                        break
                    yield A.TextCharacters.SPACE
                if self.agent._observation['misc'][1] and 'What do you want to write in the' in smsg() and \
                        self._wand_types_from_message(self.agent.message) is None:
                    wrote[0] = True
                    for ch in 'Elbereth':
                        yield ch
                    yield '\r'
                return
            if 'Do you want to add to the current engraving' in smsg():
                self.agent.type_text('y')
                # assert 'You add to the writing in the dust with' in smsg(), smsg()
                # self.agent.type_text(' ')
            r = self._determine_possible_wands(smsg(), item)
            if r is not None:
                possible_wand_types.extend(r)
                if 'glows, then fades' in self.agent.message and item.glyphs:
                    # nothing learned (the empty text prompt): WAND_ENGRAVE_TEXT re-tests it once the dive has started
                    self.item_manager._engraved_textless.add(item.glyphs[0])
            else:
                # wand exploded
                skip_engraving[0] = True

        self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

        if text_path[0]:
            if item.glyphs:
                self.item_manager._engraved_text.add(item.glyphs[0])
            r = self._wand_types_after_text(msg(), item) if wrote[0] else None
            if r is None and not wrote[0]:
                try:
                    # the old rule: named before the prompt, exploded, worn out, or 'glows, then fades'
                    r = self._determine_possible_wands(msg(), item)
                    if r is None:
                        return None
                except AssertionError:
                    r = None
            if r is None:
                r = [p for p in O.possibilities_from_glyph(item.glyphs[0]) if p.name not in ['light', 'wishing']]
            self.agent.log(f'ENGRAVE {item.text!r}: {[o.name for o in r]} (text written: {wrote[0]})')
            return r

        if skip_engraving[0]:
            return None

        if 'Do you want to add to the current engraving' in smsg():
            self.agent.type_text('q')
            assert smsg().strip() == 'Never mind.', smsg()

        return possible_wand_types

    @staticmethod
    def _public_buc(item):
        return public_buc(item)

    @staticmethod
    def _public_noncursed(item):
        # ItemManager defaults an unknown BUC to UNCURSED. Public text is
        # required here: this is preparation, not wear-testing unknown gear.
        return public_noncursed(item)

    def _armor_fitting_record(self, item):
        return dict(letter=self.items.get_letter(item), glyphs=tuple(item.glyphs),
                    name=item.object.name, naming=item.naming, modifier=item.modifier)

    def _armor_fitting_item(self, record):
        matches = [i for i in self.items if i.is_armor() and i.is_unambiguous() and
                   tuple(i.glyphs) == record['glyphs'] and i.object.name == record['name'] and
                   i.naming == record['naming'] and i.count == 1 and
                   (record['modifier'] is None or i.modifier == record['modifier'])]
        if len(matches) == 1 and matches[0].shop_status == Item.NOT_SHOP and self._public_noncursed(matches[0]):
            return matches[0]
        return None

    def layering_reserved_items(self):
        """Only current owned armor in the bounded fitting/restoration plan."""
        state = getattr(self, '_armor_fitting', None)
        if state is None:
            return []
        records = [state['target']] + [record for _, record in state['originals']]
        return list(dict.fromkeys(i for r in records if (i := self._armor_fitting_item(r)) is not None))

    def _armor_fitting_choice(self, plan, *, restore_only=False):
        state = self._armor_fitting
        if restore_only:
            state['restoring'] = True

        def request(action, item, detail=()):
            result = plan(action, item, detail)
            previous = self._equipment_preparation
            signature = (action, self.items.get_letter(item), item.object.name,
                         item.modifier, item.status, detail)
            if result is None and previous is not None and previous['signature'] == signature and previous['attempts'] >= 2:
                self.agent.log(f'EQUIPMENT_PREP layers_action_exhausted action={action} item={item.text!r}')
                return False
            return result

        target = self._armor_fitting_item(state['target'])
        current = getattr(self.items, state['slot'])
        if (target is None or self.agent.blstats.time > state['until'] or state['actions'] >= 8):
            state['restoring'] = True
        if target is not None and target is current:
            # A known noncursed trial can reveal negative armor. Undo it before
            # restoring outer layers; otherwise a safe wear-test becomes a
            # permanent AC penalty. Magic properties retain their utility.
            if target.modifier is not None and target.get_ac() > 0 and not (
                    target.object.mgc or target.object.power or target.object.blk):
                state['restoring'] = True
                if not restore_only and not self._armor_fitting_blocked(state['slot']):
                    choice = request('takeoff', target)
                    if choice is not False:
                        return choice
            state['restoring'] = True
        if not state['restoring']:
            for slot, record in state['originals']:
                worn = getattr(self.items, slot)
                if worn is None:
                    continue
                if worn is not self._armor_fitting_item(record):
                    state['restoring'] = True  # an upgrade/ownership change superseded the plan
                    break
                choice = request('takeoff', worn, (state['target']['letter'], state['target']['name'],
                                                   state['target']['modifier']))
                if choice is not False:
                    return choice
                state['restoring'] = True
                break
            if not state['restoring']:
                choice = request('wear', target)
                if choice is not False:
                    return choice
                state['restoring'] = True
        # Inner layer first. Completed commands are reconciled from slots, so
        # combat preemption after TAKEOFF/WEAR never blindly repeats a command.
        for slot, record in reversed(state['originals']):
            restore = self._armor_fitting_item(record)
            if restore is not None and getattr(self.items, slot) is None and not self._armor_fitting_blocked(slot):
                choice = request('wear', restore)
                if choice is not False:
                    return choice
        self.agent.log(f'EQUIPMENT_PREP layers_finished actions={state["actions"]} '
                       f'target={state["target"]["name"]!r}')
        self._armor_fitting = None
        return None

    def _armor_fitting_blocked(self, slot):
        return ((slot in ('shirt', 'suit') and self.items.cloak is not None) or
                (slot == 'shirt' and self.items.suit is not None))

    def _begin_armor_fitting(self, candidate, slot, plan):
        # Capture all layers before the first removal. No unknown/cursed outer
        # gear, welded weapon or corpse-bearing bare-hand transition is allowed.
        slots = (['cloak', 'suit'] if slot == 'shirt' else ['cloak'] if slot == 'suit' else []) + [slot]
        originals = []
        for name in slots:
            current = getattr(self.items, name)
            if current is None:
                continue
            if (not current.is_armor() or not current.is_unambiguous() or current.count != 1 or
                    current.shop_status != Item.NOT_SHOP or not self._public_noncursed(current) or
                    '(being worn' not in (current.text or '')):
                return None
            originals.append((name, self._armor_fitting_record(current)))
        self._armor_fitting = dict(target=self._armor_fitting_record(candidate), slot=slot,
                                  originals=originals, until=int(self.agent.blstats.time) + 60,
                                  actions=0, restoring=False)
        self.agent.log(f'EQUIPMENT_PREP fitting={candidate.text!r} '
                       f'layers={[r["name"] for _, r in originals]}')
        return self._armor_fitting_choice(plan)

    def _equipment_preparation_choice(self, *, armor_only=False, allow_unknown_modifier=False):
        agent, items = self.agent, self.items
        prop, bl = agent.character.prop, agent.blstats
        if (agent.character.role != Character.WIZARD or
                any((prop.blind, prop.confusion, prop.stun, prop.hallu, prop.polymorph)) or
                bl.carrying_capacity >= 2 or agent.get_visible_monsters() or
                self.shop_debt_pending() or self._credit_clone_reserved() or self._preparation_shop_reserved()):
            return None

        def plan(action, item, detail=()):
            signature = (action, items.get_letter(item), item.object.name,
                         item.modifier, item.status, detail)
            previous = self._equipment_preparation
            if previous is not None and previous['signature'] == signature and (
                    bl.time < previous['retry_after'] or previous['attempts'] >= 2):
                return None
            return action, item, signature

        # Protection costs nutrition, but a transient Hungry state or an
        # accessible meal is not a supply shortage. Keep its AC through meals;
        # only shed our own ring when Weak or worse with no usable owned food.
        if bl.hunger_state >= Hunger.HUNGRY:
            if bl.hunger_state < Hunger.WEAK and getattr(self, '_armor_fitting', None) is not None and \
                    not agent.hands_welded():
                return self._armor_fitting_choice(plan, restore_only=True)
            if armor_only or bl.hunger_state < Hunger.WEAK or agent.survival.usable_food():
                return None
            ring = next((i for i in items if i.category == nh.RING_CLASS and i.equipped and
                         i.is_unambiguous() and i.object.name == 'protection' and
                         re.search(r'\(on (?:left|right) hand\)', i.text or '') and
                         tuple(i.glyphs) in self._preparation_protection_rings and
                         i.shop_status == Item.NOT_SHOP and self._public_noncursed(i)), None)
            if ring is not None and not agent.hands_welded() and (
                    items.gloves is None or items.gloves.status != Item.CURSED):
                return plan('remove_ring', ring)
            return None

        slots = [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'),
                 (O.ARM_BOOTS, 'boots'), (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'),
                 (O.ARM_CLOAK, 'cloak')]
        worn = [getattr(items, name) for _, name in slots if getattr(items, name) is not None]
        if items.main_hand is not None:
            worn.append(items.main_hand)
        worn += [i for i in items if i.equipped and i.category in (nh.RING_CLASS, nh.AMULET_CLASS)]
        cursed = [i for i in worn if i.equipped and self._public_buc(i) == Item.CURSED and
                  i.shop_status == Item.NOT_SHOP]
        recovery = getattr(agent, 'equipment_recovery', None)
        recovered_cursed = [i for i in items if recovery is not None and
                            recovery.retain_cursed_armor(i) and self._public_buc(i) == Item.CURSED and
                            i.is_unambiguous() and (i.object.mgc or i.object.power or
                            i.modifier is not None and i.get_ac() < 0)]
        if not armor_only and (cursed or recovered_cursed):
            # read.c: ordinary uncursed removes worn/wielded curses; blessed
            # affects the whole pack. Never confuse a carried cursed dagger
            # with a blocking worn item or spend a cursed/unknown scroll.
            scrolls = [i for i in items if i.category == nh.SCROLL_CLASS and i.is_unambiguous() and
                       i.object.name == 'remove curse' and i.shop_status == Item.NOT_SHOP and
                       i.count > 0 and self._public_noncursed(i)]
            # Only blessed remove curse reaches carried armor in 3.6.7.
            # Recovered provenance is rechecked against this top-level pack;
            # never wear/wield cursed gear to extend an ordinary scroll's scope.
            scroll = next((i for i in scrolls if recovered_cursed and
                           self._public_buc(i) == Item.BLESSED), None)
            if scroll is None and cursed:
                scroll = next(iter(scrolls), None)
            if scroll is not None and all(i.shop_status == Item.NOT_SHOP for i in flatten_items(items)):
                targets = cursed + (recovered_cursed if self._public_buc(scroll) == Item.BLESSED else [])
                return plan('remove_curse', scroll, tuple(sorted(items.get_letter(i) for i in targets)))

        if agent.hands_welded():
            return None
        if getattr(self, '_armor_fitting', None) is not None:
            return self._armor_fitting_choice(plan)
        magicbane = self.owned_magicbane()
        if not armor_only and magicbane is not None and magicbane is not items.main_hand and not magicbane.equipped and \
                (items.main_hand is None or items.main_hand.status != Item.CURSED):
            return plan('wield_magicbane', magicbane)
        safe_armor = [i for i in items if i.is_armor() and i.is_unambiguous() and
                      i.shop_status == Item.NOT_SHOP and self._public_noncursed(i) and
                      (i.modifier is not None or allow_unknown_modifier) and
                      not re.search(r'\b(rusty|corroded|burnt|rotted)\b', i.text or '') and
                      i.object.name not in ('dunce cap', 'helm of opposite alignment',
                                            'gauntlets of fumbling', 'fumble boots', 'levitation boots')]
        best = self.get_best_armorset(items=safe_armor, armor_up=True)
        for slot, name in slots:
            candidate, current = best[slot], getattr(items, name)
            if candidate is None or candidate is current or candidate.equipped:
                continue
            if candidate.modifier is None and current is not None:
                continue  # learn safe armor enchantments only in an empty slot
            if slot == O.ARM_SHIELD and items.main_hand is not None and getattr(items.main_hand.objs[0], 'bi', False):
                continue
            if slot == O.ARM_GLOVES and items.main_hand is not None and items.main_hand.status == Item.CURSED:
                continue
            if slot == O.ARM_GLOVES and any(i.equipped and i.is_corpse() for i in items):
                continue  # never expose bare hands while wielding a corpse
            if current is not None and (not current.is_armor() or not current.is_unambiguous() or current.modifier is None or
                    '(being worn' not in (current.text or '') or
                    current.shop_status != Item.NOT_SHOP or not self._public_noncursed(current) or
                    ((current.object.mgc or current.object.power or current.object.blk) and
                     current.object.name != candidate.object.name)):
                continue
            if candidate.get_ac() >= (current.get_ac() if current is not None else 0) and not (
                    current is None and (candidate.modifier is None or candidate.object.mgc or
                                         candidate.object.power or candidate.object.blk)):
                continue
            # Retain get_best_armorset's Wizard suit/glove/shield exclusions,
            # and avoid adding a metal helm/boot penalty where none existed.
            previous = self._equipment_preparation
            continuing_swap = (current is None and previous is not None and
                               previous.get('metal_slot') == slot and previous['signature'][0] == 'takeoff' and
                               previous['signature'][-1] == (items.get_letter(candidate), candidate.object.name,
                                                            candidate.modifier))
            if not continuing_swap and slot in (O.ARM_HELM, O.ARM_BOOTS) and O.IRON <= candidate.object.metal <= O.MITHRIL and \
                    candidate.object.name != 'helm of brilliance' and (current is None or
                    not O.IRON <= current.object.metal <= O.MITHRIL or current.object.name == 'helm of brilliance'):
                continue
            layered = name in ('shirt', 'suit') and self._armor_fitting_blocked(name)
            if layered or candidate.modifier is None:
                return self._begin_armor_fitting(candidate, name, plan)
            return plan('takeoff', current, (items.get_letter(candidate), candidate.object.name,
                                            candidate.modifier)) if current is not None else plan('wear', candidate)

        if armor_only:
            return None

        # A known positive protection ring is useful before contact as well as
        # during combat. Keep a free finger; no speculative ring swapping.
        if sum(i.equipped and i.category == nh.RING_CLASS for i in items) < 2 and (
                items.gloves is None or items.gloves.status != Item.CURSED):
            rings = [i for i in items if i.category == nh.RING_CLASS and not i.equipped and
                     i.is_unambiguous() and i.object.name == 'protection' and i.modifier is not None and
                     i.modifier > 0 and i.shop_status == Item.NOT_SHOP and self._public_noncursed(i)]
            if rings:
                return plan('put_on', max(rings, key=lambda i: i.modifier))
        return None

    @utils.debug_log('inventory.prepare_owned_equipment')
    @Strategy.wrap
    def prepare_owned_equipment(self, *, armor_only=False, allow_unknown_modifier=False):
        """One safe, observed equipment action; replan after each real result."""
        options = dict(armor_only=armor_only, allow_unknown_modifier=allow_unknown_modifier)
        if self._equipment_preparation_choice(**options) is None:
            yield False
            return
        yield True
        choice = self._equipment_preparation_choice(**options)
        if choice is None:
            return
        action, item, signature = choice
        previous = self._equipment_preparation
        attempts = previous['attempts'] if previous is not None and previous['signature'] == signature else 0
        state = dict(signature=signature, attempts=attempts + 1,
                     retry_after=int(self.agent.blstats.time) + 100, outcome='interrupted')
        if action == 'takeoff' and O.IRON <= item.object.metal <= O.MITHRIL:
            state['metal_slot'] = item.object.sub
        self._equipment_preparation = state
        if getattr(self, '_armor_fitting', None) is not None and action in ('wear', 'takeoff'):
            self._armor_fitting['actions'] += 1
        before = tuple((i.text, i.count, i.status, i.equipped) for i in self.items)
        letter = self.items.get_letter(item)
        self.agent.log(f'EQUIPMENT_PREP action={action} item={item.text!r} attempt={state["attempts"]}')
        if action == 'remove_curse':
            state['targets_before'] = [(self.items.get_letter(i), i.text) for i in self.items
                                       if self.items.get_letter(i) in signature[-1]]
            self.agent.log(f'EQUIPMENT_PREP uncurse_targets={state["targets_before"]}')
        try:
            if action == 'takeoff':
                self.takeoff(item)
            elif action == 'wear':
                self.wear(item, smart=False)
            elif action == 'wield_magicbane':
                self.wield(item, smart=False)
            else:
                command, prompt = {'remove_curse': (A.Command.READ, 'What do you want to read?'),
                                   'put_on': (A.Command.PUTON, 'What do you want to put on?'),
                                   'remove_ring': (A.Command.REMOVE, 'What do you want to remove?')}[action]
                finger = 'r' if any(i.equipped and '(on left hand)' in (i.text or '') for i in self.items) else 'l'
                if action == 'put_on':
                    self._preparation_protection_rings.add(tuple(item.glyphs))
                # Persist ownership before an atomic callback can preempt us;
                # the combat-only manager must not remove this ring when quiet.
                if action in ('put_on', 'remove_ring'):
                    self._module_worn.difference_update(item.glyphs)

                def answers():
                    if prompt in self.agent.single_message:
                        yield letter
                    if action == 'put_on' and 'Which ring-finger' in self.agent.single_message:
                        yield finger

                with self.agent.atom_operation():
                    self.agent.step(command, answers())
            self.items.update()  # observed inventory only; never force-open containers
            after = tuple((i.text, i.count, i.status, i.equipped) for i in self.items)
            state['outcome'] = 'observed_change' if after != before else 'no_observed_change'
            if action == 'remove_curse':
                state['targets_after'] = [(self.items.get_letter(i), i.text, self._public_buc(i))
                                         for i in self.items if self.items.get_letter(i) in signature[-1]]
                self.agent.log(f'EQUIPMENT_PREP uncurse_observed={state["targets_after"]}')
        finally:
            self.agent.log(f'EQUIPMENT_PREP outcome={state["outcome"]} action={action} '
                           f'retry_after={state["retry_after"]}')

    @utils.debug_log('inventory.read_enchant_armor')
    @Strategy.wrap
    def read_enchant_armor(self):
        """Legacy scheduler: the same public scroll/target safety as maintenance."""
        from ..armor_maintenance import _safe_scroll, _safe_worn
        agent = self.agent

        def choice():
            prop, bl = agent.character.prop, agent.blstats
            if (not jf_config.ARMOR_UP or
                    getattr(self, '_armor_fitting', None) is not None or
                    any((prop.blind, prop.confusion, prop.stun, prop.hallu, prop.polymorph)) or
                    bl.time < getattr(self, '_enchant_read_until', -1) or
                    bl.hunger_state > Hunger.NOT_HUNGRY or bl.carrying_capacity >= 2 or
                    agent.get_visible_monsters() or self.shop_debt_pending() or
                    self._credit_clone_reserved() or self._preparation_shop_reserved() or
                    _safe_worn(self.items) is None):
                return None
            return next((i for i in self.items if _safe_scroll(i)), None)

        if choice() is None:
            yield False
            return
        yield True
        scroll = choice()
        if scroll is None:
            return
        self._enchant_read_until = agent.blstats.time + 20
        letter = self.items.get_letter(scroll)
        agent.log(f'ARMOR_UP reading {scroll.text!r} (worn: {[i.text for i in _safe_worn(self.items)]})')

        def gen():
            if 'What do you want to read?' in agent.single_message:
                yield letter
        with agent.atom_operation():
            agent.step(A.Command.READ, gen())
        self.items.update()

    @utils.debug_log('inventory.wear_best_stuff')
    @Strategy.wrap
    def wear_best_stuff(self):
        if self.agent.character.role == Character.WIZARD:
            # Gathering previously bypassed preparation's ownership/BUC and
            # magic-property checks. Use its bounded one-action policy here
            # too; an empty slot may safely reveal a noncursed item's modifier.
            yield from self.prepare_owned_equipment(
                armor_only=True, allow_unknown_modifier=True).strategy()
            return
        if self.agent.hands_welded():
            yield False   # armor can't come off (or go on over it) with the hands welded
            return
        yielded = False
        while 1:
            best_armorset = self.get_best_armorset(armor_up=jf_config.ARMOR_UP)

            # TODO: twoweapon
            for slot, name in [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'),
                               (O.ARM_BOOTS, 'boots'), (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'),
                               (O.ARM_CLOAK, 'cloak')]:
                if best_armorset[slot] == getattr(self.items, name) or \
                        (getattr(self.items, name) is not None and getattr(self.items, name).status == Item.CURSED):
                    continue
                additional_cond = True
                if jf_config.ROBUST_FIXES:
                    # a refused wear/take-off (see _wear_refused) waits before the slot is tried again
                    additional_cond = self.agent.blstats.time >= \
                        getattr(self.agent, '_wear_blocked_until', {}).get(slot, 0)
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
                        done = self.takeoff(self.items.cloak)
                    elif slot == O.ARM_SHIRT and self.items.suit is not None:
                        done = self.takeoff(self.items.suit)
                    elif getattr(self.items, name) is not None:
                        done = self.takeoff(getattr(self.items, name))
                    else:
                        assert best_armorset[slot] is not None
                        done = self.wear(best_armorset[slot])
                    if not done and jf_config.ROBUST_FIXES:
                        # any refusal (the cloak or suit in the way, "Don't even bother.", a cursed item) passes
                        # no turn: without a wait this slot's step repeats forever in this very loop
                        blocked = getattr(self.agent, '_wear_blocked_until', {})
                        blocked[slot] = max(blocked.get(slot, 0), self.agent.blstats.time + 20)
                        self.agent._wear_blocked_until = blocked
                    break
            else:
                break

        if not yielded:
            yield False

    ITEM_INSPECTION_RETRY_TURNS = 100
    ITEM_INSPECTION_MAX_FAILURES = 64

    def _inspection_backoff_active(self, key):
        failure = self._item_inspection_failures.get(key)
        if failure is None:
            return False
        until, blocker = failure
        expired = self.agent.blstats.time >= until
        if not expired and blocker is not None:
            # A remembered item glyph at range is not evidence a monster left.
            # Match sessile-memory's nearby, sighted observation requirement.
            y, x = blocker
            bl = self.agent.blstats
            prop = self.agent.character.prop
            expired = (not prop.blind and not prop.hallu and
                       not getattr(self.agent, '_terrain_view', False) and
                       max(abs(bl.y - y), abs(bl.x - x)) <= 1 and
                       not self.agent.monster_tracker.monster_mask[y, x] and
                       ((bl.y, bl.x) == blocker or
                        self.agent.glyphs[y, x] not in G.MONS | G.PETS | G.INVISIBLE_MON))
        if expired:
            del self._item_inspection_failures[key]
            return False
        return True

    def _record_inspection_failure(self, key, error):
        """Only navigation's explicit failures count; preemption is not failure."""
        reason = str(error)
        monster = re.fullmatch(r'Monster on a next tile when moving: \((\d+),(\d+)\)', reason)
        if monster is None and reason not in (
                'end point is no longer accessible', 'go_to: path blocked at its first step'):
            return
        if self.agent.current_level().key() != key[:2]:
            return
        if (self.agent.blstats.y, self.agent.blstats.x) == key[2:]:
            return
        blocker = tuple(map(int, monster.groups())) if monster is not None else None
        # When the endpoint itself became occupied, retain that concrete
        # obstacle too. Other failed paths expire on the bounded clock.
        if blocker is None and self.agent.monster_tracker.monster_mask[key[2:]]:
            blocker = key[2:]
        now = int(self.agent.blstats.time)
        for old_key, (until, _) in list(self._item_inspection_failures.items()):
            if until <= now:
                del self._item_inspection_failures[old_key]
        until = now + self.ITEM_INSPECTION_RETRY_TURNS
        self._item_inspection_failures[key] = (until, blocker)
        while len(self._item_inspection_failures) > self.ITEM_INSPECTION_MAX_FAILURES:
            oldest = min(self._item_inspection_failures, key=lambda k: self._item_inspection_failures[k][0])
            del self._item_inspection_failures[oldest]
        self.agent.log(f'ITEM_INSPECTION defer target={key} until={until} reason={reason}')

    @utils.debug_log('inventory.check_items')
    @Strategy.wrap
    def check_items(self, navigation_backoff=True):
        if self._preparation_shop_reserved():
            yield False
            return
        mask = utils.isin(self.agent.glyphs, G.OBJECTS, G.BODIES, G.STATUES)
        if not mask.any():
            yield False

        dis = self.agent.bfs()

        mask &= self.agent.current_level().item_count == 0
        if not mask.any():
            yield False

        mask &= dis > 0
        level_key = self.agent.current_level().key()
        if navigation_backoff:
            for y, x in zip(*mask.nonzero()):
                if self._inspection_backoff_active((*level_key, int(y), int(x))):
                    mask[y, x] = False
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]
        target_key = (*level_key, int(target_y), int(target_x))

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            try:
                self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))
            except AgentPanic as error:
                if navigation_backoff:
                    self._record_inspection_failure(target_key, error)
                raise
            else:
                self._item_inspection_failures.pop(target_key, None)

    @utils.debug_log('inventory.go_to_unchecked_containers')
    @Strategy.wrap
    def go_to_unchecked_containers(self):
        if self._credit_clone_reserved() or self._preparation_shop_reserved():
            yield False
            return
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
            items = self.agent.current_level().items[y, x]
            quarantined = self._quarantined_floor_items((int(y), int(x)), items)
            available = [item for item in items if item not in quarantined]
            if not available:
                mask[y, x] = False
                continue
            for item in available:
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
        if self._credit_clone_reserved() or self._preparation_shop_reserved():
            yield False
            return
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
        quarantined = self._quarantined_floor_items()
        for item in self.agent.inventory.items_below_me:
            if item in quarantined:
                continue
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
                    if fail_msg is not None and fail_msg.startswith('BOX_TRAP_SAFE'):
                        # BOX_TRAP_SAFE: never open it; the square is skipped from now on (as a multi-container
                        # square: check_containers and go_to_unchecked_containers leave it alone)
                        self.multi_container_squares.add(self._here())
                        continue
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

    def _food_for_sale(self, dis, *, shop_id=None, position=None):
        level = self.agent.current_level()
        gold = self.agent.blstats.gold
        best = None
        for y, x in zip(*(level.shop_interior & (level.item_count > 0)).nonzero()):
            if position is not None and (y, x) != position:
                continue
            if dis[y, x] == -1 or level.shops.shop_at(y, x) is None:
                continue
            if shop_id is not None and level.shops.shop_at(y, x).shop_id != shop_id:
                continue
            for item in level.items[y, x]:
                if item.shop_status != Item.FOR_SALE or not item.is_unambiguous():
                    continue
                nutrition = self.BUY_FOOD_NUTRITION.get(item.object.name)
                price = quantity_price(item)
                shop = level.shops.shop_at(y, x)
                if nutrition is None or price is None or price > gold + self.shop_credit(shop):
                    continue
                score = nutrition / price - dis[y, x] / 1000
                if best is None or score > best[0]:
                    best = (score, int(y), int(x), item.object.name, price)
        return best

    def _unpaid_items(self):
        return [i for i in flatten_items(self.items) if i.shop_status == Item.UNPAID]

    def shop_debt_pending(self):
        return self._shop_transaction is not None or bool(self._unpaid_items())

    def _transaction_shop(self):
        transaction = self._shop_transaction
        level = self.agent.current_level()
        if transaction is None or level.key() != transaction.level_key:
            return None
        return level.shops.records.get(transaction.shop_id)

    def shop_exit_blocked(self, from_y, from_x, to_y, to_x):
        """Ordinary navigation must settle before leaving the owning shop.

        Combat movement deliberately does not use this guard: escaping an
        immediate threat can interrupt a transaction without erasing its debt.
        """
        shop = self._transaction_shop()
        if shop is None:
            return False
        inside = lambda y, x: shop.interior[y, x] or (y, x) in shop.entrances
        return bool(inside(from_y, from_x) and not inside(to_y, to_x))

    def _remember_unpaid(self):
        if self._shop_transaction is None and self._unpaid_items():
            bl = self.agent.blstats
            shop = self.agent.current_level().shops.shop_at(bl.y, bl.x, include_entrance=True)
            if shop is not None:
                self._shop_transaction = ShopTransaction(
                    self.agent.current_level().key(), shop.shop_id, (bl.y, bl.x))

    def _visible_shop_keeper(self, shop):
        """Historical keeper positions cannot establish the current owner."""
        if shop is None:
            return None
        area = shop.interior.copy()
        for position in shop.entrances:
            area[position] = True
        candidates = list(zip(*(area & self.agent.monster_tracker.peaceful_monster_mask &
                                utils.isin(self.agent.glyphs, G.SHOPKEEPER)).nonzero()))
        return candidates[0] if len(candidates) == 1 else None

    def _visible_transaction_keeper(self):
        """Require an observed owner, including when PAY selects automatically."""
        return self._visible_shop_keeper(self._transaction_shop())

    def invalidate_shop_credit(self, shop=None):
        credits = getattr(self, '_shop_credits', {})
        if shop is None:
            credits.clear()
        else:
            credits.pop((self.agent.current_level().key(), shop.shop_id), None)

    def shop_credit(self, shop):
        """Known credit is owner scoped, and is only a provisional budget."""
        level = self.agent.current_level()
        if shop is None or level.shops.records.get(shop.shop_id) is not shop:
            return 0
        entry = getattr(self, '_shop_credits', {}).get((level.key(), shop.shop_id))
        if entry is None or entry[0] is not shop:
            return 0
        prop = getattr(self.agent.character, 'prop', None)
        if getattr(prop, 'hallu', False) or getattr(prop, 'blind', False) or \
                self._visible_shop_keeper(shop) is None:
            self.invalidate_shop_credit(shop)
            return 0
        return entry[1]

    def refresh_shop_credit(self, shop):
        """Query `$` inside this shop; ambiguous reports invalidate the cache.

        This query does not pay, sell, move, or create a transaction.  A live
        query is required again before using credit to fund a purchase.
        """
        self.invalidate_shop_credit(shop)
        agent = self.agent
        level = agent.current_level()
        prop = getattr(agent.character, 'prop', None)
        if shop is None or level.shops.records.get(shop.shop_id) is not shop or \
                level.shops.shop_at(agent.blstats.y, agent.blstats.x) is not shop or \
                self._visible_shop_keeper(shop) is None or \
                getattr(prop, 'hallu', False) or getattr(prop, 'blind', False):
            return None
        origin = (level.key(), agent.blstats.y, agent.blstats.x)
        amount = None
        with agent.atom_operation():
            agent.step('$')
            if origin == (agent.current_level().key(), agent.blstats.y, agent.blstats.x) and \
                    agent.current_level().shops.records.get(shop.shop_id) is shop and \
                    self._visible_shop_keeper(shop) is not None:
                amount = current_shop_credit(agent.message)
                if amount is not None:
                    if not hasattr(self, '_shop_credits'):
                        self._shop_credits = {}
                    self._shop_credits[(level.key(), shop.shop_id)] = (shop, amount)
                    agent.log(f'SHOP credit shop={shop.shop_id} amount={amount}')
        return amount

    def _pay_shop_bill(self):
        self.invalidate_shop_credit(self._transaction_shop())
        # A sole visible keeper can belong to another shop. Its automatic
        # zero-debt reply says nothing about this transaction's owner's bill.
        if self._visible_transaction_keeper() is None:
            return ''

        def answers():
            for _ in range(100):
                message = self.agent.single_message
                if 'Pay whom?' in message:
                    # With multiple keepers visible NetHack asks for a map
                    # target. Only the current owner is eligible; historical
                    # keeper positions can now contain a different keeper.
                    owner = self._visible_transaction_keeper()
                    if owner is None:
                        yield A.Command.ESC
                        return
                    target_y, target_x = owner
                    directions = {(-1, -1): 'y', (-1, 0): 'k', (-1, 1): 'u',
                                  (0, -1): 'h', (0, 1): 'l',
                                  (1, -1): 'b', (1, 0): 'j', (1, 1): 'n'}
                    # getpos uses map coordinates, like the existing travel
                    # cursor driver. Complete selection before inventory reads.
                    for _ in range(sum(self.agent.glyphs.shape)):
                        y, x = self.agent.cursor_pos
                        if (y, x) == (target_y, target_x):
                            yield '.'
                            break
                        yield directions[(int(np.sign(target_y - y)), int(np.sign(target_x - x)))]
                    else:
                        yield A.Command.ESC
                        return
                elif 'Itemized billing?' in message:
                    yield 'n'  # buy the whole bill; no per-item menu can strand us
                elif 'Pay?' in message:
                    yield 'y'
                elif self.agent._observation['misc'][2]:
                    yield A.MiscAction.MORE
                else:
                    return

        self.agent.step(A.Command.PAY, answers())
        message = self.agent.message
        self.items.update(force=True)
        return message

    def pay_or_drop_unpaid(self):
        """Resume payment or return goods inside their originating shop.

        Keep intent until an explicit zero-debt response: consumed merchandise
        and debit can remain on the bill with no unpaid inventory entry.
        """
        self._remember_unpaid()
        transaction = self._shop_transaction
        shop = self._transaction_shop()
        if transaction is None or shop is None or self._shop_settling:
            return
        self._shop_settling = True
        try:
            bl = self.agent.blstats
            if not shop.interior[bl.y, bl.x]:
                distance = self.agent.bfs()
                positions = list(zip(*(shop.interior & (distance >= 0)).nonzero()))
                if not positions:
                    transaction.retry_after = bl.time + 10
                    return
                self.agent.go_to(*min(positions, key=lambda p: distance[p]))
                bl = self.agent.blstats
                if self.agent.current_level().key() != transaction.level_key or not shop.interior[bl.y, bl.x]:
                    return
            # Finish bookkeeping before callbacks may eat, arrange, or move.
            with self.agent.atom_operation():
                transaction.retry_after = self.agent.blstats.time + 10
                message = self._pay_shop_bill()
                bl = self.agent.blstats
                if self.agent.current_level().key() != transaction.level_key or not shop.interior[bl.y, bl.x]:
                    return  # e.g. teleportitis during payment; never return to another owner
                unpaid = self._unpaid_items()
                if unpaid:
                    returnable = [i for i in unpaid if i in self.items.all_items and
                                  i.can_be_dropped_from_inventory()]
                    if returnable:
                        self.agent.log(f'SHOP returning unpaid goods: {[i.text for i in returnable]}')
                        self.drop(returnable, smart=False)
                        self.items.update(force=True)
                bl = self.agent.blstats
                if self.agent.current_level().key() != transaction.level_key or not shop.interior[bl.y, bl.x]:
                    return
                if not self._unpaid_items():
                    # Successful purchase/return alone cannot exclude an older
                    # consumed-item bill. PAY's zero-debt reply does.
                    if not bill_cleared(message):
                        message = self._pay_shop_bill()
                    if bill_cleared(message) and not self._unpaid_items():
                        self._shop_transaction = None
                        self.agent.log('SHOP bill settled')
                        if transaction.credit_budgeted:
                            # PAY invalidates the old balance.  Establish any
                            # remainder anew only after the whole bill clears.
                            self.refresh_shop_credit(shop)
        finally:
            self._shop_settling = False

    @utils.debug_log('inventory.settle_shop_debt')
    @Strategy.wrap
    def settle_shop_debt(self):
        self._remember_unpaid()
        transaction = self._shop_transaction
        if transaction is None or self._shop_settling or self._transaction_shop() is None or \
                self.agent.blstats.time < transaction.retry_after or self.agent.get_visible_monsters():
            yield False
            return
        yield True
        self.pay_or_drop_unpaid()

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
        if self._credit_clone_reserved() or self._preparation_shop_reserved() or self.shop_debt_pending() or not jf_config.SELL_PRICE_ID or agent.character.prop.hallu or agent.character.prop.blind or \
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
        owner = level.shops.shop_at(bl.y, bl.x)
        if owner is None:
            yield False
            return
        free = owner.interior & (level.item_count == 0) & (dis != -1)
        if not free.any():
            yield False
            return
        yield True
        if self._credit_clone_reserved() or self._preparation_shop_reserved():
            return
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

    def food_purchase_target(self):
        """A modest ongoing supply goal, independent of the first branch visit."""
        agent = self.agent
        route = getattr(getattr(agent, 'global_logic', None), 'wizard_route', None)
        preparation_target = getattr(route, 'preparation_food_target', None)
        if preparation_target is not None:
            return min(jf_config.BUY_FOOD_UNTIL, preparation_target)
        character = agent.character
        if getattr(character, 'role', None) == Character.WIZARD and \
                getattr(character, 'race', None) == Character.GNOME and \
                (agent._carries_digging_tool() or self._preparation_shop_reserved()):
            # An interrupted/returning live visit owns the time and inventory
            # budget. Once it finishes, ordinary resupply becomes available.
            # The reservation keeps this target stable while the pick is on
            # the floor. Ordinary no-tool provisioning retains its old limit.
            if route is not None and getattr(route, 'active', False):
                return None
            from ..wizard_route import FOOD_TARGET
            return min(jf_config.BUY_FOOD_UNTIL, FOOD_TARGET)
        return None if agent._carries_digging_tool() else jf_config.BUY_FOOD_UNTIL

    def _food_buying_allowed(self, preparation_visit=None):
        """Recheck supply demand and live safety before exposing shop debt."""
        agent = self.agent
        bl = agent.blstats
        if preparation_visit is not None:
            from .. import shop_preparation
            if not shop_preparation.permits_purchase(agent, preparation_visit):
                return False
        elif self._preparation_shop_reserved():
            return False
        # Deposited principal may need this owner's credit for a safe rollback.
        # A paused clone keeps that reservation until recovery or abandonment.
        if self._credit_clone_reserved() or self.shop_debt_pending() or not jf_config.BUY_FOOD or \
                bl.hunger_state >= Hunger.FAINTING:
            return False
        target = self.food_purchase_target()
        if target is None or self.carried_nutrition() >= target or agent.get_visible_monsters():
            return False
        if jf_config.SHOP_GUARD and agent.character.teleportitis and not agent.character.teleport_control:
            # a random teleport between the pickup and the payment takes the goods out unpaid: Kops and an
            # angry shopkeeper (base4-jf14 s10, dead)
            return False
        return True

    @utils.debug_log('inventory.buy_food')
    @Strategy.wrap
    def buy_food(self, *, preparation_visit=None):
        agent = self.agent
        bl = agent.blstats
        if not self._food_buying_allowed(preparation_visit):
            yield False
        level = agent.current_level()
        if not level.shop_interior.any():
            yield False
        dis = agent.bfs()
        target_level_key = level.key()
        target = self._food_for_sale(dis, shop_id=preparation_visit.shop_id,
                                    position=(bl.y, bl.x)) if preparation_visit is not None else \
            self._food_for_sale(dis)
        if target is None:
            yield False
        _, y, x, name, price = target
        key = (level.dungeon_number, level.level_number, y, x)
        if jf_config.BUY_FOOD_GIVEUP and self._buy_food_blocked.get(key, (0, -1))[1] > bl.time:
            yield False
        yield True
        if not self._food_buying_allowed(preparation_visit):
            return
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
        if not self._food_buying_allowed(preparation_visit):
            return
        # Walking may spend gold, change a quote, reveal different stock or be
        # interrupted. Purchase only from the live owner and live floor list.
        self.get_items_below_me()
        bl = agent.blstats
        level = agent.current_level()
        shop = level.shops.shop_at(bl.y, bl.x)
        if level.key() != target_level_key or (bl.y, bl.x) != (y, x) or shop is None or \
                not self._food_buying_allowed(preparation_visit):
            return
        budget = int(bl.gold)
        if self.shop_credit(shop):
            # Credit may have been spent by an interrupted operation.  Never
            # make unpaid stock ours on the strength of an old balance alone.
            credit = self.refresh_shop_credit(shop)
            bl = agent.blstats
            if level is not agent.current_level() or (bl.y, bl.x) != (y, x) or \
                    not self._food_buying_allowed(preparation_visit):
                return
            self.get_items_below_me()
            budget = int(bl.gold) + (credit if credit is not None else 0)
        items = [i for i in self.items_below_me
                 if i.shop_status == Item.FOR_SALE and i.is_unambiguous() and i.object.name == name
                 and quantity_price(i) is not None and quantity_price(i) <= budget]
        if not items or not self._food_buying_allowed(preparation_visit):
            return
        price = quantity_price(items[0])
        agent.log(f'SHOP buying {name} for {price} (gold {agent.blstats.gold}, budget {budget})')
        # Persist before pickup: its atom exit can preempt this strategy.
        with agent.atom_operation():
            self._shop_transaction = ShopTransaction(
                level.key(), shop.shop_id, (bl.y, bl.x), credit_budgeted=budget > int(bl.gold))
            self.pickup(items[0], 1, shop_purchase=True)
            self.items.update(force=True)
        self.pay_or_drop_unpaid()

    @utils.debug_log('inventory.go_to_item_to_pickup')
    @Strategy.wrap
    def go_to_item_to_pickup(self):
        if self._preparation_shop_reserved():
            yield False
            return
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
        if self._preparation_shop_reserved():
            yield False
            return
        # TODO: free (no charge) items
        self.item_manager.price_identification()
        if self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            yield False
        if self.unreachable_items_until.get(self._here(), -1) > self.agent.blstats.time:
            yield False
        if len(self.items_below_me) == 0:
            yield False

        yield from self.arrange_items().strategy()


# RING_MODULE (jf_config): CleverShovel's ring/amulet module (ring_amulet_logic.py, scroll_identify.py)
from . import ring_amulet_logic as _ring_amulet_logic  # noqa: E402
_ring_amulet_logic.install(Inventory)
