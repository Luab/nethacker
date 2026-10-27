"""Finite Oracle fountain attempts using public knowledge and an inert dipper.

NetHack 3.6.7 fountain.c: inert-item dipping and ordinary drinking both reach
the demon branch with probability 1/30.  Dipping avoids drinking's whole-pack
curse, poisonous water and hunger losses.  It still risks summons and floods;
this is a bounded experiment, not a guaranteed safe source of wishes.
"""
import re

from nle import nethack as nh
from nle.nethack import actions as A

from . import jf_config, objects as O, power_route, utils
from .character import Character
from .glyph import G, Hunger, SS
from .item import Item
from .item.utils import public_noncursed
from .level import Level
from .strategy import Strategy


MAX_DIPS = 6
MAX_DIPS_PER_FOUNTAIN = 3
MAX_WALKS = 12
MAX_TURNS = 60
MAX_INPUTS = 180
FOOD_RESERVE = 800
ROOM = frozenset((SS.S_room, SS.S_darkroom))
CORRIDOR = frozenset((SS.S_corr, SS.S_litcorr))


class OracleFountains:
    def __init__(self, agent):
        self.agent = agent
        self.state = None
        self.attempts = self.wishes = 0
        self.blocked = None
        self._counts = {}
        self._retired = set()
        self._escape = None
        self.escape_outcome = None
        self.poison_intrinsic = False
        self._history = 0  # public prior messages may already prove an intrinsic
        self._observed = None
        self.skip_counts = {}
        self.last_skip = None
        self._skip_stamp = None

    def _key(self):
        return tuple(map(int, self.agent.current_level().key()))

    def _here(self):
        return int(self.agent.blstats.y), int(self.agent.blstats.x)

    def enabled(self):
        a = self.agent
        oracle = getattr(a.global_logic, 'oracle_level', None)
        return (a.character.role == Character.WIZARD and a.character.race == Character.GNOME and
                a.current_level().dungeon_number == Level.DUNGEONS_OF_DOOM and
                oracle is not None and self._key() == tuple(oracle))

    def _skip(self, reason):
        stamp = (int(self.agent.step_count), reason)
        if self._skip_stamp != stamp:
            self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1
            self._skip_stamp = stamp
        if reason != self.last_skip:
            self.agent.log(f'ORACLE_FOUNTAIN unavailable={reason}')
        self.last_skip = reason
        return False

    @staticmethod
    def _inert(item):
        if (item.category != nh.GEM_CLASS or item.shop_status != Item.NOT_SHOP or
                item.equipped or item.at_ready or item.count <= 0 or item.naming or item.comment):
            return False
        # Colored gems/glass are unchanged by water and harmless if cursed.
        # An ambiguous gray stone may be a loadstone/luckstone/touchstone.
        return bool(item.objs) and all(isinstance(obj, O.Gem) or
            isinstance(obj, O.Rock) and obj.name in ('rock', 'flint') for obj in item.objs)

    def _dipper(self):
        choices = [i for i in self.agent.inventory.items if self._inert(i)]
        return min(choices, key=lambda i: (i.unit_weight(), i.text), default=None)

    def reserved_items(self):
        if not self.enabled() or self.blocked or self.attempts >= MAX_DIPS:
            return []
        item = self._dipper()
        return [] if item is None else [item]

    def _poison_resistant(self):
        a = self.agent
        if self.poison_intrinsic or any(getattr(obj, 'poison_resistance', False)
                                        for obj in (a.character, a.character.prop)):
            return True
        return any(i.equipped and re.search(r'\((?:being worn|on (?:left|right) hand)\)', i.text or '') and
                   i.is_unambiguous() and
                   getattr(i.object, 'power', None) == O.POISON_RES
                   for i in a.inventory.items)

    def _escape_item(self):
        a = self.agent
        # The separate controlled-teleport UI is not part of this transaction.
        if power_route.tc_known(a) or getattr(a, '_tc_intrinsic', False):
            return None
        for i in a.inventory.items:
            if not i.is_unambiguous() or i.shop_status != Item.NOT_SHOP or i.count <= 0:
                continue
            if i.category == nh.SCROLL_CLASS and i.object.name == 'teleportation' and public_noncursed(i):
                return 'scroll', i
            if i.category == nh.WAND_CLASS and i.object.name == 'teleportation':
                charges = re.search(r'\(\d+:(\d+)\)', i.text or '')
                if charges and int(charges.group(1)) > 0 and not a.inventory.is_known_empty(i):
                    return 'wand', i
        return None

    def _neighbors(self, pos):
        height, width = self.agent.current_level().objects.shape
        y, x = pos
        return [(y+dy, x+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and 0 <= y+dy < height and 0 <= x+dx < width]

    def _next_to_door(self, pos):
        level = self.agent.current_level()
        return any(level.objects[p] in G.DOORS or level.objects[p] == SS.S_ndoor
                   for p in self._neighbors(pos))

    def _can_flood(self, pos, fountain):
        if pos == fountain or (sum(pos) % 2) or self._next_to_door(pos):
            return False
        # A remembered trap glyph replaces the ROOM beneath it. gush() can
        # delete that trap and flood the square, so it cannot certify dry land.
        return self.agent.current_level().objects[pos] in ROOM | G.TRAPS | {-1}

    def _safe_geometry(self, fountain):
        a, level = self.agent, self.agent.current_level()
        tracker = a.monster_tracker
        friendly = tracker.peaceful_monster_mask | utils.isin(a.glyphs, G.PETS)
        for y, x in zip(*friendly.nonzero()):
            pos = int(y), int(x)
            if max(abs(pos[0]-fountain[0]), abs(pos[1]-fountain[1])) <= 7 and self._can_flood(pos, fountain):
                return 'friendly_in_possible_flood'
        safe_exit = any(level.walkable[p] and not tracker.monster_mask[p] and
                        level.objects[p] in ROOM | CORRIDOR | {SS.S_ndoor} and
                        not self._can_flood(p, fountain) and not level.forbidden[p] and
                        not level.shop[p] for p in self._neighbors(fountain))
        return None if safe_exit else 'no_dry_adjacent_exit'

    def _ready(self):
        a, inv = self.agent, self.agent.inventory
        p, bl, logic = a.character.prop, a.blstats, a.global_logic
        if self.blocked or self.attempts >= MAX_DIPS:
            return self._skip(self.blocked or 'total_attempt_budget')
        if any(getattr(p, name, False) for name in ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) or \
                getattr(bl, 'prop_mask', 0) & nh.BL_MASK_LEV:
            return self._skip('impaired_or_levitating')
        if (bl.experience_level < 5 or bl.hitpoints < max(30, 0.8 * bl.max_hitpoints) or
                bl.armor_class > 5 or bl.energy < 10):
            return self._skip('hp_level_armor_or_energy')
        if bl.hunger_state > Hunger.NOT_HUNGRY or bl.carrying_capacity != 0:
            return self._skip('hunger_or_burden')
        food = sum(inv.BUY_FOOD_NUTRITION.get(i.object.name, 0) * i.count
                   for i in a.edible_carried_food() if i in inv.items.all_items and
                   i.shop_status == Item.NOT_SHOP and i.status != Item.CURSED and
                   i.is_unambiguous() and not i.is_corpse() and
                   not re.search(r'\b(?:partly|partially) eaten\b', i.text or ''))
        if food < FOOD_RESERVE:
            return self._skip('food_reserve')
        if a.get_visible_monsters():
            return self._skip('visible_hostile')
        if inv.shop_debt_pending() or inv._credit_clone_reserved() or inv._preparation_shop_reserved() or \
                a.current_level().shop[self._here()]:
            return self._skip('shop_or_debt')
        controllers = (getattr(logic, 'altar_preparation', None),
                       getattr(logic, 'equipment_identification', None),
                       getattr(logic, 'oracle_preparation', None),
                       getattr(a, 'equipment_recovery', None))
        if (any(getattr(c, 'state', None) is not None for c in controllers) or
                getattr(inv, '_armor_fitting', None) is not None or
                getattr(getattr(logic, 'wizard_route', None), 'phase', None) in ('return', 'recover') or
                getattr(a, 'wish_text', None)):
            return self._skip('other_preparation_reserved')
        pet = getattr(a, '_pet_care', None) or {}
        if pet.get('pending') is not None or pet.get('hunger_warning') is not None or \
                bl.time <= getattr(a, '_pet_starving_until', -1):
            return self._skip('pet_need')
        if not self._poison_resistant():
            return self._skip('no_observed_poison_resistance')
        if self._escape_item() is None:
            return self._skip('no_ready_teleport_escape')
        if self._dipper() is None:
            return self._skip('no_owned_inert_dipper')
        return True

    def _target(self):
        a, level = self.agent, self.agent.current_level()
        distances = a.bfs()
        candidates = []
        for y, x in zip(*utils.isin(level.objects, G.FOUNTAIN).nonzero()):
            pos = int(y), int(x)
            key = self._key(), pos
            if key in self._retired or self._counts.get(key, 0) >= MAX_DIPS_PER_FOUNTAIN or level.shop[pos] or \
                    distances[pos] < 0 or distances[pos] > MAX_WALKS:
                continue
            reason = self._safe_geometry(pos)
            if reason is not None:
                self._skip(reason)
                continue
            candidates.append(pos)
        return min(candidates, key=lambda p: (distances[p], p), default=None)

    def _message(self, text):
        if re.search(r'You feel (?:especially )?healthy\.', text):
            self.poison_intrinsic = True
        if 'You feel a little sick!' in text:
            self.poison_intrinsic = False
        if self.state is None or self.state['phase'] != 'dip':
            return
        if 'grants you a wish' in text or 'You may wish for an object.' in text:
            self.wishes = 1
            self.blocked = 'wish_received'
            self._escape = None  # grateful demon is removed by mongrantswish()
            return
        reason = None
        if 'You unleash a water demon' in text:
            reason = 'water_demon'
            dive = self.agent.global_logic.dive
            dive._demon_vigil_until = max(dive._demon_vigil_until,
                                         int(self.agent.blstats.time) + jf_config.DEMON_VIGIL_TURNS)
        elif 'You attract a water nymph' in text:
            reason = 'water_nymph'
        elif 'An endless stream of snakes' in text or 'You hear hissing!' in text:
            reason = 'water_snakes'
        elif 'Water gushes forth from the overflowing fountain!' in text:
            reason = 'flood'
        elif 'stop using that fountain' in text or 'flow reduces to a trickle' in text:
            reason = 'town_warning'
        if reason is not None and self.blocked != 'wish_received':
            self.blocked = reason
            if self._escape is None:
                self._escape = dict(level=self.state['level'], turn=int(self.agent.blstats.time),
                                    step=int(self.agent.step_count), reason=reason, attempted=False)
            self.agent.log(f'ORACLE_FOUNTAIN hazard={reason}; further attempts stopped')

    def observe(self):
        """Raw public messages before automatic wish/prompt handling; no actions."""
        text = self.agent.single_message or ''
        stamp = int(self.agent.step_count), text
        if stamp != self._observed:
            self._observed = stamp
            self._message(text)

    def update(self):
        history = getattr(self.agent, '_message_history', [])
        for text in history[min(self._history, len(history)):]:
            self._message(text)
        self._history = len(history)
        if self.state is not None:
            s = self.state
            if self._key() != s['level'] or self.agent.blstats.time - s['turn'] >= MAX_TURNS or \
                    self.agent.step_count - s['step'] >= MAX_INPUTS:
                self._retired.add((s['level'], s['target']))
                self.agent.log('ORACLE_FOUNTAIN visit ended at level/time/input boundary')
                self.state = None
        if self._escape is not None:
            e = self._escape
            if self._key() != e['level'] or self.agent.blstats.time - e['turn'] > 10 or \
                    self.agent.step_count - e['step'] > 100 or e['attempted']:
                self._escape = None

    @Strategy.wrap
    def strategy(self):
        self.update()
        if not self.enabled() or not self._ready():
            yield False
            return
        if self.state is None and self._target() is None:
            self._skip('no_eligible_known_fountain')
            yield False
            return
        yield True
        if not self.enabled() or not self._ready():
            return
        if self.state is None:
            target = self._target()
            if target is None:
                return
            self.state = dict(level=self._key(), target=target, turn=int(self.agent.blstats.time),
                              step=int(self.agent.step_count), walks=0, phase='approach')
        while self.state is not None and self.enabled() and self._ready():
            self.update()
            s = self.state
            if s is None:
                return
            a, target = self.agent, s['target']
            key = s['level'], target
            distance = a.bfs()[target]
            if (a.current_level().objects[target] not in G.FOUNTAIN or a.current_level().shop[target] or
                    self._counts.get(key, 0) >= MAX_DIPS_PER_FOUNTAIN or distance < 0 or
                    distance > MAX_WALKS - s['walks'] or self._safe_geometry(target) is not None):
                self._retired.add(key)
                self.state = None
                return
            if self._here() != target:
                s['walks'] += 1
                before = a.step_count
                a.go_to(*target, max_steps=1)
                if a.step_count == before:
                    self._retired.add(key)
                    self.state = None
                continue
            # Persist a spent attempt before any prompt/callback can interrupt.
            item = self._dipper()
            letter = a.inventory.items.get_letter(item)
            self.attempts += 1
            self._counts[key] = self._counts.get(key, 0) + 1
            s['phase'] = 'dip'
            a.log(f'ORACLE_FOUNTAIN dip={self.attempts}/{MAX_DIPS} target={target} item={item.text!r}')
            old_purpose = getattr(a, 'wish_purpose', None)
            had_purpose = hasattr(a, 'wish_purpose')
            a.wish_purpose = 'oracle_preparation'
            confirmed = [False]

            def answers():
                if 'What do you want to dip?' in a.single_message:
                    yield letter
                if ' into the fountain?' in a.single_message and a.single_message.startswith('Dip '):
                    confirmed[0] = True
                    yield 'y'
                elif 'What do you want to dip ' in a.single_message:
                    yield A.Command.ESC  # never select a potion if the fountain disappeared

            try:
                with a.atom_operation():
                    a.step(A.Command.DIP, answers())
            finally:
                if had_purpose:
                    a.wish_purpose = old_purpose
                else:
                    del a.wish_purpose
            self.update()
            if not confirmed[0]:
                self._retired.add(key)
                self.state = None
                return
            if self.state is not None:
                self.state['phase'] = 'approach'
        if self.blocked:
            self.state = None

    @Strategy.wrap
    def escape_strategy(self):
        self.update()
        if self._escape is None:
            yield False
            return
        a = self.agent
        p = a.character.prop
        escape = self._escape_item()
        if self._escape is None or escape is None or any(getattr(p, key, False) for key in
                ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) or a.inventory.shop_debt_pending():
            yield False
            return
        yield True
        self.update()
        escape = self._escape_item()
        if self._escape is None or escape is None or any(getattr(p, key, False) for key in
                ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) or a.inventory.shop_debt_pending():
            return
        self._escape['attempted'] = True
        self.escape_outcome = 'attempted'
        kind, item = escape
        a.log(f'ORACLE_FOUNTAIN escape={kind} item={item.text!r}')
        if kind == 'wand':
            a.zap(item, '.')
        else:
            letter = a.inventory.items.get_letter(item)

            def answers():
                if 'What do you want to read?' in a.single_message:
                    yield letter
                if 'Do you wish to teleport?' in a.single_message:
                    yield 'y'

            with a.atom_operation():
                a.step(A.Command.READ, answers())
        self._escape = None

    def snapshot(self):
        return dict(state=self.state, attempts=self.attempts, wishes=self.wishes, blocked=self.blocked,
                    poison_intrinsic_observed=self.poison_intrinsic, escape=self._escape,
                    escape_outcome=self.escape_outcome, last_skip=self.last_skip,
                    skip_counts=dict(self.skip_counts),
                    fountain_attempts=[dict(level=key[0], position=key[1], attempts=count)
                                       for key, count in sorted(self._counts.items())])
