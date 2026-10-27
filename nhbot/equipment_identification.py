"""Bounded altar BUC testing with an owned-item drop/recovery transaction.

All decisions use public inventory, floor and map observations.  In particular,
ItemManager's UNKNOWN-to-UNCURSED fallback is not evidence of known beatitude.
"""
from nle import nethack as nh

from . import objects as O
from .character import Character
from .glyph import G, Hunger
from .item import Item
from .item.utils import public_buc
from .strategy import Strategy


MAX_DISTANCE = 8
MAX_WALKS = 12
MAX_TURNS = 40
MAX_INPUTS = 160
MAX_TESTS_PER_LEVEL = 4
_BAD_ARMOR = frozenset(('dunce cap', 'helm of opposite alignment',
                        'gauntlets of fumbling', 'fumble boots', 'levitation boots'))
_SCROLLS = frozenset(('remove curse', 'enchant armor'))


class EquipmentIdentification:
    def __init__(self, agent):
        self.agent = agent
        self.state = None
        self._attempted = set()
        self._level_attempts = {}
        self.tested = self.recovered = 0
        self.last_outcome = None
        self.last_unrecovered = None

    public_buc = staticmethod(public_buc)

    def _key(self):
        return tuple(map(int, self.agent.current_level().key()))

    def _here(self):
        return int(self.agent.blstats.y), int(self.agent.blstats.x)

    @staticmethod
    def _identity(item):
        return (int(item.category), item.object.name, tuple(map(int, item.glyphs)),
                item.naming, item.modifier, int(item.count))

    @classmethod
    def _matches(cls, identity, item):
        return item.is_unambiguous() and cls._identity(item) == identity

    @staticmethod
    def _same_kind(identity, item):
        # Do not merge with, or later mistake, pre-existing floor goods.  A
        # broader comparison than exact identity also rejects unknown modifiers.
        return (int(item.category) == identity[0] and
                any(obj.name == identity[1] for obj in item.objs) and
                item.naming == identity[3])

    def _safe(self, recovering=False):
        a, inv = self.agent, self.agent.inventory
        p, bl = a.character.prop, a.blstats
        logic = getattr(a, 'global_logic', None)
        altar = getattr(logic, 'altar_preparation', None)
        route = getattr(logic, 'wizard_route', None)
        recovery = getattr(a, 'equipment_recovery', None)
        pet = getattr(a, '_pet_care', None)
        if not recovering and (bl.hitpoints < 0.8 * bl.max_hitpoints or
                               getattr(route, 'phase', None) in ('return', 'recover') or
                               getattr(inv, '_armor_fitting', None) is not None or
                               getattr(altar, 'state', None) is not None or
                               getattr(recovery, 'state', None) is not None or
                               pet is not None and pet.get('pending') is not None):
            return False
        return (a.character.role == Character.WIZARD and a.character.race == Character.GNOME and
                not any(getattr(p, name, False) for name in
                        ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) and
                not (getattr(bl, 'prop_mask', 0) & nh.BL_MASK_LEV) and
                bl.hunger_state < (Hunger.WEAK if recovering else Hunger.HUNGRY) and
                bl.carrying_capacity < 2 and not a.get_visible_monsters() and
                not inv.shop_debt_pending() and not inv._credit_clone_reserved() and
                not inv._preparation_shop_reserved() and
                not a.current_level().shop[self._here()])

    def _candidate(self, item):
        if (item.shop_status != Item.NOT_SHOP or item.equipped or item.at_ready or
                not item.is_unambiguous() or not item.can_be_dropped_from_inventory() or
                self.public_buc(item) != Item.UNKNOWN or item.count <= 0):
            return False
        if item.category == nh.SCROLL_CLASS:
            return item.object.name in _SCROLLS
        if not item.is_armor() or item.count != 1 or item.object.name in _BAD_ARMOR:
            return False
        # Wizard spellcasting makes most metal armor an unsuitable investment.
        return (not O.IRON <= item.object.metal <= O.MITHRIL or
                item.object.name == 'helm of brilliance')

    def _choice(self):
        level = self.agent.current_level()
        key = self._key()
        if self._level_attempts.get(key, 0) >= MAX_TESTS_PER_LEVEL:
            return None
        # Iterating InventoryItems only visits the direct pack, never bag contents.
        choices = [i for i in self.agent.inventory.items if self._candidate(i) and
                   (key, self._identity(i)) not in self._attempted and
                   sum(self._same_kind(self._identity(i), other)
                       for other in self.agent.inventory.items) == 1]
        if not choices:
            return None
        distance = self.agent.bfs()
        altars = set(level.altars)
        for value in G.ALTAR:
            altars.update(zip(*(level.objects == value).nonzero()))
        reachable = [tuple(map(int, pos)) for pos in altars if
                     not level.shop[pos] and 0 <= distance[pos] <= MAX_DISTANCE]
        if not reachable:
            return None
        target = min(reachable, key=lambda pos: (distance[pos], pos))
        choices.sort(key=lambda i: (i.category != nh.SCROLL_CLASS,
                                   i.object.name != 'remove curse', i.text))
        return target, choices[0]

    def _carried(self):
        identity = self.state['identity']
        matches = [i for i in self.agent.inventory.items if self._matches(identity, i)]
        return matches[0] if len(matches) == 1 and matches[0].shop_status == Item.NOT_SHOP else None

    def _finish(self, outcome):
        self.last_outcome = outcome
        if self.state is not None:
            if self.state['phase'] in ('retrieve', 'pickup') and outcome not in (
                    'identified', 'recovered_unclassified'):
                self.last_unrecovered = dict(level=self.state['level'], target=self.state['target'],
                                             text=self.state['text'], reason=outcome)
                self.agent.log(f'EQUIPMENT_ID unrecovered_owned_item={self.last_unrecovered}')
            self.agent.log(f'EQUIPMENT_ID finish={outcome} item={self.state["text"]!r} '
                           f'walks={self.state["walks"]}')
        self.state = None

    def update(self):
        """Reconcile fresh inventory even when an atomic action was preempted."""
        state = self.state
        if state is None:
            return
        if self._key() != state['level']:
            self._finish('level_changed')
            return
        item = self._carried()
        if state['phase'] == 'dropping' and item is None:
            state['phase'] = 'retrieve'
            self.tested += 1
        elif state['phase'] in ('retrieve', 'pickup') and item is not None:
            self.recovered += 1
            self.agent.log(f'EQUIPMENT_ID recovered={item.text!r} public_buc={self.public_buc(item)}')
            self._finish('identified' if self.public_buc(item) != Item.UNKNOWN else 'recovered_unclassified')
            return
        if (self.agent.blstats.time - state['turn'] >= MAX_TURNS or
                self.agent.step_count - state['step'] >= MAX_INPUTS):
            self._finish('turn_or_input_budget')

    @Strategy.wrap
    def strategy(self):
        self.update()
        recovering = self.state is not None and self.state['phase'] in ('retrieve', 'pickup')
        if not self._safe(recovering=recovering):
            yield False
            return
        choice = None if self.state is not None else self._choice()
        if self.state is None and choice is None:
            yield False
            return
        yield True
        if self.state is None:
            # Strategy conditions may be suspended while a higher priority
            # action changes the current level, inventory or safety state.
            if not self._safe():
                return
            choice = self._choice()
            if choice is None:
                return
            target, item = choice
            identity = self._identity(item)
            self._attempted.add((self._key(), identity))
            self._level_attempts[self._key()] = self._level_attempts.get(self._key(), 0) + 1
            self.state = dict(level=self._key(), target=target, identity=identity, text=item.text,
                              turn=int(self.agent.blstats.time), step=int(self.agent.step_count),
                              walks=0, phase='approach', pickups=0)
            self.agent.log(f'EQUIPMENT_ID begin target={target} item={item.text!r}')
        while self.state is not None:
            self.update()
            state = self.state
            if state is None:
                return
            recovering = state['phase'] in ('retrieve', 'pickup')
            if not self._safe(recovering=recovering):
                return
            target, inv = state['target'], self.agent.inventory
            level = self.agent.current_level()
            distance = self.agent.bfs()[target]
            if (level.shop[target] or distance < 0 or
                    distance > MAX_WALKS - state['walks']):
                self._finish('target_unreachable')
                return
            if self._here() != target:
                state['walks'] += 1
                before = self.agent.step_count
                self.agent.go_to(*target, max_steps=1)
                if self.agent.step_count == before:
                    self._finish('navigation_stalled')
                continue
            if state['phase'] == 'dropping':
                # No observed removal: do not issue a second drop command.
                self._finish('drop_unconfirmed')
                return
            below = inv.get_items_below_me()
            if self.state is not state or self._here() != target or not self._safe(recovering=recovering):
                return
            if state['phase'] == 'approach':
                item = self._carried()
                if (item is None or not self._candidate(item) or
                        target not in level.altars and level.objects[target] not in G.ALTAR):
                    self._finish('candidate_or_altar_changed')
                    return
                if any(self._same_kind(state['identity'], other) for other in below):
                    self._finish('existing_matching_floor_item')
                    return
                state['phase'] = 'dropping'
                self.agent.log(f'EQUIPMENT_ID drop={item.text!r} target={target}')
                inv.drop(item, smart=False)
                self.update()
                continue
            matches = [i for i in below if self._matches(state['identity'], i)]
            if (len(matches) != 1 or matches[0].shop_status != Item.NOT_SHOP or
                    matches[0] in inv._quarantined_floor_items() or inv.items.free_slots() <= 0 or
                    inv.items.total_weight + matches[0].weight() > self.agent.character.carrying_capacity):
                self._finish('owned_floor_item_unavailable')
                return
            if state['pickups'] >= 2:
                self._finish('pickup_attempt_budget')
                return
            state['phase'] = 'pickup'
            state['pickups'] += 1
            inv.pickup(matches[0])
            self.update()

    def snapshot(self):
        return dict(state=self.state, tested=self.tested, recovered=self.recovered,
                    last_outcome=self.last_outcome, attempts=len(self._attempted),
                    last_unrecovered=self.last_unrecovered)
