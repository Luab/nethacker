"""Recover our observed equipment after an involuntary change of form.

No engine inventory ids: an old equipped item only supplies a local search
origin. A fresh, unique, owned floor match is required before taking it.
"""
import re

import nle.nethack as nh

from .character import Character
from .glyph import Hunger
from .item import Item
from .strategy import Strategy


MAX_WALKS = 12
MAX_NORMAL_TURNS = 60
MAX_INPUTS = 200
_FORM = re.compile(r"You turn into an? (?:male |female )?([a-z' -]+?)!")
_RETURN = re.compile(r'You return to .+? form[!.]|You feel like a new .+?!')
_SLOTS = ('main_hand', 'shirt', 'suit', 'cloak', 'off_hand', 'helm', 'gloves', 'boots')
_BAD_ARMOR = {'dunce cap', 'helm of opposite alignment', 'gauntlets of fumbling',
              'fumble boots', 'levitation boots'}


class EquipmentRecovery:
    def __init__(self, agent):
        self.agent = agent
        self.state = None
        self.form = None
        self._seen_event = None
        self._last_clock = None
        self._normal_clock = False
        self.captured = self.picked_up = self.equipped = 0
        self.last_outcome = None
        self._retained_armor = {}  # at most one observed recovered piece per slot

    def enabled(self):
        c = self.agent.character
        return c.role == Character.WIZARD and c.race == Character.GNOME

    @staticmethod
    def _identity(item):
        return dict(category=int(item.category), glyphs=tuple(map(int, item.glyphs)),
                    naming=item.naming, modifier=item.modifier)

    @staticmethod
    def _matches(record, item):
        key = record['identity']
        return (item.count == 1 and int(item.category) == key['category'] and
                tuple(map(int, item.glyphs)) == key['glyphs'] and item.naming == key['naming'] and
                (item.modifier is None or key['modifier'] is None or item.modifier == key['modifier']))

    @staticmethod
    def _owned(item):
        # Curse status does not transfer ownership. Formerly worn armor can
        # remain useful after uncursing; taking it into the pack is safe.
        # Equipping has the separate fresh public-noncursed requirement below.
        return item.shop_status == Item.NOT_SHOP

    def observe(self, observation):
        """Every raw observation, before prompts or InventoryItems replace old gear.

        Coordinates belong to this observation, never to the previous blstats.
        This observer does no input and is safe inside an atomic operation.
        """
        if not self.enabled():
            return
        raw = observation['blstats']
        # Public NLE blstats layout, also named by Agent.BLStats.
        now, level = int(raw[20]), (int(raw[23]), int(raw[24]))
        position = (int(raw[1]), int(raw[0]))
        steps = int(self.agent.step_count)
        msg = self.agent.single_message or ''
        changes, returns = list(_FORM.finditer(msg)), list(_RETURN.finditer(msg))
        match = changes[-1] if changes else None
        returned = returns[-1] if returns else None
        if self.state is not None:
            if level != self.state['level']:
                self._finish('level_changed')
            elif self._last_clock is not None and self._normal_clock:
                self.state['normal_turns'] += max(0, now - self._last_clock[0])
                self.state['inputs'] += max(0, steps - self._last_clock[1])
        if match and self._seen_event != (now, match.group(0)):
            self._seen_event = (now, match.group(0))
            self.form = match.group(1)
            self._capture(level, position, now)
        if returned and (match is None or returned.start() > match.start()):
            self.form = None
            if self.state is not None and self.state['returned_turn'] is None:
                self.state['returned_turn'] = now
                self.agent.log(f'EQUIPMENT_RECOVERY returned; origin={self.state["level"]}')
        self._last_clock = (now, steps)
        self._normal_clock = self.form is None and self.state is not None and \
            self.state['returned_turn'] is not None
        if self.state is not None and (self.state['normal_turns'] >= MAX_NORMAL_TURNS or
                                       self.state['inputs'] >= MAX_INPUTS):
            self._finish('normal_turn_or_input_budget')

    def _capture(self, level, position, now):
        inventory = self.agent.inventory
        captured = []
        for slot in _SLOTS:
            item = getattr(inventory.items, slot, None)
            if item is None or not item.equipped or item.count != 1 or not self._owned(item) or \
                    not item.is_unambiguous():
                continue
            if slot == 'main_hand' and not item.is_weapon():
                continue  # only ordinary weapons, never wielded corpses/tools
            if slot != 'main_hand' and (not item.is_armor() or item.object.name in _BAD_ARMOR):
                continue
            captured.append(dict(slot=slot, identity=self._identity(item), text=item.text,
                                 position=position, phase='seek', attempts=0))
        if not captured:
            return
        if self.state is None:
            self.state = dict(level=level, born=now, returned_turn=None, normal_turns=0,
                              inputs=0, walks=0, entries=[])
        for entry in captured:
            # A later transformation can shed an item just restored. Preserve
            # the episode budget, but replace that item's old pickup location.
            self.state['entries'] = [r for r in self.state['entries'] if
                                     not (r['slot'] == entry['slot'] and r['identity'] == entry['identity'])]
            self.state['entries'].append(entry)
        self.captured += len(captured)
        inventory.items_below_me = inventory.letters_below_me = inventory.engraving_below_me = None
        self.agent.log(f'EQUIPMENT_RECOVERY capture observed_turn={now} form={self.form} level={level} '
                       f'position={position} items={[r["text"] for r in captured]}')

    def _finish(self, reason):
        self.last_outcome = reason
        if self.state is not None:
            self.agent.log(f'EQUIPMENT_RECOVERY finish={reason} normal_turns={self.state["normal_turns"]} '
                           f'walks={self.state["walks"]} inputs={self.state["inputs"]}')
        self.state = None

    def _safe(self):
        a, inv = self.agent, self.agent.inventory
        p, bl = a.character.prop, a.blstats
        return (self.enabled() and self.form is None and
                not any(getattr(p, k, False) for k in
                        ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned')) and
                not (getattr(bl, 'prop_mask', 0) & nh.BL_MASK_LEV) and
                bl.hunger_state < Hunger.WEAK and bl.carrying_capacity < 2 and
                not a.get_visible_monsters() and not inv.shop_debt_pending() and
                not inv._credit_clone_reserved() and not inv._preparation_shop_reserved() and
                not a.current_level().shop[bl.y, bl.x])

    def _here(self):
        return int(self.agent.blstats.y), int(self.agent.blstats.x)

    def _carried(self, record):
        matches = [i for i in self.agent.inventory.items if self._matches(record, i)]
        return matches[0] if len(matches) == 1 and self._owned(matches[0]) else None

    def _still_useful(self, slot, item):
        current = getattr(self.agent.inventory.items, slot)
        if current in (None, item):
            return True
        # A temporary wrapping is no replacement for the recovered source of
        # magic resistance. Keep this one provenance-bound reservation within
        # the ordinary food-first weight budget until it can be uncursed.
        return (slot == 'cloak' and item.is_unambiguous() and
                item.object.name == 'cloak of magic resistance' and
                current.is_unambiguous() and current.object.name == 'mummy wrapping')

    def _reconcile(self):
        for slot, record in list(self._retained_armor.items()):
            item = self._carried(record)
            if item is None or not self._still_useful(slot, item):
                del self._retained_armor[slot]
        if self.state is None:
            return
        for record in self.state['entries']:
            item = self._carried(record)
            if item is not None and record['phase'] == 'pickup':
                self.picked_up += 1
                record['phase'] = 'carried'
                if item.is_armor():
                    self._retained_armor[record['slot']] = dict(
                        slot=record['slot'], identity=self._identity(item))
                self.agent.log(f'EQUIPMENT_RECOVERY pickup_confirmed origin={record["position"]} '
                               f'item={item.text!r}')
            if item is not None and record['phase'] == 'equip' and \
                    getattr(self.agent.inventory.items, record['slot']) is item:
                self.equipped += 1
                record['phase'] = 'done'
                self.agent.log(f'EQUIPMENT_RECOVERY equipped_confirmed slot={record["slot"]} '
                               f'item={item.text!r} ac_before={record.get("ac_before")} '
                               f'ac_after={self.agent.blstats.armor_class}')

    def update(self):
        """Fresh inventory reconciliation; no actions, also after preemption."""
        self._reconcile()

    def retain_cursed_armor(self, item):
        """Keep recovered cursed armor for uncursing after food and usable gear.

        Never grants floor pickup permission or displaces a current replacement.
        Leaving inventory, changing owner, or finding a useful replacement
        retires the reservation, preventing later lookalikes from inheriting it.
        """
        if item not in self.agent.inventory.items.all_items or not self._owned(item) or \
                not item.is_armor() or item.equipped or not (
                    item.status == Item.CURSED or re.search(r'\bcursed\b', item.text or '')):
            return False
        return any(self._carried(record) is item and self._still_useful(slot, item)
                   for slot, record in self._retained_armor.items())

    def _can_equip(self, record, item):
        a, inv = self.agent, self.agent.inventory
        items, slot = inv.items, record['slot']
        if (getattr(items, slot) is not None or not self._owned(item) or
                not inv._public_noncursed(item) or a.hands_welded()):
            return False
        if slot == 'main_hand':
            return item.is_weapon() and (not item.object.bi or items.off_hand is None)
        if not item.is_armor() or item.object.name in _BAD_ARMOR:
            return False
        if slot in ('shirt', 'suit') and items.cloak is not None or slot == 'shirt' and items.suit is not None:
            return False
        if slot == 'off_hand' and items.main_hand is not None and items.main_hand.object.bi:
            return False
        if a.blstats.time < getattr(a, '_wear_blocked_until', {}).get(item.object.sub, 0):
            return False
        return True

    @Strategy.wrap
    def strategy(self):
        self._reconcile()
        if self.state is None or self.state['returned_turn'] is None or not self._safe():
            yield False
            return
        yield True
        # The caller resumes its old navigator when this returns. Complete the
        # bounded local sequence here, so it cannot walk away between pieces.
        while self.state is not None and self._safe():
            self._reconcile()
            a, inv, state = self.agent, self.agent.inventory, self.state
            if a.current_level().key() != state['level']:
                self._finish('level_changed')
                return
            if state['normal_turns'] >= MAX_NORMAL_TURNS or state['inputs'] >= MAX_INPUTS:
                self._finish('normal_turn_or_input_budget')
                return
            entries = [r for r in state['entries'] if r['phase'] != 'done']
            if not entries:
                self._finish('sequence_complete')
                return
            entries.sort(key=lambda r: _SLOTS.index(r['slot']))
            record = entries[0]
            item = self._carried(record)
            current = getattr(inv.items, record['slot'])
            if current is not None:
                record['phase'] = 'done'  # never displace a replacement
                continue
            if item is not None:
                if not self._can_equip(record, item) or record['attempts'] >= 1:
                    record['phase'] = 'done'
                    continue
                # Record before atomic callbacks: completed-but-preempted
                # WIELD/WEAR must never issue an unconditional second action.
                record['phase'] = 'equip'
                record['attempts'] += 1
                record['ac_before'] = int(a.blstats.armor_class)
                self.agent.log(f'EQUIPMENT_RECOVERY equip slot={record["slot"]} item={item.text!r}')
                if record['slot'] == 'main_hand':
                    inv.wield(item, smart=False)
                else:
                    inv.wear(item, smart=False)
                self._reconcile()
                record['phase'] = 'done'
                continue
            if record['phase'] in ('pickup', 'equip'):
                record['phase'] = 'done'  # no confirmed result, no blind retry
                continue
            pos = record['position']
            dis = a.bfs()
            if (a.current_level().shop[pos] or dis[pos] < 0 or
                    dis[pos] > MAX_WALKS - state['walks']):
                record['phase'] = 'done'
                continue
            if self._here() != pos:
                state['walks'] += 1
                before = a.step_count
                a.go_to(*pos, max_steps=1)
                if a.step_count == before:
                    record['phase'] = 'done'
                continue
            below = inv.get_items_below_me()
            if self.state is not state or not self._safe() or self._here() != pos:
                return
            matches = [i for i in below if self._matches(record, i)]
            if len(matches) != 1 or not self._owned(matches[0]) or \
                    matches[0] in inv._quarantined_floor_items() or inv.items.free_slots() <= 0 or \
                    inv.items.total_weight + matches[0].weight() > a.character.carrying_capacity:
                record['phase'] = 'done'
                continue
            record['phase'] = 'pickup'
            self.agent.log(f'EQUIPMENT_RECOVERY pickup slot={record["slot"]} item={matches[0].text!r}')
            inv.pickup(matches[0])
            self._reconcile()

    def snapshot(self):
        return dict(state=self.state, observed_form=self.form, captured=self.captured,
                    picked_up=self.picked_up, equipped=self.equipped, last_outcome=self.last_outcome,
                    retained_armor=list(self._retained_armor.values()))
