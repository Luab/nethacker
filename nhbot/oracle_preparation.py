"""One-time Oracle experience purchases from public observations only.

3.6.7 rumors.c: a full first major gives cost/10 XP before a minor, cost/25
after one. An underfunded major takes all carried gold, gives no XP, and burns
the first-major flag. Never accept a discounted/underfunded consultation.
"""
import re

from nle.nethack import actions as A

from .character import Character
from .glyph import G, Hunger
from .item import Item, flatten_items
from .strategy import Strategy


GOLD_RESERVE = 200
FOOD_RESERVE = 800
MAX_DISTANCE = MAX_WALKS = 8
MAX_TURNS = 30
MAX_INPUTS = 120
MAX_ATTEMPTS = 2
RETRY_TURNS = 200
_MINOR = re.compile(r'Wilt thou settle for a minor consultation\?"?\s+\((\d+) zorkmids?\)')
_MAJOR = re.compile(r'Then dost thou desire a major one\?"?\s+\((\d+) zorkmids?\)')


class OraclePreparation:
    def __init__(self, agent):
        self.agent = agent
        self.state = None
        self.completed = {'major': False, 'minor': False}
        self.accepted = set()  # write before sending yes; ambiguity never authorizes another payment
        self.attempts = {'major': 0, 'minor': 0}
        self.retry_after = 0
        self.seen = set()
        self.last_reason = None
        self.last_outcome = None
        self.purchases = []

    def _key(self):
        return tuple(map(int, self.agent.current_level().key()))

    def _here(self):
        return int(self.agent.blstats.y), int(self.agent.blstats.x)

    def _stats(self):
        # During an atomic prompt, normal blstats can lag the raw public
        # observation. Read the latter when available, never engine internals.
        raw = getattr(self.agent, '_observation', {}).get('blstats')
        if raw is not None and len(raw) > 20:
            return dict(gold=int(raw[13]), xl=int(raw[18]), xp=int(raw[19]), turn=int(raw[20]))
        bl = self.agent.blstats
        return dict(gold=int(bl.gold), xl=int(bl.experience_level),
                    xp=int(bl.experience_points), turn=int(bl.time))

    def _food(self):
        a, inv = self.agent, self.agent.inventory
        return sum(inv.BUY_FOOD_NUTRITION.get(i.object.name, 0) * i.count
                   for i in a.edible_carried_food() if i in inv.items.all_items and
                   i.shop_status == Item.NOT_SHOP and i.status != Item.CURSED and
                   i.is_unambiguous() and not i.is_corpse() and
                   not re.search(r'\b(?:partly|partially) eaten\b', i.text or ''))

    def _visible_positions(self):
        if self.agent.character.prop.hallu:
            return []
        a = self.agent
        return [tuple(map(int, pos)) for glyph in G.ORACLE for pos in zip(*(a.glyphs == glyph).nonzero())]

    def _positions(self):
        return [pos for pos in self._visible_positions() if self.agent.monster_tracker.peaceful_monster_mask[pos]]

    def _safe_reason(self):
        a, inv, bl = self.agent, self.agent.inventory, self.agent.blstats
        c, p, logic = a.character, a.character.prop, a.global_logic
        if c.role != Character.WIZARD or c.race != Character.GNOME:
            return 'character'
        if any(getattr(p, name, False) for name in
               ('blind', 'confusion', 'stun', 'hallu', 'polymorph', 'stoned', 'deaf', 'strangled', 'swallowed')):
            return 'impaired'
        if bl.hitpoints < .8 * bl.max_hitpoints or a.get_visible_monsters():
            return 'health_or_threat'
        if bl.hunger_state > Hunger.NOT_HUNGRY or self._food() < FOOD_RESERVE:
            return 'food_reserve'
        if bl.carrying_capacity >= 2:
            return 'burden'
        if (inv.shop_debt_pending() or inv._credit_clone_reserved() or inv._preparation_shop_reserved() or
                a.current_level().shop[self._here()] or
                any(i.shop_status != Item.NOT_SHOP for i in flatten_items(inv.items))):
            return 'shop_obligation'
        if getattr(getattr(logic, 'wizard_route', None), 'phase', None) in ('return', 'recover'):
            return 'route_return'
        if (getattr(inv, '_armor_fitting', None) is not None or
                any(getattr(getattr(logic, name, None), 'state', None) is not None for name in
                    ('altar_preparation', 'equipment_identification')) or
                getattr(getattr(a, 'equipment_recovery', None), 'state', None) is not None):
            return 'equipment_or_altar_transaction'
        pet = getattr(a, '_pet_care', None) or {}
        if pet.get('pending') is not None or pet.get('hunger_warning') is not None or \
                bl.time <= getattr(a, '_pet_starving_until', -1):
            return 'pet_need'
        return None

    def _reject(self, reason):
        if reason != self.last_reason:
            self.last_reason = reason
            if self.seen:
                self.agent.log(f'ORACLE_PREP unavailable={reason} cash={self._stats()["gold"]} '
                               f'food={self._food()} completed={self.completed}')

    def _kind(self):
        # Preserve the larger first-major reward rather than spending 50 for
        # five XP and permanently cutting the major reward to 40 percent.
        kind = 'minor' if self.completed['major'] else 'major'
        if kind in self.accepted or self.completed[kind] or self.attempts[kind] >= MAX_ATTEMPTS:
            return None
        return kind

    def _price(self, kind):
        return 50 if kind == 'minor' else 500 + 50 * self._stats()['xl']

    def _choice(self):
        # This strategy is checked frequently during ordinary exploration.
        # No inventory/nutrition scan or BFS before a fresh peaceful sighting.
        if not self._positions():
            self._reject('no_fresh_reachable_peaceful_oracle')
            return None
        reason = self._safe_reason()
        kind = self._kind()
        if reason is None and kind is None:
            reason = 'completed_or_attempt_budget'
        if reason is None and self.agent.blstats.time < self.retry_after:
            reason = 'retry_cooldown'
        if reason is None and self._stats()['gold'] < self._price(kind) + GOLD_RESERVE:
            reason = 'gold_reserve'
        if reason is not None:
            self._reject(reason)
            return None
        a, level = self.agent, self.agent.current_level()
        distance = a.bfs()
        options = []
        for oracle in self._positions():
            if level.shop[oracle]:
                continue
            for y in range(max(0, oracle[0] - 1), min(a.glyphs.shape[0], oracle[0] + 2)):
                for x in range(max(0, oracle[1] - 1), min(a.glyphs.shape[1], oracle[1] + 2)):
                    pos = y, x
                    if (pos == oracle or level.shop[pos] or a.monster_tracker.monster_mask[pos] or
                            not level.walkable[pos] or not 0 <= distance[pos] <= MAX_DISTANCE):
                        continue
                    options.append((int(distance[pos]), oracle, pos))
        if not options:
            self._reject('no_fresh_reachable_peaceful_oracle')
            return None
        _, oracle, target = min(options)
        self.last_reason = None
        return kind, oracle, target

    def _finish(self, outcome):
        self.last_outcome = outcome
        if self.state is not None:
            self.agent.log(f'ORACLE_PREP finish={outcome} kind={self.state["kind"]} '
                           f'walks={self.state["walks"]} cash={self._stats()["gold"]}')
        self.state = None
        self.retry_after = int(self.agent.blstats.time) + (0 if outcome == 'purchased_with_xp' else RETRY_TURNS)

    def update(self):
        for pos in self._visible_positions():
            key = self._key(), pos
            if key not in self.seen:
                self.seen.add(key)
                self.agent.log(f'ORACLE_PREP seen level={key[0]} position={pos} '
                               f'peaceful={bool(self.agent.monster_tracker.peaceful_monster_mask[pos])} '
                               f'cash={self._stats()["gold"]} food={self._food()}')
        state = self.state
        if state is None:
            return
        if state['phase'] == 'accepted':
            after, before = self._stats(), state['before']
            paid = before['gold'] - after['gold'] == state['price']
            xp = after['xp'] > before['xp'] or after['xl'] > before['xl']
            # Receipt/XP evidence survives callback preemption. Never rely on
            # the score or on a chat command returning normally.
            if paid and xp:
                self.completed[state['kind']] = True
                purchase = dict(kind=state['kind'], price=state['price'], before=before, after=after)
                self.purchases.append(purchase)
                self.agent.log(f'ORACLE_PREP purchase_confirmed={purchase}')
                self._finish('purchased_with_xp')
                return
            if state.get('input_finished'):
                self._finish('accepted_payment_unconfirmed')
                return
        if self._key() != state['level']:
            self._finish('level_changed')
        elif (self.agent.blstats.time - state['turn'] >= MAX_TURNS or
              self.agent.step_count - state['step'] >= MAX_INPUTS):
            self._finish('turn_or_input_budget')

    def _can_pay(self, state, kind, price):
        return (self.state is state and self._safe_reason() is None and
                state['oracle'] in self._positions() and self._key() == state['level'] and
                max(abs(a - b) for a, b in zip(self._here(), state['oracle'])) == 1 and
                kind == state['kind'] and kind not in self.accepted and
                price == self._price(kind) and self._stats()['gold'] >= price + GOLD_RESERVE)

    def _chat(self, state):
        a = self.agent
        state['phase'] = 'chat'
        direction = a.calc_direction(*self._here(), *state['oracle'])

        def accept(kind, price):
            self.accepted.add(kind)
            state.update(phase='accepted', price=price, before=self._stats(), input_finished=False)
            a.log(f'ORACLE_PREP accepting kind={kind} price={price} public_before={state["before"]}')

        def answers():
            if 'Talk to whom? (in what direction)' not in a.single_message:
                yield A.Command.ESC
                return
            yield getattr(A.CompassDirection, direction.upper())
            minor = _MINOR.search(a.single_message)
            if minor is None:
                yield A.Command.ESC
                return
            if int(minor.group(1)) != 50:
                yield 'q'
                return
            if state['kind'] == 'minor':
                if self._can_pay(state, 'minor', 50):
                    accept('minor', 50)
                    yield 'y'
                else:
                    yield 'q'
                return
            if not self._can_pay(state, 'major', self._price('major')):
                yield 'q'
                return
            yield 'n'
            major = _MAJOR.search(a.single_message)
            if major is None:
                yield A.Command.ESC
            elif self._can_pay(state, 'major', int(major.group(1))):
                accept('major', int(major.group(1)))
                yield 'y'
            else:
                yield 'n'

        try:
            with a.atom_operation():
                a.step(A.Command.CHAT, answers())
                state['input_finished'] = True
        finally:
            self.update()
            if self.state is state and state['phase'] == 'chat':
                self._finish('chat_declined_or_unexpected')

    @Strategy.wrap
    def strategy(self):
        self.update()
        if self.state is None and self._choice() is None:
            yield False
            return
        if self._safe_reason() is not None:
            yield False
            return
        yield True
        if self.state is None:
            choice = self._choice()
            if choice is None:
                return
            kind, oracle, target = choice
            self.attempts[kind] += 1
            self.state = dict(kind=kind, oracle=oracle, target=target, level=self._key(),
                              turn=int(self.agent.blstats.time), step=int(self.agent.step_count),
                              walks=0, phase='approach')
            self.agent.log(f'ORACLE_PREP begin kind={kind} oracle={oracle} target={target} '
                           f'price={self._price(kind)} cash={self._stats()["gold"]} food={self._food()}')
        while self.state is not None:
            self.update()
            state = self.state
            if state is None:
                return
            if self._safe_reason() is not None:
                return
            if state['phase'] != 'approach':
                self._finish('interrupted_chat_no_replay')
                return
            if state['oracle'] not in self._positions():
                self._finish('oracle_no_longer_observed')
                return
            if self._stats()['gold'] < self._price(state['kind']) + GOLD_RESERVE:
                self._finish('funds_changed')
                return
            if max(abs(a - b) for a, b in zip(self._here(), state['oracle'])) == 1:
                self._chat(state)
                return
            distance = self.agent.bfs()[state['target']]
            if (state['walks'] >= MAX_WALKS or distance < 0 or
                    distance > MAX_WALKS - state['walks'] or
                    self.agent.current_level().shop[state['target']] or
                    self.agent.monster_tracker.monster_mask[state['target']]):
                self._finish('approach_unreachable_or_budget')
                return
            state['walks'] += 1
            before = self.agent.step_count
            self.agent.go_to(*state['target'], max_steps=1)
            if self.agent.step_count == before:
                self._finish('navigation_stalled')
                return

    def snapshot(self):
        return dict(state=self.state, completed=dict(self.completed), accepted=sorted(self.accepted),
                    attempts=dict(self.attempts), seen=sorted(self.seen),
                    cash=self._stats()['gold'], major_price=self._price('major'), food=self._food(),
                    gold_reserve=GOLD_RESERVE, food_reserve=FOOD_RESERVE,
                    last_reason=self.last_reason, last_outcome=self.last_outcome,
                    purchases=list(self.purchases))
