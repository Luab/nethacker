"""Bounded, opportunistic offerings from public corpse and altar observations."""
import re

from nle import nethack as nh
from nle.nethack import actions as A

from .character import Character
from .glyph import Hunger, MON
from .item import Item, flatten_items
from .nhmodel.prayer import PrayerModel
from .strategy import Strategy


MAX_AGE = 50
MAX_TURNS = 30
MAX_WALK = 12
MAX_INPUTS = 160
GIFT_TURNS = 100
_KEEP = frozenset(('lichen', 'lizard', 'wraith', 'nurse', 'floating eye', 'newt',
                  'mind flayer', 'master mind flayer', 'quantum mechanic', 'stalker',
                  'cockatrice', 'chickatrice', 'Medusa', 'Death', 'Famine', 'Pestilence'))
_DOMESTIC = frozenset(('kitten', 'housecat', 'large cat', 'little dog', 'dog', 'large dog',
                      'pony', 'horse', 'warhorse'))


class AltarPreparation:
    def __init__(self, agent):
        self.agent = agent
        self.mode = 'seeking'
        self.state = None
        self.gift_observed = False
        self.gift = None
        self.reconciled_turn = None
        self.attempts = 0
        self.rejections = {}
        self._seen = set()
        self._history = len(getattr(agent, '_message_history', []))
        self._pet_warning_until = -1
        self._gift_pending = None
        self._discovered = set()
        self._last_rejection = None
        self.sacrifices = 0
        self.gift_equipped = False
        self.recovered_gift = None
        self._last_message = None

    def enabled(self):
        c = self.agent.character
        return c.role == Character.WIZARD and c.race == Character.GNOME and c.alignment == Character.NEUTRAL

    def _key(self):
        return tuple(map(int, self.agent.current_level().key()))

    def _here(self):
        return int(self.agent.blstats.y), int(self.agent.blstats.x)

    def _messages(self):
        history = getattr(self.agent, '_message_history', [])
        start = min(self._history, len(history))
        messages = []
        for message in history[start:]:
            if message != self._last_message:
                messages.append(message)
            self._last_message = message
        self._history = len(history)
        return messages

    @staticmethod
    def _weapon_key(item):
        if not item.is_weapon() or not item.is_unambiguous() or item.count != 1:
            return None
        # The name, enchantment and appearance must survive pickup unchanged.
        name = item.naming or ('Magicbane' if re.search(r'\bMagicbane\b', item.text or '') else '')
        return (item.object.name, name, tuple(map(int, item.glyphs)),
                None if item.modifier is None else int(item.modifier))

    @classmethod
    def _weapon_appearance(cls, item):
        key = cls._weapon_key(item)
        return key[:3] if key is not None else None

    def update(self):
        now = int(self.agent.blstats.time)
        if self.enabled():
            for pos, alignment in self.agent.current_level().altars.items():
                if alignment is None or alignment == Character.UNKNOWN:
                    continue
                key = (self._key(), tuple(map(int, pos)), int(alignment))
                if key not in self._discovered:
                    self._discovered.add(key)
                    self.agent.log(f'ALTAR_PREP discovered={key} coaligned={alignment == self.agent.character.alignment}')
        if not self.gift_observed and any(re.search(r'\bMagicbane\b', i.text or '')
                and i.shop_status == Item.NOT_SHOP for i in self.agent.inventory.items):
            self.mode = 'complete'
        for text in self._messages():
            if re.search(r'confused from hunger|starves|feel worried about', text):
                self._pet_warning_until = max(self._pet_warning_until, now + 200)
            if 'You feel a malignant aura surround you.' in text or 'You feel like you need some help.' in text:
                # rndcurse can curse an unknown-BUC gift without changing its
                # inventory text. Public curse evidence revokes the exception.
                self.gift = None
                if self.state is not None:
                    self.state['gift_buc_invalidated'] = True
                if self._gift_pending is not None:
                    self._gift_pending['buc_invalidated'] = True
            gift = PrayerModel._ARTIFACT_GIFT_RE.search(text)
            if gift:
                self.gift_observed = True
                self.mode = 'reconcile_after_gift'
                self.agent.global_logic._got_artifact = True
                self.agent.log('ALTAR_PREP observed artifact gift; cooldown repair pending')
                if self.state is not None and self.state['phase'] == 'offering':
                    self.state['phase'] = 'gift'
                    self.state['gift_turn'] = now
                    self.state['gift_step'] = int(self.agent.step_count)
            reconciled = re.search(r'(?:^|(?<=[.!?])\s+|(?<=[.!?]")\s+)You have a feeling of reconciliation\.(?=$|\s)', text)
            clover = PrayerModel._CLOVER_RE.search(text)
            if (reconciled or clover and self.gift_observed) and not gift:
                self.reconciled_turn = now
                if self.gift_observed:
                    self.mode = 'complete'
                    self.agent.log('ALTAR_PREP observed post-gift cooldown cleared')
        if self.state is not None and self.state['phase'] == 'pickup':
            self._record_carried()
        if self.gift is not None:
            matches = [i for i in self.agent.inventory.items if self._weapon_key(i) == self.gift['key']]
            if (self._key() != self.gift['level'] or now - self.gift['turn'] > GIFT_TURNS or
                    len(matches) != 1 or matches[0].shop_status != Item.NOT_SHOP or
                    matches[0].status == Item.CURSED or re.search(r'\bcursed\b', matches[0].text or '') or
                    matches[0].equipped):
                if len(matches) == 1 and matches[0].equipped:
                    self.gift_equipped = True
                self.gift = None

    def _reject(self, reason):
        if reason != self._last_rejection:
            self._last_rejection = reason
            self.rejections[reason] = self.rejections.get(reason, 0) + 1
            if self._discovered:
                self.agent.log(f'ALTAR_PREP unavailable={reason} mode={self.mode}')

    def confirmed_gift(self, item):
        self.update()
        return self.gift is not None and item in self.agent.inventory.items.all_items and \
            self._weapon_key(item) == self.gift['key']

    def _food_reserve(self):
        inventory = self.agent.inventory
        carried = inventory.items.all_items
        # Count ordinary meals at their known nutrition. Tripe is pet food;
        # partly eaten food has no publicly known remaining nutrition.
        return sum(inventory.BUY_FOOD_NUTRITION.get(i.object.name, 0) * i.count
                   for i in self.agent.edible_carried_food()
                   if i in carried and i.shop_status == Item.NOT_SHOP and i.status != Item.CURSED and
                   not i.is_corpse() and i.is_unambiguous() and
                   not re.search(r'\b(?:partly|partially) eaten\b', i.text or ''))

    def _safe(self, *, recovering_gift=False):
        a, bl = self.agent, self.agent.blstats
        p, inv, level = a.character.prop, a.inventory, a.current_level()
        route = getattr(a.global_logic, 'wizard_route', None)
        pet = getattr(a, '_pet_care', {}) or {}
        return self.enabled() and not any(getattr(p, k, False) for k in
            ('blind', 'confusion', 'stun', 'hallu', 'polymorph')) and \
            not (getattr(bl, 'prop_mask', 0) & nh.BL_MASK_LEV) and \
            bl.hitpoints >= .8 * bl.max_hitpoints and \
            bl.hunger_state <= (Hunger.HUNGRY if recovering_gift else Hunger.NOT_HUNGRY) and \
            bl.carrying_capacity == 0 and not a.get_visible_monsters() and \
            not inv.shop_debt_pending() and not inv._credit_clone_reserved() and \
            not inv._preparation_shop_reserved() and not level.shop[self._here()] and \
            all(i.shop_status == Item.NOT_SHOP for i in flatten_items(inv.items)) and \
            getattr(route, 'phase', None) not in ('return', 'recover') and \
            pet.get('hunger_warning') is None and bl.time > self._pet_warning_until and \
            bl.time > getattr(a, '_pet_starving_until', -1) and \
            (recovering_gift or self._food_reserve() >= 800)

    @staticmethod
    def _species_allowed(mid):
        m = MON.permonst(int(mid) + nh.GLYPH_MON_OFF)
        return m.mname not in _KEEP | _DOMESTIC and 'unicorn' not in m.mname and \
            not (m.mflags2 & (MON.M2_GNOME | MON.M2_DOMESTIC | MON.M2_GIANT)) and \
            not m.mconveys and \
            not (m.mflags1 & (MON.M1_TPORT | MON.M1_TPORT_CNTRL))

    def _corpse(self, item, mid):
        return item.is_corpse() and item.monster_id == mid and item.count == 1 and \
            item.shop_status == Item.NOT_SHOP and not item.naming and not item.comment and \
            not re.search(r'partly|partially|\bnamed\b|\bcalled\b', item.text or '') and \
            self._species_allowed(mid)

    def reserved_corpse(self, item):
        s = self.state
        if s is None or s['phase'] not in ('pickup', 'carrying', 'offering') or not self._visit_valid() or \
                item not in self.agent.inventory.items.all_items or not self._safe():
            return False
        self._record_carried()
        return self._carried() is item

    def _visit_valid(self):
        s = self.state
        return s is not None and self._key() == s['level'] and \
            0 <= self.agent.blstats.time - s['born'] < MAX_AGE and \
            self.agent.blstats.time - s['started'] < MAX_TURNS and \
            self.agent.step_count - s['start_step'] < MAX_INPUTS

    def _record_carried(self):
        s = self.state
        if s is None or 'carried_letter' in s:
            return
        matches = [i for i in self.agent.inventory.items if i.is_corpse() and i.monster_id == s['mid']]
        if len(matches) == 1 and self._corpse(matches[0], s['mid']):
            s['carried_letter'] = self.agent.inventory.items.get_letter(matches[0])
            s['carried_text'] = matches[0].text

    def _carried(self):
        s = self.state
        matches = [i for i in self.agent.inventory.items if i.is_corpse() and i.monster_id == s['mid']]
        if len(matches) == 1 and self._corpse(matches[0], s['mid']) and \
                matches[0].text == s.get('carried_text') and \
                self.agent.inventory.items.get_letter(matches[0]) == s.get('carried_letter'):
            return matches[0]

    def reserved_floor_corpse(self, y, x, mid):
        s = self.state
        return s is not None and s['phase'] in ('approach', 'pickup') and self._visit_valid() and self._safe() and \
            self._key() == s['level'] and (int(y), int(x)) == s['corpse_pos'] and mid == s['mid']

    def _finish(self, reason):
        if self.state is not None:
            self._seen.add(self.state['token'])
            if len(self._seen) > 64:
                self._seen.remove(min(self._seen, key=lambda x: (x[-1], x)))
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        self.agent.log(f'ALTAR_PREP finish={reason} mode={self.mode}')
        self.state = None

    def _select(self):
        if self.mode == 'complete':
            self._reject('seeking_complete')
            return None
        if self.mode == 'seeking' and self.agent.blstats.experience_level < 3:
            # pray.c cannot grant an artifact before XL3. Do not divert the
            # fragile opening toward an offering that cannot achieve the goal.
            self._reject('experience_below_artifact_minimum')
            return None
        if not self._safe():
            self._reject('safety_or_food_reserve')
            return None
        a, level, now = self.agent, self.agent.current_level(), int(self.agent.blstats.time)
        dis = a.bfs()
        altars = sorted(p for p, alignment in level.altars.items()
                        if alignment == a.character.alignment and dis[p] >= 0 and not level.shop[p])
        if not altars:
            self._reject('no_confirmed_reachable_coaligned_altar')
            return None
        for pos, species in sorted(level.corpses_to_eat.items()):
            if dis[pos] < 0 or dis[pos] > MAX_WALK or level.shop[pos]:
                continue
            for mid, born in sorted(species.items()):
                token = (self._key(), tuple(map(int, pos)), int(mid), int(born))
                if token in self._seen or not 0 <= now - born < MAX_AGE or not self._species_allowed(mid):
                    continue
                if any(i.is_corpse() and i.monster_id == mid for i in flatten_items(a.inventory.items)):
                    continue  # Never merge provenance with an existing carried corpse.
                altar = min(altars, key=lambda p: (max(abs(p[0]-pos[0]), abs(p[1]-pos[1])), p))
                if dis[pos] + max(abs(altar[0]-pos[0]), abs(altar[1]-pos[1])) > MAX_WALK:
                    continue
                if now - born + MAX_WALK + 3 >= MAX_AGE:
                    continue
                weight = MON.permonst(mid + nh.GLYPH_MON_OFF).cwt
                if a.inventory.items.total_weight + weight > a.character.carrying_capacity:
                    continue
                self._last_rejection = None
                return dict(phase='approach', level=self._key(), corpse_pos=token[1], mid=int(mid),
                            born=int(born), token=token, altar=tuple(map(int, altar)), started=now,
                            start_step=int(a.step_count), walks=0, before_weapons=())
        self._reject('no_surplus_fresh_single_candidate')
        return None

    def _walk(self, pos):
        if self._here() == pos:
            return True
        s, a = self.state, self.agent
        if s['walks'] >= MAX_WALK or a.bfs()[pos] < 0:
            self._finish('unreachable_or_walk_budget')
            return False
        s['walks'] += 1  # Preemption never renews this allowance.
        a.go_to(*pos, max_steps=1)
        return self._here() == pos

    def _confirm_altar(self):
        a = self.agent
        start = len(getattr(a, '_message_history', []))
        with a.atom_operation():
            a.step(A.Command.LOOK)
            text = ' '.join(list(getattr(a, '_message_history', [])[start:]) +
                            [a.message or ''] + list(getattr(a, 'popup', [])))
        expected = next(k for k, v in Character.name_to_alignment.items() if v == a.character.alignment)
        return re.search(r'altar to .+?\(' + expected + r'\) here\.', text) is not None

    def _offer(self):
        a, s = self.agent, self.state
        s['before_weapons'] = tuple(self._weapon_appearance(i) for i in a.inventory.get_items_below_me()
                                    if self._weapon_appearance(i) is not None)
        if not self._safe() or self._here() != s['altar'] or self._key() != s['level'] or \
                int(a.blstats.time) - s['born'] >= MAX_AGE or \
                a.current_level().altars.get(s['altar']) != a.character.alignment:
            self._finish('offer_revalidation')
            return
        current = self._carried()
        if current is None:
            self._finish('carried_identity_changed')
            return
        s['phase'] = 'offering'
        self.attempts += 1
        a.log(f'ALTAR_PREP offer species={s["mid"]} age={int(a.blstats.time)-s["born"]} mode={self.mode}')
        with a.atom_operation():
            a.step(A.Command.OFFER)
            for _ in range(4):
                if ('There is ' in a.message or 'There are ' in a.message) and \
                        ('sacrifice it?' in a.message or 'sacrifice one?' in a.message):
                    a.type_text('n')
                else:
                    break
            if 'What do you want to sacrifice?' not in a.message:
                a.step(A.Command.ESC)
                self._finish('unexpected_offer_prompt')
                return
            a.type_text(a.inventory.items.get_letter(current))
            if 'Your sacrifice is consumed' in a.message:
                self.sacrifices += 1
            self.update()
        # A completed effect can preempt at atomic exit. The persisted phase
        # prevents a second OFFER when the strategy is later re-entered.
        if self.state is s and s['phase'] == 'offering':
            self._finish('offered_once')

    def _pickup_gift(self):
        a, s = self.agent, self.state
        if self._here() != s['altar']:
            if not self._walk(s['altar']):
                return
        below = a.inventory.get_items_below_me()
        new = [i for i in below if self._weapon_appearance(i) is not None and
               self._weapon_appearance(i) not in s['before_weapons'] and i.shop_status == Item.NOT_SHOP]
        if not self._safe(recovering_gift=True) or self._key() != s['level'] or self._here() != s['altar']:
            self._finish('gift_pickup_revalidation')
            return
        if len(new) != 1 or any(self._weapon_appearance(i) == self._weapon_appearance(new[0]) for i in a.inventory.items):
            self._finish('gift_identity_ambiguous')
            return
        self._gift_pending = dict(appearance=self._weapon_appearance(new[0]), level=s['level'],
                                 turn=int(a.blstats.time), observed_modifier=new[0].modifier,
                                 buc_invalidated=s.get('gift_buc_invalidated', False))
        s['phase'] = 'gift_pickup'
        a.inventory.pickup(new[0])
        self._reconcile_gift_pickup()

    def _reconcile_gift_pickup(self):
        g = self._gift_pending
        if g is None:
            self._finish('gift_pickup_unconfirmed')
            return
        matches = [i for i in self.agent.inventory.items if self._weapon_appearance(i) == g['appearance']]
        if len(matches) == 1 and matches[0].shop_status == Item.NOT_SHOP and self._key() == g['level'] and \
                self.agent.blstats.time - g['turn'] <= GIFT_TURNS and \
                (g['observed_modifier'] is None or matches[0].modifier == g['observed_modifier']):
            g['key'] = self._weapon_key(matches[0])
            self.recovered_gift = g['key']
            if not g['buc_invalidated']:
                self.gift = g
            self.agent.log(f'ALTAR_PREP gift recovered={matches[0].text!r}')
        self._gift_pending = None
        self._finish('gift_recovered' if self.gift is g else 'gift_pickup_unconfirmed')

    @Strategy.wrap
    def strategy(self):
        self.update()
        if not self.enabled():
            yield False
        if self.state is not None and self.state['phase'] == 'gift':
            s = self.state
            if self._key() != s['level'] or self.agent.blstats.time - s['gift_turn'] > GIFT_TURNS or \
                    self.agent.step_count - s['gift_step'] >= MAX_INPUTS:
                self._finish('gift_recovery_budget')
                yield False
            if not self._safe(recovering_gift=True):
                # Preserve the observed gift while higher-priority food/combat
                # acts; a ready strategy that does no input would block them.
                yield False
        if self.state is None:
            selected = self._select()
            if selected is None:
                yield False
            yield True
            self.state = selected
        else:
            yield True
        # Finish the bounded visit before yielding back to the underlying
        # navigator: Agent.preempt resumes it once before checking us again.
        while self.state is not None:
            if self.state['phase'] == 'gift' and not self._safe(recovering_gift=True):
                return
            self._advance()

    def _advance(self):
        s, a = self.state, self.agent
        if s['phase'] == 'gift_pickup':
            self._reconcile_gift_pickup()
            return
        if s['phase'] == 'offering':
            self._finish('interrupted_offer_not_retried')
            return
        if s['phase'] == 'gift':
            if self._key() != s['level'] or a.blstats.time - s['gift_turn'] > GIFT_TURNS or \
                    a.step_count - s['gift_step'] >= MAX_INPUTS:
                self._finish('gift_recovery_budget')
                return
            self._pickup_gift()
            return
        if not self._safe() or self._key() != s['level'] or \
                a.blstats.time - s['started'] >= MAX_TURNS or a.step_count - s['start_step'] >= MAX_INPUTS:
            self._finish('safety_or_budget')
            return
        if a.blstats.time - s['born'] >= MAX_AGE or \
                a.current_level().altars.get(s['altar']) != a.character.alignment:
            self._finish('stale_corpse_or_altar')
            return
        if s['phase'] in ('approach', 'pickup'):
            self._record_carried()
            if s['phase'] == 'pickup' and self._carried() is not None:
                s['phase'] = 'carrying'
            elif s['phase'] == 'pickup':
                self._finish('pickup_unconfirmed')
                return
            else:
                if not self._walk(s['corpse_pos']):
                    return
                below = a.inventory.get_items_below_me()
                matches = [i for i in below if i.is_corpse() and i.monster_id == s['mid']]
                age = a.current_level().corpses_to_eat.get(s['corpse_pos'], {}).get(s['mid'])
                if len(matches) != 1 or not self._corpse(matches[0], s['mid']) or age != s['born'] or \
                        not self._safe() or not self._visit_valid() or self._here() != s['corpse_pos']:
                    self._finish('floor_provenance_changed')
                    return
                s['phase'] = 'pickup'
                a.inventory.pickup(matches[0], counts=1)
                return
        if not self._walk(s['altar']):
            return
        if not self._confirm_altar():
            self._finish('altar_not_confirmed')
            return
        self._offer()

    def snapshot(self):
        return dict(mode=self.mode, state=self.state, gift_observed=self.gift_observed,
                    gift=self.gift, reconciled_turn=self.reconciled_turn,
                    attempts=self.attempts, sacrifices=self.sacrifices,
                    discovered_altars=sorted(self._discovered),
                    usable_altars=sum(alignment == self.agent.character.alignment
                                      for alignment in self.agent.current_level().altars.values()),
                    gift_equipped=self.gift_equipped,
                    recovered_gift=self.recovered_gift,
                    rejections=dict(self.rejections))
