"""Wizard hunger recovery from observable action results, independent of strategy restarts."""

from . import objects as O
from .character import Character
from .glyph import G, Hunger
from .item import Item


class HungerRecovery:
    def __init__(self, agent):
        self.agent = agent
        self.episode_started = None
        self.relief_count = 0
        self.prayer_unresolved = False
        self.pending_prayer = None
        self._relieved_prayer = None
        self.last_outcome = None
        self.food_retry_after = -1
        self.decision = None
        self.forage_blocked = {}

    @property
    def enabled(self):
        return getattr(self.agent.character, 'role', None) == Character.WIZARD

    def observe(self):
        if not self.enabled:
            return
        bl = self.agent.blstats
        if bl.hunger_state >= Hunger.WEAK:
            if self.episode_started is None:
                self.episode_started = int(bl.time)
        elif self.episode_started is not None:
            self.relief_count += 1
            self.episode_started = None
            self.prayer_unresolved = False
            self.food_retry_after = -1
            self.forage_blocked.clear()
            self._relieved_prayer, self.pending_prayer = self.pending_prayer, None
            self.note('resume', 'observed_hunger_relief')

    def note(self, action, reason):
        decision = {'action': action, 'reason': reason, 'episode_started': self.episode_started}
        if decision != self.decision:
            self.decision = decision
            self.agent.log(f'SURVIVAL decision {decision}')

    def begin_prayer(self):
        self.observe()
        self._relieved_prayer = None
        if not self.enabled or self.episode_started is None:
            return
        self.pending_prayer = {'episode_started': self.episode_started,
                               'turn': int(self.agent.blstats.time),
                               'hunger_before': int(self.agent.blstats.hunger_state)}

    def cancel_prayer(self):
        self.pending_prayer = self._relieved_prayer = None

    def finish_prayer(self, accepted, failed, began=True):
        pending = self.pending_prayer or self._relieved_prayer
        self.cancel_prayer()
        if pending is None:
            return
        if not began:
            self.last_outcome = dict(pending, action='pray', result='not_started')
            return
        self.observe()
        relieved = self.agent.blstats.hunger_state < Hunger.WEAK
        if not relieved:
            self.prayer_unresolved = True
        self.last_outcome = dict(pending, action='pray', accepted=bool(accepted), failed=bool(failed),
                                 hunger_after=int(self.agent.blstats.hunger_state),
                                 hunger_relieved=relieved)
        self.agent.log(f'SURVIVAL outcome {self.last_outcome}')
        self.note('resume' if relieved else 'find_food',
                  'observed_hunger_relief' if relieved else 'prayer_did_not_relieve_hunger')

    def blocks_hunger_override(self):
        self.observe()
        return self.enabled and self.episode_started is not None and \
            (self.prayer_unresolved or self.pending_prayer is not None)

    def usable_food(self):
        """Immediately accessible owned meals; defer tins and unpacking to normal inventory logic."""
        if not self.enabled or self.agent.blstats.hunger_state < Hunger.WEAK or \
                self.agent.blstats.carrying_capacity >= 4 or self.agent.character.prop.polymorph:
            return []
        inventory = self.agent.inventory
        return [item for item in self.agent.edible_carried_food()
                if item in inventory.items and self._safe_meal(item)]

    def _safe_meal(self, item):
        return item.shop_status == Item.NOT_SHOP and item.status != Item.CURSED and \
            item.is_unambiguous() and (item.is_corpse() or
                                      item.object.name in self.agent.inventory.BUY_FOOD_NUTRITION)

    def urgent_food(self):
        self.observe()
        if not self.prayer_unresolved or self.agent.blstats.time < self.food_retry_after:
            return None
        if self.agent.character.prop.confusion or self.agent.character.prop.stun:
            return None
        bl = self.agent.blstats
        if any(max(abs(y - bl.y), abs(x - bl.x)) <= 1
               for _, y, x, _, _ in self.agent.get_visible_monsters()):
            return None
        food = self.usable_food()
        return food[0] if food else None

    def eat(self, item):
        before = int(self.agent.blstats.hunger_state)
        self.food_retry_after = int(self.agent.blstats.time) + 10
        self.note('eat', 'usable_owned_food')
        try:
            self.agent.inventory.eat(item)
        finally:
            # Inventory actions may finish before an update callback preempts their caller.
            self.observe()
            self.last_outcome = {'action': 'eat', 'hunger_before': before,
                                 'hunger_after': int(self.agent.blstats.hunger_state),
                                 'hunger_relieved': self.agent.blstats.hunger_state < Hunger.WEAK}
            self.agent.log(f'SURVIVAL outcome {self.last_outcome}')

    def nearby_food(self):
        return self.forage_target() is not None

    def _can_forage(self):
        agent = self.agent
        prop = agent.character.prop
        return self.enabled and self.prayer_unresolved and agent.blstats.hunger_state >= Hunger.WEAK and \
            not (prop.blind or prop.hallu or prop.confusion or prop.stun or prop.polymorph) and \
            agent.blstats.carrying_capacity < 4

    def forage_target(self):
        """A bounded walk to public food evidence; ownership is checked again on arrival."""
        agent = self.agent
        if not self._can_forage():
            return None
        level = agent.current_level()
        candidates = {position: True for position in level.corpses_to_eat}
        for y, x in zip(*((agent.glyphs >= min(G.FOOD_OBJECTS)) &
                         (agent.glyphs <= max(G.FOOD_OBJECTS))).nonzero()):
            glyph = int(agent.glyphs[y, x])
            if glyph in G.FOOD_OBJECTS and all(o.name in agent.inventory.BUY_FOOD_NUTRITION
                                              for o in O.possibilities_from_glyph(glyph)):
                candidates[int(y), int(x)] = False
        if any(self._safe_meal(item) and not item.is_corpse()
               for item in (agent.inventory.items_below_me or ())):
            candidates[int(agent.blstats.y), int(agent.blstats.x)] = False
        if not candidates:
            return None
        dis = agent.bfs()
        targets = []
        for (y, x), corpse_only in candidates.items():
            key = (level.key(), y, x)
            if level.shop[y, x] or not 0 <= dis[y, x] <= 12 or \
                    agent.blstats.time < self.forage_blocked.get(key, -1):
                continue
            corpses = level.corpses_to_eat.get((y, x))
            if corpse_only and corpses is not None and not any(agent._is_corpse_editable(mid, age) and
                                               not agent.reserve_corpse(mid) for mid, age in corpses.items()):
                continue
            targets.append((int(dis[y, x]), int(y), int(x)))
        return min(targets)[1:] if targets else None

    def forage(self, target):
        agent = self.agent
        level = agent.current_level()
        key = (level.key(), *target)
        self.note('forage', 'reachable_food')
        if (agent.blstats.y, agent.blstats.x) != target:
            before = (agent.blstats.y, agent.blstats.x)
            try:
                agent.go_to(*target, max_steps=1)
            finally:
                if (agent.blstats.y, agent.blstats.x) == before:
                    self.forage_blocked[key] = int(agent.blstats.time) + 20
            if agent.current_level().key() != level.key() or (agent.blstats.y, agent.blstats.x) != target:
                return
            # Agent.preempt resumes its base strategy after this invocation;
            # returning on arrival would let a dive walk off the food before
            # checking this emergency again. Complete the arrived meal here.
        if not self._can_forage() or agent.get_visible_monsters() or level.shop[target]:
            return
        # A glyph identifies food, not its owner or a corpse's freshness.
        for item in agent.inventory.items_below_me:
            if not self._safe_meal(item):
                continue
            if item.is_corpse():
                age = level.corpses_to_eat.get(target, {}).get(item.monster_id)
                if item.count != 1 or age is None or not agent._is_corpse_editable(item.monster_id, age) or \
                        agent.reserve_corpse(item.monster_id):
                    continue
            self.forage_blocked[key] = int(agent.blstats.time) + 20
            self.eat(item)
            return
        self.forage_blocked[key] = int(agent.blstats.time) + 20

    def release_idle_guard(self, near):
        self.observe()
        bl = self.agent.blstats
        if not self.enabled or self.episode_started is None or \
                any(max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1 for m in near):
            return False
        if self.prayer_unresolved or self.nearby_food():
            self.note('forage', 'unresolved_hunger' if self.prayer_unresolved else 'reachable_food')
            return True
        return False

    def needs_resource_exit(self):
        self.observe()
        result = self.enabled and self.episode_started is not None and self.prayer_unresolved and \
            not self.usable_food() and not self.nearby_food()
        if result:
            self.note('leave_floor', 'hunger_unresolved_without_food')
        return result

    def snapshot(self):
        return {'episode_started': self.episode_started, 'prayer_unresolved': self.prayer_unresolved,
                'relief_count': self.relief_count,
                'pending_prayer': self.pending_prayer, 'last_outcome': self.last_outcome,
                'decision': self.decision}
