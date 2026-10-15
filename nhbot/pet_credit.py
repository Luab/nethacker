"""Bounded shop credit cloning with an owned gold sack and an observed pet.

Only ordinary observations and commands are used. Shop credit is not cash:
the principal must come back outside, and the owner's balance is queried
before it becomes available to the normal food-purchase transaction.
"""
from collections import deque
from dataclasses import dataclass

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from .glyph import G, Hunger, MON, SS
from . import jf_config
from .item import Item
from .shop_transactions import quantity_price
from .strategy import Strategy


MAX_TURNS = 160
CLEANUP_TURNS = 24
MAX_DECISIONS = 240
MAX_CYCLES = 2
MIN_PRINCIPAL = 50
MAX_PRINCIPAL = 400
MAX_QUOTE_SQUARES = 4
MAX_QUOTE_TURNS = 24
_DOMESTIC = {'kitten', 'housecat', 'large cat', 'little dog', 'dog', 'large dog'}
_GROUND = G.FLOOR | G.DOOR_OPENED
_DONE = {'done', 'abandoned'}


@dataclass
class Clone:
    level_key: tuple
    shop_id: int
    bag_id: str
    stage: tuple
    entrance: tuple
    outside: tuple
    principal: int
    target_credit: int
    started: int
    total_gold_before: int
    bag_glyphs: tuple = ()
    phase: str = 'query'
    credit_before: int = 0
    credit_after: int = 0
    cycles: int = 0
    decisions: int = 0
    cleanup_started: int = 0
    failure: str = ''
    recovery_target: tuple = None
    quote_started: int = 0
    quotes_seen: tuple = ()


def reserves_inventory(agent):
    state = getattr(agent, '_credit_clone', None)
    return state is not None and state.phase not in _DONE


def _here(agent):
    return int(agent.blstats.y), int(agent.blstats.x)


def _shop(agent, state):
    level = agent.current_level()
    if level.key() != state.level_key:
        return None
    return level.shops.records.get(state.shop_id)


def _bag(agent, state):
    matches = [item for item in agent.inventory.items
               if item.container_id == state.bag_id and item.is_container()]
    return matches[0] if len(matches) == 1 else None


def _gold_in(bag):
    if bag is None or bag.content.locked or any(
            item.category != nh.COIN_CLASS or item.shop_status != Item.NOT_SHOP
            for item in bag.content.items):
        return None
    return sum(int(item.count) for item in bag.content.items)


def _usable_bag(bag):
    return (bag.is_container() and bag.is_unambiguous() and
            bag.object.name in ('sack', 'oilskin sack') and bag.container_id is not None and
            bag.shop_status == Item.NOT_SHOP and bag.status in (Item.BLESSED, Item.UNCURSED) and
            not bag.equipped and bag.count == 1 and _gold_in(bag) is not None and
            bag.weight() <= 40)


def _pet_positions(agent):
    positions = []
    for y, x in zip(*np.isin(agent.glyphs, tuple(G.PETS)).nonzero()):
        if MON.permonst(int(agent.glyphs[y, x])).mname not in _DOMESTIC:
            return []
        positions.append((int(y), int(x)))
    return positions if len(positions) == 1 else []


def _safe(agent, *, cleanup=False):
    bl, character = agent.blstats, agent.character
    if character.role != character.WIZARD:
        return False
    if any(getattr(character.prop, name, False) for name in
           ('blind', 'confusion', 'stun', 'hallu', 'polymorph')):
        return False
    if int(bl.prop_mask) & nh.BL_MASK_LEV:
        return False
    return (bl.hitpoints >= 6 and bl.hitpoints * 5 >= bl.max_hitpoints * 4 and
            bl.hunger_state <= (Hunger.HUNGRY if cleanup else Hunger.NOT_HUNGRY) and
            bl.carrying_capacity < 2 and not agent.get_visible_monsters() and
            not character.teleportitis and not agent.hands_welded() and
            not agent.inventory.shop_debt_pending() and not agent._carries_digging_tool())


def _peaceful_owner(agent, shop):
    area = shop.interior.copy()
    for point in shop.entrances:
        area[point] = True
    return np.count_nonzero(area & agent.monster_tracker.peaceful_monster_mask &
                            np.isin(agent.glyphs, tuple(G.SHOPKEEPER))) == 1


def _floor(agent, position):
    level = agent.current_level()
    y, x = position
    return (0 <= y < level.objects.shape[0] and 0 <= x < level.objects.shape[1] and
            int(level.objects[position]) in _GROUND and level.walkable[position] and
            not level.forbidden[position] and
            not (getattr(level, 'forbidden_until', np.zeros_like(level.objects))[position] >
                 agent.blstats.time) and
            not (getattr(level, 'petrify_until', np.zeros_like(level.objects))[position] >
                 agent.blstats.time))


def _layout(agent, shop):
    if len(shop.entrances) != 1:
        return None
    entrance = next(iter(shop.entrances))
    if not _floor(agent, entrance):
        return None
    layouts = []
    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        points = [tuple(entrance[i] + n * (dy, dx)[i] for i in (0, 1))
                  for n in (-2, -1, 1, 2)]
        if not all(_floor(agent, point) for point in points):
            continue
        outside, threshold_out, threshold_in, stage = points
        if (shop.interior[threshold_in] and shop.interior[stage] and
                not agent.current_level().shop_interior[outside] and
                not agent.current_level().shop_interior[threshold_out] and
                agent.current_level().item_count[stage] == 0):
            # Wait immediately outside the doorway. Further retreat can pull
            # the pet out of sight of an off-axis sack, defeating its goal scan.
            layouts.append((stage, tuple(entrance), threshold_out))
    return layouts[0] if len(layouts) == 1 else None


def _useful_credit(agent, shop, principal):
    """Fund at most the ordinary food reserve, never an unbounded balance."""
    inv, level = agent.inventory, agent.current_level()
    needed = max(0, jf_config.BUY_FOOD_UNTIL - inv.carried_nutrition())
    prices = []
    known = shop.interior & (level.item_count > 0)
    if shop.interior[_here(agent)] and inv.items_below_me is not None:
        known[_here(agent)] = True
    for y, x in zip(*known.nonzero()):
        rows = inv.items_below_me if (y, x) == _here(agent) else level.items[y, x]
        for item in rows or []:
            if item.shop_status != Item.FOR_SALE or not item.is_unambiguous():
                continue
            nutrition = inv.BUY_FOOD_NUTRITION.get(item.object.name, 0)
            price = quantity_price(item)
            if nutrition and price is not None:
                prices.extend([(price / nutrition, price, nutrition)] * min(item.count, 4))
    total = 0
    for _, price, nutrition in sorted(prices):
        if needed <= 0:
            break
        total += price
        needed -= nutrition
    return min(total, principal * MAX_CYCLES)


def _quote_target(agent, state, shop):
    """Inspect a few visible food piles before sizing the shop-food budget."""
    level = agent.current_level()
    targets = []
    for y, x in zip(*(shop.interior & np.isin(agent.glyphs, tuple(G.FOOD_OBJECTS))).nonzero()):
        point = int(y), int(x)
        if (point in state.quotes_seen or not _floor(agent, point) or
                max(abs(point[i] - state.entrance[i]) for i in (0, 1)) > 3 or
                agent.monster_tracker.monster_mask[point]):
            continue
        rows = level.items[point] if level.item_count[point] > 0 else []
        if any(item.category == nh.FOOD_CLASS and item.is_unambiguous() and
               item.shop_status == Item.FOR_SALE and quantity_price(item) is not None
               for item in rows):
            continue
        targets.append(point)
    return min(targets, key=lambda p: (abs(p[0] - _here(agent)[0]) +
                                      abs(p[1] - _here(agent)[1]), p)) if targets else None


def _replacement_stage(agent, state, shop):
    """Reconsider observed empty floor when stock moves onto the deposit spot."""
    level = agent.current_level()
    pets = _pet_positions(agent)
    if not pets:
        return None
    candidates = []
    for y, x in zip(*(shop.interior & (level.item_count == 0)).nonzero()):
        point = int(y), int(x)
        distance = max(abs(point[i] - state.entrance[i]) for i in (0, 1))
        # The owner's home is next to the door and is not chargeable. Stay
        # beyond that ring, close enough for the existing bounded walk/fetch.
        if (point != state.stage and 2 <= distance <= 3 and _floor(agent, point) and
                not agent.monster_tracker.monster_mask[point] and
                max(abs(point[i] - pets[0][i]) for i in (0, 1)) <= 5):
            candidates.append(point)
    return min(candidates, key=lambda p: (abs(p[0] - _here(agent)[0]) +
                                         abs(p[1] - _here(agent)[1]), p)) if candidates else None


def _plan(agent):
    pets = _pet_positions(agent)
    if not _safe(agent) or not pets:
        return None
    from . import pet_care
    if not pet_care.ready_for_fetch(agent):
        # Unknown nutrition is a reason to prepare a real job, not to feed
        # every pet encountered while exploring. A tiny meat treat cannot
        # establish reserve for the whole bounded job.
        food = pet_care.available_food(agent)
        if food is None or food.object.name != 'tripe ration':
            return None
    level = agent.current_level()
    shop = level.shops.shop_at(*_here(agent), include_entrance=True)
    if shop is None or not _peaceful_owner(agent, shop):
        return None
    if (level.key(), shop.shop_id) in getattr(agent, '_credit_clone_attempts', set()):
        return None
    layout = _layout(agent, shop)
    if layout is None:
        return None
    if max(abs(pets[0][i] - layout[0][i]) for i in (0, 1)) > 5:
        return None
    bags = [item for item in agent.inventory.items if _usable_bag(item) and
            sum(other.container_id == item.container_id for other in agent.inventory.items) == 1]
    for bag in bags:
        gold = _gold_in(bag)
        principal = min(MAX_PRINCIPAL, int(agent.blstats.gold) + gold)
        # Never expose other valuables or a large prefilled money bag.
        if principal < MIN_PRINCIPAL or gold > principal:
            continue
        target = _useful_credit(agent, shop, principal)
        if target <= agent.inventory.shop_credit(shop) or target == 0:
            continue
        return Clone(level.key(), shop.shop_id, bag.container_id, *layout,
                     principal, target, int(agent.blstats.time), int(agent.blstats.gold) + gold,
                     bag_glyphs=tuple(bag.glyphs))
    return None


def _finish(agent, state, reason, success=False):
    if not success and _bag(agent, state) is None:
        # Generic gathering must not later perform the debt-producing pickup
        # which rollback refused. Outside delivery remains safe to recover.
        unrecovered = getattr(agent, '_credit_clone_unrecovered', {})
        unrecovered[(state.level_key, state.bag_id)] = {
            'shop_id': state.shop_id, 'principal': state.principal}
        agent._credit_clone_unrecovered = unrecovered
    state.failure = reason
    state.phase = 'done' if success else 'abandoned'
    agent.log(f'PET_CREDIT {state.phase}: {reason}; cycles={state.cycles} '
              f'principal={state.principal} credit={state.credit_after}')


def _cleanup(agent, state, reason):
    if state.phase != 'cleanup':
        state.phase = 'cleanup'
        state.cleanup_started = int(agent.blstats.time)
        state.failure = reason
        agent.inventory.invalidate_shop_credit()
        agent.log(f'PET_CREDIT cleanup: {reason}')


def _walk(agent, state, target):
    """One ordinary cardinal step on observed safe floor, in a small local area."""
    start = _here(agent)
    if start == target:
        return True
    for approach in (False, True):
        queue, parent = deque([start]), {start: None}
        while queue:
            point = queue.popleft()
            for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                nxt = point[0] + dy, point[1] + dx
                if nxt in parent or max(abs(nxt[i] - state.entrance[i]) for i in (0, 1)) > 4:
                    continue
                if not _floor(agent, nxt) or (not approach and agent.monster_tracker.monster_mask[nxt]):
                    continue
                parent[nxt] = point
                if nxt == target:
                    while parent[nxt] != start:
                        nxt = parent[nxt]
                    # A keeper can stand near the door until we approach.
                    # Prefer a completely clear route, then use static floor
                    # only to approach a distant obstruction. Never step onto
                    # an occupied tile, attack, or swap through this fallback.
                    if agent.monster_tracker.monster_mask[nxt]:
                        agent.search()
                        return False
                    agent._settle_before_shop_exit(*nxt)
                    agent.move(*nxt)
                    return _here(agent) == target
                queue.append(nxt)
    agent.search()  # Let the keeper/pet move, charged to the bounded budget.
    return False


def _drop_bag(agent, state, bag):
    letter = agent.inventory.items.get_letter(bag)
    def answers():
        if 'What do you want to drop?' not in agent.single_message:
            return
        yield letter
        for _ in range(8):
            if agent._observation['misc'][0]:
                yield 'n'  # Gold credit is automatic; never sell the sack.
            elif agent._observation['misc'][2]:
                yield A.MiscAction.MORE
            else:
                return
    state.phase = 'deposit_check'  # Persist before the atomic exit can preempt.
    agent.inventory.invalidate_shop_credit()
    with agent.atom_operation():
        agent.step(A.Command.DROP, answers())
        agent.inventory.items.update()


def _outside_bag_target(agent, state, shop):
    # A matching bag glyph is a reason to inspect, never proof of identity.
    level, here = agent.current_level(), _here(agent)
    if _recovery_square(agent, here, state):
        for item in agent.inventory.items_below_me or []:
            if item.container_id == state.bag_id:
                return here
    candidates = []
    for y, x in zip(*np.isin(agent.glyphs, tuple(G.OBJECTS)).nonzero()):
        p = int(y), int(x)
        if (not _recovery_square(agent, p, state) or
                max(abs(p[i] - state.outside[i]) for i in (0, 1)) > 3):
            continue
        known = level.items[p] if level.item_count[p] else []
        if any(item.container_id == state.bag_id for item in known):
            return p
        if int(agent.glyphs[p]) in state.bag_glyphs:
            candidates.append(p)
    return min(candidates, key=lambda p: abs(p[0]-here[0])+abs(p[1]-here[1])) if candidates else None


def _recovery_square(agent, position, state):
    level = agent.current_level()
    if not _floor(agent, position) or level.shop_interior[position]:
        return False
    entrances = {p for shop in level.shops.records.values() for p in shop.entrances}
    # NetHack's actual doorway is outside the costly room. A room-floor
    # bottleneck inferred by ShopModel is not proof of a free pickup square.
    return position not in entrances or (position == state.entrance and
            int(level.objects[position]) in G.DOOR_OPENED | {SS.S_ndoor})


def _run(agent, state):
    inv = agent.inventory
    while state.phase not in _DONE:
        state.decisions += 1
        shop = _shop(agent, state)
        if shop is None:
            _finish(agent, state, 'level or owner changed')
            return
        if state.decisions > MAX_DECISIONS:
            _finish(agent, state, 'decision budget exhausted')
            return
        if not _safe(agent, cleanup=state.phase == 'cleanup'):
            _cleanup(agent, state, 'hunger, threat, health or transaction changed')
            return
        if state.phase != 'cleanup' and agent.blstats.time - state.started >= MAX_TURNS:
            _cleanup(agent, state, 'fetch deadline')
        if state.phase not in ('cleanup', 'query', 'quote', 'prepare_pet'):
            from . import pet_care
            remaining = max(1, MAX_TURNS - int(agent.blstats.time) + state.started)
            if not pet_care.ready_for_fetch(agent, budget=remaining):
                if pet_care.ready_for_fetch(agent, budget=remaining, allow_eating=True):
                    agent.search()
                    continue
                _cleanup(agent, state, 'pet feeding reserve no longer adequate')
        bag = _bag(agent, state)
        if state.phase == 'query':
            if not shop.interior[_here(agent)]:
                _walk(agent, state, state.stage)
                continue
            balance = inv.refresh_shop_credit(shop)
            if balance is None or not _peaceful_owner(agent, shop):
                _finish(agent, state, 'no authoritative clean shop balance')
                return
            state.credit_before = balance
            if balance >= state.target_credit:
                _finish(agent, state, 'existing credit already sufficient', success=True)
                return
            state.quote_started = int(agent.blstats.time)
            state.phase = 'quote'
        elif state.phase == 'quote':
            target = _quote_target(agent, state, shop)
            if (target is None or len(state.quotes_seen) >= MAX_QUOTE_SQUARES or
                    agent.blstats.time - state.quote_started >= MAX_QUOTE_TURNS):
                state.target_credit = max(state.target_credit, _useful_credit(agent, shop, state.principal))
                from . import pet_care
                remaining = max(1, MAX_TURNS - int(agent.blstats.time) + state.started)
                state.phase = ('pack' if pet_care.ready_for_fetch(agent, budget=remaining)
                               else 'prepare_pet')
                continue
            if not _walk(agent, state, target):
                continue
            inv.get_items_below_me()
            state.quotes_seen += (target,)
            # Include the fresh LOOK even before the next map-cache update.
            state.target_credit = max(state.target_credit, _useful_credit(agent, shop, state.principal))
        elif state.phase == 'prepare_pet':
            from . import pet_care
            remaining = max(1, MAX_TURNS - int(agent.blstats.time) + state.started)
            if pet_care.ready_for_fetch(agent, budget=remaining):
                state.phase = 'pack'
                continue
            if pet_care.ready_for_fetch(agent, budget=remaining, allow_eating=True):
                agent.search()
                continue
            food = pet_care.available_food(agent)
            if not _pet_positions(agent) or food is None or food.object.name != 'tripe ration':
                _finish(agent, state, 'no visible pet or suitable preparation food')
                return
            # Leave room for an adjacent pet to stand outside the protected
            # doorway. Fetching itself still waits only one square outside.
            target = tuple(2 * state.outside[i] - state.entrance[i] for i in (0, 1))
            if not _floor(agent, target) or agent.current_level().shop[target]:
                _finish(agent, state, 'no safe outside feeding square')
                return
            if not _walk(agent, state, target):
                continue
            if not pet_care.feed_pet(agent, for_fetch=True).run(return_condition=True):
                # The pet may be a square behind or still in the doorway.
                # Waiting, failed throws and preemption retain this deadline.
                agent.search()
        elif state.phase == 'pack':
            if bag is None or not _usable_bag(bag):
                _finish(agent, state, 'owned safe sack unavailable')
                return
            # Reinspect after any interrupted transfer before computing a delta.
            inv.check_container_content(bag)
            bag = _bag(agent, state)
            if bag is None or _gold_in(bag) is None:
                _finish(agent, state, 'unverified bag contents')
                return
            missing = state.principal - _gold_in(bag)
            if missing < 0:
                _finish(agent, state, 'bag contains excess principal')
                return
            if missing:
                coin = next((i for i in inv.items if i.category == nh.COIN_CLASS and
                             i.shop_status == Item.NOT_SHOP and i.count >= missing), None)
                if coin is None:
                    _finish(agent, state, 'owned cash unavailable')
                    return
                # Resume PACK and re-read content, never replay a stale quantity.
                inv.use_container(bag, [coin], [], items_to_put_counts=[missing])
                continue
            state.phase = 'deposit'
        elif state.phase == 'deposit':
            if not _walk(agent, state, state.stage):
                continue
            inv.get_items_below_me()
            bag = _bag(agent, state)
            if bag is None or not _usable_bag(bag) or _gold_in(bag) != state.principal:
                _finish(agent, state, 'deposit sack changed')
                return
            if inv.items_below_me:
                replacement = _replacement_stage(agent, state, shop)
                if replacement is None:
                    _finish(agent, state, 'no observed empty deposit square remains')
                    return
                state.stage = replacement
                continue
            if not _peaceful_owner(agent, shop):
                _cleanup(agent, state, 'keeper no longer observed peaceful')
                continue
            _drop_bag(agent, state, bag)
        elif state.phase == 'deposit_check':
            if bag is not None:
                _finish(agent, state, 'drop did not release sack')
                return
            balance = inv.refresh_shop_credit(shop)
            if balance is None or balance < state.credit_before + state.principal:
                _cleanup(agent, state, 'deposit not confirmed')
                continue
            state.credit_after = balance
            state.phase = 'exit'
            agent.log(f'PET_CREDIT deposited {state.principal}; credit={balance}')
        elif state.phase == 'exit':
            if _walk(agent, state, state.outside):
                state.phase = 'fetch'
        elif state.phase == 'fetch':
            if bag is not None:
                state.phase = 'verify'
                continue
            inv.get_items_below_me()
            target = _outside_bag_target(agent, state, shop)
            if target is not None:
                if not _walk(agent, state, target):
                    continue
                inv.get_items_below_me()
                found = next((i for i in inv.items_below_me if i.container_id == state.bag_id and
                              i.shop_status == Item.NOT_SHOP), None)
                if found is not None and _recovery_square(agent, _here(agent), state):
                    state.phase = 'verify'
                    inv.pickup(found)
                    inv.items.update()
                    continue
            if _here(agent) != state.outside:
                _walk(agent, state, state.outside)
            else:
                agent.search()
        elif state.phase == 'verify':
            if bag is None:
                state.phase = 'fetch'  # Pickup may have been interrupted before selection.
                continue
            inv.check_container_content(bag)
            bag = _bag(agent, state)
            if bag is None or not _usable_bag(bag) or _gold_in(bag) != state.principal or \
                    int(agent.blstats.gold) + _gold_in(bag) < state.total_gold_before:
                _finish(agent, state, 'returned principal not verified')
                return
            state.phase = 'verify_credit'
        elif state.phase == 'verify_credit':
            if not _walk(agent, state, state.stage):
                continue
            bag = _bag(agent, state)
            if bag is None or _gold_in(bag) != state.principal:
                _finish(agent, state, 'principal changed during return')
                return
            balance = inv.refresh_shop_credit(shop)
            if balance is None or balance < state.credit_before + state.principal:
                _finish(agent, state, 'credit not retained after retrieval')
                return
            state.cycles += 1
            state.credit_after = balance
            state.target_credit = max(state.target_credit, _useful_credit(agent, shop, state.principal))
            agent.log(f'PET_CREDIT recovered principal={state.principal}; retained_credit={balance}')
            if balance >= state.target_credit or state.cycles >= MAX_CYCLES:
                _finish(agent, state, 'principal recovered and spendable credit verified', success=True)
                return
            state.credit_before = balance
            state.phase = 'deposit'
        elif state.phase == 'cleanup':
            if bag is not None:
                _finish(agent, state, state.failure + '; sack in inventory')
                return
            if agent.blstats.time - state.cleanup_started >= CLEANUP_TURNS:
                _finish(agent, state, state.failure + '; recovery deadline, no success claimed')
                return
            # Rollback only the originally staged, still-owned sack, after an
            # authoritative balance proves its gold cannot create new debit.
            if not _walk(agent, state, state.stage):
                continue
            inv.get_items_below_me()
            found = next((i for i in inv.items_below_me if i.container_id == state.bag_id and
                          i.shop_status == Item.NOT_SHOP), None)
            if found is None:
                _finish(agent, state, state.failure + '; sack not at observed origin')
                return
            balance = inv.refresh_shop_credit(shop)
            if balance is None or balance < state.principal:
                _finish(agent, state, state.failure + '; rollback would risk debt')
                return
            inv.invalidate_shop_credit(shop)
            inv.pickup(found)
            inv.items.update()
            inv.refresh_shop_credit(shop)
            _finish(agent, state, state.failure + '; reclaimed original sack')
            return


@Strategy.wrap
def clone_credit(agent):
    state = getattr(agent, '_credit_clone', None)
    if state is not None and state.phase not in _DONE:
        if agent.current_level().key() != state.level_key or \
                agent.blstats.time - state.started > MAX_TURNS + CLEANUP_TURNS:
            agent.inventory.invalidate_shop_credit()
            _finish(agent, state, 'attempt expired or level changed during interruption')
            yield False
            return
        if not _safe(agent, cleanup=True):
            yield False
            return
    else:
        state = _plan(agent)
        if state is None:
            yield False
            return
    yield True
    if not reserves_inventory(agent):
        # Revalidate after selection, before reserving inventory or issuing input.
        state = _plan(agent)
        if state is None:
            return
        agent._credit_clone = state
        attempts = getattr(agent, '_credit_clone_attempts', set())
        attempts.add((state.level_key, state.shop_id))
        agent._credit_clone_attempts = attempts
        agent.log(f'PET_CREDIT begin shop={state.shop_id} principal={state.principal}')
    _run(agent, state)
