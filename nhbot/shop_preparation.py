"""One bounded preparation shop visit with an owned tool left outside.

The tool reservation and its recovery obligation survive strategy preemption.
Only public item descriptions, observed shop geometry and ordinary commands
are used.  Financial settlement remains Inventory's separate transaction.
"""
from dataclasses import dataclass, field
import re

import nle.nethack as nh
import numpy as np

from . import jf_config
from .glyph import G, Hunger, SHOP, SS
from .item import Item, flatten_items
from .shop_transactions import quantity_price
from .strategy import Strategy
from .wizard_route import TOTAL_TURNS


MAX_TURNS = 150
CLEANUP_TURNS = 24
MAX_DECISIONS = 220
MAX_QUOTES = 4
MAX_SURVEY_POINTS = 3
MAX_BLOCKED_WAITS = 8
_TOOLS = {'pick-axe', 'dwarvish mattock'}
_DONE = {'done', 'abandoned'}


@dataclass
class Visit:
    level_key: tuple
    shop_id: int
    owner: object
    signature: tuple
    deposit: tuple
    entrance: tuple
    started: int
    deadline: int
    phase: str = 'deposit'
    deposited: bool = False
    decisions: int = 0
    blocked: int = 0
    quotes: set = field(default_factory=set)
    surveyed: set = field(default_factory=set)
    target: tuple = None
    surveying: bool = False
    reason: str = ''


def reserves_inventory(agent):
    state = getattr(agent, '_shop_preparation', None)
    return state is not None and state.phase not in _DONE


def _here(agent):
    return int(agent.blstats.y), int(agent.blstats.x)


def _signature(item):
    if not item.is_unambiguous() or item.object.name not in _TOOLS or not item.text:
        return None
    # ItemManager optimistically maps UNKNOWN BUC to UNCURSED. Keep the
    # actual public description instead, including unknown enchantment/BUC.
    text = re.sub(r' \((?:weapon in [^)]*|wielded|at the ready|in quiver|'
                  r'alternate weapon; not wielded|no charge)\)', '', item.text)
    text = re.sub(r'^(?:a|an|the|1) ', '', text)
    return item.object.name, text, int(item.count)


def _owned_match(item, state):
    return item.shop_status == Item.NOT_SHOP and _signature(item) == state.signature


def _carried(agent, state):
    matches = [item for item in agent.inventory.items if _owned_match(item, state)]
    return matches[0] if len(matches) == 1 else None


def _tool(agent):
    tools = [item for item in flatten_items(agent.inventory.items)
             if any(obj.name in _TOOLS for obj in item.objs)]
    if len(tools) != 1:
        return None
    item = tools[0]
    if item not in agent.inventory.items.all_items or item.count != 1 or \
            item.shop_status != Item.NOT_SHOP or _signature(item) is None or \
            not item.can_be_dropped_from_inventory():
        return None
    # A cursed unwielded tool can be dropped. A wielded one must be known
    # releasable; hands_welded() only tests free hands and is insufficient.
    if item.equipped and not re.match(r'^(?:a|an|the|1) (?:blessed|uncursed) ', item.text):
        return None
    return item


def _demand(agent):
    route = getattr(agent.global_logic, 'wizard_route', None)
    target = getattr(route, 'preparation_food_target', None)
    return target is not None and jf_config.BUY_FOOD and \
        agent.inventory.carried_nutrition() < min(target, jf_config.BUY_FOOD_UNTIL)


def _safe(agent):
    bl, prop = agent.blstats, agent.character.prop
    return (bl.hunger_state < Hunger.WEAK and bl.hitpoints >= max(10, 0.8 * bl.max_hitpoints)
            and bl.carrying_capacity <= 1 and not int(bl.prop_mask) & nh.BL_MASK_LEV
            and not any(getattr(prop, key, False) for key in
                        ('blind', 'hallu', 'confusion', 'stun', 'polymorph', 'stoned'))
            and not agent.character.teleportitis and not agent.get_visible_monsters())


def _shop(agent, state):
    level = agent.current_level()
    if level.key() != state.level_key or level.shops.records.get(state.shop_id) is not state.owner:
        return None
    return state.owner


def _ground(agent, point, *, allow_occupied=False):
    level = agent.current_level()
    y, x = point
    return (0 <= y < level.objects.shape[0] and 0 <= x < level.objects.shape[1]
            and level.seen[point] and int(level.objects[point]) in G.FLOOR
            and level.walkable[point] and not level.forbidden[point]
            and not (level.forbidden_until[point] > agent.blstats.time)
            and not (level.petrify_until[point] > agent.blstats.time)
            and int(agent.glyphs[point]) not in G.BOULDER
            and (allow_occupied or point == _here(agent) or
                 int(agent.glyphs[point]) not in G.MONS | G.PETS | G.INVISIBLE_MON))


def _outside(agent, point, *, allow_occupied=False):
    level = agent.current_level()
    return (_ground(agent, point, allow_occupied=allow_occupied)
            and not level.shop[point] and not level.shop_interior[point]
            and all(point not in shop.entrances for shop in level.shops.records.values()))


def _food_points(agent, shop):
    level = agent.current_level()
    visible = shop.interior & np.isin(agent.glyphs, tuple(G.FOOD_OBJECTS))
    points = set(zip(*visible.nonzero()))
    for point in zip(*(shop.interior & (level.item_count > 0)).nonzero()):
        if any(item.is_unambiguous() and item.object.name in agent.inventory.BUY_FOOD_NUTRITION
               for item in level.items[point] or []):
            points.add(point)
    return {tuple(map(int, point)) for point in points}


def _plan(agent):
    if not _demand(agent) or not _safe(agent) or agent.blstats.gold <= 0 or \
            agent.inventory.shop_debt_pending() or agent.inventory._credit_clone_reserved():
        return None
    tool = _tool(agent)
    if tool is None:
        return None
    level, dis = agent.current_level(), agent.bfs()
    attempts = getattr(agent, '_shop_preparation_attempts', set())
    for shop in level.shops.records.values():
        if (level.key(), shop.shop_id) in attempts or len(shop.entrances) != 1 or \
                agent.inventory._visible_shop_keeper(shop) is None:
            continue
        if not _food_points(agent, shop) and shop.shop_type not in (
                SHOP.name2id['delicatessen'], SHOP.name2id['health food store']):
            continue
        entrance = tuple(next(iter(shop.entrances)))
        if int(level.objects[entrance]) not in G.DOOR_OPENED | {SS.S_ndoor} and \
                entrance not in level.refused_doors:
            continue
        deposits = []
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            inside = entrance[0] + dy, entrance[1] + dx
            outside = entrance[0] - dy, entrance[1] - dx
            if not (0 <= inside[0] < dis.shape[0] and 0 <= inside[1] < dis.shape[1]):
                continue
            if shop.interior[inside] and _outside(agent, outside) and \
                    level.item_count[outside] == 0 and 0 <= dis[outside] <= 12:
                deposits.append(outside)
        if len(deposits) != 1:
            continue
        now = int(agent.blstats.time)
        route = agent.global_logic.wizard_route
        deadline = min(now + MAX_TURNS, int(route.started) + TOTAL_TURNS)
        if deadline - now <= CLEANUP_TURNS:
            continue
        return Visit(tuple(level.key()), shop.shop_id, shop, _signature(tool),
                     deposits[0], entrance, now, deadline)
    return None


def permits_purchase(agent, state):
    return (state is not None and state is getattr(agent, '_shop_preparation', None)
            and state.phase == 'buy' and _shop(agent, state) is not None
            and agent.inventory._visible_shop_keeper(state.owner) is not None
            and state.target == _here(agent) and state.owner.interior[_here(agent)]
            and int(agent.blstats.time) < state.deadline - CLEANUP_TURNS
            and _demand(agent) and _safe(agent))


def _finish(agent, state, reason, recovered=False):
    state.reason = reason
    state.phase = 'done' if recovered or not state.deposited else 'abandoned'
    if state.deposited and not recovered:
        unresolved = getattr(agent, '_shop_preparation_unrecovered', {})
        unresolved[(state.level_key, state.deposit)] = state.signature
        agent._shop_preparation_unrecovered = unresolved
    agent.log(f'SHOP_PREP {state.phase} reason={reason} level={state.level_key} '
              f'deposit={state.deposit} tool={state.signature} elapsed={agent.blstats.time-state.started}')


def quarantined_items(agent, items, position):
    """Never accidentally recover a missing outside tool from inside a shop."""
    unresolved = getattr(agent, '_shop_preparation_unrecovered', {})
    if not unresolved:
        return set()
    level = agent.current_level()
    if not level.shop[position] and not level.shop_interior[position] and not any(
            position in shop.entrances for shop in level.shops.records.values()):
        return set()
    signatures = {signature for (key, _), signature in unresolved.items() if key == level.key()}
    return {item for item in items or [] if _signature(item) in signatures}


def _recover(agent, state, reason):
    if state.phase != 'recover':
        state.phase, state.reason, state.target, state.blocked = 'recover', reason, None, 0
        agent.log(f'SHOP_PREP recover reason={reason} deposit={state.deposit}')


def _walk(agent, state, target):
    before = _here(agent)
    dis = agent.bfs()
    if dis[target] < 0 or (target != before and
            int(agent.glyphs[target]) in G.MONS | G.PETS | G.INVISIBLE_MON):
        # A keeper can block the far exit while the hero is still across the
        # room. Approach through currently reachable floor before waiting;
        # ordinary BFS/go_to still enforce occupants, doors and debt guards.
        distance = lambda point: abs(point[0] - target[0]) + abs(point[1] - target[1])
        candidates = [tuple(map(int, point)) for point in zip(*(state.owner.interior & (dis > 0)).nonzero())
                      if _ground(agent, tuple(point)) and distance(point) < distance(before)]
        if candidates:
            approach = min(candidates, key=lambda point: (distance(point), dis[point], point))
            agent.go_to(*approach, max_steps=1)
            state.blocked = 0 if _here(agent) != before else state.blocked + 1
            return state.blocked <= MAX_BLOCKED_WAITS
        state.blocked += 1
        if state.blocked > MAX_BLOCKED_WAITS:
            return False
        agent.search()
        return True
    agent.go_to(*target, max_steps=1)
    state.blocked = 0 if _here(agent) != before else state.blocked + 1
    return state.blocked <= MAX_BLOCKED_WAITS


def _run(agent, state):
    inv = agent.inventory
    while state.phase not in _DONE:
        state.decisions += 1
        if state.phase == 'drop_pending':
            # DROP may have completed before a command exception. Reconcile
            # before any owner, safety or deadline exit; never repeat DROP.
            inv.items.update()
            if _carried(agent, state) is not None:
                _finish(agent, state, 'interrupted drop retained tool', recovered=True)
                return
            _recover(agent, state, 'interrupted drop needs recovery')
        if state.phase == 'recover':
            inv.items.update()
            if _carried(agent, state) is not None:
                _finish(agent, state, 'tool observed in inventory', recovered=True)
                return
        shop = _shop(agent, state)
        if shop is None or state.decisions > MAX_DECISIONS or agent.blstats.time >= state.deadline:
            _finish(agent, state, 'owner/level changed or visit budget exhausted')
            return
        if not _safe(agent):
            # Combat/hero recovery can preempt. Never hold navigation forever
            # when safely retrieving the tool is no longer possible.
            _finish(agent, state, 'unsafe to continue')
            return
        if inv._credit_clone_reserved():
            _finish(agent, state, 'another inventory reservation')
            return
        if state.phase != 'recover' and (not _demand(agent) or
                agent.blstats.time >= state.deadline - CLEANUP_TURNS):
            _recover(agent, state, 'supply objective ended or cleanup reserve')
        if inv.shop_debt_pending():
            # This separate intent is never cleared by our timeout/finish.
            inv.pay_or_drop_unpaid()
            if inv.shop_debt_pending():
                agent.search()
            continue
        if state.phase == 'deposit':
            if _here(agent) != state.deposit:
                if not _walk(agent, state, state.deposit):
                    _finish(agent, state, 'outside deposit unreachable')
                continue
            with agent.atom_operation():
                inv.get_items_below_me()
                tool = _tool(agent)
                if not _safe(agent) or not _demand(agent) or not _outside(agent, state.deposit) or \
                        inv.items_below_me or tool is None or _signature(tool) != state.signature:
                    _finish(agent, state, 'deposit revalidation failed')
                    return
                state.phase = 'drop_pending'
                # This is an obligation, not a claim that DROP succeeded.
                # Only a fresh inventory observation can discharge it.
                state.deposited = True
                inv.drop([tool], smart=False)
                inv.items.update()
                inv.get_items_below_me()
                if _carried(agent, state) is not None:
                    _finish(agent, state, 'tool drop refused', recovered=True)
                    return
                matches = [item for item in inv.items_below_me if _owned_match(item, state)]
                if len(matches) != 1:
                    _finish(agent, state, 'deposited tool not uniquely observed')
                    return
                state.phase = 'buy'
                agent.log(f'SHOP_PREP deposited outside={state.deposit} tool={state.signature}')
            continue
        if state.phase == 'recover':
            if not state.deposited or not _outside(agent, state.deposit, allow_occupied=True):
                _finish(agent, state, 'outside provenance unavailable')
                return
            if _here(agent) != state.deposit:
                if not _walk(agent, state, state.deposit):
                    _finish(agent, state, 'outside recovery unreachable')
                continue
            with agent.atom_operation():
                inv.get_items_below_me()
                matches = [item for item in inv.items_below_me if _owned_match(item, state)]
                if not _outside(agent, state.deposit) or inv.shop_debt_pending() or len(matches) != 1:
                    _finish(agent, state, 'tool missing or ownership ambiguous')
                    return
                inv.pickup(matches[0], 1)
                inv.items.update()
                _finish(agent, state, 'outside tool recovered', recovered=_carried(agent, state) is not None)
            return
        if state.phase == 'buy':
            if inv._visible_shop_keeper(shop) is None:
                # Leaving the pick just outside can also leave line of sight
                # to the previously observed owner. Reobserve from this exact
                # entrance before inspecting stock or exposing any debt.
                here = _here(agent)
                if here == state.deposit and agent.bfs()[state.entrance] >= 0 and \
                        int(agent.glyphs[state.entrance]) not in G.MONS | G.PETS | G.INVISIBLE_MON:
                    agent.go_to(*state.entrance, max_steps=1)
                    if _here(agent) != here:
                        continue
                if here in (state.deposit, state.entrance):
                    state.blocked += 1
                    if state.blocked <= MAX_BLOCKED_WAITS:
                        agent.search()
                        continue
                _recover(agent, state, 'owner no longer observed peaceful')
                continue
            # Inspect public food glyphs even when no floor quote existed at
            # the blocked entrance. Quotes never authorize an unpriced pickup.
            if state.target is None:
                points = _food_points(agent, shop) - state.quotes
                dis = agent.bfs()
                state.surveying = False
                if not points and not state.quotes and len(state.surveyed) < MAX_SURVEY_POINTS and \
                        shop.shop_type in (SHOP.name2id['delicatessen'], SHOP.name2id['health food store']):
                    # A greeting identifies a food shop before the blocking
                    # keeper permits a view of its stock. Inspect only known
                    # interior floor through ordinary movement, without pickup.
                    points = {tuple(map(int, point)) for point in zip(*shop.interior.nonzero())
                              if tuple(point) not in state.surveyed and _ground(agent, tuple(point))}
                    state.surveying = True
                if not points or len(state.quotes) >= MAX_QUOTES:
                    _recover(agent, state, 'no more food quotes')
                    continue
                reachable = [point for point in points if dis[point] >= 0]
                if not reachable:
                    # A peaceful keeper can vacate its blocking home only
                    # after the tool-free hero approaches the actual doorway.
                    # BFS still decides whether that step is legal.
                    if _here(agent) != state.entrance and dis[state.entrance] >= 0:
                        if not _walk(agent, state, state.entrance):
                            _recover(agent, state, 'shop entrance blocked')
                        continue
                    state.blocked += 1
                    if state.blocked > MAX_BLOCKED_WAITS:
                        _recover(agent, state, 'keeper or terrain blocks food')
                    else:
                        agent.search()
                    continue
                state.target = min(reachable, key=lambda point: dis[point])
            if _here(agent) != state.target:
                if not _walk(agent, state, state.target):
                    _recover(agent, state, 'food approach blocked')
                continue
            inv.get_items_below_me()
            if _shop(agent, state) is not shop or _here(agent) != state.target:
                _recover(agent, state, 'food inspection moved or changed owner')
                continue
            if state.surveying:
                state.surveyed.add(state.target)
                state.target, state.surveying = None, False
                continue
            budget = agent.blstats.gold + inv.shop_credit(shop)
            affordable = [item for item in inv.items_below_me
                          if item.shop_status == Item.FOR_SALE and item.is_unambiguous()
                          and item.object.name in inv.BUY_FOOD_NUTRITION
                          and quantity_price(item) is not None and quantity_price(item) <= budget]
            if affordable and permits_purchase(agent, state):
                before = inv.carried_nutrition()
                inv.buy_food(preparation_visit=state).run()
                if inv.carried_nutrition() > before or inv.shop_debt_pending():
                    continue  # Remaining stack can supply the same objective.
            state.quotes.add(state.target)
            state.target = None


@Strategy.wrap
def visit(agent):
    state = getattr(agent, '_shop_preparation', None)
    plan = None
    if not reserves_inventory(agent):
        plan = _plan(agent)
        if plan is None:
            yield False
            return
    yield True
    if plan is not None:
        plan = _plan(agent)  # Live revalidation after strategy selection.
        if plan is None:
            return
        state = agent._shop_preparation = plan
        attempts = getattr(agent, '_shop_preparation_attempts', set())
        attempts.add((state.level_key, state.shop_id))
        agent._shop_preparation_attempts = attempts
        agent.log(f'SHOP_PREP start shop={state.shop_id} deposit={state.deposit} '
                  f'deadline={state.deadline} tool={state.signature}')
    _run(agent, state)
