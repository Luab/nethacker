"""Public-observation hunger care for a single visible Wizard cat or dog.

NLE 1.3.0 dog.c/initedog starts hungrytime at world time + 1000;
dogmove.c/dog_eat credits nutrition at meal start, before meating finishes.
Hunger confusion starts after hungrytime + 500, death after + 750. We keep
a lower bound from public history, never a hidden hunger value or monster ID.
"""
import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import objects as O, utils
from .glyph import G, Hunger, MON, SS
from .item import Item
from .strategy import Strategy


PETS = frozenset(('kitten', 'housecat', 'large cat', 'little dog', 'dog', 'large dog'))
FAVORITES = frozenset(('tripe ration', 'meatball', 'meat stick', 'meat ring'))
FETCH_MARGIN = 50
CARE_MARGIN = 210
TRUST_TURNS = 1000
THROW_COOLDOWN = 8
MAX_UNCONFIRMED_THROWS = 2
VISIBILITY_GRACE = 8
_PET = r'(?:[Tt]he|[Yy]our) (' + '|'.join(sorted(PETS, key=len, reverse=True)) + r')'
_EVENT = re.compile(_PET + r' (?:(eats|devours) (.+?)\.|(is confused from hunger|starves)\.)')
_WORRY = re.compile(r'You feel worried about (?:your )?(' + '|'.join(PETS) + r')\.')
_PREFIX = re.compile(r'^(?:(?:an?|the|\d+|uncursed|blessed|cursed) )+')
_FLOOR = frozenset((SS.S_room, SS.S_darkroom, SS.S_corr, SS.S_litcorr))


def _active(agent):
    return agent.character.role == agent.character.WIZARD


def _state(agent):
    state = getattr(agent, '_pet_care', None)
    if state is None:
        state = dict(pet=None, level=None, last_seen=-1, hungry_at=None, trusted_until=-1,
                     eating_until=-1, last_step=None, last_message=None,
                     last_meal=None, last_throw=-10000, unconfirmed_throws=0,
                     pending=None, hunger_warning=None,
                     outcome='unknown_pet', reason='unknown_pet')
        agent._pet_care = state
    return state


def _pet_positions(agent):
    if agent.character.prop.blind or agent.character.prop.hallu:
        return []
    positions = list(zip(*utils.isin(agent.glyphs, G.PETS).nonzero()))
    return positions


def _visible_pet(agent):
    positions = _pet_positions(agent)
    if len(positions) != 1:
        return None
    y, x = map(int, positions[0])
    name = MON.permonst(int(agent.glyphs[y, x])).mname
    return (y, x, name) if name in PETS else None


def _family(name):
    return 'cat' if name in ('kitten', 'housecat', 'large cat') else 'dog'


def _meal(text, species, devoured=False):
    """Known full meal nutrition and an upper bound on eating duration.

    Partly eaten portions have no public remaining quantity; do not invent it.
    Corpse age controls whether eating is safe, not its full nutrition once
    an actual meal has been observed. No corpse is selected for throwing.
    """
    text = _PREFIX.sub('', text.strip())
    if 'partly eaten' in text or 'partially eaten' in text:
        return None
    try:
        if text.endswith(' corpse'):
            mon = MON.permonst(MON.from_name(text[:-7]))
            nutrition, delay = int(mon.cnutrit), 3 + (int(mon.cwt) >> 6)
        else:
            food = O.from_name(text, nh.FOOD_CLASS)
            nutrition, delay = int(food.nutrition), int(food.delay)
        size = int(MON.permonst(MON.from_name(species)).msize)
        nutrition *= {0: 8, 1: 6, 2: 5, 3: 4, 4: 3, 7: 2}.get(size, 5)
        if devoured:
            nutrition = nutrition * 3 // 4
            delay = max(1, delay // 2)
        return nutrition, delay + 2
    except (AssertionError, AttributeError, KeyError, ValueError):
        return None


def observe(agent):
    """No actions. Call with the original message and current glyphs/blstats."""
    if not _active(agent):
        return
    state = _state(agent)
    # update_state may run repeatedly for the same completed observation.
    stamp = (int(agent.step_count), int(agent.blstats.time), agent.message or '')
    if state['last_step'] == stamp:
        return
    previous_turn = state['last_step'][1] if state['last_step'] is not None else None
    state['last_step'] = stamp
    now = int(agent.blstats.time)
    pet = _visible_pet(agent)
    level = tuple(agent.current_level().key())
    previous = state['pet']
    positions = _pet_positions(agent)
    impaired = agent.character.prop.blind or agent.character.prop.hallu
    recent = previous is not None and now - state['last_seen'] <= VISIBILITY_GRACE
    same_level = state['level'] == level
    # A closed sightline for a few turns does not change the nutrition clock.
    # Ambiguity, long separation, or changing levels does: public glyphs cannot
    # establish a stable monster identity. No new clone can start unseen.
    continuous = (not impaired and same_level and recent and
                  ((pet is None and len(positions) == 0) or
                   (pet is not None and _family(previous[2]) == _family(pet[2]) and
                    max(abs(pet[0] - previous[0]), abs(pet[1] - previous[1])) <=
                    2 * max(1, now - state['last_seen']) + 1)))
    if not continuous:
        state['hungry_at'] = None
        state['trusted_until'] = -1
        state['hunger_warning'] = None
        state['reason'] = 'pet_identity_uncertain'
    if pet is not None:
        state['pet'], state['last_seen'] = pet, now
    elif not continuous:
        state['pet'] = None
    state['level'] = level
    # Only the observed starting kitten/puppy receives the initialization bound.
    if previous is None and pet is not None and now <= 2 and pet[2] in ('kitten', 'little dog'):
        state.update(hungry_at=1000, trusted_until=1000, outcome='starting_pet')
    message = agent.message or ''
    if message == state['last_message']:
        return
    state['last_message'] = message
    worry = _WORRY.search(message)
    if worry and state['pet'] is not None and worry.group(1) == state['pet'][2]:
        state.update(hungry_at=None, trusted_until=-1, hunger_warning=now,
                     outcome='observed_hunger_warning')
    if pet is None:
        return
    for match in _EVENT.finditer(message):
        species, verb, food, warning = match.groups()
        if species != pet[2]:
            continue
        if warning:
            state.update(hungry_at=None, trusted_until=-1,
                         hunger_warning=now if warning == 'is confused from hunger' else None,
                         outcome='observed_' + warning.replace(' ', '_'))
            continue
        meal = _meal(food, species, verb == 'devours')
        # Unknown portions still prove an eating occupation; reserve a bounded
        # conservative interval but never authorize fetching from unknown fuel.
        state['eating_until'] = now + (meal[1] if meal else 30)
        state['pending'] = None
        state['unconfirmed_throws'] = 0
        state['hunger_warning'] = None
        known = meal is not None and previous_turn is not None
        state['last_meal'] = dict(turn=now, earliest=previous_turn, food=food,
                                  species=species, known=known)
        if known and meal[0] > 0:
            # Do not accumulate several message copies or multiple ambiguous
            # meals. A multi-turn hero occupation can delay this observation;
            # the preceding observed world time bounds when the meal began.
            state['hungry_at'] = max(state['hungry_at'] or 0, previous_turn + meal[0])
            state['trusted_until'] = previous_turn + TRUST_TURNS
            state['outcome'] = 'observed_meal'
            agent.log(f'PET_CARE meal={food!r} reserve_until={state["hungry_at"]}')
        else:
            state['outcome'] = 'observed_unknown_meal'


def ready_for_fetch(agent, budget=160, *, allow_eating=False):
    """Known reserve for the whole job; at most eight turns of lost sight.

    The clone planner separately requires a currently visible pet. This small
    grace period is only useful while that already-started job crosses a door.
    allow_eating only permits the caller to wait inside its original deadline;
    it never treats an unconfirmed throw as a known meal.
    """
    if not _active(agent) or budget < 0:
        return False
    state = getattr(agent, '_pet_care', None)
    pet = _visible_pet(agent)
    if (state is None or state['pet'] is None or
            tuple(agent.current_level().key()) != state['level'] or
            agent.character.prop.blind or agent.character.prop.hallu):
        return False
    now = int(agent.blstats.time)
    if pet is None:
        if _pet_positions(agent) or now - state['last_seen'] > VISIBILITY_GRACE:
            return False
    elif pet != state['pet']:
        return False
    if state['pending'] is not None or (not allow_eating and now <= state['eating_until']):
        state['reason'] = 'meal_unconfirmed_or_eating'
        return False
    if (state['hungry_at'] is None or now + budget + FETCH_MARGIN >= state['hungry_at'] or
            now + budget >= state['trusted_until']):
        state['reason'] = 'insufficient_known_food_reserve'
        return False
    state['reason'] = 'known_food_reserve' if pet else 'recent_pet_temporarily_unseen'
    return True


def available_food(agent):
    """Best known owned favorite in accessible inventory; no action or position test."""
    foods = [item for item in agent.inventory.items if item.category == nh.FOOD_CLASS and
             item.is_unambiguous() and item.object.name in FAVORITES and
             item.shop_status == Item.NOT_SHOP and item.status in (Item.UNCURSED, Item.BLESSED) and
             item.count > 0 and not item.equipped and
             not any(part in (item.text or '') for part in ('partly eaten', 'partially eaten', 'greased'))]
    return max(foods, key=lambda item: item.object.nutrition, default=None)


def _feeding_demand(agent, state, *, for_fetch):
    now = int(agent.blstats.time)
    if for_fetch:
        # The controller reserves a real shop opportunity before requesting a
        # meal. Its original clock includes the whole preparation, not just fetch.
        from .pet_credit import MAX_TURNS
        job = getattr(agent, '_credit_clone', None)
        return (job is not None and job.phase == 'prepare_pet' and
                tuple(job.level_key) == tuple(agent.current_level().key()) and
                0 <= now - int(job.started) < MAX_TURNS)
    warning = state['hunger_warning']
    if warning is not None and 0 <= now - warning <= TRUST_TURNS:
        return True
    # Lack of a clock, or expiry of its evidence, is uncertainty rather than
    # hunger. Do not spend food merely to restore an exploration-time estimate.
    return (state['hungry_at'] is not None and now <= state['trusted_until'] and
            now + CARE_MARGIN >= state['hungry_at'])


def _plan(agent, *, for_fetch=False):
    if not _active(agent):
        return None
    state = getattr(agent, '_pet_care', None)
    pet = _visible_pet(agent)
    prop, bl = agent.character.prop, agent.blstats
    if (state is None or pet is None or pet != state['pet'] or
            state['level'] != tuple(agent.current_level().key()) or
            int(bl.time) - state['last_seen'] > VISIBILITY_GRACE or
            any((prop.blind, prop.hallu, prop.confusion, prop.stun, prop.polymorph)) or
            int(bl.prop_mask) & nh.BL_MASK_LEV or
            bl.hunger_state > Hunger.NOT_HUNGRY or bl.carrying_capacity >= 2 or
            agent.get_visible_monsters() or ready_for_fetch(agent) or
            int(bl.time) <= state['eating_until'] or
            int(bl.time) < state['last_throw'] + THROW_COOLDOWN or
            state['unconfirmed_throws'] >= MAX_UNCONFIRMED_THROWS):
        return None
    if not _feeding_demand(agent, state, for_fetch=for_fetch):
        return None
    inv, level = agent.inventory, agent.current_level()
    if inv.shop_debt_pending():
        return None
    y, x = int(bl.y), int(bl.x)
    py, px, _ = pet
    if max(abs(py - y), abs(px - x)) != 1:
        return None
    # Only ordinary dry floor, outside shop protection and known traps. A
    # refused catch must not strand food in water or transfer it to a shop.
    for pos in ((y, x), (py, px)):
        if level.shop[pos] or not level.walkable[pos] or int(level.objects[pos]) not in _FLOOR:
            return None
    food = available_food(agent)
    if food is None:
        return None
    # A tripe ration sustains the job; tiny meat treats only prove their small
    # actual reserve. Keep the bounded module from repeatedly feeding crumbs.
    return food, agent.calc_direction(y, x, py, px)


@Strategy.wrap
def feed_pet(agent, *, for_fetch=False):
    if _plan(agent, for_fetch=for_fetch) is None:
        yield False
        return
    yield True
    plan = _plan(agent, for_fetch=for_fetch)
    if plan is None:
        return
    item, direction = plan
    state = _state(agent)
    state['last_throw'] = int(agent.blstats.time)
    state['unconfirmed_throws'] += 1
    state['pending'] = dict(turn=state['last_throw'], food=item.object.name, pet=state['pet'])
    state['outcome'] = 'attempting_throw'
    letter = agent.inventory.items.get_letter(item)

    def answers():
        if 'What do you want to throw?' not in agent.single_message:
            yield A.Command.ESC
            return
        yield letter
        if 'In what direction?' in agent.single_message:
            yield getattr(A.CompassDirection, direction.upper())
        else:
            yield A.Command.ESC

    agent.log(f'PET_CARE throw={item.object.name!r} pet={state["pet"]} '
              f'reason={"fetch_preparation" if for_fetch else "observed_hunger"}')
    try:
        with agent.atom_operation():
            agent.step(A.Command.THROW, answers())
    finally:
        if state['outcome'] == 'attempting_throw':
            state['outcome'] = 'throw_without_observed_meal'
        # Preserve the cooldown even when the callback preempts after THROW.
        # A later observed meal may still confirm a dropped, initially refused
        # treat; no inventory disappearance is interpreted as nutrition.
        state['pending'] = None


def snapshot(agent):
    state = getattr(agent, '_pet_care', None)
    if state is None:
        return None
    return {key: state[key] for key in ('pet', 'level', 'last_seen', 'hungry_at', 'trusted_until',
            'eating_until', 'last_meal', 'last_throw', 'unconfirmed_throws', 'pending',
            'hunger_warning', 'outcome', 'reason')}
