"""Knight only: keep the saddled pony fed, so that it never turns on us (gene kni-feed-steed).

What goes wrong without it (replays of the champion, kni-hum-law-mal screen 202091 and 814013):
the pony is never fed.  At T~1500 'The saddled pony is confused from hunger.' and it starts kicking the
Knight, who keeps searching/swapping next to it (a tame glyph is never fought): 25 -> 0 HP in 15 turns at
XL2.  Where the pony foraged (lichen corpses) it grew into a warhorse, then went hungry at T7779 and killed
a 32/32 HP XL4 Knight in 21 turns -- while the Knight had eaten the kit's carrots itself.  5 of ~48 cached
Knight games end 'killed by a pony/warhorse' on Dlvl 1; in the others the pony starves at T~1750 and the
Knight grinds to XL8 alone.

NetHack 3.6.6 source:
- u_init.c Knight[] (l.73-83): the kit has 10 apples and 10 carrots.  role.c (l.211): the Knight's pet is
  a pony, dog.c makedog (l.191-193): saddled.
- dog.c initedog (l.47): hungrytime = 1000 + monstermoves.
- dogmove.c dog_hunger (l.356-397): past hungrytime + 500 a herbivore gets mconf = 1 and a third of its
  max HP ('is confused from hunger'); past + 750 it starves.
- mon.c mfndpos (l.1368-1370): a confused monster may move onto every square, ALLOW_U included, and
  dogmove.c dog_move (l.1189-1195) then calls mattacku: the pet attacks us (pony d6 kick + d2 bite; warhorse
  d10 + d4 at speed 24).  monmove.c (l.415): the confusion wears off with 1/50 per move.
- dogmove.c dog_eat (l.206-241): hungrytime = max(hungrytime, moves) + nutrition, mconf = 0, HP penalty
  removed.  dog_nutrition (l.142): oc_nutrition x5 for a pony (MZ_MEDIUM), x4 for a horse/warhorse
  (MZ_LARGE); corpses give cnutrit.  So a meal eaten while the pet is already hungry forgives the deficit:
  feeding late (but before + 500) makes every apple last longer.
- dog.c dogfood (l.820-823): apples and carrots are DOGFOOD for herbivores; a starving pet
  (mhpmax_penalty) takes any other plain food as ACCFOOD (l.830-832), never meat, eggs or tins.
- dothrow.c thitmonst (l.1769-1776) -> dog.c tamedog (l.895-917): a tame pet that is not confused and not
  eating catches thrown DOGFOOD and eats it at once.  A confused one doesn't: the food misses and lands on
  its square (dothrow.c l.1418), and dogmove.c dog_invent (l.403-443) / dog_move's food search eat it on
  its next move.

Killing or hitting the pet instead is ruinous for a prayer-fed grind: mon.c xkilled (l.2473-2476, 2506)
gives Luck -5 and alignment -15, and uhitm.c (l.1200-1202) abuse_dog lowers tameness.

So: throw an apple or carrot at the pony once it is estimated hungry (seen meals are counted), throw any
plain food at it when it is 'confused from hunger', and with nothing left to throw step out of its reach
while it is confused.  Every other role returns at the first check; any exception means 'do nothing'.
"""
import re

import nle.nethack as nh

from . import objects as O
from . import utils
from .character import Character
from .glyph import G, MON
from .strategy import Strategy

HORSES = {'pony': 5, 'horse': 4, 'warhorse': 4}     # dog_nutrition size multiplier (MZ_MEDIUM / MZ_LARGE)
VEGGIES = ('carrot', 'apple')                       # DOGFOOD for a herbivore: always caught
# small plain food a starving herbivore eats (dogfood default branch / banana / garlic; never meat, eggs,
# tins). Rations stay ours: they are the Knight's insurance after a failed prayer (the step-away covers it).
# Never a cream pie: dothrow.c thitmonst (l.1752-1756) hmon()s the pet with it before the food branch.
STARVING_FOOD = frozenset((
    'kelp frond', 'eucalyptus leaf', 'orange', 'pear', 'melon', 'banana', 'clove of garlic', 'slime mold',
    'lump of royal jelly', 'candy bar', 'fortune cookie', 'pancake'))

FEED_AFTER = 200        # feed once the pet is estimated this many turns past its hungrytime (< 500)
URGENT_AFTER = 470      # past this, treat it as starving even without the message
STARVING_FOR = 300      # the 'confused from hunger' state lasts at most 250 turns (starves at + 750)
THROW_COOLDOWN = 8
MAX_STEP_AWAYS = 60     # per starvation episode (the confusion wears off at 1/50 per move)
MAX_THROW_DIST = 3
HOSTILE_RADIUS = 5

_HORSE = r'(?:[Tt]he|[Yy]our) (?:saddled )?(pony|horse|warhorse)'
_EAT_RE = re.compile(_HORSE + r' (eats|devours) (.+?)\.')
_CONF_RE = re.compile(_HORSE + r' is confused from hunger')
_WORRY_RE = re.compile(r'You feel worried about (?:your )?(?:saddled )?(pony|horse|warhorse)')
_STARVE_RE = re.compile(_HORSE + r' starves')
_ARTICLE_RE = re.compile(r'^(?:an?|the|\d+) ')
_BUC_RE = re.compile(r'^(?:uncursed|blessed|cursed) ')


def _food_nutrition(text):
    """oc_nutrition / cnutrit of the thing named in a pet's 'eats ...' message (0 when unknown)."""
    t = _ARTICLE_RE.sub('', text.strip())
    t = _BUC_RE.sub('', t)
    frac = 1.0
    if t.startswith('partly eaten '):
        t = t[len('partly eaten '):]
        frac = 0.5
    if t.endswith(' corpse'):
        name = t[:-len(' corpse')]
        try:
            return frac * getattr(MON.permonst(MON.from_name(name)), 'cnutrit', 0)
        except Exception:
            return 0
    try:
        return frac * O.from_name(t, nh.FOOD_CLASS).nutrition
    except Exception:
        return 0


class SteedKeeper:
    def __init__(self, agent):
        self.agent = agent
        self.hungry_at = 1001          # dog.c initedog: 1000 + monstermoves (moves = 1 at the start)
        self.starving_since = None     # turn of 'confused from hunger' (or its estimate)
        self.last_throw = -10 ** 9
        self.urgent_throws = 0
        self.step_aways = 0
        self._last_step = None
        self._pets = []                # [(y, x, name)] horse pets on screen this step

    def active(self):
        ch = self.agent.character
        # cmd.c getdir (l.5095): a stunned (or confused) hero throws / steps in a random direction
        return ch.role == Character.KNIGHT and not ch.prop.hallu and not ch.prop.blind and \
            not ch.prop.confusion and not ch.prop.stun

    # ---- per-step observation (GlobalLogic.update); never raises

    def update(self):
        try:
            self._update()
        except Exception:
            self._pets = []

    def _update(self):
        agent = self.agent
        if agent.character.role != Character.KNIGHT:
            return
        if self._last_step == agent.step_count:
            return
        self._last_step = agent.step_count
        turn = agent.blstats.time
        msg = agent.message or ''
        if msg:
            if _CONF_RE.search(msg) or _WORRY_RE.search(msg):
                self.starving_since = turn
                self.urgent_throws = 0
                self.step_aways = 0
                self.hungry_at = min(self.hungry_at, turn - 500)
            if _STARVE_RE.search(msg):
                self.starving_since = None
            for m in _EAT_RE.finditer(msg):
                nut = _food_nutrition(m.group(3)) * HORSES[m.group(1)]
                if m.group(2) == 'devours':
                    nut = nut * 3 // 4
                self.hungry_at = max(self.hungry_at, turn) + int(nut)
                self.starving_since = None
                self.urgent_throws = 0
        if self.starving_since is not None and turn - self.starving_since > STARVING_FOR:
            self.starving_since = None
        pets = []
        if not agent.character.prop.hallu:
            for y, x in zip(*utils.isin(agent.glyphs, G.PETS).nonzero()):
                name = MON.permonst(agent.glyphs[y, x]).mname
                if name in HORSES:
                    pets.append((int(y), int(x), name))
        self._pets = pets

    # ---- decision

    def _starving(self, turn):
        return self.starving_since is not None or turn >= self.hungry_at + URGENT_AFTER

    def _hostile_near(self, radius):
        agent = self.agent
        mt = agent.monster_tracker
        mask = mt.monster_mask & ~mt.peaceful_monster_mask
        if not mask.any():
            return False
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        for y, x in zip(*mask.nonzero()):
            if max(abs(int(y) - y0), abs(int(x) - x0)) <= radius:
                return True
        return False

    def _throw_line(self, py, px):
        """(dy, dx) unit direction if a thrown object reaches the pet at (py, px) first, else None."""
        agent = self.agent
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        dy, dx = int(py) - y0, int(px) - x0
        dist = max(abs(dy), abs(dx))
        if dist == 0 or dist > MAX_THROW_DIST:
            return None
        if not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            return None
        sy, sx = int(dy > 0) - int(dy < 0), int(dx > 0) - int(dx < 0)
        level = agent.current_level()
        for k in range(1, dist):
            y, x = y0 + k * sy, x0 + k * sx
            g = agent.glyphs[y, x]
            # bhit (zap.c l.3320-3430) stops at the first monster, wall or closed door on the way
            if not level.walkable[y, x] or g in G.MONS or g in G.PETS or g in G.INVISIBLE_MON or \
                    g in G.BOULDER or level.objects[y, x] in G.DOOR_CLOSED:
                return None
        return sy, sx

    def _food(self, starving):
        """A carrot/apple (always caught); with the pet confirmed starving, any plain food (it lands under
        the confused pet, which eats it as ACCFOOD)."""
        # never unpaid shop food: a pet eating it bills us (dogmove.c dog_eat 'That ... will cost you')
        items = [i for i in self.agent.inventory.items
                 if i.category == nh.FOOD_CLASS and not i.is_corpse() and len(i.objs) == 1 and
                 getattr(i, 'shop_status', 0) != 2]
        for name in VEGGIES:
            for item in items:
                if item.objs[0].name == name:
                    return item
        if starving:
            other = [i for i in items if i.objs[0].name in STARVING_FOOD]
            if other:
                return min(other, key=lambda i: i.objs[0].nutrition)
        return None

    def _plan(self):
        """('throw', item, (dy, dx)) / ('step', (y, x)) / None."""
        agent = self.agent
        if not self._pets or not self.active() or agent.character.prop.polymorph:
            return None
        turn = agent.blstats.time
        confirmed = self.starving_since is not None
        urgent = self._starving(turn)
        if not urgent and turn < self.hungry_at + FEED_AFTER:
            return None
        food = self._food(confirmed) if turn - self.last_throw >= THROW_COOLDOWN else None
        if food is not None and (not confirmed or self.urgent_throws < 4):
            if not urgent and self._hostile_near(HOSTILE_RADIUS):
                return None
            y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
            for py, px, _ in sorted(self._pets, key=lambda p: max(abs(p[0] - y0), abs(p[1] - x0))):
                d = self._throw_line(py, px)
                if d is not None:
                    return 'throw', food, d
            return None
        if not confirmed or self.step_aways >= MAX_STEP_AWAYS:
            return None
        # nothing to give while it is confused: keep out of its reach (it attacks only from an adjacent square)
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        if not any(utils.adjacent((y0, x0), (py, px)) for py, px, _ in self._pets):
            return None
        if self._hostile_near(1):
            return None
        dis = agent.bfs()
        best = None
        for y, x in agent.neighbors(y0, x0, shuffle=False):
            if dis[y, x] != 1:
                continue
            g = agent.glyphs[y, x]
            if g in G.MONS or g in G.PETS or g in G.INVISIBLE_MON or g in G.BOULDER:
                continue
            gap = min(max(abs(y - py), abs(x - px)) for py, px, _ in self._pets)
            if gap < 2:
                continue
            if best is None or gap > best[0]:
                best = (gap, y, x)
        if best is None:
            return None
        return 'step', (best[1], best[2])

    @Strategy.wrap
    def strategy(self):
        try:
            plan = self._plan()
        except Exception:
            plan = None
        if plan is None:
            yield False
            return
        yield True
        agent = self.agent
        if plan[0] == 'throw':
            _, item, (dy, dx) = plan
            self.last_throw = agent.blstats.time
            if self.starving_since is not None:
                self.urgent_throws += 1
            direction = agent.calc_direction(agent.blstats.y, agent.blstats.x,
                                             agent.blstats.y + dy, agent.blstats.x + dx)
            agent.log(f'KNI steed: throwing {item.objs[0].name} {direction} '
                      f'(hungry_at {self.hungry_at}, starving {self.starving_since})')
            agent.fire(item, direction)
        else:
            y, x = plan[1]
            self.step_aways += 1
            agent.log('KNI steed: stepping away from the starving pet')
            agent.move(y, x)
