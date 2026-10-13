"""Prayer model: NetHack 3.6.6 pray.c semantics -- will a prayer made now be answered, and what will it fix?

Shared by the prayer rules in agent.py (emergency_strategy, is_safe_to_pray) and by other islands
(hunger prayers read rnz_cdf). Everything is derived from vendor/nethack-3.6.6/src:

* Timeout u.ublesscnt. u_init.c:644 sets 300 at turn 1; allmain.c:184-185 takes 1 off every turn (no
  Luck term in 3.6.6). A pleased() prayer ends with ublesscnt = rnz(350) (pray.c:1220); a failed one
  goes through angrygods(), which ends with ublesscnt = rnz(300) (pray.c:738). rnz is rnd.c:219-240
  with rne(4) capped at 5 below XL 15 (rnd.c:196-206). A wish adds rn1(100, 50) (zap.c:5350); a
  sacrifice's 'feeling of reconciliation' sets it to 0 (pray.c:1735-1740).
* can_pray() (pray.c:1794-1839): p_type 0 'too soon' if ublesscnt > 200 in major trouble, > 100 in
  minor trouble, > 0 in none; else p_type 1 'naughty' (always fails) if Luck < 0, u.ugangr or the
  alignment record < 0; else answered (p_type 3, and u.uinvulnerable during the 3 prayer turns).
* A 'too soon' prayer: Luck -3 and gods_upset() -> u.ugangr++ (pray.c:1911-1916, 1284-1292). ugangr
  never decays; only a sacrifice lowers it ('seems mollified', pray.c:1687-1706). While it is > 0
  every prayer is p_type 1. A p_type 1 prayer made for Luck < 0 only (ugangr 0) leaves ugangr at 0,
  so prayers work again once Luck is back to its base.
* Troubles (pray.c:in_trouble 183-265): stoned > slimed > strangled > lava > sick > starving (Weak
  or worse) > region > critically low HP (pray.c:102-140) > lycanthropy ... ; minor: Hungry, blind,
  stunned, confused, hallucinating, ...
* pleased() (pray.c:975-1005): off an altar action = rn1(Luck + 2, 1) capped at 3; a record below
  STRIDENT (4) forces action 1 (record > 0) or 0/1 (record 0). Action 1 fixes only the WORST major
  trouble, action >= 2 every major one.
* Luck: base +1 on a full moon, -1 on Friday 13th (allmain.c:48-57, messages at game start); it
  moves one step toward the base on turns divisible by 600, or 300 while ugangr > 0
  (timeout.c:486-505). Visible hits: own pet killed -1 (mon.c:2473, 'You hear the rumble of distant
  thunder' mon.c:2507-2510), 'You murderer!' -2 (mon.c:2463-2469), co-aligned unicorn -5 ('You feel
  guilty...', mon.c:2475-2478), cannibalism -2..-5 (eat.c:670), a mirror broken in melee -2
  (uhitm.c:883, "That's bad luck!").
* Alignment record: starts at the role's initrecord (role.c: 10 for Arc Bar Hea Kni Mon Rog Ran Sam,
  0 for Cav Pri Tou Val Wiz); a kill adds the victim's malign (mon.c:2516, makemon.c:set_malign
  2055-2101), -15 for a pet (mon.c:2506), -5 for a peaceful; hypocrite -5 (mon.c:2899-2908); caitiff
  -1 (uhitm.c:246-247); gains are capped at ALIGNLIM = 10 + moves/200 (align.h, attrib.c:1121-1134).
* Threat (mhitu.c:mattacku 572-581, hitmu 1699-1702): attack i hits if
  10 + AC_VALUE(u.uac) + m_lev > rnd(20 + i); AT_HUGS hits when the two previous attacks hit or the
  hero is already held; negative AC takes rnd(-AC) off each hit (min 1). m_lev per makemon.c:adj_lev.

The model never acts: the agent asks it, and every call site falls back to the old rules on an error.
"""
import bisect
import functools
import math
import re

from . import mondata

# ---------------------------------------------------------------- troubles (pray.c:25-50 ranks)
TROUBLE_STONED = 14
TROUBLE_SLIMED = 13
TROUBLE_STRANGLED = 12
TROUBLE_SICK = 10
TROUBLE_STARVING = 9
TROUBLE_HIT = 7
TROUBLE_LYCANTHROPE = 6
LETHAL_STATUS_TROUBLES = (TROUBLE_STONED, TROUBLE_SLIMED, TROUBLE_STRANGLED, TROUBLE_SICK)

# botl.h condition bits (NLE blstats[NLE_BL_CONDITION])
BL_MASK_STONE = 0x1
BL_MASK_SLIME = 0x2
BL_MASK_STRNGL = 0x4
BL_MASK_FOODPOIS = 0x8
BL_MASK_TERMILL = 0x10
BL_MASK_BLIND = 0x20
BL_MASK_STUN = 0x80
BL_MASK_CONF = 0x100
BL_MASK_HALLU = 0x200

HUNGER_HUNGRY, HUNGER_WEAK = 2, 3  # glyph.Hunger values of NLE's hunger_state

STRIDENT = 4  # pray.c:58
DEVOUT = 14

# role.c initrecord, indexed by Character role constants (Arc Bar Cav Hea Kni Mon Pri Ran Rog Sam Tou Val Wiz)
_INIT_RECORD = {0: 10, 1: 10, 2: 0, 3: 10, 4: 10, 5: 10, 6: 0, 7: 10, 8: 10, 9: 10, 10: 0, 11: 0, 12: 0}


# ---------------------------------------------------------------- rnz distribution (rnd.c:219-240)
@functools.lru_cache(maxsize=8)
def _rnz_table(i, cap):
    """Exact distribution of rnz(i) with rne(4) capped at `cap` -> (sorted values, cdf)."""
    pmf = {}
    for k in range(1, cap + 1):
        pk = (0.25 ** (k - 1)) * (0.75 if k < cap else 1.0)
        w = pk * 0.5 / 1000.0
        for r in range(1000):
            tmp = (1000 + r) * k
            a = i * tmp // 1000
            b = i * 1000 // tmp
            pmf[a] = pmf.get(a, 0.0) + w
            pmf[b] = pmf.get(b, 0.0) + w
    vals = sorted(pmf)
    cdf, s = [], 0.0
    for v in vals:
        s += pmf[v]
        cdf.append(s)
    return vals, cdf


def rnz_cdf(i, x, xl=1):
    """P(rnz(i) <= x) for a hero of experience level xl (rne's cap is max(XL/3, 5))."""
    cap = 5 if xl < 15 else max(5, xl // 3)
    vals, cdf = _rnz_table(i, cap)
    idx = bisect.bisect_right(vals, x)
    return min(1.0, cdf[idx - 1]) if idx else 0.0


_rnz_table(350, 5)  # ~10k iterations at import, then cached
_rnz_table(300, 5)


def critically_low_hp(hp, maxhp, xl):
    """pray.c:critically_low_hp(FALSE): low HP counts as major trouble."""
    maxhp = min(maxhp, 15 * xl)
    rank = 0 if xl <= 2 else ((xl + 2) // 4 if xl <= 30 else 8)  # botl.c:xlev_to_rank
    divisor = 5 if rank <= 1 else 6 if rank <= 3 else 7 if rank <= 5 else 8 if rank <= 7 else 9
    return hp <= 5 or hp * divisor <= maxhp


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- threat (mhitu.c)
_AT_MELEE = frozenset((1, 2, 3, 4, 5, 6, 16, 254))  # CLAW BITE KICK BUTT TUCH STNG TENT WEAP
AT_HUGS, AT_EXPL, AT_WEAP, AT_MAGC = 7, 13, 254, 255
AD_CLRC, AD_SPEL = 240, 241


def adj_lev(mlevel, depth, xl):
    """makemon.c:adj_lev with level_difficulty ~ depth."""
    if mlevel > 49:
        return 50
    tmp = mlevel
    diff = depth - mlevel
    if diff < 0:
        tmp -= 1
    else:
        tmp += diff // 5
    if xl - mlevel > 0:
        tmp += (xl - mlevel) // 4
    upper = min(49, (3 * mlevel) // 2)
    return max(0, min(tmp, upper))


def _p_hit(ac, mlev, i, helpless=False):
    """P(10 + AC_VALUE(ac) + mlev (+4 while the hero is helpless: fainted, praying) > rnd(20 + i));
    AC_VALUE(ac) = -rnd(-ac) for negative AC (mhitu.c:571-575)."""
    n = 20 + i
    base = 10 + mlev + (4 if helpless else 0)
    tmps = (base + ac,) if ac >= 0 else tuple(base - r for r in range(1, -ac + 1))
    tot = 0.0
    for tmp in tmps:
        tot += min(max(tmp - 1, 0), n) / n
    return tot / len(tmps)


@functools.lru_cache(maxsize=2048)
def _dmg_moments(n, d, ac, weapon=False):
    """Mean and second moment of one hit: d(n, d) (+ a d6 weapon), minus rnd(-AC) when AC < 0, min 1."""
    dist = {0: 1.0}
    dice = [d] * max(n, 0) if d > 0 else []
    if weapon:
        dice.append(6)
    for sides in dice:
        new = {}
        for v, p in dist.items():
            for k in range(1, sides + 1):
                new[v + k] = new.get(v + k, 0.0) + p / sides
        dist = new
    if ac < 0:
        red = -ac
        new = {}
        for v, p in dist.items():
            if v <= 0:
                new[v] = new.get(v, 0.0) + p
                continue
            for r in range(1, red + 1):
                x = max(1, v - r)
                new[x] = new.get(x, 0.0) + p / red
        dist = new
    m1 = sum(v * p for v, p in dist.items())
    m2 = sum(v * v * p for v, p in dist.items())
    return m1, m2


@functools.lru_cache(maxsize=4096)
def monster_turn_damage(name, ac, depth, xl, stuck=False, helpless=False):
    """Mean and variance of one monster move's melee damage against the hero, and its moves per turn
    (mmove / 12, mon.c:mcalcmove)."""
    entry = mondata.MONS.get(name)
    if entry is None:
        # unknown / unseen attacker: a level-(depth/2) monster with one d8 hit
        mlev, speed, attacks = max(1, depth // 2), 12, ((1, 0, 1, 8),)
    else:
        lvl, speed, _mac, _aln, _f2, attacks = entry
        mlev = adj_lev(lvl, depth, xl)
    mean = var = 0.0
    p_prev = [0.0, 0.0]
    for i, (at, ad, n, d) in enumerate(attacks):
        m1 = m2 = 0.0
        if at in _AT_MELEE:
            p = _p_hit(ac, mlev, i, helpless)
            m1, m2 = _dmg_moments(n, d, ac, at == AT_WEAP)
        elif at == AT_HUGS:
            p = 1.0 if stuck else (p_prev[0] * p_prev[1] if i >= 2 else 0.0)
            m1, m2 = _dmg_moments(n, d, ac)
        elif at == AT_EXPL:
            p = 1.0
            m1, m2 = _dmg_moments(n, d, 0)
        elif at == AT_MAGC and ad in (AD_CLRC, AD_SPEL):
            # mcastu.c: about half the casts are damage spells of d(ml/2 + 1, 6)
            p = 0.5
            m1, m2 = _dmg_moments(min(mlev // 2 + 1, 12), 6, 0)
        elif at == AT_MAGC:
            p = 0.5
            m1, m2 = _dmg_moments(n, d, 0)
        else:
            p = 0.0
        mean += p * m1
        var += p * m2 - (p * m1) ** 2
        p_prev = [p, p_prev[0]]
    return mean, max(var, 0.0), speed / 12.0


# ---------------------------------------------------------------- the model
class PrayerModel:
    _M_FULL_MOON = 'You are lucky!  Full moon tonight.'
    _M_FRIDAY13 = 'Bad things can happen on Friday the 13th.'
    _GUILTY_MEAT_RE = re.compile(r"You feel guilty\.(?!\.)")
    _KILL_RE = re.compile(r"You (?:kill|destroy) (?:the )?(poor )?([^!.]+?)!")

    def __init__(self, agent):
        self.agent = agent
        self.disabled = False
        self.errors = 0
        # Luck: joint distribution over (base, luck); the base is 0 unless the start message says otherwise
        # (the arena's first observation is not always seen: a full moon then stays unknown, which only
        # makes the model a little pessimistic about multi-trouble fixes)
        self.luck = {(0, 0): 1.0}
        self.p_ugangr = 0.0          # P(u.ugangr > 0)
        self.record = None           # alignment record estimate (None until the role is known)
        self.last_turn = None
        self.timeout_kind = 'start'  # 'start' | 'pleased' (rnz(350)) | 'angry' (rnz(300))
        self.timeout_turn = 1
        self.timeout_xl = 1
        self.timeout_shift = 0       # wishes since the last prayer
        self.timeout_zero_turn = None
        self.hp_history = []         # (turn, lowest hp seen that turn), last 8 turns
        self._last_msg_step = -1
        self.events = []             # (turn, event) for the log
        self.prayers = 0
        self.failures = 0
        self._cache_key = None
        self._cache = None
        self.last_p_hp = 0.0
        self.last_p_die = 0.0

    # ------------------------------------------------------------ state carried across a driver restart
    def adopt(self, agent):
        self.agent = agent
        self._last_msg_step = -1
        self._cache_key = None

    # ------------------------------------------------------------ observation
    def initial_message(self, msg):
        """The game's first message (allmain.c:48-57): full moon -> Luck base +1, Friday 13th -> -1."""
        if not msg:
            return
        base = None
        if self._M_FULL_MOON in msg:
            base = 1
        elif self._M_FRIDAY13 in msg:
            base = -1
        if base is not None:
            self.luck = {(base, base): 1.0}
            self.events.append((1, 'luck base %d' % base))

    def _init_record(self):
        role = getattr(self.agent.character, 'role', None)
        if role is None or role < 0:
            return
        self.record = _INIT_RECORD.get(role, 0)

    def _shift_luck(self, delta, prob=1.0):
        new = {}
        for (b, l), p in self.luck.items():
            l2 = max(-10, min(10, l + delta))
            new[(b, l2)] = new.get((b, l2), 0.0) + p * prob
            if prob < 1.0:
                new[(b, l)] = new.get((b, l), 0.0) + p * (1.0 - prob)
        self.luck = new

    def _clear_bad_luck(self):
        new = {}
        for (b, l), p in self.luck.items():
            new[(b, max(l, 0))] = new.get((b, max(l, 0)), 0.0) + p
        self.luck = new

    def _luck_timeouts(self, t0, t1):
        """timeout.c:486-505: one step toward the base on turns divisible by 600 (300 while ugangr)."""
        period = 300 if self.p_ugangr >= 0.5 else 600
        steps = min(t1 // period - t0 // period, 25)
        if steps <= 0:
            return
        new = {}
        for (b, l), p in self.luck.items():
            if l > b:
                l = max(b, l - steps)
            elif l < b:
                l = min(b, l + steps)
            new[(b, l)] = new.get((b, l), 0.0) + p
        self.luck = new

    def _malign(self, name):
        """makemon.c:set_malign for a hostile monster of this species (the record change when killed)."""
        entry = mondata.MONS.get(name)
        if entry is None:
            return 0
        mal, f2 = entry[3], entry[4]
        usgn = {0: -1, 1: 0, 2: 1}.get(getattr(self.agent.character, 'alignment', None), 0)
        if mal == mondata.A_NONE:
            return 20
        coaligned = ((mal > 0) - (mal < 0)) == usgn
        if f2 & mondata.M2_PEACEFUL:
            return 3 * max(5, abs(mal))  # a renegade (hostile always-peaceful)
        if f2 & mondata.M2_HOSTILE:
            return 0 if coaligned else max(5, abs(mal))
        return max(3, abs(mal)) if coaligned else abs(mal)

    def _adjalign(self, n, turn):
        if self.record is None:
            self._init_record()
            if self.record is None:
                return
        if n < 0:
            self.record += n
        elif n > 0:
            self.record = min(self.record + n, 10 + turn // 200)  # attrib.c:1129-1132, ALIGNLIM

    def observe(self):
        """Once per observation (agent.update): turn bookkeeping and message parsing."""
        agent = self.agent
        if agent.step_count == self._last_msg_step:
            return
        self._last_msg_step = agent.step_count
        bl = agent.blstats
        turn = bl.time
        if self.last_turn is None:
            self.last_turn = turn
        elif turn > self.last_turn:
            self._luck_timeouts(self.last_turn, turn)
            self.last_turn = turn
        hp = bl.hitpoints
        if not self.hp_history or self.hp_history[-1][0] != turn:
            self.hp_history.append((turn, hp))
            if len(self.hp_history) > 8:
                del self.hp_history[:-8]
        else:
            self.hp_history[-1] = (turn, min(hp, self.hp_history[-1][1]))
        if self.record is None:
            self._init_record()
        msg = agent.message
        if msg and ('You' in msg or 'mollified' in msg or 'bad luck' in msg):
            self._parse(msg, turn)

    def _parse(self, msg, turn):
        if 'You hear the rumble of distant thunder' in msg or 'You hear the studio audience applaud' in msg:
            self._shift_luck(-1)        # own pet killed (mon.c:2473-2474, 2505-2510)
            self._adjalign(-15, turn)
            self.events.append((turn, 'pet killed'))
        if 'You murderer!' in msg:
            self._shift_luck(-2)
            self._adjalign(-5 - 15, turn)  # peaceful -5 and its malign -3 * max(5, |align|)
            self.events.append((turn, 'murderer'))
        if 'You feel guilty...' in msg:
            self._shift_luck(-5)
            self.events.append((turn, 'unicorn'))
        if 'You feel guilty about losing your pet like this.' in msg:
            self.p_ugangr = 1.0          # hack.c:1883-1887 u.ugangr++
            self._adjalign(-15, turn)
            self.events.append((turn, 'god angered (pet lost)'))
        if 'You cannibal!' in msg:
            new = {}
            for d in (-2, -3, -4, -5):  # eat.c:670 change_luck(-rn1(4, 2))
                for (b, l), p in self.luck.items():
                    l2 = max(-10, l + d)
                    new[(b, l2)] = new.get((b, l2), 0.0) + p * 0.25
            self.luck = new
            self.events.append((turn, 'cannibal'))
        if "That's bad luck!" in msg:
            self._shift_luck(-2)
            self.events.append((turn, 'mirror'))
        if self._GUILTY_MEAT_RE.search(msg):
            self._adjalign(-1, turn)  # eat.c:1163 a Monk eating meat
        if 'You feel like a hypocrite.' in msg:
            self._adjalign(-5 if (self.record or 0) > 5 else -3, turn)
        if 'You caitiff!' in msg or 'You dishonorably attack the innocent!' in msg:
            self._adjalign(-1, turn)
        if 'seems mollified.' in msg and 'slightly' not in msg:
            self.p_ugangr = 0.0          # pray.c:1700-1706: anger gone, bad Luck reset to 0
            self._clear_bad_luck()
            self.events.append((turn, 'god mollified'))
        elif 'seems slightly mollified.' in msg:
            self._shift_luck_up_if_negative()
        if 'You have a hopeful feeling.' in msg:
            self._shift_luck_up_if_negative()
        if 'You have a feeling of reconciliation.' in msg:
            self.timeout_zero_turn = turn   # pray.c:1735-1740
            self._clear_bad_luck()
        if 'You may wish for an object.' in msg:
            self.timeout_shift += 150    # zap.c:5350 += rn1(100, 50), upper bound
        if 'You kill' in msg or 'You destroy' in msg:
            for m in self._KILL_RE.finditer(msg):
                name = m.group(2).strip()
                if name == 'it' or m.group(1):
                    continue  # unseen, or the own pet (the thunder message covers it)
                self._adjalign(self._malign(name), turn)

    def _shift_luck_up_if_negative(self):
        new = {}
        for (b, l), p in self.luck.items():
            l2 = l + 1 if l < 0 else l
            new[(b, l2)] = new.get((b, l2), 0.0) + p
        self.luck = new

    # ------------------------------------------------------------ prayers
    def on_prayer(self, turn_before, messages, answered, failed, limit):
        """After agent.pray(). `limit` is the ublesscnt bound pray.c applied (200/100/0) as estimated
        before the prayer; used for the anger posterior when it failed."""
        agent = self.agent
        self.prayers += 1
        turn = agent.blstats.time
        if failed:
            self.failures += 1
            p_soon = 1.0 - self.p_timeout_ok(limit, turn=turn_before)
            p_bad = self.naughty(include_anger=False)
            denom = p_soon + (1.0 - p_soon) * p_bad
            # P(this failure was 'too soon', i.e. gods_upset -> ugangr++): the timeout posterior against
            # Luck/record trouble; an unexplained failure counts as even odds
            p_new = p_soon / denom if denom > 0.02 else 0.5
            self.p_ugangr = 1.0 - (1.0 - self.p_ugangr) * (1.0 - p_new)
            if p_new > 0.5:
                self._shift_luck(-3)  # pray.c:1915 change_luck(-3)
            self.timeout_kind, self.timeout_turn = 'angry', turn  # angrygods(): ublesscnt = rnz(300)
        elif answered:
            # pleased() needs ugangr 0, Luck >= 0 and record >= 0
            self.p_ugangr = 0.0
            self._clear_bad_luck()
            if self.record is not None and self.record < 0:
                self.record = 0
            try:
                from .. import jf_config
                record_fix = jf_config.RECORD_MODEL_FIX
            except Exception:
                record_fix = True
            if record_fix:
                # pray.c:941 pleased(): 'else if (u.ualign.record < 2 && trouble <= 0) adjalign(1)' -- only a prayer
                # made without major trouble (limit < 200) earns the point. The old test (+1 unless 'You feel much
                # better') also counted Weak/Fainting prayers, so a record-0 hero (Cav/Pri/Tou/Val/Wiz start at 0;
                # a neutral gains nothing from killing the always-hostile alignment-0 jackals, rats, newts) read as
                # record 1-2 while each of its prayers still had pleased()'s action-0 coin flip (record < STRIDENT:
                # action = record > 0 || !rnl(2) ? 1 : 0): v2a cav-gno-neu-mal s205/s206 ('satisfied', hunger unfixed)
                if self.record is not None and self.record < 2 and limit < 200:
                    self.record += 1
            elif self.record is not None and self.record < 2 and 'You feel much better' not in messages:
                self.record += 1  # pray.c:936-937 (no trouble fixed)
            self.timeout_kind, self.timeout_turn = 'pleased', turn
        else:
            # neither message seen (e.g. interrupted parsing): assume a normal answered prayer
            self.timeout_kind, self.timeout_turn = 'pleased', turn
        self.timeout_xl = agent.blstats.experience_level
        self.timeout_shift = 0
        self.events.append((turn, 'prayer %s' % ('FAILED' if failed else 'ok')))

    # ------------------------------------------------------------ estimates
    def p_timeout_ok(self, limit, turn=None):
        """P(u.ublesscnt <= limit) at `turn` (default: now)."""
        if turn is None:
            turn = self.agent.blstats.time
        if self.timeout_zero_turn is not None and self.timeout_zero_turn >= self.timeout_turn:
            return 1.0
        elapsed = turn - self.timeout_turn - self.timeout_shift
        if self.timeout_kind == 'start':
            return 1.0 if 300 - elapsed <= limit - 2 else 0.0  # u_init.c:644, 2 turns of margin
        return rnz_cdf(350 if self.timeout_kind == 'pleased' else 300, limit + elapsed, self.timeout_xl)

    def p_luck_nonneg(self):
        return sum(p for (_, l), p in self.luck.items() if l >= 0)

    def p_record_nonneg(self):
        if self.record is None or self.record >= 0:
            return 1.0
        return 0.5 if self.record >= -2 else 0.0

    def naughty(self, include_anger=True):
        """P(p_type 1 whatever the timeout): god angry, Luck < 0 or record < 0 (pray.c:1823-1824)."""
        p_ok = self.p_luck_nonneg() * self.p_record_nonneg()
        if include_anger:
            p_ok *= (1.0 - self.p_ugangr)
        return 1.0 - p_ok

    def troubles(self):
        """Observable troubles in pray.c:in_trouble order -> (worst, [major], minor?)."""
        agent = self.agent
        bl = agent.blstats
        cond = int(bl.prop_mask)
        major = []
        if cond & BL_MASK_STONE:
            major.append(TROUBLE_STONED)
        if cond & BL_MASK_SLIME:
            major.append(TROUBLE_SLIMED)
        if cond & BL_MASK_STRNGL:
            major.append(TROUBLE_STRANGLED)
        if cond & (BL_MASK_FOODPOIS | BL_MASK_TERMILL):
            major.append(TROUBLE_SICK)
        if bl.hunger_state >= HUNGER_WEAK:
            major.append(TROUBLE_STARVING)
        if critically_low_hp(bl.hitpoints, bl.max_hitpoints, bl.experience_level):
            major.append(TROUBLE_HIT)
        if getattr(agent.character, 'is_lycanthrope', False):
            major.append(TROUBLE_LYCANTHROPE)
        minor = bl.hunger_state == HUNGER_HUNGRY or bool(cond & (BL_MASK_STUN | BL_MASK_CONF | BL_MASK_HALLU |
                                                                    BL_MASK_BLIND))
        return (major[0] if major else 0), major, minor

    def trouble_limit(self):
        _, major, minor = self.troubles()
        return 200 if major else (100 if minor else 0)

    def p_answered(self, limit=None):
        """P(p_type 3) for a prayer made now."""
        if limit is None:
            limit = self.trouble_limit()
        return self.p_timeout_ok(limit) * (1.0 - self.naughty())

    def p_fix(self, target):
        """P(target trouble fixed | answered): pray.c:pleased 975-1005."""
        worst, major, _ = self.troubles()
        if target not in major:
            return 0.0
        rec = self.record if self.record is not None else 10
        lucks = [(l, p) for (_, l), p in self.luck.items() if l >= 0]
        tot = sum(p for _, p in lucks) or 1.0
        if target == worst:
            if rec >= 1:
                return 1.0
            return 0.5  # record 0: action = !rnl(2) ? 1 : 0
        if rec < STRIDENT:
            return 0.0
        # action = rn1(Luck + 2, 1) >= 2 (capped at 3 off an altar): P = (L + 1) / (L + 2)
        return sum(p * (l + 1.0) / (l + 2.0) for l, p in lucks) / tot

    # ------------------------------------------------------------ threat
    def death_probability(self, turns=3, helpless=False):
        """P(the hostiles in reach deal >= current HP within `turns` turns), mhitu.c melee rules."""
        agent = self.agent
        bl = agent.blstats
        hp, ac, depth, xl = bl.hitpoints, bl.armor_class, bl.depth, bl.experience_level
        dive = agent.global_logic.dive
        on_elbereth = False
        try:
            eng = agent.inventory.engraving_below_me
            on_elbereth = eng is not None and eng.lower() == 'elbereth'
        except Exception:
            pass
        msgs = ' '.join(agent._message_history[-3:])
        stuck = 'You are being crushed' in msgs or 'grabs you' in msgs or 'cannot escape from' in msgs
        mean = var = 0.0
        n_adj = 0
        for _, y, x, mon, _ in agent.get_visible_monsters():
            d = max(abs(int(y) - bl.y), abs(int(x) - bl.x))
            if d > 3:
                continue
            if on_elbereth and d <= 1 and not dive._ignores_elbereth(mon):
                continue  # monmove.c:onscary -- it flees and does not attack (dochug 646)
            name = getattr(mon, 'mname', 'unknown')
            m1, v1, spd = monster_turn_damage(name, ac, depth, xl, stuck and d == 1, helpless)
            eff = max(0, turns - (d - 1))
            if d == 1:
                n_adj += 1
            mean += m1 * spd * eff
            var += v1 * spd * eff
        # what the last turns actually cost (covers weapons, spells and anything the table misses)
        if n_adj and len(self.hp_history) >= 2:
            (t0, h0), (t1, h1) = self.hp_history[max(0, len(self.hp_history) - 4)], self.hp_history[-1]
            if t1 > t0 and h0 > h1:
                mean = max(mean, 0.75 * (h0 - h1) / (t1 - t0) * turns)
        if mean <= 0:
            return 0.0
        sd = math.sqrt(max(var, 1.0))
        return 1.0 - _phi((hp - 0.5 - mean) / sd)

    # ------------------------------------------------------------ decisions
    def hp_decision(self, safe_p, doom_margin=0.1, doom_min_p=0.3):
        """For critically low HP: 'hp' (as safe as the old 500-turn rule), 'hp-doom' (the old rule says
        wait, but P(a prayer restores HP) beats P(surviving 3 more turns) by doom_margin), or None."""
        agent = self.agent
        key = (agent.step_count, agent.blstats.time, agent.blstats.hitpoints)
        if self._cache_key == key:
            return self._cache
        res = self._hp_decision(safe_p, doom_margin, doom_min_p)
        self._cache_key, self._cache = key, res
        return res

    def _hp_decision(self, safe_p, doom_margin, doom_min_p):
        worst, major, _ = self.troubles()
        if TROUBLE_HIT not in major:
            return None
        naughty = self.naughty()
        if naughty >= 0.5:
            return None
        p_time = self.p_timeout_ok(200)
        if p_time >= safe_p:
            return 'hp'
        p_hp = p_time * (1.0 - naughty) * self.p_fix(TROUBLE_HIT)
        self.last_p_hp = p_hp
        if p_hp < doom_min_p:
            return None
        p_die = self.death_probability(turns=3)
        self.last_p_die = p_die
        if p_hp > (1.0 - p_die) + doom_margin:
            return 'hp-doom'
        return None

    def faint_decision(self, safe_p, margin, turns=8):
        """Fainting (pray.c TROUBLE_STARVING; eat.c:2977-2996 faints for 10 - uhunger/10 turns at a time, and a
        helpless hero is hit at +4, mhitu.c:573-574) with hostiles in reach: pray once the timeout posterior is
        as safe as the old HP rule and P(answered) beats P(surviving the faints) by `margin`. Returns
        (pray?, p_answered, p_die)."""
        _, major, _ = self.troubles()
        if TROUBLE_STARVING not in major:
            return False, 0.0, 0.0
        p_ok = self.p_answered(200)
        if p_ok < safe_p:
            return False, p_ok, 0.0
        p_die = self.death_probability(turns=turns, helpless=True)
        return p_ok > (1.0 - p_die) + margin, p_ok, p_die

    def status_decision(self):
        """A lethal status trouble (stoned, slimed, strangled, sick): P(the prayer cures it), or 0."""
        worst, major, _ = self.troubles()
        status = [t for t in major if t in LETHAL_STATUS_TROUBLES]
        if not status:
            return 0.0
        return self.p_answered(200) * self.p_fix(status[0])

    def summary(self):
        return 'ok200=%.2f naughty=%.2f ugangr=%.2f luck>=0=%.2f record=%s' % (
            self.p_timeout_ok(200), self.naughty(), self.p_ugangr, self.p_luck_nonneg(), self.record)
