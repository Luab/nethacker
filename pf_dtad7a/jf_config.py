"""Feature switches for A/B experiments.

Dev runs may override them with JF_CFG='{"TOUR_FIXES": false, ...}'; the arena never sets
JF_CFG, so submissions always run these defaults.
"""
import json
import os

# Fixes that change the levelling tour. Split by WHEN they first change a game:
# LATE: only at a specific hazard (gas spore next to the pet, cockatrice corpse, passive-damage
#   monster, deadly status, empty wand) -- the elite's early game is untouched until then.
# EARLY: prayer at pray.c's critically_low_hp instead of 'HP < 12', eat carried food before a
#   hunger prayer below XL 5 -- these reshuffle games from the first prayer on.
EARLY_FIXES = False
LATE_FIXES = True
# the rarest-hazard subset of LATE_FIXES (gas spore next to the pet, cockatrice-family corpse
# squares, spotted/ochre jelly and gelatinous cube melee): these first fire close to the deaths
# they prevent, so they barely perturb the elite's public trajectories
HAZARD_FIXES = False
# master switch kept for older experiment configs: sets both
TOUR_FIXES = None
# Excalibur dips only at >= 90% HP with a prayer ready (astra); changes the tour
SAFE_DIPS = False
# when Weak or worse with HP > 40, poisonous/acidic corpses are acceptable food
STARVING_EATS = True
# astra's survival layer (Elbereth rest, retreat upstairs) also during the levelling tour
SURVIVAL_IN_TOUR = True
# at critically low HP with no safe prayer and a hostile adjacent: stairs, unknown wands/potions/scrolls
LAST_RESORT = True
# Komershan's verified leader (0.2033 on hidden seeds): dwarves and gnomes skip Sokoban and walk the
# peaceful Mines from Minetown straight to Mines' End (Dlvl 10-13: 0.126-0.26 banked early)
SKIP_SOKOBAN = False
# pray for HP only at pray.c's critically_low_hp (DT6A's 'HP < 12' prays with no trouble to fix: no
# heal, and a failure if the timeout isn't 0), and allow the first prayer from turn 100 (the timeout
# starts at 300; major trouble needs <= 200)
EXACT_PRAYER = False
# minimum turns since the last prayer for a hunger prayer while Fainting (DT6A: 400). Most first prayer
# failures were Fainting prayers 900-1100 turns after the last one (rnz(350) timeout: ~6% fail there,
# ~2% past 1100); a longer gap means fainting longer instead
FAINT_PRAYER_GAP = 1100
# AutoAscend's periodic 'eat corpses' preempt was meant to walk to edible corpses on the level, but
# only_below_me defaults to True, so it only ever eats what lies underfoot; the kills' corpses beside
# us go to the pet (it ate ~40% of the grind's corpses)
EAT_NEARBY_CORPSES = False
# Weak/Fainting with nothing to eat and no safe prayer yet: wait on Elbereth instead of wandering (many
# grind deaths came while fainted: rats, bats, ants; nearly everything on Dlvl 1 respects Elbereth).
# OFF: calibrated against the frozen s13 on the same 60 games (jf14/jf16/jf25/jf26) the shelter +
# starvation clock + 1400-turn Weak gap regime scored 0.207 vs s13's 0.251; with the three reverted
# (the other fixes kept) 0.252. Sheltering stops the hunt/eat loop that keeps the grind fed.
FAINT_SHELTER = False
# Fainting prayers by the starvation clock instead of a fixed gap: eat.c kills at uhunger <
# -(100 + 10 * Con) and uhunger drops at most 1 per turn once Fainting (almost not at all while
# fainted), so death is at least 100 + 10 * Con turns after Fainting begins. Pray once the gap is
# FAINT_PRAYER_GAP_LONG (= the Weak rule's gap), or STARVE_MARGIN turns before that deadline whatever
# the gap (the fixed 1000-turn gap could starve a character whose last prayer was an HP one, and it
# prays where rnz(350) still fails ~5.5% of the time; ~2.3% at 1200).
# hypothesis: Tourists (weak, often Fainting in the grind; seed 6 starved) survive longer when the Fainting
# prayer follows eat.c's starvation deadline instead of a fixed 1100-turn gap (local: 0.248 -> 0.259)
STARVE_CLOCK = True
# give up looking for the Mines entrance after this many turns and go on to Sokoban (0: never)
MINES_SEARCH_TURNS = 3500
# buy food in shops (never while carrying a digging tool) until carrying BUY_FOOD_UNTIL nutrition
BUY_FOOD = True
BUY_FOOD_UNTIL = 2400
# with at least FOOD_FIRST_MIN nutrition carried, eat instead of hunger-praying below FOOD_FIRST_GAP turns
# (0: off). Off: the 20 games where it fired (mostly the grind, on found food) fell 5.11 -> 3.97 in total --
# DT6A's reserve (eat only when no safe prayer) saves riskier Fainting prayers later.
FOOD_FIRST_MIN = 0
FOOD_FIRST_GAP = 1400
# from this XL the grind detours to the Mines (level PICK_TRIP_LEVEL, PICK_TRIP_TURNS at most) for a dwarf's
# pick-axe and keeps it (0: off)
PICK_TRIP_XL = 0
PICK_TRIP_LEVEL = 2
PICK_TRIP_TURNS = 2500
# a tool-less trip ends once the character reaches this XL (0: never): see _pick_trip_active
PICK_TRIP_END_XL = 0
# a tour headed for a shallower main-dungeon level (the grind, after a trip) explores for up staircases
# only: after its trip, pt6-public seed 9 fell through a trap door to Dlvl 4, took that level's unexplored
# '>' to Dlvl 5 and met soldier ants at XL 7 (the old rule takes any unexplored staircase, 50/50)
UPWARD_RETURN = False
# hypothesis: a trap door on Dlvl 1 drops the XL 1-3 grind a few levels (fem s1 fell to Dlvl 3 at T7), and the
# tour then explores each of those levels to exhaustion on its way back -- items, every unexplored staircase --
# at 10-20 HP among monsters generated for their depth (s1: killed by a hobbit at T395, 0.018). Fallen below
# the grind level in the main dungeon, head straight home: explore for '<' only (as UPWARD_RETURN does) and
# read one of the Tourist's 4 identified scrolls of magic mapping on a level whose '<' isn't known, so the
# walk to it replaces the search (measured: s5 0.021 -> 0.206, s9 0.554 -> 0.507; s1 gets home by T252 but
# still dies on Dlvl 1)
# sources: https://nethackwiki.com/wiki/Trap_door, https://nethackwiki.com/wiki/Scroll_of_magic_mapping,
#          https://nethackwiki.com/wiki/Tourist, /refs/top/1c4099e80253 (explore until the stairs appear)
FALL_HOME = True
# from this XL the Dlvl 1 grind moves to Dlvl GRIND_DEEP_LEVEL (0: never)
GRIND_DEEP_XL = 0
GRIND_DEEP_LEVEL = 3
# no Excalibur dips while carrying a digging tool (a dip's silent curse welds the sword and locks the pick out)
NO_DIP_WITH_TOOL = False
# the grind's main-dungeon level by XL, {min XL: Dlvl} (empty: Dlvl 1 throughout; overrides GRIND_DEEP_*):
# e.g. {5: 3, 7: 2} keeps the random-monster cap (depth + XL) / 2 at 4 from XL 5 (global_logic._grind_level)
GRIND_LEVELS = {}
# the tour skips to its next milestone after this many turns within 8 squares of one spot on one level
# (0: never). Stalls held 12 of 90 games for 1500-14000 turns, fainting through hunger prayers.
TOUR_STALL_TURNS = 1500
FAINT_PRAYER_GAP_LONG = 1400
STARVE_MARGIN = 60
# Weak hunger prayers wait for this gap (DT6A/s13: 1200). Measured over ~2600 prayers, 900-1399-turn
# gaps failed 3.5-5.4% of the time, 1400-1799 only 1.1% and 1800+ 0.6% -- but 1400 lost more games
# to fainting than it saved from failed prayers (see FAINT_SHELTER).
WEAK_PRAYER_GAP = 1200
# corpses older than this (turns since the kill) are not eaten (AutoAscend: 50; tainting starts above 50)
CORPSE_MAX_AGE = 30
FAINT_ESTIMATE_MARGIN = 90
# Gehennom (dive_logic.valley_step, valley_sneak, valley_retreat, gehennom_escape): walk the Valley of the Dead
# to its '>' (fixed map in valley.py), digging through its three locked secret doors with the pick-axe; never
# pray (pray.c: the god can't help there and may get angry) or rely on Elbereth (onscary: Inhell) in Gehennom;
# the Valley's '<' only as the retreat to the castle's east edge (the castle mode rests and drops back through
# the trap door behind the back door); a wand of digging zapped down as the escape; dig-dive below. Everything
# keys on dnum == 1, which no game reached before the castle passage (msgs byte-identical on vs off).
GEHENNOM_DIVE = True
# Castle passage (castle_logic.py): on the castle level (the dig-dive's floor) walk to the west courtyard,
# try every ring/potion that may be levitation (or freeze the moat with a cold ray), float round the moat
# to the back door (56,08) and drop through the trap door behind it into the Valley (castle depth + 1).
# Dying on the castle level costs nothing: the castle is as deep as a dig can go.
CASTLE_PASSAGE = False
# engulfed: wield the best melee weapon before fighting out (a dig-diver is swallowed with its pick-axe)
ENGULF_WIELD = False
# a Hungry (or worse) dive walks to fresh edible corpses within DIVE_EAT_RADIUS (BFS steps) and eats them
DIVE_EAT = False
DIVE_EAT_RADIUS = 8
# LAST_RESORT: a known wand of digging is zapped down first (an escape that also banks a level)
# hypothesis: letting the last-resort escape dig down (when HP is critical, prayer isn't safe and a
# hostile is adjacent) saves deep Tourists that otherwise die cornered (local fem seed 14 0.353->0.554)
LAST_RESORT_DIG = True
# LAST_RESORT: pray at critical HP beside a hostile once this many turns have passed since the last prayer
# (0: off; the ordinary low-HP prayer waits 500)
DESPERATE_PRAYER_GAP = 0
# LAST_RESORT: zap each unknown wand once (one still unknown after a zap at a monster is no attack wand)
# hypothesis: at critical HP beside an attacker, re-zapping an unknown wand that already did nothing visible at a
# monster wastes the last turns (fem s10: the same oak wand zapped 5 times from 7 to 1 HP, housecat): an attack wand
# names itself when its ray/beam hits (zap.c learn_it), so move on to the unknown potions/scrolls instead
# sources: https://nethackwiki.com/wiki/Wand, https://nethackwiki.com/wiki/Engrave-identification, NetHack 3.6.6 zap.c
LR_WAND_ONCE = True
# --- power: what the character carries to the Castle (power.py) ---
# Unidentified boots of the magic appearances (combat/jungle/hiking/mud/buckled/riding/snow) are 2/7 levitation
# or water walking, the Castle's moat crossing. The bot never picked them up (get_best_armorset skips ambiguous
# armour, and ItemPriority's unknown-status branch is dead: the parser maps UNKNOWN to UNCURSED); 26 of 90
# base-* games walked over a pair. Keep them unworn for castle_logic; never wear levitation/fumble boots.
KEEP_MAGIC_BOOTS = False
# a known wand of wishing keeps rnd(3) - 1 charges after the engrave-test wish (51 of 3158 games got one):
# zap them (power.wish_text: GDSM, then an amulet of life saving worn at once, then a ring of levitation for
# the Castle, then speed boots). ON (coordinator, train 2): it fires only in wish games (~1%).
SPARE_WISHES = True
# scrolls that may be scare monster are never dropped by arrange_items (a heavy armour swap dropped all light
# loot and picked it up again: 250 of 3158 games turned one to dust), a scroll we dropped is never picked up
# again, and a dust event names the scare monster label (power.scare_scrolls for castle/gehennom)
SCARE_KEEP = False
# while diving, ItemPriority keeps food and then the passage candidates (potions/rings that may be levitation,
# ...) ahead of the thrown weapons: 23 of 90 base games dropped potion types for good, mostly for daggers
# and the dive's pick-axe/mattock
KEEP_POTIONS = False
# inside a shop that buys them, drop each unknown potion/ring/candidate boots, read the shopkeeper's offer
# (base/2, or 3/8 of it) and decline: the price group narrows levitation to 1 of 5 potions / 1 of 7 rings.
# PROTOTYPE, keep off: in power-sell1-jf16 a ring's offer worked (100 -> the 200 zm group) but no potion
# offer was captured, and one game (s13) looped with 4114 tracebacks before dying.
SELL_PRICE_ID = False
# a hunger prayer from this gap (instead of FAINT_PRAYER_GAP / WEAK_PRAYER_GAP) when a hostile within
# THREAT_RADIUS can reach us while Fainting, or while Weak with at most THREAT_WEAK_MARGIN nutrition left
# (0: off). 9 of 10 Dlvl-1 deaths in the base runs were fainted next to ordinary monsters at gaps 950-1110.
THREAT_PRAYER_GAP = 0
THREAT_RADIUS = 5
THREAT_WEAK_MARGIN = 15
THREAT_MIN_DIFFICULTY = 4
# this many walking hostiles within THREAT_RADIUS count as a threat (99: only difficulty / Elbereth-ignorers / HP)
THREAT_MIN_COUNT = 2
# ... or our HP below this fraction (a faint on a smudged Elbereth took a 90-HP XL7 to 49 in one faint)
THREAT_HP_FRAC = 0.5
# find the kill square of our melee/thrown kills from the attack itself, and of pack kills from the corpse
# glyph, when the glyph-disappearance test misses it (27% of kills: their corpses were never eaten)
CORPSE_TRACK = False
# walk to fresh (<= CLAIM_MAX_AGE turns) edible corpses within CLAIM_DIST steps and eat them, before the pet
CLAIM_CORPSES = False
CLAIM_DIST = 3
CLAIM_MAX_AGE = 15
# eat poisonous corpses (not only when Weak) at HP >= max(POISON_EATS_MIN_HP, 60%) during the tour
POISON_EATS = False
POISON_EATS_MIN_HP = 40
# carry up to this many lichen/lizard corpses as a food reserve instead of eating them off the floor while not
# Weak (0: off)
LICHEN_RESERVE = 0
# the Dlvl 1 grind ends (DIVE_XL) only fed: Not Hungry within DIVE_FED_GAP turns of the last hunger prayer, or
# carrying >= DIVE_FED_FOOD nutrition; else it waits for the next hunger prayer (at most DIVE_FED_MAX_WAIT turns)
DIVE_FED = False
DIVE_FED_GAP = 500
DIVE_FED_FOOD = 400
DIVE_FED_MAX_WAIT = 2000
# longer hunger-prayer gaps in the tour only (0: WEAK_PRAYER_GAP / FAINT_PRAYER_GAP): with FAINT_GUARD(_IDLE)
# holding Elbereth through faints, rnz(350) fails 2.3% of prayers at a 1200 gap, 1.8% at 1400, 1.0% at 1700
TOUR_WEAK_PRAYER_GAP = 0
TOUR_FAINT_PRAYER_GAP = 0
# per-XL tour gaps [[min_xl, weak_gap, faint_gap], ...] (the highest min_xl <= XL wins; overrides TOUR_*)
TOUR_GAPS_BY_XL = []
# the low-HP prayer only at pray.c's critically_low_hp (EXACT_PRAYER's HP rule without its turn-100 first prayer)
LOWHP_EXACT = False
# hunger-prayer gaps while diving at depth >= DIVE_GAP_MIN_DEPTH (0: WEAK_PRAYER_GAP / FAINT_PRAYER_GAP)
DIVE_WEAK_PRAYER_GAP = 0
DIVE_FAINT_PRAYER_GAP = 0
DIVE_GAP_MIN_DEPTH = 5
# without STARVE_CLOCK: a Fainting prayer whatever the gap when the faint-length hunger estimate nears starvation
STARVE_DEADLINE = False
# log corpse bookkeeping (kills recorded, corpses we stand on and their known age); diagnostics only
CORPSE_DEBUG = False

# robustness (loops and stalls):
# path() walks the BFS distances back obeying the BFS's own diagonal rule (a door entered diagonally was
# retried 90 times in one jf26 game): only changes a path that would have failed
PATH_DIAG_FIX = True
# fight2 lets go of monsters it has faced this many turns without contact, a kill or damage (asleep, immobile,
# out of reach), or after FIGHT_STALL_MOVES consecutive moves among <= 3 squares (the dance), for
# FIGHT_IGNORE_TURNS or until they come adjacent / anything hurts us (0: off; < 0: log only)
# ON (train 2) at 150 in the dive (the tour keeps FIGHT_STALL_TOUR_TURNS): robustness b2 guard 0.3480 vs 0.3479
FIGHT_STALL_TURNS = 150
FIGHT_STALL_MOVES = 30
FIGHT_IGNORE_TURNS = 300
# in the tour (the Dlvl 1 grind) only stalls this long are broken, and not by the dance trigger
FIGHT_STALL_TOUR_TURNS = 400
# the same in the Valley of the Dead only (GEHENNOM_DIVE; used while FIGHT_STALL_TURNS is 0)
VALLEY_FIGHT_STALL_TURNS = 25
# walking through known traps (AutoAscend's search relent, the dive's cut-off stairs, TRAP_LAST_RESORT) never
# enters polymorph/fire/sleeping gas/magic/anti-magic/rust traps, nor trap doors/holes/level teleporters in the tour
# ON (train 2): robustness b2 guard; +0.14 over the 2 games it fired in (b1)
SAFE_TRAP_WALK = True
# exploration with nothing left but searching, while unexplored ground lies beyond a known trap: walk through
# (the safe kinds of) traps at once instead of after ~80 visits' worth of searching one square
# ON (train 2): robustness b2 guard (dive only, after TRAP_RESORT_MIN_TURNS on the level)
TRAP_LAST_RESORT = True
TRAP_RESORT_MIN_TURNS = 1000   # ...once the dive has spent this long on the level
# remember where molds/jellies/floating eyes/gas spores sit and keep the BFS off those squares while they are out of
# sight (a mold on an item pile looks like the pile from afar: check_items looped on it for thousands of turns)
SESSILE_MEMORY = False
# go_to() re-plans to its real target after a path is blocked mid-way (its loop variables overwrote the target,
# so the next round aimed at the blocked square: 'end point is no longer accessible' and a strategy restart)
GOTO_TARGET_FIX = False
# the panic-loop breaker closes a square for 100 turns, not for the rest of the game (the failing moves are
# usually ours: a bear trap, a web; a permanent forbid boxed a Mines dive in for 5000+ turns)
# hypothesis: a permanent forbid walls the grind/dive into a pocket for thousands of turns (starving, praying
# on Dlvl 1); expiring it after 100 turns lets the bot leave and reach deeper milestones
TEMP_FORBID = True
# boxed in by diagonal squeezes while carrying > 600: drop to 550 for a while (a corridor bend held a grind 8000 turns)
UNSQUEEZE = False
# after 3 failed use_container attempts on a floor container, leave that square's containers alone (a take-out
# menu that never matched was retried 45,453 times in one game)
# ON (train 2): the take-out loop (robustness B004) hit jf14 s0 in the train-2 smoke: 11399 panics and 317k steps; with the fix 152 and 36k
CONTAINER_LOOP_FIX = True
# climbing out of a branch (or on the tool quest), known trap doors and holes stay closed: the stairs-cut-off walk
# stepped onto them again and again (base-jf26 s14 fell 42 times in 10k turns climbing out of Mines' End)
CLIMB_NO_FALL = False
# no fight2 moves while we are still in a pit (each try costs a turn and the attacker hits for free)
PIT_AWARE_FIGHT = False
# Lycanthropy (466 of 3158 dev games were infected; uncured Dlvl-1 lycanthropes died ~21% per 1000
# turns): detect 'You dream that you feel feverish' and were changes, never eat our were family's corpses
# (cannibalism: Luck -2..-5, the next prayer fails -- public s13, jf25 s10), treat a were form's HP as
# a buffer (no HP prayers/last resort for it: jf16 s11 and jf25 s9 spent their hunger prayer on the
# rat's HP and starved), and drop the load a rat can't carry so it can eat (public s4 starved Overloaded
# with 5 food items)
LYCAN_FIXES = True
# no lycanthropy cure prayer while Hungry without food (wait for the Weak hunger prayer; see cure_disease)
LYCAN_CURE_WAIT = False
# Weak/Fainting in the tour with no prayer due and a monster within FAINT_GUARD_RADIUS: hold on Elbereth
# instead of fighting (dive_logic.faint_guard; fainted melee deaths were 8 of 18 Dlvl-1 grind deaths)
FAINT_GUARD = True
FAINT_GUARD_RADIUS = 4
# with FAINT_GUARD: also hold on Elbereth with nothing in view, from FAINT_GUARD_IDLE_WEAK turns into Weak and
# while Fainting, until the prayer is due (monsters arriving during a faint get no conscious turn to react to)
FAINT_GUARD_IDLE = False
FAINT_GUARD_IDLE_WEAK = 30
# the reactive guard also in the rescue dive after a failed prayer (no idle hold there: the dive must go on)
FAINT_GUARD_RESCUE = False
# one-action hold strategies (Elbereth rest, water demon vigil, faint shelter/guard) repeat while their own
# condition holds, instead of letting one step of a lower strategy run between two hold actions
# (dive_logic._hold_loop)
HOLD_LOOP = False
# fight2 keeps only attack actions while we are stuck in a pit (failed moves out of a pit are free rounds for
# the monsters around us; melee from a pit is unrestricted)
FIGHT_PIT_FIX = False
# Water demon vigil fixes (dive_logic.water_demon_vigil): step off the fountain before engraving (the dip
# leaves us on it: 'You can't write on the fountain!', so the vigil never held), hold while a demon is
# within DEMON_VIGIL_RADIUS (not 2) for DEMON_VIGIL_TURNS. 74 of 975 dipping games released a demon, 22
# of them died within 300 turns.
DEMON_FIX = True
DEMON_VIGIL_RADIUS = 5
DEMON_VIGIL_TURNS = 400
# fight2 never melees a floating eye we can see (the exploration stall breaker's attack-all mode did: 401
# paralysis events in 223 dev games, 35 games died frozen)
FEYE_FIX = False
# no Excalibur dips during a water demon's vigil window (the bot went back to the fountain next to the demon)
DEMON_NO_REDIP = False
# the last resort (unknown wands/potions/scrolls) yields to the Elbereth rest while everything close respects
# Elbereth and we are on one or can engrave (a zap erased it, a bounced ray / potion of sickness killed at 2-3 HP)
# hypothesis: preserve usable Elbereth protection against susceptible enemies
# instead of erasing it with an unknown wand or drinking a harmful potion.
LR_ELBERETH = True

# Never dig or zap digging down on a staircase (rescue agent, 78a30e1; ported by hand for train 2): the square
# under '@' is unknown on arrival, so _diggable_spot took the arrival '<' for floor, and a wand of digging
# zapped there only says 'The beam bounces off the stairs' -- the dive zapped again until the wand was empty
# (6 of 90 baseline games, up to 5 charges = 5 levels each; jf16/5, jf27/1).
WAND_STAIRS_FIX = True

_raw = os.environ.get('JF_CFG')
if _raw:
    for _name, _value in json.loads(_raw).items():
        if _name in globals() and not _name.startswith('_'):
            globals()[_name] = _value

# JSON object keys are strings
GRIND_LEVELS = {int(_k): int(_v) for _k, _v in (GRIND_LEVELS or {}).items()}

if TOUR_FIXES is not None:
    EARLY_FIXES = LATE_FIXES = bool(TOUR_FIXES)
if LATE_FIXES:
    HAZARD_FIXES = True
