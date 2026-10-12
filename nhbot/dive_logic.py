"""Depth-first plan for the BALROG progression score.

The episode score is the best milestone ever reached (death costs nothing):
depth dominates (Dlvl 12 = 0.21, Dlvl 20 = 0.38, Dlvl 25 = 0.47) and stepping
into the Quest home through the magic portal on Dlvl 11-16 is worth 0.366 on
its own. So instead of AutoAscend's Mines/Sokoban tour, this plan:

  * explores the first levels fully while under-levelled (items, XP),
  * then dives the Dungeons of Doom by the nearest '>' / trap door / hole,
  * on the Quest portal level (announced by a telepathic message) sweeps the
    candidate rooms until the hidden portal fires, then walks back out,
  * and keeps diving afterwards.

The plan is a restartable loop of short tasks: fights, eating, prayer etc.
preempt it at any step, so all decisions are re-derived from game state.
"""

import functools
import re

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A
from scipy import ndimage

from . import objects as O

from . import jf_config, jf_log, power, utils, valley, valley_walk
from .castle_logic import CastlePassage
from .character import Character
from .exceptions import AgentPanic
from .glyph import C, G, MON, SS, Hunger
from .level import Level
from .item import Item, flatten_items
from .strategy import Strategy

ROOM_FLOOR = frozenset({SS.S_room, SS.S_darkroom})
PLAIN_FLOOR = frozenset({SS.S_room, SS.S_darkroom, SS.S_corr, SS.S_litcorr})
WET = frozenset({SS.S_pool, SS.S_water, SS.S_lava})
CORRIDORS = frozenset({SS.S_corr, SS.S_litcorr})
DOORWAYS = frozenset({SS.S_ndoor}) | G.DOORS
FALL_TRAPS = frozenset({SS.S_trap_door, SS.S_hole})
PORTAL = frozenset({SS.S_magic_portal})

GEHENNOM = 1
MAIN_LINE = (Level.DUNGEONS_OF_DOOM, GEHENNOM)

PORTAL_MESSAGES = ('telepathic message', 'pleading for help', 'demanding your attendance')

# XP gate: while XL < REQUIRED_XL[depth of the next level], the current level is explored
# fully first (items + XP), within FULL_EXPLORE_TURNS. Dlvl 1-4 are always explored fully.
# Deeper than the table: the last entry applies.
REQUIRED_XL = {2: 99, 3: 99, 4: 99, 5: 99, 6: 6, 7: 7, 8: 8, 9: 8, 10: 9, 11: 9, 12: 10}
# trap doors and holes can drop several levels at once
TRAPDOOR_LOOKAHEAD = 3
# rest (search) to this fraction of max HP before taking a way down (astra: 95-100%)
REST_BEFORE_DESCEND = 0.95   # astra left a level at >= 95% HP on 97% of descents
# rest (search) whenever below this and nothing hostile is in view (astra guard.py: 2/3)
REST_BELOW = 0.66
# Dwarves and gnomes find nearly every Mines inhabitant peaceful: bank Mines' End depth
# (Dlvl 10-13, up to 0.26) safely before the main-dungeon dive.
MINES_ROUTE = True
# MINES_TOOL_TRIP: a planned dive with no digging tool takes the Mines route whatever our race (MINES_ROUTE was for
# dwarves and gnomes only, whose Mines are peaceful). Humans, elves and orcs otherwise dove the main dungeon by stairs
# and met a pick-carrying dwarf only by chance: 24 of 39 tool-less planned Wizard dives never reached the Mines
# (EARLY_DETOUR needs a Mines level we already know, and the Dlvl-1 grind shows none), and tool-less dives averaged
# 0.06 progress against 0.34 with a tool. In the Mines the camp (MINES_CAMP) hunts levels 1-4 for a dwarf (3 in 8
# carry a pick-axe or a mattock, makemon.c m_initweap); with a tool the dive climbs back and digs the main dungeon,
# without one it goes on down to Mines' End. Rescue dives keep the main stairs (see should_dive).
# ON: A/B mt0/mt1 (23 human/elf/orc identity types x 12 held-out seeds): +0.0178 +- 0.0065 per game, 30 wins / 16 losses.
MINES_TOOL_TRIP = True
MINES_BRANCH_MAX_DEPTH = 4     # the Mines branch staircase is on Dlvl 2-4
MINES_MIN_LEVELS = 8           # dungeon.def: the Mines have 8-9 levels, Mines' End is the last
# XP gate inside the Mines: before going to Mines level k, explore the current level fully while
# XL < MINES_REQUIRED_XL[k] (hostile orcs/ants there are the XP). Empty = no gate.
MINES_REQUIRED_XL = {}
# astra: retreat onto Elbereth at 45-65% HP, rest there with searches, never attack from it
# hand-over from AutoAscend's levelling tour to the dive
DIVE_XL = 8
# sam-grind-exit: a Samurai leaves the Dlvl-1 grind for the dive at this XL instead of DIVE_XL (None: off).
# The grind is spawn-limited: allmain.c:124 makes one random monster per 70 turns, and makemon.c:1550 rndmonst
# caps it at difficulty (level_difficulty() + u.ulevel) / 2 = 3-4 on Dlvl 1, while exper.c:14 newuexp doubles
# the exp needed per level (320 -> 640 -> 1280 for XL 6 -> 7 -> 8). So XL 6 -> 8 is ~10k grind turns (sam
# replays: XL 6 at T7-9.6k, XL 8 at T15-23k), fed by ~5-7 hunger prayers (sam starts with no food, u_init.c:132).
# Each prayer resets the timeout to rnz(350) (pray.c:1220) and a Weak prayer fails while it is > 200
# (pray.c:1819), ~2.3% per prayer at the 1200-turn gap: failures send the game into a starving rescue dive or
# a faint (sam 814007: fainted, eaten by an iguana at XL 5). 7 of 17 cached champion sam games died in the
# grind (0.02-0.05) while the dives scored 0.10-0.60. The Samurai is the best-armoured starter (splint mail,
# AC 4), intrinsically Fast (attrib.c:72) with d8+d2+Con HP per level (role.c:449, attrib.c:1000), i.e.
# ~55-69 HP at XL 6. XL 6 is where jawfish's own XP gate opens (REQUIRED_XL[6] = 6): the stairs dive still
# explores Dlvl 2-4 and then every level whose next depth needs more XL (6 at XL 6, 7 at XL 7) fully, on fresh
# levels where the same cap admits more and bigger monsters (exp, corpses, dwarves' picks: difficulty 4 <=
# (d + 6) / 2 from Dlvl 2). A lower XL also lowers that cap at every depth (a digger with a pick skips the gate
# and digs at once, as the arc-early-dig gene showed pick dives to be XL-insensitive).
SAM_DIVE_XL = 6
DIVE_TURN = 10 ** 9
ELBERETH_REST_BELOW = 0.4
ELBERETH_REST_UNTIL = 0.85
# breathers, spitters and casters: Elbereth doesn't stop them hurting you from a distance
RANGED_MONSTERS = frozenset((
    'winter wolf cub', 'winter wolf', 'hell hound pup', 'hell hound', 'red naga', 'black naga',
    'golden naga', 'guardian naga', 'cobra', 'lich', 'demilich', 'master lich', 'arch-lich',
    'kobold shaman', 'orc shaman', 'gnomish wizard', 'energy vortex', 'yellow light', 'black light',
    'mind flayer', 'master mind flayer', 'titan', 'couatl', 'ki-rin', 'Aleax', 'Angel',
    'nalfeshnee', 'pit fiend', 'balrog', 'djinni', 'storm giant'))
# XP farm: under-levelled on a fully explored level, wait on the up staircase (escape route)
# for random spawns instead of descending. 0 disables.
FARM_TURNS = 0
# retreat: below this HP fraction, with prayer/Elbereth unavailable, climb the up stairs if close
RETREAT_BELOW = 0.35
RETREAT_MAX_DISTANCE = 12
ARRIVAL_WATCH_TURNS = 30       # how long after arriving a crowd still sends us back up
# ...or when this fraction of max HP was lost within RETREAT_WINDOW turns
RETREAT_FAST_LOSS = 0.3
RETREAT_WINDOW = 3
# arrival check: back up the stairs at once into a crowd (a b1 dive died 12 turns after walking
# onto Dlvl 12 among a leocrotta, ogre king, soldier ants, killer bee, Woodland-elf, ...)
CROWD_SIZE = 3
CROWD_RADIUS = 6
BOSS_MONSTERS = ('minotaur', 'ettin', 'titan', 'lich', 'demilich', 'master lich', 'arch-lich',
                 'purple worm', 'green slime', 'cockatrice', 'Olog-hai', 'ogre king', 'soldier',
                 'sergeant', 'lieutenant', 'captain', 'mind flayer', 'master mind flayer',
                 'energy vortex', 'black dragon', 'red dragon', 'white dragon', 'blue dragon',
                 'green dragon', 'yellow dragon', 'orange dragon', 'silver dragon', 'gray dragon')
ARRIVAL_RETREAT_REST = 150     # turns to wait upstairs before trying that staircase again
# ARRIVAL_FIX: the crowded-arrival retreat almost never fired (31 retreats in 2678 dive stairs descents over
# base/base2/base3/dsafe runs). descend() recorded the arrival AFTER agent.move('>'), but move() ends an atomic
# operation whose update_state() runs the preempt checks, so anything preempting on arrival (fight2 with monsters
# in view: exactly a crowd) raised AgentChangeStrategy out of move() and the arrival was never recorded
# (base3-jf14 s7: down the stairs to Dlvl 10 into Green-elves, a Grey-elf and a straw golem, dead 3 turns later,
# no retreat). And the '<' we arrive on shows us, not the stairs, so the retreat found no up staircase either.
# With the fix: the arrival is recorded by update() from a pending note taken before the move, and the square
# we arrived on counts as the up staircase.
# ON (train 3): castle A034 v3 (74ed55b), bug B005: the crowded-arrival retreat fired 31 times in 2678 stair descents
ARRIVAL_FIX = True
FULL_EXPLORE_TURNS = 2500      # per level, while under-levelled
PORTAL_SWEEP_TURNS = 3000      # per portal level visit
STUCK_EXPLORE_TURNS = 4000     # searching for a hidden way down before trying other things
# Digging down (pick-axe, or a wand of digging): a few turns per level instead of hundreds spent
# finding the '>', and the only way past Medusa's island without levitation (4 of 8 s4 dives ended
# on her level, walking into the water). A '>' this close is still taken (it keeps an up staircase
# under us on arrival).
DIG_STAIRS_RADIUS = 8
# Dig before fighting: fight2 engages anything within 7 squares, but a hole takes a dwarf only 3-4 dig
# steps and a monster interrupts the dig only once it attacks or first comes into view (monmove.c
# disturb, mhitu.c): with no hostile within DIG_FIRST_RADIUS, keep digging out instead.
DIG_FIRST = True
DIG_FIRST_RADIUS = 2
DIG_FIRST_MIN_HP = 0.35
# Elbereth before each dig step when hostiles are in view: a monster that respects it neither attacks
# nor, on first sight, interrupts the dig (monmove.c disturb() checks onscary). Digging the pit wipes it
# (dig.c del_engr_at), so it is engraved again from inside the pit (can_reach_floor allows it there).
# The early dig-dive deaths are crowd landings: ogre lord + owlbear, titan + warhorse, soldier ants...
ELBERETH_DIG = False
# dig under Elbereth: engrave before digging when an Elbereth-respecting hostile is within ELBERETH_DIG_RADIUS,
# and keep digging (not fighting) while every monster within DIG_FIRST_RADIUS respects it; after
# SHELTERED_DIG_STUCK turns on one level (and nothing hurting us) dig regardless
SHELTERED_DIG = True
SHELTERED_DIG_STUCK = 200
# engrave Elbereth before every hole, monsters in view or not (CleverShovel's Archeologist dig-dive, 0.41):
# whatever arrives during the ~4 dig turns then can't melee us
ELBERETH_ALWAYS = False
ELBERETH_DIG_RADIUS = 6
# DIG_ESCAPE: with a digging tool, the hole is the way out of nearly every fight above the castle. In the 90
# HEAD baseline games (base-*), 20 of the 34 tool dives whose castle depth is known died above it. Most of
# them fought instead of digging:
#  * ravens on Medusa-3 (7 games). Every square of its arrival island borders water, and dig_first
#    accepted only dry squares, so fight2 took over; the ravens blind you and swarm.
#  * snakes and cobras on Medusa-4. The cobra counted as 'ranged', which blocked sheltered digging.
#  * the titan on Medusa-2. Titans cast only in melee range (buzzmu() does nothing for AD_SPEL), and
#    there Elbereth stops them.
#  * a treasure zoo, and landing in a crowd.
# With DIG_ESCAPE:
#  * dig_first judges our square by the level's driest square, like try_dig_down;
#  * only an ADJACENT @ or minotaur (onscary() ignores Elbereth for them) makes us fight instead;
#  * there is no HP floor (emergency prayers preempt dig_first anyway);
#  * Elbereth is engraved before every dig step on Medusa's level (hidden snakes, 30 fast ravens);
#  * digging the pit erases the engraving (maketrap -> unearth_objs -> del_engr_at), so the engrave cap is
#    counted separately before and after the pit. A dust Elbereth gets a typo 28% of the time (engrave.c:
#    1 in 25 per letter), so 3 tries per square left about 1 hole in 5 finished with no Elbereth
#    (base-jf25 s0: bitten to death in its pit on Medusa-4).
# ON (train 2): dive-safety A2, 45 games divergence +0.65; Medusa-4 3-4/7 vs 1/7, Medusa-3 2/7 vs 0/7
DIG_ESCAPE = True
ELBERETH_TRIES_ESCAPE = 4      # engravings per square and dig phase (before / after the pit)
ELBERETH_TRIES_BLIND = 6       # the same while blind, each after being hurt since the last one
# DIVE_REST: a digger rests only below DIG_REST_BELOW, never to 95% before stairs, never while its pit is
# half dug, and on Elbereth. At XL 8 HP comes back at 1 per 5 turns (allmain.c), and the deep rests were
# fatal:
#  * base-jf25 s3 rested at 61% in its pit, 4 dig turns from the hole, and met a dwarf mummy and a xorn;
#  * base-jf25 s13 rested 180 turns at a '>' on Dlvl 14 (soldier ant, yellow light, fire vortex);
#  * base-public s12 rested at 62% on Medusa's level (pit viper).
# ON (train 2): dive-safety A2
DIVE_REST = True
# Medusa's level: the first Dungeons-of-Doom level at depth >= 21 showing this many pool/water squares
# (Medusa's maps are ~70% moat; the castle comes later and is found undiggable). Shared with the castle work.
MEDUSA_MIN_DEPTH = 21
MEDUSA_WET_SQUARES = 30
# MEDUSA_TITAN_DETECT: Medusa-2 drops a fall (and the '<') into its closed, unlit west room (medusa.des REGION
# (02,03,05,16) unlit, walls at x 1 and 6), so the 30 wet squares are never in view and the level went unrecognised:
# no Medusa-level digging rules there, no below_medusa() for the mazes under it (WAND_RESERVE's arrival zap). The
# room holds an awake titan (02,05) -- infravisible, so a dwarf sees it in the dark -- and a titan is never a random
# monster for an XL-7..9 diver (monstr 20 > (depth + XL) / 2 <= 18, makemon.c): one in view at depth >= 21 means
# Medusa-2. (jf41 s5/s13 with WAND_RESERVE: undetected on arrival by stairs, a boulder the titan threw into the dig
# pit, killed by the room's gremlin with the wand still reserved.)
MEDUSA_TITAN_DETECT = True
# jf_config.MEDUSA_TITAN_MSG: the titan named in a message (a word: 'titanothere' is no titan)
_TITAN_MSG = re.compile(r'\btitan\b')
# DIG_BOULDER_FIX: a boulder that lands in our dig pit ('The boulder falls into the pit with you': a titan's throw,
# a rolling boulder trap; trap.c flooreffects leaves it on our square) blocks the dig for good (dig.c dig_check:
# "There isn't enough room to dig here"), but that message comes a step after the apply's 'You continue digging
# downward.', so dig_with_tool never saw it and re-applied on the spot every turn: 106-157 turns of it until death in
# cmp-s22-jf42 s3 (Medusa-2, an invisible gremlin), base10arm-jf14 s6 (140), base10arm-jf25 s11 (108), and the
# WAND_RESERVE games jf41 s5/s13. With the flag the message history since the last apply on this square is checked:
# the square is given up (a bad dig spot) and the dig goes on elsewhere.
DIG_BOULDER_FIX = True
# MEDUSA_BLIND_HOLD: blind on Medusa's level (a raven's claw, a cobra's spit), don't walk and don't fight: hold the
# square on Elbereth until sight returns, then walk and dig as usual. Blind, the island's ravens are unseen: each
# step into one is 'Wait! There's something there you can't see!' -- a lost turn, and an attack (uhitm.c
# attack_checks -> wakeup -> setmangry) that erases our own Elbereth ('You feel like a hypocrite'); fight2 wields the
# long sword and swings at air. Harness Medusa-3 pick-only (mm-m3-pick): 15 of 16 deaths were blinded, 6 fought blind,
# most walked blind to the driest square. A blind dust Elbereth keeps its 8 letters only (24/25 * 10/11)^8 = 34% of
# the time (engrave.c: 1/25 per letter in dust, 1/11 more when blind) and can't be read back, so it is written again
# whenever we were hurt since the last one; a raven's blinding lasts d4 turns per claw (mhitu.c AD_BLND), so once it
# holds, sight comes back within a few turns.
MEDUSA_BLIND_HOLD = True
MEDUSA_BLIND_TRIES = 20        # blind engravings per square before giving the hold up
MEDUSA_BLIND_HOLD_MAX = 40     # ...and turns of one blind spell (a cobra's spit keeps us blind from range)
# MEDUSA_BLIND_DIG: blind on Medusa's level, keep digging where we stand -- in our own pit, or on a square the dig would
# pick anyway -- instead of waiting the blindness out. With no monster in view (blind, the snakes are unseen until
# one bites) dig_first stood aside and global_logic's wait_out_unexpected_state rested '.' turn after turn: the M4
# pick-only harness (mm-bh-m4-pick / mm-m4p-bh14) spent blind spells of 5-25 turns (a cobra spits from up to 7
# squares away, rnd(25) turns per hit, mthrowu.c) with 0 applies in 25 of 29 of them, 9 of them in a pit whose hole
# was 2 dwarf dig turns away (dig.c: effort doubles per call), then took the bites that killed 11 of 20 wet-islet
# losses. Digging needs no sight; a bite or spit only interrupts the turn it lands (stop_occupation), and the blind
# Elbereth before each dig phase is rewritten after every hurt (see _elbereth_before_digging_escape). Also: when
# sight returns, the square is looked at again before any new Elbereth -- the blind look had read no engraving, so a
# good one was wiped and rewritten (1 turn in a cobra's line, and a 28% typo chance).
# NEUTRAL (parked): harness M4 catalog seeds +2/+1 of 46, M3 -2/-1 of 49 (R138): a blind flood crawls out blind, and
# under constant attack the dig makes no progress anyway.
MEDUSA_BLIND_DIG = False
# MEDUSA_LIFT: on Medusa's level, stranded where no dry square is reachable on foot, a KNOWN ring of levitation that
# isn't known cursed (one free ring finger) floats us over the moat to the nearest dry square -- no moat in its 3x3, so
# fillholetyp can't flood the pit or the hole -- where the ring comes off again ('You float gently to the ground') and
# the pick digs as on any floor. On foot the wet islets pass 36-57% even with no monster about (F074, mm/pick: every
# flood turns the dug square into moat and the islet shrinks); the float is 1-14 steps to Medusa-4's NE hut and 7-12 to
# Medusa-3's dry land beside the south building (mm/pick/floatdry.py). Never a potion: it can't be ended at will (a
# blessed one aside) and it is the castle's lift. Levitating we can't engrave (the ravens get their attacks) and an eel's
# wrap still drowns us (mhitu.c AD_WRAP has no levitation check), so the float goes straight there. A ring that turns
# out cursed is prayed off (pray.c TROUBLE_CURSED_LEVITATION). Lead: kit-builder F072 -- with ALTAR_PICKUP a known
# ring of levitation reached a Medusa-3 level and went unused (arm cfpd-s13, died to the ravens).
MEDUSA_LIFT = True
MEDUSA_LIFT_MAX_STEPS = 25     # farthest dry square worth a float
# MEDUSA_POTION_LIFT (off, deep-arrivals F097 follow-up): stranded on a wet islet as MEDUSA_LIFT means it, with no ring
# of levitation and none of the flood reducers left (a known wand of digging with charges, a known cold source, a
# scroll of charging for an empty wand of digging), quaff potions that may be levitation, one at a time: a known potion
# of levitation first, then the unknown ones most likely to be it. First an intact Elbereth under us and nothing
# adjacent that ignores it (sleeping/paralysis are 25-34 helpless turns; ravens, snakes and eels respect it). A lift
# floats us to MEDUSA_LIFT's dry square and hovers there until it ends ('>' first: a blessed potion ends at will); then
# the pick digs on dry floor. Confused, stunned or hallucinating from one of these quaffs, the square is held until it
# clears (a confused apply digs a random way; hallucinated '@'s and minotaurs draw fights) while HP >= 50%. Why: the
# pick-only wet islets pass 36-57% on their flood lottery; ~17% of arrivals carry an unknown potion of levitation
# (4.6% of the ~4 unknown potions a Medusa arrival carries; objects.c probabilities, cand-g kits).
# REJECTED (keep off), harness F099: dive-medusa + 4 unidentified potions drawn from objects.c (potkits.py), flag off
# vs on, same kits: Medusa-4 (121 catalog seeds x 2 kits) 132 -> 75 of 242 passed, Medusa-3 (49 x 3, with
# MEDUSA_POTION_LIFT_M3) 37 -> 28 of 147.
# An uncursed potion's lift can't be ended, so the hover over the dry square lasts 10-149 turns with no Elbereth to be
# had: 0 of 19 hovers survived M4's 14 snakes and 8 nagas (the 2 blessed potions landed at once and passed). The
# other effects kill too (M4 off -> on): sleeping 11 -> 2 of 20, paralysis 9 -> 3 of 27, blindness 10 -> 0 of 22,
# hallucination 9 -> 0 of 22, confusion/booze 17 -> 6 of 37 -- the Elbereth under us leaks on Medusa-4: a cobra's spit
# that misses us blinds the monster next to us ('The killer tomato is blinded by the splash of venom') and a blind
# monster ignores Elbereth (monmove.c onscary: !mcansee). Kits where nothing was quaffed: 46 -> 46 of 48.
MEDUSA_POTION_LIFT = False
MEDUSA_POTION_LIFT_M3 = False  # ...on Medusa-3 too (MEDUSA_LIFT's ring floats lost 37 of 39 floaters to its ravens)
MEDUSA_POTION_LIFT_HP = 0.5    # no quaff below this share of max HP
MEDUSA_POTION_HOLD_MAX = 1100  # turns a quaff's confusion/stun/hallucination is held at most (cursed hallu: 900-1099)
# MEDUSA_MAP: on Medusa's level, a square the bot never saw (level.objects -1) takes its terrain from the variant's
# fixed map (medusa_maps.py, dat/medusa.des; the variant is the one whose moat agrees with what we saw). A square first
# seen under a monster stays unknown and even walkable in agent.update_level, and on the islands those are the moat
# squares eels, jellyfish and ravens sit on: the dig took a square beside one for dry and flooded (MEDUSA_LIFT smoke
# jf14 s24, g6 jf14 s2). MEDUSA_UNKNOWN_WET counted every unknown neighbour as moat (neutral on M3: it also spoiled
# squares beside unseen land); the map counts only the real moat. MEDUSA_LIFT uses it for floats toward floor we
# never had in view (Medusa-4's NE hut from the other islets).
MEDUSA_MAP = True
MEDUSA_MAP_MIN_SEEN = 40       # known squares compared before a variant is trusted
MEDUSA_MAP_MAX_BAD = 0.03      # ...and the largest share of them that may disagree
# MEDUSA_DIG_ORDER: on Medusa's level with no dry square to dig from, of the squares with the fewest moat neighbours
# dig the one with the best flood-only expectimax value, not the nearest. A flood turns the dug square into moat, so
# the first choice decides which good squares are left: on Medusa-3's island two of its four k=1 squares are worth
# 0.56 and two 0.46-0.48 (the rest of the dig included), on Medusa-4's west islet one k=2 square 0.28 and three 0.36
# (mm/pick/optdig.py, optshow.py). Flood-only Monte Carlo from the landing squares (mm/pick/dpcheck.py): Medusa-3
# nearest 0.539 -> 0.568, Medusa-4 west islet 0.345 -> 0.392, south islet 0.402 = 0.403; ~4 more steps walked.
# Harness pick-only (dive-medusa, M3+M4 catalog seeds jf14/16/40/41/42, 248 episodes): off 99, MEDUSA_MAP 100,
# MAP+DIG_ORDER 106 passed (jf14/16 44 vs 39; jf40-42 62 vs 60) -- a small lean, not significant (paired vs off
# 29 gained / 22 lost): the walks to the better square cost bites.
MEDUSA_DIG_ORDER = False
MEDUSA_DIG_ORDER_DEPTH = 5     # floods looked ahead
MEDUSA_DIG_ORDER_MAX = 40      # islet squares at most (bigger land always has a dry square anyway)
# MEDUSA_FREEZE: on Medusa's level, freeze the dig square's moat neighbours with a known wand of cold (or frost horn)
# before digging. A cold ray turns every moat square on its path into ice (zap.c zap_over_floor ZT_COLD: typ = ICE,
# melting no sooner than MIN_ICE_TIME = 50 turns), and fillholetyp counts only moat, pool and lava squares -- so with
# no wet neighbour left the pick's two flood rolls (or a wand of digging's one) always come up dry. One zap per wet
# neighbour (each lies in its own direction from us); the ray also hits what flies or swims on its path (6d6 cold, a
# raven has ~4d8 HP), and a Valkyrie is cold resistant if it bounces back. Medusa-3's raven island has no dry square
# (k = 1-6 moat neighbours; flood-only Monte Carlo on its map, mm/m3mc.py: 52% for a pick digger, 84-90% with a wand
# of digging): from a k=1 square a single zap makes the hole certain.
MEDUSA_FREEZE = True
MEDUSA_FREEZE_MAX = 8          # cold zaps per Medusa level
MEDUSA_FREEZE_BLIND = True     # ...also while blind, toward the water the map remembers (see _freeze_action)
# MEDUSA_UNKNOWN_WET: on Medusa's level a never-seen square next to a dig candidate counts as moat when choosing where
# to dig. agent.update_level leaves the terrain of a square first seen under a monster unknown (-1, and even marks it
# walkable), and on Medusa's islands those are the water squares ravens, eels and jellyfish sit over: the dig choice
# took them for dry (g6 jf14 s2: a wand hole flooded from a neighbour a raven had covered since the landing).
MEDUSA_UNKNOWN_WET = False
# COLD_RESERVE: fight2 never zaps a known wand of cold at a monster (fight_heur.get_potential_wand_usages) unless
# below WAND_RESERVE_HP with a hostile within 2 (_reserve_emergency), and not before Gehennom -- the wand is kept for
# MEDUSA_FREEZE and the castle crossing's ice (castle_logic._cold_step). With WAND_ENGRAVE_TEXT the engrave test names
# cold ('A few ice cubes drop from the wand.': 6 of 30 g3 games on jf41/jf42, 0 before), and fight2 then spent it on
# leprechauns, ponies and giant ants on Dlvl 1-3 (mm-g3-jf42 s3/s11): 0 MEDUSA_FREEZE firings in those 30 games.
COLD_RESERVE = True
# DIG_TRY_FIX: DIG_MAX_TRIES counts only applies that weren't interrupted. On jf27 s3 Dlvl 9 a bat interrupted
# all 20 applies ("You stop digging"), and a diggable level was marked undiggable.
# ON (train 2): dive-safety A2
DIG_TRY_FIX = True
DIG_MAX_APPLIES = 80           # hard cap on all applies on one level (interrupted ones included)
DIG_WALK_RADIUS = 6            # DIG_ESCAPE walks this far (BFS steps) to a square it can dig
MEDUSA_WALK_RADIUS = 30        # ... and this far on Medusa's level (ESCAPE_V2)
# ESCAPE_V2 (with DIG_ESCAPE): the refinements found after A2 (train 2):
#  * dig_first keeps acting while the escape applies. Returning after one action let one step of the strategy
#    underneath leak in before this hook fired again: fight2's swing from our fresh Elbereth (grind-combat's
#    HOLD_LOOP finding);
#  * an eel/kraken that wraps us drowns us with its next hit: Elbereth at once (monflee releases the hero);
#  * with monsters adjacent, engrave before walking to a dig square (a scared monster flees, not attacks);
#  * on Medusa's level walk up to MEDUSA_WALK_RADIUS to a dry dig square rather than gamble on a flood.
# ON (train 3): dive-safety A3, +0.233 pooled over 45 arm64 games vs base3arm, no new failure mode
ESCAPE_V2 = True
DIG_ESCAPE_ACTIONS = 40        # dig_first's escape acts at most this often per activation (see dig_first)
MEDUSA_REROLL_WET = 2          # stranded where every square has this many moat neighbours: climb and re-fall
MEDUSA_REROLLS = 3
# MEDUSA_SKIP (with DIG_ESCAPE): reroll from Medusa's level (also Medusa-3) once a dig there has flooded
# MEDUSA_SKIP_FLOODS times or the best square has MEDUSA_REROLL_WET moat neighbours; each hole dug from the level
# above skips Medusa's level 1 time in 4 (see _medusa_reroll_stairs)
MEDUSA_SKIP = False
# MEDUSA_HOP: on Medusa's level (variant known, MEDUSA_MAP) with no dry diggable square on our own land, step into a
# one-square moat channel toward land that has one. trap.c drown(): an unprotected hero in water crawls out at once to a
# random free land square next to it (crawl_dest_ok), so the channel is crossed with odds (far-side squares) / (all
# land squares around it), and a miss lands us back where we were. Costs a soaking (potions dilute, scrolls blank,
# iron rusts). Medusa-4: 33% of arrivals land on the islet next to the north-east hut (6 dry squares) and cross with
# 3/4 odds per try; Medusa-3's island has no such channel (see scripts/medusa_hop.py in the workspace).
MEDUSA_HOP = False
MEDUSA_HOP_MAX = 6             # soakings per Medusa level
MEDUSA_SKIP_FLOODS = 1
MEDUSA_SKIP_REROLLS = 8
MEDUSA_SKIP_FIRST = False      # on Medusa-3 (ravens) reroll at once, before any dig, while the '<' is close
MEDUSA_SKIP_FIRST_STEPS = 4
# MEDUSA_HOLE_CYCLE (with DIG_ESCAPE): a hole we dig drops exactly one level (dig.c digactualhole: dlevel + 1 -- so
# MEDUSA_SKIP's reroll never skipped her level), but stepping into an existing hole or trap door falls 1 + Geom(1/4)
# levels (trap.c fall_through: newlevel++ while !rn2(4)), with no stop at Medusa's level. Every variant puts her
# level's '<' inside or beside the region a fall lands in (medusa.des '<' / fall region: medusa-1 (5,14) / (1-5,1-17),
# medusa-2 (4,9) / (2-5,3-16), medusa-3 (32-39,1-7) / (33-38,2-7), medusa-4 (67-74,1-20) / (64-74,1-17)). So on her
# level climb her '<' at once (ravens, snakes and titans don't follow: mon.c levl_follower wants M2_STALK), dig one hole
# beside the '>' up there (that fall lands us on her level again), and from then on follow every climb by a step into
# that hole: 1 time in 4 the fall carries us past her level, 3 in 4 it lands us beside her '<' again. No dig on her
# islands (no flood roll), a few steps among her ravens per landing, a rest up there between landings. Medusa passes
# before (devruns tr0/p0/v2a/mt*/a*): medusa-1 86%, -2 51%, -3 40%, -4 58%.
# REJECTED (mcb0/mcb1, 98 deterministic pairs of Medusa games): -0.0147 +- 0.0050 per game, 6 better / 20 worse. The skip
# works (7 of 42 plunges, 17%), but the extra landings and the rests up there cost more: passes medusa-1 12 -> 7 of 15,
# medusa-4 16 -> 10 of 28, medusa-2 11 -> 12 of 21, medusa-3 12 -> 11 of 24. (Also: a seen hole is escaped 1 time in 5,
# trap.c dotrap -- the plunge should then use '>' (TOOKPLUNGE) instead of giving the hole up.)
MEDUSA_HOLE_CYCLE = False
MEDUSA_HOLE_CYCLE_MAX = 20          # climbs off Medusa's level
MEDUSA_HOLE_CYCLE_STEPS = 16        # her '<' this many BFS steps away at most (else the usual dig)
MEDUSA_HOLE_CYCLE_HOLE_STEPS = 10   # our hole up there this many steps away at most (else the dive digs one)
MEDUSA_HOLE_CYCLE_REST = 0.9        # rest up there to this HP fraction before the plunge (or a new dig)
# RAVEN_CYCLE: Medusa-3's raven island. Every island square borders water, so a pick-axe hole succeeds only
# 1/(n+1)^2 (1 in 4 at best, n = 1) and each flood drowns a square of the island: a pick digger needs 2-4
# tries (Monte Carlo on the map: 44% by the 2nd, 51% by the 4th, 65% at most), and the 30 ravens (speed 20,
# blinding claws) kill an XL-8 digger in ~20-40 turns -- 32% of 115 real Medusa-3 arrivals got through vs 68% on
# the other variants. Ravens don't follow up stairs (no M2_STALK), and the island's '<' is a few steps away:
# climb after a flood, below RAVEN_CYCLE_HP or blind and bleeding, rest upstairs to RAVEN_CYCLE_REST with sight
# back, and come down for a fresh try at full strength.
# OFF (rejected): harness dive-medusa, 20 Medusa-3 seeds: off 6/20 past the level; v1 (climb after a flood / low
# HP, came back by digging) 3/20; v2 (stair-dance: hold the '<', dig only when quiet) 1/20 (med3-cyc3, after the
# 7f470c7 dig fix; the first v2 run's 0/20 had digging disabled) -- the ravens camp at the island's '<', and every
# return at ~75/80 HP lost ~40 HP and its sight on the first step.
RAVEN_CYCLE = False
RAVEN_CYCLE_HP = 0.5
RAVEN_CYCLE_REST = 0.9
RAVEN_CYCLE_STEPS = 8
RAVEN_CYCLE_MAX = 8
RAVEN_DANCE_RADIUS = 5         # hostiles this close send us to the '<' to hold it
RAVEN_BLIND_HP = 0.7           # blind on the '<': climb below this
RAVEN_HOLD_IDLE = 6            # turns with nothing next to us / hurting us before the dig plan resumes
# DROWN_GUARD (with DIG_ESCAPE): a hole that fills with water drops us into it, and we crawl out only to a free
# adjacent land square (trap.c drown -> hack.c crawl_destination: no monster on it, no diagonal squeeze); with none
# we drown on the spot. base3 lost 7 Medusa digs this way ('You sink like a rock', no 'try to crawl out'). Dig a
# wet square only with DROWN_GUARD_EXITS such squares around it: walk to one if the ground is the problem, wait
# on Elbereth up to DROWN_GUARD_WAIT turns if monsters are.
DROWN_GUARD = False
DROWN_GUARD_EXITS = 2
DROWN_GUARD_WAIT = 15
# DIG_WAND_ESCAPE (with DIG_ESCAPE): a known wand of digging, zapped down, holes the floor at once (zap.c
# zap_dig -> dighole: one flood roll, no pit phase). Used where the pick-axe can't work -- an adjacent @, or
# bitten with no Elbereth to be had -- and on Medusa's wet islands. (power: 4% of XL-8 castle arrivals carry
# one, 13% at XL 10; try_dig_down only ever zapped it with no pick-axe in the pack.)
# ON (train 2): dive-safety A2
DIG_WAND_ESCAPE = True
# WAND_FIRST (with DIG_WAND_ESCAPE): with hostiles in view, zap a known wand of digging down on a dry square
# before digging with the pick (instant hole), and as the escape of a dive that has only the wand
# ON (train 3): harness Big Room with a wand only 7/7 vs 5/7; inert without a known wand of digging
WAND_FIRST = True
# WAND_RESERVE (with DIG_WAND_ESCAPE): the known wand of digging is kept for Medusa's level and the mazes below it.
# 18 of 90 cmp-main games (99e4eb7) carried one, but WAND_FIRST zapped it at whatever hostile was in view from Dlvl 2
# on -- newts, geckos, kittens, lichens, floating eyes, 3-9 zaps a game (a wand holds 4-8, zap.c rn1(5,4)) -- where
# the pick-axe digs as safely under Elbereth; by Medusa it was empty. Below it the wand is worth most:
#  * Medusa's level: every raven-island square borders the moat, and a pick-axe hole rolls fillholetyp twice (the
#    pit, then the hole: P(no flood) = 1/(k+1)^2 for k moat neighbours) over ~8 exposed actions; a zap rolls once
#    (dig.c zap_dig -> dighole: 1/(k+1)) and holes the floor at once. Medusa-3 passes 43% of 77 distinct arrivals in
#    the recent trains (ravens 31, drowned 11 deaths), the other variants 72% of 109 (drowned 10, snakes 15);
#  * the filler mazes between Medusa and the castle hold 0-2 minotaurs each (mkmaze.c rn2(3)), which ignore Elbereth
#    and stop the dig with every attack: 17% of 190 distinct maze visits died (25 minotaurs), the minotaur's first
#    sign came 0-8 turns after the landing and a pick-axe hole takes ~6 turns; a zap takes one.
# Above Medusa (or before her level is known) a reserved wand fires only in an emergency: below WAND_RESERVE_HP with
# a hostile within 2, next to an @/minotaur, or bitten with no Elbereth to be had (no pick-axe: as before).
WAND_RESERVE = True
WAND_RESERVE_HP = 0.5
# ...and on a level below Medusa's the zap comes on arrival, unless the level may be the castle: its bottom floor
# can't be holed (Can_dig_down false: the zap only digs a pit), and its charges are castle_power's/CFP_ZAP's there.
# The castle's falls land in levregion(1,0,10,20) (castle.des), bot x <= 9, a filler maze's anywhere (u_on_rndspot
# with no region); from a landing at x <= 9 the zap waits WAND_CASTLE_WAIT turns for the castle's door sounds
# (soldiers: 'You hear a door open.', castle-first-pass CFP_SENSE: 101 of 101 castle arrivals, 1 of 185 filler
# levels) and the pick-axe digs meanwhile (the pit, then 'too hard to dig', tells the castle too). 4, not 3: the first
# sound often comes at exactly +3 (castle-first-pass).
WAND_CASTLE_WAIT = 4
# Resting to 95% on a deep level lets its monsters come to us (an s7 dig-dive rested for 150 turns on
# Dlvl 15 until a leocrotta took it to 2 HP): with a digging tool, rest only below this.
# arc-early-dig (daglar c131c7c EARLY_DIG_XL 3; Komershan/DT6A 5): an Archeologist starts with a pick-axe
# and leaves the Dlvl-1 grind for the dig-dive at this XL, keeping the pick through the tour. Random
# monsters are capped at difficulty (level_difficulty() + u.ulevel) / 2 (makemon.c mkclass/rndmonst), so a
# low-XL digger meets weaker ones; a sheltered hole costs a few turns per level. None: DIG_DIVE_XL.
# zarutskiysy A/B a3/a5 (4 Archeologist types x 60 held-out seeds): XL 5 beats XL 3 by +0.024 +- 0.013 (101 wins / 77
# losses): the Dlvl-1 grind to XL 5 is short and safe, and ~45 HP survive the Dlvl 9-15 elves and Medusa better than ~25.
# XL 6 lost (-0.063 +- 0.028): GRIND_LEVELS sends XL 5+ to Dlvl 3 before that dive.
ARC_DIG_DIVE_XL = 5
# rog-early-dig (port of DT6A d956abf global_logic: EARLY_DIG_XL + ItemPriority.add_pick -- any pick holder leaves the
# grind for the dig-dive, and the pick is always kept). Rogue-gated. A Rogue's Dlvl-1 grind meets its digging tools
# at XL 7 (dwarf difficulty 4, monst.c:421, vs the random-monster cap (1 + XL) / 2 on Dlvl 1, makemon.c:1550; 3 in 8
# dwarves carry a pick-axe or mattock, makemon.c:372-386; lawful dwarves are always hostile to a chaotic Rogue,
# makemon.c:2024 peace_minded) but then waited for XL 8 with the tool on the floor: XL 7 -> 8 takes a Rogue 5,000-7,000 turns of
# prayer-fed grind (no starting food, u_init.c Rogue[]), and a sixth of its games die there at Xp 7 (.051).
# The threshold is the Archeologist's (ARC_DIG_DIVE_XL, confirmed) instead of DT6A's hand-set 5: a Rogue has the
# same hit dice (role.c:30 vs :323, 10-11 + d8 per level), digs as fast per turn (dig.c:299 effort 10 + rn2(5) +
# abon(); Dex 16-18 gives abon +2..+4, weapon.c:911-919, where the Archeologist's intrinsic Fast gives 4/3 actions,
# allmain.c:141-144, attrib.c:26-27) and has the same Stealth (attrib.c:68). None: DIG_DIVE_XL.
ROG_DIG_DIVE_XL = 3
DIG_REST_BELOW = 0.6
DIG_MAX_TRIES = 20             # applies on one level without falling through: floor can't be holed
# A cursed pick-axe digs like any other (dig.c: the curse only matters on the Plane of Earth); applied, it
# welds to the hand, which a dig-dive can live with. Without this, the first apply of an unidentified cursed
# pick welded it, digging_tool() then rejected it as CURSED and the dive went looking for another tool
# (eg-trip5 public seed 3: 'The pick-axe welds itself to your hand!' -> 'no digging tool at depth 1: early
# detour to the Mines' -> killed at XL 6 on Mines 1, 0.037; the same seed without the trip scored 0.507).
# A known-cursed mattock is still refused unless it is already welded: it takes both hands, leaving none
# to engrave Elbereth with (agent.hands_welded).
# ON (train 2): 56 of 74 games that welded a pick never dug again (0.163 vs 0.411)
CURSED_PICK_OK = True
# A dig try leaves a pit (dig.c: effort > 50 makes it and ends the occupation; the next apply digs on from
# inside it to a hole). Once a #terrain check had shown that pit on our square, the square no longer counted
# as plain floor and try_dig_down walked off to dig a fresh pit next door -- 'You are still in a pit', a
# panic, another forced #terrain that recorded the new pit... A jf25 s12 digger marched pit by pit across
# Dlvl 4-11, 20 tries per level, marked each one undiggable and took the stairs (~1000 turns a level; dead
# on Dlvl 12 at T29790). A pit under us is where the hole gets finished.
DIG_IN_PITS = True
PITS = frozenset({SS.S_pit, SS.S_spiked_pit})
# The budgets and task checks in plan_step (FULL_EXPLORE_TURNS, DWARF_SEARCH_TURNS, STUCK_EXPLORE_TURNS, the
# tool quest's stuck test...) only run between tasks, and one exploration run can last thousands of turns (public
# s9: a 2500-turn 'explore fully' budget overrun by 800). With EXPLORE_BUDGETS a dive exploration hands control
# back once its budget is spent, and at least every EXPLORE_REPLAN_TURNS turns.
# ON (train 2): robustness b2 guard 0.3480 vs 0.3479; +0.38 over the 4 games it fired in (b1)
EXPLORE_BUDGETS = True
EXPLORE_REPLAN_TURNS = 500
# standing on a known trap door that didn't trigger, press '>' (the dive asserted in a loop there before)
TRAPDOOR_PLUNGE = True
FETCH_TOOL_TURNS = 3000        # budget for walking back to a pick-axe the tour dropped
# Dwarves carry a pick-axe or a mattock 37.5% of the time (makemon.c) and are peaceful to a dwarf:
# with no digging tool yet, the dive kills the peaceful dwarves it meets in the Mines (never in
# Minetown: the Watch). Each kill costs Luck -1 half of the time, so prayer waits 600 turns per kill.
DWARF_HUNT = True
DWARF_HUNT_MAX_KILLS = 6
HUNT_IN_TOUR = False           # the tour hunts too (in the Mines) from HUNT_MIN_XL
HUNT_MIN_XL = 8
# With a digging tool the dive is a few turns per level and XL matters much less (s7 public seed 10:
# Dlvl 4 -> 26 in ~200 turns of digging, past Medusa): dive as soon as one is in hand from this XL,
# and keep one during the tour (it drops them for lighter loot).
# train 3 (early-game A027): a tool holder dives at XL 7 (public +0.056; 45 games neutral; grind losses 10 -> 5)
DIG_DIVE_XL = 7
KEEP_TOOL_IN_TOUR = False
# A planned dive (the XL or tool trigger, never a rescue) waits until this many turns have passed since the last
# prayer (0: off), so the emergency HP prayer is there on the way down: it needs the prayer timeout <= 200
# (pray.c), and the timeout is rnz(350) right after a prayer. eg-t2glh7 XL-7 tool dives whose first dig came
# < 800 turns after a prayer: 4 of 12 died above Dlvl 20 (Dlvl 10-17 crowds: soldiers and killer bees, elves, a
# centaur's wand of fire), mean 0.414; >= 800 turns: 0 of 13, mean 0.513. base/base2 XL-8 dives: 28% vs 14%.
# ON (train 3.1, early-game A044/A049): a tool holder digs only >= 800 turns after its last prayer; A049 0.399 vs base4 0.378, mid-dive deaths 1/32 vs 4/37
DIVE_PRAYER_GAP = 800
# ...and (DIVE_PRAYER_FOOD) not Hungry or carrying DIVE_PRAYER_FOOD_MIN nutrition: an HP prayer made while only
# Hungry fixes the HP but not the hunger (pray.c fix_worst_trouble: Hungry is a minor trouble) and restarts the
# clock, so the next hunger prayer is ~1100 turns away (rescue, ledger F032: 20-35 faints). The wait is capped at
# DIVE_PRAYER_MAX_WAIT turns (a Hungry grinder with no food would otherwise wait through every prayer cycle).
DIVE_PRAYER_FOOD = False
# ...only when a digging tool is already in hand (the dive digs at once). A tool-less dive heads for the Mines camp
# for thousands of turns, so its wait only reshuffles the hunt: eg-dpg lost 3 XL-8 tool-less dives that way.
# ON (train 3.1): the wait only with a digging tool in hand (A044)
DIVE_PRAYER_TOOL_ONLY = True
DIVE_PRAYER_FOOD_MIN = 400
DIVE_PRAYER_MAX_WAIT = 2500
# DIVE_START_HP: a planned dive (the XL or tool trigger, never a rescue) starts with at least this fraction of max HP
# (0: off); below it the grind goes on (and rests as usual) for up to DIVE_START_HP_WAIT turns. base5-8 dives that
# started below 50% HP all died within 300 turns (3 of 3: base6-8 jf14 s8 dove at 32/74 right after a hunger prayer
# and died on Dlvl 3 116 turns later), 50-80% scored 0.397 (13 dives), 95%+ 0.475 (269)
# t35: jf16 s1 0.507 -> 0.075 -- the 'wait' is the grind, which fought on from 66/84 to 11/84 and prayed at gap 607;
# a version would have to rest -- off
DIVE_START_HP = 0
DIVE_START_HP_WAIT = 300
# The portal sweep (Home 1 = 0.366) costs ~1500 turns of exploring the level; digging reaches
# Dlvl 20+ (0.38+) within a few hundred turns, so no sweep while holding a digging tool.
SWEEP_WITH_TOOL = False
# Tool run (off: None): end the tour's Dlvl 1 grind at this XL instead of XL 8 and head for the Mines
# to take a dwarf's pick-axe (HUNT_MIN_XL / DIG_DIVE_XL follow it). The XL 5-8 grind is where unseen
# games starve (9 of 30 died on Dlvl 1 at XL 3-7).
TOOL_RUN_XL = None
# Rescue dive: a failed prayer during the Dlvl 1 grind leaves the god angry and the game starving (92
# past games: median survival ~1,050 turns after the first failure, 24 of 40 first failures on Dlvl 1).
# Such a game dives at once: down the Mines (peaceful to a dwarf; Mines' End is Dlvl 10-13, 0.13-0.26),
# hunting dwarves on the way, digging in the main dungeon if it gets a pick-axe. Fires only in games
# that are otherwise lost.
RESCUE_DIVE = True
# The same later in the tour: a prayer failed (the god is angry or Luck < 0, so no more hunger prayers)
# and the character is Weak with nothing to eat. The tour would starve on the spot (a clock-jf6 XL8
# starved in the Mines 1700 turns after an unlucky prayer); the dive at least banks depth on the way.
LATE_RESCUE = True
# rescue dives take the main-dungeon stairs (not the Mines route): see should_dive
RESCUE_MAIN_DUNGEON = True
# Experiment: skip the Dlvl 1 grind altogether (38% of unseen games are lost there: prayer failures,
# fainting deaths) and dive from EARLY_DIVE_TURN on the rescue route -- down the Mines (peaceful to a
# dwarf), hunting dwarves for a pick-axe, digging the main dungeon once one is in hand. Even a failed
# early dive that reaches Dlvl 10-12 scores 0.13-0.21, above a grind death (0.02-0.07).
EARLY_DIVE = False
EARLY_DIVE_TURN = 1
# planned early dive from this XL (0: off): see should_dive
EARLY_DIVE_XL = 0
# Ditch the pet for the Dlvl 1 grind (off: experiment). On 15 unseen grinds the pet ate ~40% of the
# corpses (497 meals vs our 732) and made ~10% of the kills (no XP for us); food is what the grind runs
# out of (hunger prayers, their failures, starvation). Take it down to Dlvl 2 and come back up alone
# (a pet only follows when adjacent, and can't climb stairs on its own).
DITCH_PET = False
DITCH_PET_AFTER = 300          # turns into the game (Dlvl 1 explored, its '>' known)
DITCH_PET_BUDGET = 300
# Knights and Samurai ditch their starting pet for the grind (vendor/nethack-3.6.6/src), as soon as the
# Dlvl 1 '>' is known and reachable (no turn gate):
# - The pet's first hunger crisis is fixed by the source: dog.c:47 initedog hungrytime = 1000 + moves, and
#   dogmove.c:361 dog_hunger at hungrytime + 500 sets mconf, cuts max HP to 1/3 and prints 'X is confused
#   from hunger.' / 'You feel worried about X.'; mon.c:1368 mfndpos gives a confused monster ALLOW_ALL
#   (incl. ALLOW_U) and dogmove.c:578 appr = 0, so dog_move's newdogpos (dogmove.c:1189) picks our square
#   and mattacku()s. The ditch runs well before turn 1500.
# - Knight (role.c:211 PM_PONY): the pony is M1_HERBIVORE only (monst.c:825) and Dlvl 1 has almost no
#   vegetarian food, so it starves every grind; champion games died to their own pony at T1512-1596 (XL 2)
#   and to a grown warhorse (kick d10 + bite d4, speed 24, monst.c:851) at XL 4-7 on Dlvl 1 (5 of ~35
#   distinct knight dungeons in the cache). A pony that does find lichens grows into that warhorse and
#   takes the grind's kills: a monster killed by a pet gives us no experience (only xkilled calls
#   more_experienced, mon.c:2481; mhitm kills go through monkilled). One such knight was XL 4 at T7800.
# - Samurai (role.c:436 PM_LITTLE_DOG): starts with no food at all (u_init.c:132-140) and grinds on
#   corpses; the carnivorous dog ate 20-30 of them per replayed grind and made 10-15 of the kills, while
#   the samurai fainted 30-80 times. Its hunger-confused dog also bites (56 -> 24 HP in one replay).
# The pet left on Dlvl 2 is met again only at the dive (XL 8): after ~15k turns away dog.c:524-545
# mon_catchup_elapsed_time makes it wild, and a pony or little dog is harmless to an XL 8 fighter.
DITCH_PET_ROLES = frozenset((Character.KNIGHT, Character.SAMURAI))
DITCH_PET_TRIES = 3            # attempts when the pet didn't follow us down (keepdogs needs it adjacent)
DITCH_RETRY_WAIT = 400         # DITCH_RETRY: turns before another try after one ran out of time
# dogmove.c:361-395: confused at hungrytime + 500, starves at + 750; don't herd a pet that is in that
# window to the stairs (it bites an adjacent hero at random)
PET_HUNGER_WINDOW = 750 - 500
DWARF_HUNT_TURNS = 400         # per level
# a dive leaving the Mines without a digging tool explores each Mines level (not Minetown) this long
# looking for dwarves before climbing on (about 2 dwarves per Mines filler level, 37.5% armed with one)
DWARF_SEARCH_TURNS = 300
# Tool quest: a tool-less dive stuck this deep with no way down (Medusa's water, half of all dives)
# has banked its depth, so the long trip back to the Mines for a dwarf's pick-axe can only add.
TOOL_QUEST_DEPTH = 18
TOOL_QUEST_STUCK_TURNS = 800
TOOL_QUEST_TURNS = 12000
# Early detour: a dive with no digging tool that is still shallow (it started in Sokoban or the main
# dungeon, never climbing out through the Mines) first visits Mines levels 1-3 for a dwarf's pick-axe:
# ~50% find one, and a digger's dive is worth ~+0.1 over a stairs dive.
EARLY_DETOUR = True
EARLY_DETOUR_DEPTH = 12
EARLY_DETOUR_MINES_LEVELS = 3
EARLY_DETOUR_TURNS = 2500
DWARF_NAMES = ('dwarf', 'dwarf lord', 'dwarf king')
# Gehennom (jf_config.GEHENNOM_DIVE): with neither prayer nor Elbereth there, a wand of digging zapped down is
# the one-turn escape from a monster we can't outfight, and it banks the next level. Which monsters count:
# the maze fillers' minotaurs (0-2 per level; 116 of 550 games that reached Dlvl 25+ in our runs died to one)
# and anything of level >= GEHENNOM_THREAT_LEVEL, plus the instadeath/brain/digestion attackers.
GEHENNOM_THREAT_LEVEL = 10
# Valley walk: attack only what blocks the next step while nothing hurts us (valley_sneak). At XL 10 (15
# valley-x10 seeds) it walked into the crowds and died after 116 turns on average, vs 537 fighting them
# (fight2) where they come, and neither variant got out; but with a strong kit (power agent: XL 14, gray
# dragon scale mail, speed boots) 4 of 12 got past the '>' with it and 0 of 5 without.
VALLEY_SNEAK = True
# The Valley's '<' is the lifeline: below VALLEY_RETREAT_BELOW of max HP (or VALLEY_CROWD_BELOW with
# CROWD_SIZE hostiles close) with a hostile near and the '<' within VALLEY_RETREAT_REACH steps, climb up to
# the castle's east edge, where Elbereth and prayer work; the castle mode (castle agent) rests there and
# comes back down through the trap door behind the castle's back door (dive.valley_retreats / _turn tell
# it we came up to heal). 15 of 15 valley-x10 seeds (XL 10, 99 HP, AC -2) died in the Valley without it,
# 3 of them swarmed at the landing spot (vampire bats, vampire lords in bat form, mummies, ghosts).
VALLEY_RETREAT = True
VALLEY_RETREAT_BELOW = 0.5
VALLEY_CROWD_BELOW = 0.75
VALLEY_RETREAT_REACH = 15
# below REST_BELOW in the Valley, rest on the '<' when within VALLEY_REST_REACH steps, up to VALLEY_REST_UNTIL
VALLEY_REST_REACH = 25
VALLEY_REST_UNTIL = 0.9
# Below the Valley a digger rests only under this HP fraction (DIG_REST_BELOW elsewhere, and no 2/3-HP rest):
# a hole is ~5 turns and banks a level, while resting in place brings the level to us (a valley+1 dig-dive
# rested at 47% on Juiblex's level and was engulfed: 'You feel deathly sick' -- no prayer to cure it there)
GEHENNOM_DIG_REST_BELOW = 0.3
GEHENNOM_THREATS = frozenset(('minotaur', 'mind flayer', 'master mind flayer', 'cockatrice', 'chickatrice',
                              'green slime'))   # stoning and sliming: only prayer (none in Gehennom) cures them
# Gehennom: a scroll of scare monster on our square scares every monster but the temple priest, lawful minions,
# Riders and Rodney (monmove.c onscary: 'the scare monster scroll doesn't have any of the below restrictions'),
# so unlike Elbereth also in Gehennom and against @ and minotaurs; a scared monster neither attacks (dochug:
# !scared) nor stops our dig occupation (allmain monster_nearby: !onscary). When a fight goes badly (below
# GEHENNOM_SCARE_BELOW with a hostile adjacent, or GEHENNOM_SCARE_CROWD adjacent below GEHENNOM_SCARE_CROWD_BELOW)
# drop one (power.scare_scrolls) and hold on it: dig down from it, or in the Valley rest on it to
# GEHENNOM_SCARE_UNTIL. It stays behind (picked up again it turns to dust, pickup.c).
GEHENNOM_SCARE = False
GEHENNOM_SCARE_BELOW = 0.5
GEHENNOM_SCARE_CROWD = 2
GEHENNOM_SCARE_CROWD_BELOW = 0.8
GEHENNOM_SCARE_UNTIL = 0.9
# with no known scroll, drop every carried unknown scroll that may be scare monster (a gamble: ~35/1000 of
# random scrolls; power: a castle arrival carries 6-8 unknown types, ~25% that one is scare monster)
GEHENNOM_SCARE_GAMBLE = False
# Valley camp: first stand on the '<' (1-13 steps from every landing square) and take the wanderers there, one
# group at a time, with the retreat up the '<' at hand; leave for the doors once no hostile has been in view for
# VALLEY_CAMP_QUIET turns and HP >= VALLEY_REST_UNTIL, or after VALLEY_CAMP_MAX turns.
VALLEY_CAMP = False
VALLEY_CAMP_QUIET = 60
VALLEY_CAMP_MAX = 800
# The Valley's graveyards (valley.GRAVEYARD, 122 squares) hold a sleeping undead on every square, and a
# Valkyrie's stealth keeps them asleep as we pass (monmove.c disturb: !Stealth). fight2 took on every one that came
# into view (sneak or not, once anything hurt us): a monster standing on a graveyard square counts for the fight
# only when it attacked us in the last VALLEY_ATTACKER_TURNS turns (its name in a hit/miss message); path
# blockers are the sneak's business. gg1 deaths clustered at the graveyards' columns (21-26, 39-48).
VALLEY_GRAVE_FILTER = False
VALLEY_ATTACKER_TURNS = 5
# jf_config.VALLEY_SPRINT: adjacent monsters faster than this are fought (they can't be outrun: vampire bats 20)
VALLEY_SPRINT_OUTRUN = 15
# jf_config.VALLEY_SPRINT: start resting (valley_step) only below this HP fraction (REST_BELOW otherwise): at XL 8 a
# rest is ~1 HP per 5 turns, and a stop lets the pursuers catch up
VALLEY_SPRINT_REST_BELOW = 0.45
# jf_config.VALLEY_SPRINT router: extra step cost of a graveyard square not yet seen empty; climbs for a new landing
VALLEY_SPRINT_GRAVE_COST = 6
VALLEY_SPRINT_REROLLS = 4
# jf_config.VALLEY_FORT (valley_fort): drop every carried scroll that may be scare monster on the '<' when none is
# known (a fake pile shows itself: something next to us still melees us, and the fort is given up)
VALLEY_FORT_GAMBLE = True
# leave the fort for the walk once no hostile has been in view this many turns and HP >= VALLEY_FORT_LEAVE, the
# first time not before VALLEY_FORT_MIN turns in it. The fort is a shelter, not a camp: with a strong kit (XL 14,
# GDSM, speed boots) 4 of 14 harness seeds walked out without it (vxs-strong), 0 of 14 with v4's long holds and pulls
# (vxs-strong-fort: 7 still holding or pulling back at 3000 turns)
VALLEY_FORT_QUIET = 40
VALLEY_FORT_LEAVE = 0.9
VALLEY_FORT_MIN = 150
# hurt on the walk within this many steps of the pile: back to it
VALLEY_FORT_REACH = 60
# ...back to it (the pull) below VALLEY_FORT_PULL_HP, or when an awake hostile of makemon difficulty >=
# VALLEY_FORT_DANGER (wraith 8, giant mummy 10, air elemental 10, xorn 11, vampire 12, vampire lord 14; not ghosts:
# harmless, speed 3) comes within VALLEY_FORT_DANGER_RADIUS, or two awake ones within 2 (VALLEY_FORT_CROWD); at most
# VALLEY_FORT_MAX_RETURNS times (a monster that never follows would pull us back forever). The danger and crowd pulls
# kept v4 walks within ~30 steps of the pile for thousands of turns: off (99 / False)
VALLEY_FORT_PULL_HP = 0.6
VALLEY_FORT_DANGER = 10
VALLEY_FORT_DANGER_RADIUS = 5
VALLEY_FORT_CROWD = True
VALLEY_FORT_MAX_RETURNS = 25
# ...but one monster kind pulls us back at most once per VALLEY_FORT_PULL_COOLDOWN turns: a leocrotta (speed 18) that
# flees the pile and hovers pulled v4's jf14-s1~1 back four times; the pull is for what follows us to the pile
VALLEY_FORT_PULL_COOLDOWN = 150
# the walk from the pile to the '>' (dive_logic._fort_walk_action): VALLEY_SPRINT's router and striking rules; off:
# the usual Valley play (valley_step, valley_sneak, fight2) walks
VALLEY_FORT_WALK = True
# ...or after this many turns in it (random spawns come at 1/50 per turn in Gehennom, so 'quiet' may never last)
VALLEY_FORT_MAX = 500
# on a working scroll only ranged attacks and spells hurt: climb to the castle below this HP fraction
VALLEY_FORT_CLIMB_BELOW = 0.35
# jf_config.VALLEY_XORN (valley_xorn): step costs for the wall-walker's way through the Valley's rock -- a floor square
# (the residents walk there) and a rock square next to floor (they strike into it) over plain rock
VALLEY_XORN_FLOOR = 12
VALLEY_XORN_EDGE = 3
# ...and in Gehennom, out of the form with a known wand of polymorph and polymorph control shown (xorn_repoly): zap it at
# ourselves again, at most this many times
VALLEY_XORN_REPOLY = 6
# --- dwarf hunt mechanics (pick-hunt) ---
# Every dwarf death is a pile to check, whoever killed it (us, the pet, fight2 finishing an angry witness), and
# at once: only the hunted target's pile was checked, 50 turns late (the fetch scan interval) -- base-jf26/14
# killed a dwarf in its tunnel on Mines 2, walked down to Mines 3, came back and found the pile 'unreachable'.
DWARF_PILES = False
# Prefer dwarves seen digging: standing on known rock or wall, or on a '#' in the Mines (the Mines have no
# corridors of their own: a '#' there is a dwarf's tunnel), or right after 'The dwarf wields a pick-axe!'.
# makemon gives a dwarf a pick 25% / a mattock 12.5% of the time; a digger has one for sure.
DIGGER_FIRST = False
# 'You hear crashing rock.' (a digging dwarf went through a wall on this level): search this level longer
CRASH_SEARCH_BONUS = 300
CRASH_SEARCH_MAX = 1500
# Alignment budget instead of a flat kill cap (3.6.6 source): hmon_hitmon wakes the dwarf -> setmangry
# (mpeaceful = 0, alignment -1) before any damage, so xkilled never takes the 'peaceful: Luck -1' branch;
# the kill costs adjalign(malign) = -12 (a coaligned peaceful's malign). Other peaceful humanoids in view
# turn hostile (-1 each; an angered dwarf we then kill costs its -12 too). A Valkyrie starts at alignment 0
# and it is capped at ALIGNLIM = 10 + turns/200 (hostile kills keep a grind near the cap); prayer fails
# below 0. So: ~7 kills at turn 20000, ~3 at 9000, 1 at 3000.
ALIGN_BUDGET = False
ALIGN_MARGIN = 5
# turns prayers wait after each dwarf we kill (assumed Luck -1; the Luck loss doesn't happen, see above)
KILL_PRAYER_HOLD = 600
# the Dlvl-1 grind (and the rest of the tour) hunts peaceful dwarves from this XL (0: off) and keeps the tool:
# dwarves spawn on Dlvl 1 from XL 7 (difficulty 4 <= (1 + XL) / 2); 51 of 90 baseline grinds heard one
# digging there ('You hear crashing rock.') and 5 of those dives still never got a tool
# train 3 (A027): the grind hunts dwarves from XL 5 on its Dlvl 2-3 grind levels
GRIND_HUNT_XL = 5
GRIND_HUNT_DIGGERS_ONLY = False
# the tool-less dive's Mines route explores Dlvl 2-4 only until the branch '>' is known (see _search_branch);
# the tour-mode pick trip (jf_config.PICK_TRIP_XL) too, through trip_branch_strategy
FAST_BRANCH = False
BRANCH_HIDDEN_TURNS = 500          # per level (0: off), see _search_branch
BRANCH_SEARCH_TURNS = 1500         # exploring one of Dlvl 2-4 for the branch before moving on (was FULL_EXPLORE_TURNS)
BRANCH_CLIMB_MAX = 3               # climb back from at most this many levels below Dlvl 4 (see _climb_back_for_branch)
BRANCH_CLIMB_TURNS = 1500          # per level, finding and taking its '<'
BRANCH_PATH_TURNS = 600            # jf_config.ROBUST_FIXES2: cut off from a staircase towards the branch level
# BRANCH_FIX2 (off): climb back up to 3 levels from below Dlvl 4 while the branch search there isn't done (trap
# doors, holes, a candidate that was the main '>'), a 1500-turn branch search per level, and replanning
# ON (train 3): pick-fix2arm +0.017 over 45 arm64 games, no-tool dives 6 -> 4
BRANCH_FIX2 = True
# CLIMB_FIX (off): the branch climb takes only a '<' that leads (or may lead) to the level above -- never one known
# to go elsewhere: base4-jf14 s1 climbed Sokoban's '<' from Dlvl 7 and came straight back down, every turn -- and
# each staircase at most BRANCH_STAIRS_CAP times; a branch '>' known on a level above with no known stairs there
# (we came down a hole) is climbed to: base4-jf14 s14 took the main '>' of Dlvl 4 instead and the climb back from
# Dlvl 5 brought it up again, 4 <-> 5 every turn
# ON (train 3.1, pick-hunt 7e4ac3f, B012): no climb-back ping-pong (Dlvl 7<->Sokoban, 4<->5); replays jf14 s1/s14
CLIMB_FIX = True
BRANCH_STAIRS_CAP = 3
# WAND_WAITS2 (off): WAND_FIRST's zap at any hostile in view waits too while the Mines route is pending (see
# _wand_waits); the zaps against an adjacent @ or when bitten with no Elbereth to be had stay
# ON (train 3.1): WAND_FIRST holds the digging-wand zap while the Mines route is pending (7 of 90 base4 tool-less dives zapped it at a distant hostile)
WAND_WAITS2 = True
# HIDDEN_SEARCH2 (off): the hidden-branch search searches from the squares next to the widest never-seen parts of
# the map first (see _search_toward_void), HIDDEN_SEARCH2_TURNS per level
HIDDEN_SEARCH2 = False
HIDDEN_SEARCH2_TURNS = 800
# Walk up to peaceful dwarves we see but can't reach: a digger inside its fresh tunnel (dark corridor squares
# aren't on our map) was dropped as unreachable. Crash-heard Mines levels gave 9 tools in 30 baseline visits,
# the rest 6 in 179, yet jf27/14 heard 53 crashes on Mines 2 in 3840 turns without one attack. Seeing the
# dwarf along a straight line proves the squares between aren't rock: they are marked walkable.
APPROACH_DWARVES = False
APPROACH_STEPS = 60                # approach moves per APPROACH_WINDOW turns on a level
APPROACH_WINDOW = 500
# LURE_DWARVES (off): throw at a peaceful dwarf in line but out of reach (see _lure)
LURE_DWARVES = False
LURE_RANGE = 7
# Minetown is Mines level 3 or 4 (dungeon.def minetn @ (3, 2)) and was only recognised by a shopkeeper in
# view: a hunt-v2 smoke replay (jf16/9) attacked a dwarf there first -- 'Halt! You're under arrest!', three
# watchmen killed ('You murderer!' x3: prayers held 3600 turns), fainted, killed by a fox. On Mines levels
# 3-4 hunt only once the other one is known to be the town, or this one looks like a filler: 500+ squares
# seen without a door, altar, fountain, shopkeeper, watchman or priest (500 squares: dropped, see _hunt_level_ok).
MINETOWN_GUARD = False
# A wand of digging doesn't fire on Dlvl 1-4 while the Mines route (a dwarf's pick) is pending: baseline jf27/4
# zapped from Dlvl 1 at dive start, fell past the branch level, gave the Mines up at Dlvl 5 and died tool-less
# on Dlvl 7 (4 of 62 regular dives zapped a wand early and dove without a pick: mean 0.18)
WAND_WAITS = False
# a tool-less dive gives the Mines route up after MINES_STUCK_TURNS on one level with its '>' cut off, instead of
# STUCK_EXPLORE_TURNS (4000) (walking through traps to it is robustness's TRAP_LAST_RESORT); and in the Mines a
# sleeping peaceful gnome boxing us in is attacked (see _clear_blocker)
MINES_UNSTUCK = False
MINES_STUCK_TURNS = 1500
# Mines camp: a tool-less dive doesn't leave the Mines. It sweeps Mines levels 1..MINES_CAMP_MAX_LEVEL (not
# Minetown) down and up, searching each visit CAMP_VISIT_TURNS for dwarves, for MINES_CAMP_TURNS in all; a
# failed prayer or the end of the budget sends it on down the Mines as before. Early-game measured the tool-less
# XL-8 phase: no deaths in 50k Mines turns, ~7% loss per 1000 turns back in the main dungeon without a tool,
# 0.3-0.7 tools per 1000 Mines turns; base2 had 15 tool-less dives in 75 games (mean 0.14).
# ON (train 2): with HUNT_V2 (pick-camp1); the camp ran in 12 dives and ended with a tool in 10
MINES_CAMP = True
MINES_CAMP_TURNS = 8000
# RETURN_FIX (off): return_to_main_dungeon also knows the up stairs by their recorded destination, and a dive
# boxed in with a digging tool digs sideways out (see _dig_out)
RETURN_FIX = False
DIG_OUT_REACH = 30                 # 'boxed in': the BFS reaches fewer squares than this ...
DIG_OUT_TURNS = 20                 # ... for this many turns
# The dive's first job on the level it starts from: a tool-less dive whose grind level held a digging dwarf
# (seen in rock, or 'You hear crashing rock.', within HOME_DIGGER_WINDOW turns) searches that level for it first,
# HOME_DIGGER_TURNS at most (0: off). Dwarves spawn on Dlvl 1 from XL 7; 51 of 90 baseline grinds heard one
# there, and base2-public s1/s8/s14 (tool-less dives) all had one on Dlvl 1 in the old base; hunt-v2 jf16/12
# watched a digger on Dlvl 1 for 2000 turns, then dove without a tool. Unlike GRIND_HUNT_XL it leaves the tour alone.
# ON (train 3): 600 turns; +0.035 over 45 arm64 games vs base3arm, no-tool dives 6 -> 4
HOME_DIGGER_TURNS = 600
# train 3.1: only fresh digger evidence (B012: stale 2000-4800-turn evidence on the GRIND_LEVELS level cost 5 tool-less dives in base4)
HOME_DIGGER_WINDOW = 300
MINES_CAMP_MAX_LEVEL = 4
CAMP_VISIT_TURNS = 400
CAMP_STAIRS_TURNS = 1000         # looking for the way on to the next camp level before turning around
# CAMP_FIX (off): town levels only on Mines 3-4 and not from a stair follower; camp stair timers reset per visit
# ON (train 3): with BRANCH_FIX2 (pick-fix2arm)
CAMP_FIX = True
# master switch for the hunt mechanics above (individual flags stay for ablations)
# ON (train 2): pick-camp1 45 games 0.393 vs 0.348, tool dives 85% vs 74%, no-tool dives 9 -> 5
HUNT_V2 = True


def _apply_dev_overrides():
    """Dev experiments only: JF_CFG='{"REQUIRED_XL": {...}, ...}' overrides the constants above.
    The arena never sets it, so submissions always run the defaults."""
    import json
    import os
    raw = os.environ.get('JF_CFG')
    if not raw:
        return
    for name, value in json.loads(raw).items():
        if name in ('REQUIRED_XL', 'MINES_REQUIRED_XL'):
            value = {int(k): int(v) for k, v in value.items()}
        if name in globals():
            globals()[name] = value


_apply_dev_overrides()
if HUNT_V2:
    DWARF_PILES = DIGGER_FIRST = ALIGN_BUDGET = FAST_BRANCH = APPROACH_DWARVES = MINETOWN_GUARD = WAND_WAITS = \
        MINES_UNSTUCK = True
    KILL_PRAYER_HOLD = 0


def _only_ranged_monsters():
    """Monsters fight2 never melees (passive damage): not the rest's to fight either."""
    from .combat.monster_utils import ONLY_RANGED_SLOW_MONSTERS
    return ONLY_RANGED_SLOW_MONSTERS


def _hold_loop(func):
    """HOLD_LOOP for one-action 'hold' strategies (Elbereth rest, demon vigil, faint guard).

    agent.preempt() re-runs the underlying strategy after a preempting body returns, and the upper hooks
    fire only on the next step's update: one step of a lower strategy always ran between two hold actions --
    an exploration move off the Elbereth (the IDLE guard re-engraved on a new square every turn, gc-idle
    public T10183-10185), or fight2's attack/throw from the square. With the flag the body repeats while the
    strategy's own condition holds, like fight2's loop; higher-priority preempts still interrupt it through
    their update hooks. Apply below @Strategy.wrap."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        gen = func(self, *args, **kwargs)
        if not next(gen):
            yield False
            return
        yield True
        agent = self.agent
        while True:
            steps = agent.step_count
            try:
                next(gen)
            except StopIteration:
                pass
            # no step taken (e.g. an engrave that was refused): never spin
            # HOLD_LOOP_RESCUE: the loop only once a prayer has failed (the rescue dive), the tour unchanged
            looping = jf_config.HOLD_LOOP or (jf_config.HOLD_LOOP_RESCUE and agent.prayer_failed)
            if not looping or agent.step_count == steps:
                return
            gen = func(self, *args, **kwargs)
            if not next(gen):
                return
    return wrapper


class DiveLogic:
    def __init__(self, agent):
        self.agent = agent
        self.portal_level = None       # (dnum, lnum) of the Quest portal level
        self.visited_quest = False
        self.quest_arrival = None      # (y, x) of the portal on the Quest home level
        self.level_first_turn = {}     # level key -> turn first seen
        self.fully_explored = set()    # level keys explored to exhaustion
        self.sweep_started = None      # turn the current portal sweep began
        self.sweep_given_up = set()    # portal level keys whose sweep ran out of budget
        self._last_key = None
        self._last_task = None
        self.mines_done = False        # reached the bottom of the Mines, or gave the route up
        self._elbereth_resting = False
        self.diving = False
        self.dive_start_turn = None     # DIVE_TOOL_HUNGER_GAP: the turn the dive phase began (None: set by a scenario)
        self.rescue = False                # the dive began as a rescue from a failed Dlvl 1 grind
        self.pick_trip = False             # the grind's detour to the Mines for a pick-axe (PICK_TRIP_XL)
        self.undiggable = set()            # level keys where the floor is too hard to dig
        self._hp_history = []              # (turn, hp) of the last few turns
        self._status_logged = -1
        self._murder_turn = -1
        self._demon_vigil_until = -1       # turn until which a released water demon is kept off with Elbereth
        self._prep_dip_block_until = -1    # PREP_EXCAL_DIVE: no dive dips until this turn (a failed wield)
        self.prep_mid_left = set()         # PREP_MID: level keys left early (danger)
        self._prep_picked = {}             # PREP_DIVE_PICKUP: level key -> squares already looked at / picked
        self._prep_pickup_target = None    # PREP_DIVE_PICKUP: (level key, square) being walked to
        self._prep_pickup_since = 0
        self._prep_seen_levels = set()     # PREP_DIVE_PICKUP: levels whose candidates were logged
        self._faint_start = None           # turn the current faint began (last awake observation)
        self._guard_weak_since = None      # turn the current Weak spell began (FAINT_GUARD_IDLE)
        self._guard_hold_until = -1        # keep holding the faint guard's Elbereth until this turn (IDLE)
        self._ranged_hit_turn = -10 ** 9   # last turn a missile, wand or ray hit us (RANGED_ON_ELB)
        self._hold_squares = set()         # (level key, y, x) where a hold (faint guard, Elbereth rest) stood
        self._durable_sq = {}              # DURABLE_ELBERETH: level key -> (y, x) of our engraved Elbereth there
        self._durable_walk_blocked_until = -1
        self._last_update_turn = 0
        self.pet_seen = {}                 # level key -> last turn a pet glyph was in view
        self._last_pos = None              # (level key, (y, x)) at the previous update
        self._arrived = None               # (level key, turn) of the last stairs arrival
        self._arrival_pending = None       # ARRIVAL_FIX: (level key, '>' position, turn) of a stairs descent
        self._arrival_square = None        # ARRIVAL_FIX: (level key, position) of the '<' we arrived on
        self._crowd_retreats = {}          # ARRIVAL_FIX: (level key, '>' position) above -> crowd retreats taken
        self._avoid_stairs_until = {}      # (level key, (y, x)) -> turn: don't take this '>' before
        self._retreat_blocked_until = -1   # turn until which a failed retreat isn't retried
        self._dig_tries = {}               # level key -> pick-axe applies without falling through
        self._dig_blocked_until = -1       # turn until which applying the pick-axe isn't retried
        self._fetch = None                 # (level key, (y, x), turn started) of a known pick-axe
        self._fetch_given_up = set()       # (level key, (y, x)) of pick-axes not worth another trip
        self._fetch_scan_turn = -10 ** 9   # last turn the levels were scanned for pick-axes
        self._bad_dig_spots = set()        # (level key, (y, x)) where a boulder etc. blocks digging
        self._dead_traps = set()           # (level key, (y, x)) trap doors '>' didn't take us through
        self.tool_spots = set()            # (level key, (y, x)) where a pick-axe was dropped or seen
        self._dwarves_killed = 0
        self._spot_visits = {}             # (level key, (y, x)) -> go_to attempts
        self._climb_trap_tries = {}        # level key -> times traps were opened for a climb
        self._ditch_state = 0              # pet ditch: 0 idle, 1 down with it, 2 up without it, 3 over
        self._ditch_started = None
        self._ditch_tries = 0              # ditch attempts started (DITCH_PET_TRIES)
        self._ditch_pet_came = False       # a pet was seen on Dlvl 2 during this attempt
        self._pet_hunger_turn = None       # last 'confused from hunger' / 'worried about' message (dog_hunger)
        self._hunting = False              # our last attack was on a peaceful dwarf
        self._hunt_started = {}            # level key -> turn the hunt began there
        self._search_started = {}          # level key -> turn the dwarf search began there
        self._quest_started = None         # turn the tool quest began
        self._quest_over = False
        self._quest_early = False          # the running quest is the early (shallow) Mines detour
        self._early_done = False
        self._valley_arrival = None        # turn we first stood in the Valley of the Dead
        self._valley_misplaced = False     # the Valley's '>' wasn't where valley.py puts it: search the usual way
        self._valley_undiggable_doors = set()  # door squares a pick-axe refused (dig them no more: search, kick)
        # climbs up the Valley's '<' to heal on the castle level (read by the castle mode: rest before coming back)
        self.valley_retreats = 0
        self.valley_retreat_turn = None
        self._valley_west = None           # westmost column reached in the Valley (progress log)
        self._repoly_zaps = 0              # VALLEY_XORN: xorn_repoly's self-zaps
        self._valley_resting = False       # resting in the Valley until VALLEY_REST_UNTIL
        self._scare_spot = None            # (level key, (y, x)) where we dropped a scroll of scare monster
        self._scare_drop_turn = None       # turn of that drop (CASTLE_SCARE hold cap)
        self._not_scare_keys = set()       # scroll appearances a fake pile proved not to be scare monster
        self._scare_drops = 0
        self._deep_scare = None            # DEEP_ITEMS: (level key, (y, x), turn) of our known scare scroll on Medusa's level
        self._deep_blows = {}              # DEEP_ITEMS: level key -> turn we last blew a scare instrument there
        self._deep_unihorn = {}            # DEEP_ITEMS: blind spell start turn -> unicorn horn applications in it
        self._deep_frost = set()           # DEEP_ITEMS: horn glyphs whose ray froze water (frost horns)
        self._deep_charge_refused = set()  # DEEP_ITEMS: wand texts a charging read didn't take (no loop on them)
        self._deep_lenses_tried = set()    # DEEP_ITEMS: lenses texts we tried to put on (once each)
        self._deep_bad_horns = set()       # DEEP_ITEMS: unicorn horn texts that proved cursed
        self._hostile_seen_turn = -1       # last turn a hostile was in view (Valley camp)
        self._valley_camp_start = None     # turn the Valley camp on the '<' began
        self._valley_camp_done = False
        self._valley_attackers = {}        # monster name -> last turn it attacked us (Valley)
        self.walker = valley_walk.ValleyWalker(self)   # jf_config.VALLEY_WALK (valley_walk.py)
        self._valley_grave_clear = set()   # graveyard squares seen empty (VALLEY_SPRINT router)
        self._valley_rerolls = 0           # VALLEY_SPRINT climbs to get a new landing square
        # VALLEY_FORT: the scare monster pile on the Valley's '<'
        self._fort_start = None            # turn the fort began (first Valley turn)
        self._fort_done = False            # left for the walk (re-entered when hurt near the '<')
        self._fort_dropped = False         # the drop on the '<' was decided (made or found nothing to drop)
        self._fort_pile = None             # None: no pile; 'scare': holds (so far); 'dead': something meleed us on it
        self._fort_hostile_turn = -1       # last turn a hostile was in view while in the fort
        self._fort_returns = 0             # times the walk came back to the pile to heal
        self._fort_meal = None             # (y, x, monster id): the corpse next to the pile we stepped off to eat
        self._fort_pulled = {}             # monster name -> last turn it pulled the walk back to the pile
        self.castle = CastlePassage(self)  # castle_logic.py (jf_config.CASTLE_PASSAGE)
        from .castle_front import FrontDoor
        self.front = FrontDoor(self)       # castle_front.py (jf_config.FRONT_DOOR)
        self._dwarf_seen = (None, [])      # (level key, [(y, x)]) of dwarf glyphs at the last update
        self._diggers = {}                 # level key -> {(y, x): turn} where a dwarf was seen digging
        self._crash_turn = {}              # level key -> last turn 'You hear crashing rock.'
        self._align_est = None             # estimated alignment record (ALIGN_BUDGET)
        self._approach_count = {}          # level key -> (window start turn, approach moves in that window)
        self._town_levels = set()          # Mines level keys where the Watch, a shopkeeper or a priest was seen
        self._visit_start = (None, 0)      # (level key, turn) we arrived on the current level
        self._visit_pos = (None, None)     # (level key, (y, x)) where we arrived on it (WAND_RESERVE: castle region)
        self._door_heard = {}              # level key -> last turn 'You hear a door open.' (the castle's soldiers)
        self._reserve_zaps = 0             # WAND_RESERVE: zaps spent on Medusa's level and below
        self._blind_since = None           # MEDUSA_BLIND_HOLD: first turn of the current blind spell
        self._was_blind = False            # MEDUSA_BLIND_DIG: blind at the last update
        self._sight_back = False           # MEDUSA_BLIND_DIG: sight returned since the square was last looked at
        self._medusa_variant = {}          # MEDUSA_MAP: level key -> medusa_maps variant name
        self._lift = None                  # MEDUSA_LIFT: the float under way {key, target, glyph, turn, steps}
        self._lift_done = set()            # MEDUSA_LIFT: level keys where a float ended (landed or given up)
        self._potion_tried = set()         # MEDUSA_POTION_LIFT: potion glyphs quaffed for a lift
        self._potion_level = None          # MEDUSA_POTION_LIFT: level key where it quaffed
        self._potion_hold_since = None     # MEDUSA_POTION_LIFT: first turn of the current confusion/stun/hallucination
        self._potion_waits = {}            # MEDUSA_POTION_LIFT: (level key, y, x) -> turns waited for neighbours to flee
        self._freeze_zaps = {}             # MEDUSA_FREEZE: level key -> cold zaps spent there
        self.soko_trip = None              # SOKOBAN_TRIP: (turn started, XL) of the trip, None before it
        self.soko_trip_done = False
        self._branch_hidden = False        # FAST_BRANCH: Dlvl 2-4 explored without the branch, searching walls
        self._branch_search_start = {}     # level key -> turn the wall search for the hidden branch began there
        self._branch_trap_walk = set()     # level keys explored once more past their known traps
        self._branch_skip = set()          # (level key, (y, x)) candidate branch '>' we could not reach
        self._branch_reach = {}            # (level key, (y, x)) -> turn we began trying to reach it
        self._climb_start = {}             # level key -> turn we began climbing back from it for the branch
        self._climb_count = {}             # CLIMB_FIX: (level key, (y, x)) -> times the branch climb took that '<'
        self._camp_start = None            # MINES_CAMP: turn the camp began
        self._camp_over = False
        self._camp_dir = 1                 # sweep direction: +1 down, -1 up
        self._camp_visit = None            # (level key, turn we arrived) of the current camp visit
        self._camp_moves = {}              # (level key, direction) -> turn we began trying to leave that way
        self._boxed_since = None           # (level key, turn) we have been boxed in by peacefuls since
        self._boxed_dig = None             # (level key, turn) boxed in with a digging tool since (RETURN_FIX dig-out)
        self._dig_out_bad = set()          # (level key, (y, x)) squares too hard to dig
        self._dig_out_block = -1           # turn until which the dig-out isn't retried (no direction to dig)
        self._boulder_push_failed = set()  # STAIR_BOULDER_FIX: (level key, boulder square, our square) pushes in vain
        self._stairs_blocked = {}          # STAIR_BOULDER_FIX: (level key, stairs square) -> turn to try them again
        self._boulder_spin = None          # STAIR_BOULDER_FIX: (level key, boulder square, turn, calls) this turn
        self._stall_cut = {}               # STALL_*: (level key, what) -> turn the way was first found cut off
        self._stall_tries = {}             # STALL_*: (level key, square) -> clearing attempts
        self._stall_skip = {}              # STALL_*: (level key, square) -> turn until which it is left alone
        self.dive_hunger_prayers = 0       # hunger prayers (Weak or worse) made since the dive began (CAMP_HUNGER_GAP)
        self._home_search = {}             # level key -> turn the dive began searching it for its digger
        self.medusa_level = None           # level key of Medusa's level once seen (see MEDUSA_WET_SQUARES)
        self._pit_at = None                # (level key, (y, x)) of the pit we dug and still stand in
        self._dig_applies = {}             # level key -> all pick-axe applies (DIG_TRY_FIX)
        self._max_wet_cache = None         # (turn, level key, max_wet) for _dig_max_wet
        self._hurt_on_elbereth = -1        # last turn HP fell while we stood on an intact Elbereth
        self._medusa_rerolls = 0           # climbs off a wet Medusa islet to fall in again elsewhere
        self._medusa_stranded = None       # MEDUSA_STRANDED_REROLL: (level key, (y, x)) last seen stranded there
        self._medusa_cycles = 0            # MEDUSA_HOLE_CYCLE climbs off Medusa's level
        self._dug_holes = {}               # level key -> (y, x) of the last hole we fell through there
        self._dig_walk_blocked_until = -1  # turn until which DIG_ESCAPE doesn't walk to a dig square
        self._medusa_reroll_blocked_until = -1
        self._raven_levels = set()         # Medusa's level key once ravens were seen there (Medusa-3)
        self._raven_cycles = 0             # RAVEN_CYCLE climbs off the raven island
        self._raven_arrival_turn = -1      # turn we last arrived on the current level
        self._raven_flood_turn = -10       # turn a hole last filled with water on Medusa's level
        self._raven_contact_turn = -100    # last turn a hostile was next to us / hurt us on the raven island
        self._last_level_key = None
        self._medusa_floods = 0            # holes that flooded on Medusa's level (MEDUSA_SKIP)
        self._medusa_hops = 0              # moat steps toward dry land on Medusa's level (MEDUSA_HOP)
        self._eel_hold_turn = -10          # last turn an eel/kraken grabbed us (or we failed to break free)
        self._eel_engraved = -10           # last turn we engraved Elbereth against such a hold
        self._at_obs = []                  # AT_THREAT_AVOID: (turn, level key, distance of the nearest @ in view)
        self._at_seen = None               # AT_THREAT_AVOID: (level key, turn) an @ was last seen within the radius
        self._at_holds = {}                # AT_THREAT_AVOID: level key -> turns the dig was held there
        self._at_logged = set()            # AT_THREAT_AVOID: (level key, what) already logged
        self._fed_wait_start = None     # DIVE_FED: turn the grind first reached its end XL
        self._prayer_wait_start = None  # DIVE_PRAYER_GAP: turn the dive first had to wait for the prayer
        self._hp_wait_start = None      # DIVE_START_HP: turn the dive first had to wait for its HP
        self._hp_wait_over = False      # ... and the wait hit DIVE_START_HP_WAIT (logged once)
        self._weld_hold_start = None    # WELD_HOLD: turn the dive first held for WELD_PRAY's prayer
        self._weld_hold_gave_up = False  # ... and the hold is over for good (a prayer left the weld, or WELD_HOLD_MAX_WAIT)
        self._weld_hold_prayer = None   # ... agent.last_prayer_turn when the hold began
        self._shop_dig_logged = None    # SHOP_DIG: (level key, reason) last logged for not digging (logged once)
        self._shop_since = None         # SHOP_DIG: (level key, turn) we were first seen inside a shop there
        self._fed_wait_logged = False

    # ------------------------------------------------------------------ state

    def update(self):
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        turn = agent.blstats.time
        if not self._hp_history or self._hp_history[-1][0] != turn:
            if self._hp_history and agent.blstats.hitpoints < self._hp_history[-1][1] and \
                    (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
                # hurt while standing on an intact Elbereth: whatever did it ignores the engraving
                self._hurt_on_elbereth = turn
            self._hp_history.append((turn, agent.blstats.hitpoints))
            self._hp_history = self._hp_history[-12:]
        if self._pit_at is not None and self._pit_at != (key, (agent.blstats.y, agent.blstats.x)):
            self._pit_at = None
        if self.medusa_level is None and level.dungeon_number == Level.DUNGEONS_OF_DOOM and \
                agent.blstats.depth >= MEDUSA_MIN_DEPTH and key not in self.undiggable and \
                utils.isin(level.objects, WET).sum() >= MEDUSA_WET_SQUARES:
            self.medusa_level = key
            agent.log(f'DIVE Medusa level detected: {key} depth {agent.blstats.depth}')
        if MEDUSA_TITAN_DETECT and self.medusa_level is None and level.dungeon_number == Level.DUNGEONS_OF_DOOM and \
                agent.blstats.depth >= MEDUSA_MIN_DEPTH and key not in self.undiggable:
            if DiveLogic.TITAN is None:
                DiveLogic.TITAN = MON.from_name('titan')
            # (MEDUSA_TITAN_MSG: not where the castle's soldiers were heard -- its throne room may hold a titan)
            if utils.isin(agent.glyphs, [DiveLogic.TITAN]).any() and \
                    not (jf_config.MEDUSA_TITAN_MSG and key in self._door_heard):
                self.medusa_level = key   # Medusa-2's dark landing room (see MEDUSA_TITAN_DETECT)
                agent.log(f'DIVE Medusa level detected: {key} depth {agent.blstats.depth} (Medusa-2: titan)')
        if MEDUSA_TITAN_DETECT and jf_config.MEDUSA_TITAN_MSG and self._titan_named() and \
                self.medusa_level is None and level.dungeon_number == Level.DUNGEONS_OF_DOOM and \
                agent.blstats.depth >= MEDUSA_MIN_DEPTH and key not in self.undiggable and key not in self._door_heard:
            self.medusa_level = key   # (see jf_config.MEDUSA_TITAN_MSG)
            agent.log(f'DIVE Medusa level detected: {key} depth {agent.blstats.depth} (Medusa-2: titan named)')
        if key != self._last_level_key:
            self._last_level_key = key
            self._raven_arrival_turn = turn   # RAVEN_CYCLE: floods count per visit
        if key == self.medusa_level and 'fills with water' in agent.message:
            self._raven_flood_turn = turn
        if key == self.medusa_level and key not in self._raven_levels:
            if DiveLogic.RAVEN is None:
                DiveLogic.RAVEN = MON.from_name('raven')
            if utils.isin(agent.glyphs, [DiveLogic.RAVEN]).any():
                self._raven_levels.add(key)   # Medusa-3 (medusa.des: 30 hostile ravens)
                agent.log('DIVE Medusa-3 (ravens)')
        if MEDUSA_MAP and key == self.medusa_level and self._medusa_variant_name() is not None:
            # a never-seen square the fixed map calls moat is no floor to walk or dig on (agent.update_level marks a
            # square first seen under a monster walkable, and on the islands those are the eels' and ravens' water)
            moat = self._medusa_moat_mask(level)
            if moat is not None:
                level.walkable[(level.objects == -1) & moat] = False
        if MEDUSA_BLIND_DIG:
            blind = bool(agent.character.prop.blind)
            if self._was_blind and not blind:
                self._sight_back = True   # the blind look read no engraving: look again before writing one
            elif blind:
                self._sight_back = False
            self._was_blind = blind
        if MEDUSA_BLIND_HOLD:
            if not agent.character.prop.blind:
                self._blind_since = None
            elif self._blind_since is None:
                self._blind_since = turn
        if agent.blstats.hunger_state >= Hunger.FAINTING:
            if agent._fainting_since is None:
                agent._fainting_since = turn
        else:
            agent._fainting_since = None
        if jf_config.RANGED_ON_ELB:
            # every message since the last update: volleys and zaps arrive inside other atomic operations
            hist = agent._message_history
            start = getattr(self, '_ranged_seen', 0)
            if start > len(hist):
                start = 0
            self._ranged_seen = len(hist)
            m = self._RANGED_HIT.search(' '.join(hist[start:] + [agent.message]))
            if m:
                if turn - self._ranged_hit_turn > jf_config.RANGED_BREAK_TURNS:
                    agent.log(f'RANGED hit: holds and Elbereth waits off for {jf_config.RANGED_BREAK_TURNS} turns')
                self._ranged_hit_turn = turn
        if agent.blstats.hunger_state == Hunger.WEAK:
            if agent._weak_since is None:
                agent._weak_since = turn
            if self._guard_weak_since is None:
                self._guard_weak_since = turn
        else:
            agent._weak_since = None
            self._guard_weak_since = None
        # faint length -> hunger: the faint began after the last awake observation and ends with 'You
        # regain consciousness' (often both messages arrive together after the faint)
        msg = agent.message
        try:
            # dogmove.c:376/380 dog_hunger: printed only for our own (tame) pet
            if 'is confused from hunger' in msg or 'You feel worried about' in msg:
                self._pet_hunger_turn = turn
        except Exception:
            pass
        uncertain = False
        if 'You faint from lack of food' in msg and self._faint_start is None:
            self._faint_start = self._last_update_turn
            if jf_config.FAINT_MEASURE_FIX:
                # the faint's own --More-- screen (agent._faint_msg_turn) marks its start; else the start is the
                # last update -- unknown if a counted search/occupation ran since ('You stop searching.  You faint':
                # the measured length then includes the search). t31-jf16 s5: start taken 5 turns early -> est
                # -207 instead of ~-140, a deadline prayer at gap 924 failed; t31-jf14 s8: a deadline prayer at
                # gap 124 after 'You stop searching.  You faint'.
                seen = getattr(agent, '_faint_msg_turn', None)
                if seen is not None and self._last_update_turn < seen < turn:
                    self._faint_start = seen
                elif 'You stop ' in msg:
                    uncertain = True
        if 'You regain consciousness' in msg and self._faint_start is not None:
            # moves per turn: a Valkyrie is intrinsically Fast from XL 7 (16 speed on average, 4/3), not before
            # (counting 4/3 at XL 1-6 read hunger 1.3x too low and fired deadline prayers ~800 turns early)
            speed = 4 / 3 if agent.blstats.experience_level >= 7 else 1
            moves = (turn - self._faint_start) * speed
            if not uncertain:
                agent._faint_measure = (turn, (10 - moves) * 10)
            elif agent._fainting_since == turn:
                # the spell's first faint: eat.c faints at once on the Weak->Fainting transition, at uhunger ~0
                # (a Weak hold's 3-turn search made its start uncertain; without a value the unmeasured-spell
                # rule prayed at the start of every such spell)
                agent._faint_measure = (turn, 0.0)
            self._faint_start = None
        self._last_update_turn = turn
        if turn // 500 != self._status_logged:
            # a heartbeat for stall diagnoses (a jf8 game idled 4850 turns on Dlvl 2 after its grind)
            self._status_logged = turn // 500
            top = agent._hb_gotos.most_common(1)
            if top and top[0][1] >= 50 and top[0][0][:2] == (agent.blstats.y, agent.blstats.x):
                # the same go_to target (our own square) all along: log what the BFS sees around us
                dis = agent.bfs()
                lv = agent.current_level()
                y0, x0 = agent.blstats.y, agent.blstats.x
                nb = [(int(y), int(x), int(lv.walkable[y, x]), int(lv.objects[y, x]), int(agent.glyphs[y, x]),
                       int(lv.shop[y, x]), int(lv.forbidden[y, x]))
                      for y, x in agent.neighbors(y0, x0, shuffle=False)]
                agent.log(f'STALL reachable={(dis != -1).sum()} here_obj={int(lv.objects[y0, x0])} '
                          f'shop_here={int(lv.shop[y0, x0])} refused={lv.dig_tool_refused} nb={nb}')
            acts = ' '.join(f'{a}:{n}' for a, n in agent._hb_actions.most_common(5))
            gotos = ' '.join(f'{c}{(int(y), int(x))}:{n}' for (y, x, c), n in agent._hb_gotos.most_common(3))
            agent._hb_actions.clear()
            agent._hb_gotos.clear()
            agent.log(f'STATUS milestone={agent.global_logic.milestone.name} diving={self.diving} '
                      f'hunger={agent.blstats.hunger_state} pos={(agent.blstats.y, agent.blstats.x)} '
                      f'acts=[{acts}] gotos=[{gotos}]')
        # A fall is instant, so the trap door's glyph is never seen and the map forgets it: a dive
        # climbing out of the Mines fell through the same trap door twice. Remember where we fell.
        pos = (agent.blstats.y, agent.blstats.x)
        prev = self._last_pos
        if prev is not None and prev[0] != key and (self.diving or jf_config.LATE_FIXES):
            msg_all = agent.message
            if 'trap door opens up under you' in msg_all or 'hole under you' in msg_all or \
                    'You fall through' in msg_all:
                old_level = agent.levels.get(prev[0])
                if old_level is not None:
                    old_level.objects[prev[1]] = SS.S_trap_door
                    agent.log(f'DIVE fell through a trap door at {prev[1]} on {prev[0]}; remembered')
                self._dug_holes[prev[0]] = (int(prev[1][0]), int(prev[1][1]))   # MEDUSA_HOLE_CYCLE
        if jf_config.MEDUSA_STRANDED_REROLL and prev is not None and prev[0] != key and \
                self._medusa_stranded is not None and prev[0] == self._medusa_stranded[0] and \
                tuple(int(v) for v in prev[1]) == self._medusa_stranded[1] and \
                int(key[0]) == int(prev[0][0]) and int(key[1]) == int(prev[0][1]) - 1:
            # (see MEDUSA_STRANDED_REROLL) climbed off a stranded square of Medusa's level (a retreat, KNOWN_ITEMS):
            # the '>' up here leads straight back onto it -- keep it closed, the dive digs down instead
            self._medusa_stranded = None
            self._avoid_stairs_until[(key, pos)] = 10 ** 9
            agent.log(f'DIVE climbed off a stranded Medusa square {prev[1]}: the > at {pos} stays closed, digging down')
        self._last_pos = (key, pos)
        self.castle.note_level()
        self._note_digging_tools(key, pos)
        if utils.any_in(agent.glyphs, G.PETS):
            self.pet_seen[key] = turn
        if 'You murderer!' in agent.message and self._murder_turn != turn:
            # mon.c: killing a peaceful human (even a watchman the Watch set hostile) is Luck -2, and a
            # prayer with Luck < 0 fails; Luck recovers 1 per 600 turns
            self._murder_turn = turn
            agent.prayer_hold_until = max(getattr(agent, 'prayer_hold_until', -1), turn) + 1200
            agent.log('MURDER: Luck -2, prayers held 1200 turns')
        if self._hunting and self._DWARF_KILLED.search(agent.message):
            self._hunting = False
            self._dwarves_killed += 1
            if KILL_PRAYER_HOLD > 0:
                agent.prayer_hold_until = max(getattr(agent, 'prayer_hold_until', -1), turn) + KILL_PRAYER_HOLD
            # its things lie under its corpse (so the pile shows a corpse glyph), and fight2 may have
            # killed it with a thrown dagger: check every pile within 2 squares
            y0, x0 = pos
            area = agent.glyphs[max(y0 - 2, 0):y0 + 3, max(x0 - 2, 0):x0 + 3]
            piles = [(int(py) + max(y0 - 2, 0), int(px) + max(x0 - 2, 0))
                     for py, px in zip(*utils.isin(area, G.OBJECTS, G.BODIES).nonzero())]
            for p in piles:
                self.tool_spots.add((key, p))
            if DWARF_PILES and piles:
                self._fetch_scan_turn = -10 ** 9   # check them now, not at the next 50-turn scan
            agent.log(f'DIVE killed a dwarf ({self._dwarves_killed}); piles to check: {piles}')
        if DWARF_PILES or DIGGER_FIRST or ALIGN_BUDGET or GRIND_HUNT_DIGGERS_ONLY:
            self._observe_dwarves(key, level, turn)
        if self._visit_start[0] != key:
            self._visit_start = (key, turn)
            self._visit_pos = (key, pos)
        if WAND_RESERVE:
            # every message since the last update (sounds arrive inside other atomic operations, e.g. a dig)
            hist = agent._message_history
            start = getattr(self, '_door_seen', 0)
            if start > len(hist):
                start = 0
            self._door_seen = len(hist)
            if any('You hear a door open' in m for m in hist[start:] + [agent.message]):
                self._door_heard[key] = turn
        if MINETOWN_GUARD and level.dungeon_number == Level.GNOMISH_MINES and key not in self._town_levels and \
                utils.isin(agent.glyphs, self._town_glyphs()).any() and \
                (not CAMP_FIX or self._town_sighting(level, key, turn)):
            self._town_levels.add(key)
            agent.log(f'MINETOWN people in view on {key}: no dwarf hunting here')
        if key not in self.level_first_turn:
            self.level_first_turn[key] = turn
        if ARRIVAL_FIX and self._arrival_pending is not None:
            pkey, ppos, pturn = self._arrival_pending
            if key != pkey:
                # recorded here, before the preempt checks of this update run (see ARRIVAL_FIX)
                self._arrived = (key, turn, (pkey, ppos))
                self._arrival_square = (key, pos)
                self._arrival_pending = None
            elif turn - pturn > 3:
                self._arrival_pending = None
        if key != self._last_key:
            agent.log(f'DIVE level {key} depth {agent.blstats.depth}')
            self._last_key = key
            # diagnostics (power.py): what the character would bring to the Castle
            mark = 20 if agent.blstats.depth >= 20 else 10 if agent.blstats.depth >= 10 else None
            if mark is not None and mark not in getattr(self, '_kit_logged', set()):
                self._kit_logged = getattr(self, '_kit_logged', set()) | {mark}
                try:
                    agent.log(f'POWER kit at depth {agent.blstats.depth} XL {agent.blstats.experience_level} '
                              f'AC {agent.blstats.armor_class}: {power.kit_summary(agent)}')
                except Exception as e:   # diagnostics only
                    agent.log(f'POWER kit summary failed: {e!r}')
            # diagnostics only (minotaur lane): the whole kit on every deep main-line level, for replays of the
            # maze deaths (what could have stopped the minotaur?)
            if agent.blstats.depth >= 20 and level.dungeon_number == Level.DUNGEONS_OF_DOOM:
                try:
                    inv = '; '.join(f'{agent.inventory.items.get_letter(i)} - {i.text}'
                                    for i in agent.inventory.items.all_items)
                    agent.log(f'MINO kit d{agent.blstats.depth} last_prayer={agent.last_prayer_turn}: {inv}')
                except Exception as e:   # diagnostics only
                    agent.log(f'MINO kit failed: {e!r}')
        # diagnostics only (minotaur lane): minotaur sightings (distance, HP) -- from the glyphs alone (no bfs():
        # its per-step cache must not be filled from here)
        try:
            if DiveLogic.MINOTAUR is None:
                DiveLogic.MINOTAUR = MON.from_name('minotaur')
            ys, xs = (agent.glyphs == DiveLogic.MINOTAUR).nonzero()
            if len(ys) and getattr(self, '_mino_logged', None) != (key, turn):
                self._mino_logged = (key, turn)
                d = min(max(abs(int(y) - int(pos[0])), abs(int(x) - int(pos[1]))) for y, x in zip(ys, xs))
                agent.log(f'MINO seen: {len(ys)} at distance {d}, hp {agent.blstats.hitpoints}/'
                          f'{agent.blstats.max_hitpoints}')
        except Exception as e:   # diagnostics only
            agent.log(f'MINO seen failed: {e!r}')
        if self.in_valley():
            # every step, not just when the plan runs: the retreat and fight2 need the '<' and the layout from
            # the first turn (three valley-x10 games were swarmed at the landing spot and never planned a step)
            self._valley_know_map(level)
            self.valley_progress()
            if VALLEY_CAMP and agent.get_visible_monsters():
                self._hostile_seen_turn = turn
            if jf_config.VALLEY_SPRINT or jf_config.VALLEY_FORT:
                self.valley_note_graves()
            if VALLEY_GRAVE_FILTER or jf_config.VALLEY_SPRINT or jf_config.VALLEY_FORT:
                for m in self._ATTACK_MSG.finditer(agent.message):
                    self._valley_attackers[m.group(1) or m.group(2)] = turn
            if jf_config.VALLEY_FORT:
                self._fort_watch()
            if jf_config.VALLEY_WALK:
                self.walker.update()

        msg = agent.message
        if level.dungeon_number == Level.DUNGEONS_OF_DOOM and any(m in msg for m in PORTAL_MESSAGES):
            if self.portal_level != key:
                agent.log(f'DIVE quest portal level detected: {key}')
            self.portal_level = key

        if level.dungeon_number == Level.QUEST and not self.visited_quest:
            self.visited_quest = True
            self.quest_arrival = (agent.blstats.y, agent.blstats.x)
            agent.log(f'DIVE entered the Quest home at {self.quest_arrival}')

    _DWARF_KILLED = re.compile(r"You kill (the|a|an) (poor )?dwarf( lord| king)?!")
    _TOOL_PICKED_UP = re.compile(r"\b[a-zA-Z] - (an?|\d+) [^.]*(pick-axe|dwarvish mattock)")

    def _note_digging_tools(self, key, pos):
        """Remember where pick-axes lie. The tour picks them up and drops them again for lighter loot
        (s4 public seed 9: 7 times), and the level item memory doesn't keep a big drop reliably.
        Drops happen inside atomic operations, so read every message since the last update."""
        history = self.agent._message_history
        start = getattr(self, '_history_seen', 0)
        if start > len(history):   # a fresh agent after a driver restart
            start = 0
        self._history_seen = len(history)
        msg = ' '.join(history[start:] + [self.agent.message])
        if 'pick-axe' not in msg and 'dwarvish mattock' not in msg:
            return
        picked = max((m.start() for m in self._TOOL_PICKED_UP.finditer(msg)), default=-1)
        dropped = max((msg.rfind(f'{verb} {tool}') for verb in ('You drop a', 'You see here a')
                       for tool in ('pick-axe', 'dwarvish mattock')), default=-1)
        if picked > dropped:
            self.tool_spots.discard((key, pos))
        elif dropped > picked and not self.agent.current_level().shop_interior[pos]:
            self.tool_spots.add((key, pos))

    # 'You kill the dwarf!' (us, also by a thrown weapon), 'The dwarf is killed!' (the pet, another monster)
    _DWARF_DEATH = re.compile(r"You kill (?:the |an? )?(?:poor )?dwarf(?: lord| king)?!|"
                              r"\b[Tt]he dwarf(?: lord| king)? is killed!")
    _KILL = re.compile(r"You (?:kill|destroy) (?:the |an? )?(?:poor )?([^!.]+?)!")
    _WIELDS_PICK = re.compile(r"dwarf(?: lord| king)? wields an? (?:[a-z ]+ )?(?:pick-axe|dwarvish mattock)!")
    _DWARF_GLYPHS = None
    # usually peaceful to a dwarf (killed only after they turned on us: malign 0 or worse, no alignment gain)
    _PEACEFUL_KIN = frozenset(('gnome', 'gnome lord', 'gnomish wizard', 'gnome king', 'hobbit', 'dwarf zombie'))

    def _dwarf_glyphs(self):
        if DiveLogic._DWARF_GLYPHS is None:
            DiveLogic._DWARF_GLYPHS = frozenset(MON.from_name(n) for n in DWARF_NAMES)
        return DiveLogic._DWARF_GLYPHS

    def _observe_dwarves(self, key, level, turn):
        """Dwarf deaths -> piles to check (DWARF_PILES), digging dwarves (DIGGER_FIRST), 'crashing rock', and
        the alignment estimate (ALIGN_BUDGET). Reads every message since the last update: kills and wields
        happen inside atomic operations."""
        agent = self.agent
        history = agent._message_history
        start = getattr(self, '_history_seen2', 0)
        if start > len(history):   # a fresh agent after a driver restart
            start = 0
        self._history_seen2 = len(history)
        msg = ' '.join(history[start:])
        glyphs = self._dwarf_glyphs()
        now = [(int(y), int(x)) for y, x in zip(*utils.isin(agent.glyphs, glyphs).nonzero())]

        if ALIGN_BUDGET:
            cap = 10 + turn // 200
            if self._align_est is None:
                # the grind's hostile kills keep the record near the cap; a Valkyrie starts at 0
                self._align_est = 0 if turn < 200 else cap - 10
            gain = loss = 0
            for m in self._KILL.finditer(msg):
                name = m.group(1)
                if name in DWARF_NAMES:
                    loss += {'dwarf': 12, 'dwarf lord': 15, 'dwarf king': 18}[name]
                elif name != 'it' and name not in self._PEACEFUL_KIN:
                    # conservative: an always-hostile monster gives max(5, |alignment|), an angered peaceful
                    # gnome 0 (its malign was computed while peaceful)
                    gain += 2
            loss += msg.count('gets angry!')
            if gain or loss:
                self._align_est = min(cap, self._align_est + gain) - loss
                if loss:
                    agent.log(f'ALIGN estimate {self._align_est} (cap {cap}) after -{loss}')
            if self._align_est < 0 and agent.blstats.hunger_state < Hunger.FAINTING:
                # prayer fails below alignment 0 (pray.c p_type 1): wait for hostile kills to restore it
                # (not while fainting: starving to death is no better than an angry god)
                agent.prayer_hold_until = max(getattr(agent, 'prayer_hold_until', -1), turn + 20)

        prev_key, prev = self._dwarf_seen
        if DWARF_PILES and prev_key == key and self._DWARF_DEATH.search(msg):
            spots = set()
            for y, x in prev:
                if agent.glyphs[y, x] in glyphs:
                    continue
                for yy in range(max(y - 1, 0), min(y + 2, agent.glyphs.shape[0])):
                    for xx in range(max(x - 1, 0), min(x + 2, agent.glyphs.shape[1])):
                        g = agent.glyphs[yy, xx]
                        if g in G.OBJECTS or g in G.BODIES:
                            spots.add((yy, xx))
            spots = {p for p in spots if (key, p) not in self._fetch_given_up}
            if spots:
                for p in spots:
                    self.tool_spots.add((key, p))
                self._fetch_scan_turn = -10 ** 9
                agent.log(f'DWARF died near {sorted(spots)}: piles to check')
        self._dwarf_seen = (key, now)

        if 'You hear crashing rock' in msg:
            self._crash_turn[key] = turn
        if (DIGGER_FIRST or GRIND_HUNT_DIGGERS_ONLY) and now:
            marks = self._diggers.setdefault(key, {})
            if len(marks) > 60:
                for q in [q for q, t in marks.items() if turn - t > 200]:
                    del marks[q]
            mines = level.dungeon_number == Level.GNOMISH_MINES
            for p in now:
                terrain = level.objects[p]
                if terrain in G.STONE or terrain in G.WALL or (mines and terrain in CORRIDORS):
                    if p not in marks:
                        agent.log(f'DIGGER dwarf in rock/tunnel at {p}')
                    marks[p] = turn
            if self._WIELDS_PICK.search(msg):
                y0, x0 = agent.blstats.y, agent.blstats.x
                p = min(now, key=lambda q: max(abs(q[0] - y0), abs(q[1] - x0)))
                marks[p] = turn
                agent.log(f'DIGGER dwarf wields a digging tool at {p}')

    _TOWN_GLYPHS = None

    def _town_glyphs(self):
        if DiveLogic._TOWN_GLYPHS is None:
            DiveLogic._TOWN_GLYPHS = frozenset(MON.from_name(n) for n in ('watchman', 'watch captain',
                                                                          'shopkeeper', 'aligned priest'))
        return DiveLogic._TOWN_GLYPHS

    def _town_sighting(self, level, key, turn):
        """CAMP_FIX: town people seen on this level mean Minetown only on Mines levels 3-4 (dungeon.def minetn
        @ (3, 2)), and not a watchman who followed us up or down the stairs (he arrives next to us): base3-public
        s13 marked Mines 2 and 4 as towns that way and its camp had one huntable level left."""
        if level.level_number not in (3, 4):
            return False
        visit = self._visit_start if self._visit_start[0] == key else (key, turn)
        if turn - visit[1] > 10:
            return True
        y0, x0 = self.agent.blstats.y, self.agent.blstats.x
        town = utils.isin(self.agent.glyphs, self._town_glyphs())
        return any(max(abs(int(y) - y0), abs(int(x) - x0)) > 1 for y, x in zip(*town.nonzero()))

    def _hunt_level_ok(self, level):
        """Dwarves may be attacked here: never in Minetown (the Watch), see MINETOWN_GUARD."""
        key = level.key()
        minetown = self.agent.global_logic.minetown_level
        if key == minetown:
            return False
        if not MINETOWN_GUARD or level.dungeon_number != Level.GNOMISH_MINES:
            return True
        if key in self._town_levels:
            return False
        if level.level_number not in (3, 4):
            return True
        other = (Level.GNOMISH_MINES, 7 - level.level_number)
        if minetown == other or other in self._town_levels:
            return True
        # Mines fillers have no doors, altars or fountains; only a watchman that sees the attack arrests us, and
        # one in view is recorded in _town_levels (hunt-v2 jf16/14 waited ~8000 turns for a Mines-3 filler to
        # show 500 squares while base took its pick there in 183 turns)
        return not utils.isin(level.objects, G.DOORS, G.ALTAR, G.FOUNTAIN).any()

    def _is_digger(self, key, p, recent=40):
        marks = self._diggers.get(key)
        if not marks:
            return False
        turn = self.agent.blstats.time
        return any(turn - t <= recent and max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 2 for q, t in marks.items())

    def _may_kill_dwarf(self, witnesses=0):
        """The alignment budget (ALIGN_BUDGET) or the flat kill cap allows one more dwarf kill."""
        if self._dwarves_killed >= DWARF_HUNT_MAX_KILLS and not ALIGN_BUDGET:
            return False
        if not ALIGN_BUDGET:
            return True
        if self._align_est is None:
            return False
        # this kill (-13) plus every peaceful dwarf in view that turns hostile and has to be killed (-13 each)
        return self._align_est - 13 * (1 + witnesses) >= ALIGN_MARGIN

    def should_dive(self):
        if self.diving:
            return True
        agent = self.agent
        gl = agent.global_logic
        from .global_logic import Milestone
        xl = agent.blstats.experience_level
        rescue = RESCUE_DIVE and agent.prayer_failed and gl.milestone == Milestone.BE_ON_FIRST_LEVEL
        late_rescue = LATE_RESCUE and agent.prayer_failed and gl.milestone != Milestone.BE_ON_FIRST_LEVEL and \
            agent.blstats.hunger_state >= Hunger.WEAK and not agent.edible_carried_food()
        rescue = rescue or (EARLY_DIVE and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and
                            agent.blstats.time >= EARLY_DIVE_TURN)
        # planned early dive: leave the grind at EARLY_DIVE_XL, take a dwarf's pick-axe in the Mines and dig
        # the main dungeon under Elbereth. Random monsters are capped at difficulty (depth + XL) / 2, so a
        # low-XL digger meets weaker ones (our XL-10 dives died at Dlvl 10-12 to air elementals, Elvenkings
        # and flesh golems -- difficulty 10-11, impossible below XL ~8 there)
        planned = bool(EARLY_DIVE_XL) and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and xl >= EARLY_DIVE_XL \
            and not agent.prayer_failed
        xl_trigger = xl >= DIVE_XL or self._sam_grind_over(xl) or \
            (xl >= self._dig_dive_xl() and self.digging_tool() is not None)
        if jf_config.SOKOBAN_TRIP and self._sokoban_trip(xl_trigger, rescue or late_rescue):
            return False
        if xl_trigger and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and not self.fed_for_dive():
            xl_trigger = False   # DIVE_FED: finish the hunger cycle on Dlvl 1 first
        if xl_trigger and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and not self.prayer_ready_for_dive():
            xl_trigger = False   # DIVE_PRAYER_GAP: go on grinding until an emergency prayer would work
        if xl_trigger and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and not self.hp_ready_for_dive():
            xl_trigger = False   # DIVE_START_HP: go on grinding (and resting) until the HP is back
        if xl_trigger or gl.milestone >= Milestone.GO_DOWN or agent.blstats.time >= DIVE_TURN or \
                rescue or late_rescue or planned:
            tag = ', rescue' if rescue else ', late rescue' if late_rescue else ', early' if planned else ''
            agent.log(f'DIVE phase starts (milestone {gl.milestone.name}{tag})')
            self.diving = True
            self.dive_start_turn = agent.blstats.time
            self.rescue = rescue or late_rescue
            # the tour handled the Mines; from here it's the main dungeon. A rescue used to take the Mines
            # route, but none of 21 rescue dives (abP/abPF) ever reached the Mines: they searched Dlvl 2
            # for the branch while fainting (11-72 faints) and died on Dlvl 1-2, while the two that took
            # the main stairs reached Dlvl 12 and 28. The Mines are peaceful to a dwarf: no corpses to eat.
            if rescue or late_rescue:
                self.mines_done = RESCUE_MAIN_DUNGEON
            else:
                # the Mines route (a dwarf's pick-axe) unless the tour has been through the Mines already: a
                # dive straight from the Dlvl 1 grind (DIVE_XL 8) otherwise only met dwarves by chance in
                # the main dungeon -- 7 of 33 such dives never got a digging tool
                self.mines_done = gl.milestone > Milestone.FIND_GNOMISH_MINES and not planned
        return self.diving

    def _sokoban_trip(self, xl_trigger, rescue):
        """SOKOBAN_TRIP (power-route): when the grind would hand over to the dive, first the tour's Sokoban milestones
        (FIND_SOKOBAN, SOLVE_SOKOBAN: AutoAscend's solver). sokoban.des puts a random ring and a random wand on each of
        its 4 levels, 2 scrolls of earth on the entry level and a bag of holding or an amulet of reflection at the top:
        4 rings are ~27 levels' worth of the dive's ring finds (a ring of teleport control 1 - (27/28)^4 = 13.6%, same
        for levitation) and 4 wands give a wand of wishing ~2% (WISH_TELEPORT_ROUTE: Dlvl 50). The trip ends when the
        tour leaves those milestones (solved, or TOUR_STALL_TURNS on one spot) or after SOKOBAN_TRIP_TURNS; then the
        milestone goes back to the grind's and the usual dive trigger (Mines route for a pick-axe) takes over.
        Returns True while the trip holds the dive back."""
        from .global_logic import Milestone
        agent = self.agent
        gl = agent.global_logic
        now = agent.blstats.time
        if self.soko_trip_done:
            return False
        if self.soko_trip is None and gl.milestone in (Milestone.FIND_SOKOBAN, Milestone.SOLVE_SOKOBAN):
            # a trip already under way (a harness scenario starts with the milestone set)
            self.soko_trip = (now, agent.blstats.experience_level)
            agent.log(f'SOKOBAN trip under way (milestone {gl.milestone.name}, XL {agent.blstats.experience_level})')
        if self.soko_trip is None:
            if rescue or not xl_trigger or gl.milestone != Milestone.BE_ON_FIRST_LEVEL or agent.prayer_failed:
                return False
            self.soko_trip = (now, agent.blstats.experience_level)
            gl.milestone = Milestone.FIND_SOKOBAN
            agent.log(f'SOKOBAN trip starts (XL {agent.blstats.experience_level}, turn {now})')
            return True
        in_trip = gl.milestone in (Milestone.FIND_SOKOBAN, Milestone.SOLVE_SOKOBAN)
        over_budget = now - self.soko_trip[0] > jf_config.SOKOBAN_TRIP_TURNS
        if in_trip and not over_budget and not rescue:
            return True
        self.soko_trip_done = True
        why = 'budget' if over_budget else 'rescue' if rescue else f'milestone {gl.milestone.name}'
        agent.log(f'SOKOBAN trip over after {now - self.soko_trip[0]} turns ({why}); on Sokoban level '
                  f'{agent.current_level().key()}: back to the grind milestone, then the dive')
        gl.milestone = Milestone.BE_ON_FIRST_LEVEL
        return False

    def edible_corpse_within(self, radius):
        """DIVE_EAT's condition: a known corpse on this level that eat_corpses_from_ground would eat, at most
        radius BFS steps away (cheap: the BFS is cached per step)."""
        agent = self.agent
        level = agent.current_level()
        if not level.corpses_to_eat:
            return False
        dis = agent.bfs()
        for (y, x), corpses in level.corpses_to_eat.items():
            if not 0 <= dis[y, x] <= radius or level.shop[y, x]:
                continue
            if any(agent._is_corpse_editable(mid, age) for mid, age in corpses.items()):
                return True
        return False

    def climbing(self):
        """Leaving a side branch upwards, or climbing on the tool quest: a fall through a trap door undoes it."""
        if not self.diving:
            return False
        level = self.agent.current_level()
        return (level.dungeon_number == Level.GNOMISH_MINES and not self.use_mines()) or self._quest_started is not None

    def turns_on_level(self):
        return self.agent.blstats.time - self.level_first_turn.get(self.agent.current_level().key(),
                                                                    self.agent.blstats.time)

    # ------------------------------------------------------------- targets

    def _stairs_ok(self, level, y, x):
        """A '>' is worth taking unless it is known to lead off the main line (e.g. the Mines)."""
        dest = level.stair_destination.get((y, x))
        if dest is None:
            return True
        return dest[0][0] in MAIN_LINE

    def down_targets(self):
        """Reachable ways down on this level: [(distance, y, x, kind)], nearest first."""
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number not in MAIN_LINE:
            return []
        dis = agent.bfs()
        targets = []
        for y, x in zip(*utils.isin(level.objects, G.STAIR_DOWN).nonzero()):
            if dis[y, x] != -1 and self._stairs_ok(level, y, x) and \
                    self._avoid_stairs_until.get((level.key(), (y, x)), -1) <= agent.blstats.time:
                targets.append((dis[y, x], y, x, 'stairs'))
        # trap doors and holes are not walkable for bfs: reach a neighbour, then step in
        strong = agent.blstats.experience_level >= self.required_xl(agent.blstats.depth + TRAPDOOR_LOOKAHEAD)
        for y, x in zip(*utils.isin(level.objects, FALL_TRAPS).nonzero()):
            if not strong:
                break
            if (level.key(), (y, x)) in self._dead_traps:
                continue
            d = self._neighbour_distance(dis, y, x)
            if d is not None:
                targets.append((d + 1, y, x, 'trap'))
        targets.sort()
        return targets

    def _neighbour_distance(self, dis, y, x):
        best = None
        for ny, nx in self.agent.neighbors(y, x, shuffle=False):
            if dis[ny, nx] != -1 and (best is None or dis[ny, nx] < best):
                best = dis[ny, nx]
        return best

    # ------------------------------------------------------------ strategy

    def exploration(self, search_prio_limit):
        return self.agent.global_logic.exploration_strategy(search_prio_limit)

    def _budgeted(self, condition, budget_over=None):
        """EXPLORE_BUDGETS: an exploration's until-condition that also ends the run once `budget_over()` holds
        or after EXPLORE_REPLAN_TURNS turns, handing control back to plan_step."""
        if not EXPLORE_BUDGETS:
            return condition
        t0 = self.agent.blstats.time
        task = self._last_task

        def until():
            if condition():
                return True
            if budget_over is not None and budget_over():
                self.agent.log(f'DIVE budget spent in task {task!r} after {self.agent.blstats.time - t0} turns')
                return True
            if self.agent.blstats.time - t0 > EXPLORE_REPLAN_TURNS:
                self.agent.log(f'DIVE replanning task {task!r} after {EXPLORE_REPLAN_TURNS} turns of exploring')
                return True
            return False
        return until

    @Strategy.wrap
    def strategy(self):
        yield True
        idle = 0
        while True:
            before = self.agent.step_count
            self.plan_step()
            if self.agent.step_count != before:
                idle = 0
                continue
            # a task that returns without acting would spin forever: let a turn pass
            idle += 1
            if idle >= 3:
                self.agent.log(f'DIVE no progress in task {self._last_task!r}: '
                               f'targets={self.down_targets()[:3]} -> searching')
                self.agent.search()
                idle = 0

    def _task(self, name):
        if name != self._last_task:
            self.agent.log(f'DIVE task {name}')
            self._last_task = name
        # diagnostics: a task stuck on one level for long -> dump the screen once per 1000 turns
        if jf_log.enabled() and self.turns_on_level() > 1000:
            mark = self.turns_on_level() // 1000
            if mark != getattr(self, '_last_dump_mark', None):
                self._last_dump_mark = mark
                screen = '\n'.join(bytes(row).decode('latin-1').rstrip()
                                   for row in self.agent.last_observation['tty_chars'])
                dis = self.agent.bfs()
                ups = [(int(y), int(x), int(dis[y, x])) for y, x in
                       zip(*utils.isin(self.agent.current_level().objects, G.STAIR_UP).nonzero())]
                self.agent.log(f'DIVE stuck in task {name!r} on level {self.agent.current_level().key()} '
                               f'for {self.turns_on_level()} turns; up stairs (y,x,dist)={ups}\n{screen}')

    def plan_step(self):
        agent = self.agent
        level = agent.current_level()
        dnum = level.dungeon_number

        # the Valley rests on its '<' (valley_step), one step from the castle
        if self.in_valley() and not self._valley_misplaced:
            self._task('valley')
            return self.valley_step()

        # WAND_RESERVE: a maze below Medusa's level is left at once with the kept wand of digging -- its minotaurs
        # come 0-8 turns after the landing and ignore Elbereth -- before any rest there
        maze = self._maze_zap()
        if maze is not None:
            self._task('zap down through the maze')
            self._escape_act(maze)
            return

        # MEDUSA_LIFT: stranded on a wet islet with a known ring of levitation -- float to a dry square (before any rest:
        # hovering over the moat is no place to wait)
        lift = self._medusa_lift_action()
        if lift is not None:
            self._task('float to a dry square')
            self._escape_act(lift)
            return

        # astra guard.py: don't walk on below 2/3 HP; rest while nothing hostile is in view
        # (not a Gehennom digger: see GEHENNOM_DIG_REST_BELOW)
        bl = agent.blstats
        digger = DIVE_REST and self._digger_here()
        rest_below = DIG_REST_BELOW if digger else REST_BELOW
        if bl.hitpoints < rest_below * bl.max_hitpoints and not agent.get_visible_monsters() and \
                bl.hunger_state < Hunger.WEAK and not (digger and self._in_own_pit()) and \
                not self._gehennom_digger() and not self._mino_alert():
            self._task('rest')
            if digger and self._rest_elbereth():
                return
            agent.search(20)
            return

        if dnum == Level.QUEST:
            self._task('leave quest')
            return self.leave_quest()

        # the castle level (the dig-dive's floor): try to get round the moat to the back door's trap door
        if self.castle.active():
            self._task('castle passage')
            if self.castle.plan_step():
                return
        if MINES_UNSTUCK and self._clear_blocker():
            return

        if RETURN_FIX and self._dig_out():
            return

        if self.should_fetch_digging_tool():
            self._task('fetch digging tool')
            return self.fetch_digging_tool()

        if self.should_hunt_dwarf():
            self._task('hunt dwarf')
            return self.hunt_dwarf()

        if self.should_approach_dwarf():
            self._task('approach dwarf')
            if self._approach_dwarf():
                return
            if self.should_hunt_dwarf():   # the line to it is open now
                self._task('hunt dwarf')
                return self.hunt_dwarf()

        if self.should_camp():
            return self.camp_step()

        if self.should_search_home_digger():
            return self.search_home_digger()

        if self.should_search_dwarves():
            self._task('search for dwarves')
            started = self._search_started.get(level.key(), bl.time)
            if DWARF_PILES or DIGGER_FIRST or ALIGN_BUDGET:
                # stop for a dwarf we may attack, and for a fresh pile to check (DWARF_PILES resets the scan);
                # the search budget (longer on crash levels) is should_search_dwarves' own
                self.exploration(None).until(agent, self._budgeted(
                    lambda: (self._hunt_level_ok(agent.current_level()) and bool(self._hunt_targets())) or
                    self.digging_tool() is not None or self._fetch_scan_turn < 0,
                    lambda: not self.should_search_dwarves())).run()
                return
            self.exploration(None).until(agent, self._budgeted(
                lambda: bool(self._peaceful_dwarves()) or self.digging_tool() is not None,
                lambda: agent.blstats.time - started > DWARF_SEARCH_TURNS)).run()
            return

        if self.should_tool_quest():
            self._task('tool quest')
            return self.tool_quest()

        if dnum == Level.GNOMISH_MINES and self.use_mines():
            self._task('mines descent')
            return self.mines_step()

        if dnum not in MAIN_LINE:
            self._task('return to main dungeon')
            return self.return_to_main_dungeon()

        # PREP_EXCAL_DIVE: a fountain in reach on a Dungeons level -> dip the long sword for Excalibur first
        if self.prep_fountain_dip():
            return

        # PREP_DIVE_PICKUP: a ring, wand, amulet, potion or scroll in view a few steps away -> take it first
        if self.prep_dive_pickup():
            return

        if self.should_sweep_portal():
            self._task('portal sweep')
            return self.portal_sweep()

        if self.should_explore_fully():
            if self.prep_mid_level():
                # PREP_MID: the middle game's XP and items -- explore, fight what's ordinary, dig out when it isn't
                self._task('explore fully (prep mid)')
                explored = self.exploration(0).until(agent, self._budgeted(
                    lambda: False, lambda: not self.prep_mid_level())).run(return_condition=True)
                if not explored:
                    self.fully_explored.add(level.key())
                return
            self._task('explore fully')
            if EXPLORE_BUDGETS:
                explored = self.exploration(0).until(agent, self._budgeted(
                    lambda: False, lambda: self.turns_on_level() > FULL_EXPLORE_TURNS)).run(return_condition=True)
            else:
                explored = self.exploration(0).run(return_condition=True)
            if not explored:
                self.fully_explored.add(level.key())
            return

        if self.should_farm():
            self._task('farm xp')
            return self.farm_xp()

        if self.use_mines():
            if self.go_to_mines():
                return
            if agent.blstats.depth > MINES_BRANCH_MAX_DEPTH:
                if FAST_BRANCH and dnum == Level.DUNGEONS_OF_DOOM:
                    if BRANCH_FIX2:
                        if self._climb_back_for_branch():
                            return
                    else:
                        # train 2: back up only from Dlvl 5, and only to a candidate or an open hidden search
                        ups = set(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))
                        ups |= {p for p, d in level.stair_destination.items()
                                if d is not None and d[0] == (Level.DUNGEONS_OF_DOOM, level.level_number - 1)}
                        if agent.blstats.depth == MINES_BRANCH_MAX_DEPTH + 1 and ups and \
                                (self._branch_search_open() or self._mines_branch_target() is not None) and \
                                self._take_stairs([(int(y), int(x)) for y, x in ups], '<'):
                            return
                agent.log('DIVE mines branch not found above; giving the Mines route up')
                self.mines_done = True

        if self.weld_hold():
            # WELD_HOLD: welded, waiting on this level for WELD_PRAY's prayer (emergency_strategy); fight2 answers what
            # comes (no Elbereth is possible welded)
            self._task('weld hold')
            agent.search(5)
            return

        self._task('descend')
        return self.descend()

    # ------------------------------------------------------------- elbereth

    def _ignores_elbereth(self, mon):
        # monmove.c onscary(): @ humans and elves (incl. shopkeepers, guards, priests), minotaurs,
        # peacefuls and blind monsters are not scared; nothing is in Gehennom. permonst.mlet is the
        # monster class as a character (ord() == class number), not the display symbol.
        # Also useless: monsters that hurt from range (breath, spit, spells) -- a b3a dive rested on
        # Elbereth at Dlvl 25 next to a yellow dragon and died to its acid breath.
        mlet = getattr(mon, 'mlet', '')
        cls = ord(mlet) if isinstance(mlet, str) and len(mlet) == 1 else -1
        name = getattr(mon, 'mname', '')
        return cls in (MON.S_HUMAN, MON.S_DRAGON) or name in ('minotaur', 'unknown') or name in RANGED_MONSTERS

    def _melee_ignores_elbereth(self, mon):
        """onscary() for melee only: @ humans and elves (also shopkeepers, guards, priests) and minotaurs
        fight on through Elbereth. Breathers, spitters and casters don't melee a hero standing on it
        (dochug: no mattacku while scared) -- they are no reason to fight instead of digging out. An
        unseen attacker counts only once something hurt us while we stood on an intact Elbereth."""
        mlet = getattr(mon, 'mlet', '')
        cls = ord(mlet) if isinstance(mlet, str) and len(mlet) == 1 else -1
        name = getattr(mon, 'mname', '')
        if name == 'unknown':
            return self.agent.blstats.time - self._hurt_on_elbereth <= 3
        if jf_config.LMINION_ELBERETH and self._lawful_minion(mon):
            return True
        return cls == MON.S_HUMAN or name == 'minotaur'

    @staticmethod
    def _lawful_minion(mon):
        """LMINION_ELBERETH: an M2_MINION monster -- the A class, all generated lawful -- is is_lminion() for
        monmove.c onscary, which ignores Elbereth (and a scroll of scare monster) for it."""
        return bool(getattr(mon, 'mflags2', 0) & MON.M2_MINION)

    def _at_threat(self):
        """AT_THREAT_AVOID: the distance of the nearest hostile Elbereth-ignoring meleer that holds a new pit here -- in
        view within the radius and not standing off, or seen there during the last AT_THREAT_MEMORY turns -- else None."""
        if not (jf_config.AT_THREAT_AVOID and self.diving):
            return None
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or self._wand_zone():
            return None   # Medusa's level and the mazes below her keep their own plans
        key = level.key()
        bl = agent.blstats
        now = bl.time
        if self._at_holds.get(key, (None, 0))[1] >= jf_config.AT_THREAT_MAX_HOLD:
            return None
        dwarf = agent.character.race == Character.DWARF
        radius = jf_config.AT_THREAT_RADIUS_DWARF if dwarf else jf_config.AT_THREAT_RADIUS
        y0, x0 = int(bl.y), int(bl.x)
        dists = []
        for m in agent.get_visible_monsters():
            if getattr(m[3], 'mname', '') == 'unknown' or not self._melee_ignores_elbereth(m[3]):
                continue
            # BFS steps where its square is reachable (an @ behind a wall is further away than it looks)
            d = int(m[0]) if m[0] > 0 else max(abs(int(m[1]) - y0), abs(int(m[2]) - x0))
            if d <= radius:
                dists.append(d)
        if not dists:
            seen = self._at_seen
            if seen is not None and seen[0] == key and now - seen[1] <= jf_config.AT_THREAT_MEMORY:
                return radius
            return None
        d = min(dists)
        obs = self._at_obs
        if not obs or obs[-1][:2] != (now, key):
            obs.append((now, key, d))
            del obs[:-30]
        self._at_seen = (key, now)
        # standing off: in view all along for AT_THREAT_STILL turns, and no closer now than then
        past = [o for o in obs if o[1] == key and o[0] <= now - jf_config.AT_THREAT_STILL]
        if past:
            run = [o for o in obs if o[1] == key and o[0] >= past[-1][0]]
            if all(b[0] - a[0] <= 2 for a, b in zip(run, run[1:])) and d >= past[-1][2]:
                return None
        return d

    def _at_hold(self):
        """AT_THREAT_AVOID: (True, action) while an @ holds the dig here -- action is a wand of digging's zap down when
        one may be spent, else None (fight2 fights the @, try_dig_down waits for it); (False, None): dig as usual. In
        our own pit the dig goes on, though a wand still takes us through at once. Only for a pick-axe digger: without
        one the wand is the dig, and WAND_FIRST zaps it as before."""
        if self.digging_tool() is None:
            return False, None
        d = self._at_threat()
        if d is None:
            return False, None
        agent = self.agent
        bl = agent.blstats
        key = agent.current_level().key()
        wand = self._dig_wand()
        if wand is not None and not self._wand_waits() and \
                (bl.experience_level < jf_config.AT_THREAT_WAND_XL or bl.hitpoints < 0.5 * bl.max_hitpoints):
            zap = self._wand_escape(wand)
            if zap is not None and self._wet_neighbours(bl.y, bl.x) == 0:
                self._at_log(key, 'zap', f'an @ {d} away: zapping {wand.text!r} down')
                return True, zap
        dwarf = agent.character.race == Character.DWARF
        if self._in_own_pit() or d < (jf_config.AT_THREAT_NEAR_DWARF if dwarf else jf_config.AT_THREAT_NEAR):
            return False, None
        self._at_log(key, 'hold', f'no pit with an @ {d} away; hostiles at '
                                  f'{[(m[3].mname, int(m[0])) for m in agent.get_visible_monsters()[:4]]}')
        return True, None

    def _at_count(self, key):
        """AT_THREAT_AVOID: one more turn the dig waited on this level (AT_THREAT_MAX_HOLD)."""
        now = self.agent.blstats.time
        last, n = self._at_holds.get(key, (None, 0))
        if last != now:
            self._at_holds[key] = (now, n + 1)

    def _at_log(self, key, what, text):
        if (key, what) not in self._at_logged:
            self._at_logged.add((key, what))
            self.agent.log(f'DIVE AT_THREAT {text}')

    def on_medusa_level(self):
        return self.medusa_level is not None and self.agent.current_level().key() == self.medusa_level

    def _titan_named(self):
        """MEDUSA_TITAN_MSG: a message since the last call names a titan ('The titan casts a spell!', 'The invisible
        titan hits!', 'The titan turns to flee.'). Called on every update, so the history pointer never lags behind a
        level change."""
        agent = self.agent
        hist = agent._message_history
        start = self.__dict__.get('_titan_msg_seen', 0)
        if start > len(hist):
            start = 0
        self._titan_msg_seen = len(hist)
        return any(_TITAN_MSG.search(m or '') for m in hist[start:] + [agent.message])

    def below_medusa(self):
        """Deeper than Medusa's level in the Dungeons of Doom (None: Medusa's level not seen yet -- a fall can
        skip it)."""
        if self.medusa_level is None:
            return None
        level = self.agent.current_level()
        return level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number > self.medusa_level[1]

    def _in_own_pit(self):
        """Standing in the pit we dug here: the hole is 4 dig turns away (dig.c: effort 0 -> 250, doubling)."""
        agent = self.agent
        return self._pit_at == (agent.current_level().key(), (agent.blstats.y, agent.blstats.x))

    def _dig_max_wet(self):
        """Wet neighbours accepted for a dig square on this level: 0 if any reachable dry floor exists, else
        the fewest any reachable floor square has (Medusa's islands: a hole floods with probability n/(n+1)
        per dig phase for n moat neighbours, fillholetyp); None if no reachable square can be dug at all
        (every one flooded or refused -- stranded)."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        turn = agent.blstats.time
        if self._max_wet_cache is not None and self._max_wet_cache[:2] == (turn, key):
            return self._max_wet_cache[2]
        dis = agent.bfs()
        candidates = utils.isin(level.objects, PLAIN_FLOOR) | ((level.objects == -1) & level.walkable)
        if DIG_IN_PITS:
            candidates |= utils.isin(level.objects, PITS)
        floor = [p for p in zip(*candidates.nonzero()) if dis[p] >= 0]
        max_wet = 0
        if not any(self._diggable_spot(*p) for p in floor):
            wet = [self._wet_neighbours(*p) for p in floor if self._diggable_spot(*p, max_wet=8)]
            max_wet = min(wet) if wet else None
        self._max_wet_cache = (turn, key, max_wet)
        return max_wet

    def _near_hostiles(self, radius=2):
        agent = self.agent
        y0, x0 = agent.blstats.y, agent.blstats.x
        return [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - y0), abs(m[2] - x0)) <= radius]

    # mthrowu.c thitu: 'You are hit by <missile>'; buzz()/mbhitm: 'The bolt of lightning hits you!', 'The wand
    # hits you!'. Melee hits read 'The <monster> hits!' (no 'you').
    _RANGED_HIT = re.compile(r"You are hit by |\bhits you[!.]")

    def shot_recently(self):
        """RANGED_ON_ELB: a missile, wand or ray hit us within RANGED_BREAK_TURNS. Elbereth stops only melee
        (monmove.c: a scared monster doesn't melee, but m_move then 'shoot on same turn'), so a hold that
        keeps standing on its square just takes the volleys: base6-public s5 rested on Elbereth at 11/59 HP
        while hill orcs threw daggers and zapped a wand of striking at it until it died (T8806)."""
        return jf_config.RANGED_ON_ELB and \
            self.agent.blstats.time - self._ranged_hit_turn <= jf_config.RANGED_BREAK_TURNS

    def _lone_weak_deadly(self, monster):
        """LONE_WEAK_THREAT: the lone-weak exemption above keys on the base level (mlevel <= 2), which takes in the
        grind's worst killers -- rothes (3 attacks, 1d3/1d3/1d8), giant bats (speed 22), giant ants (speed 18),
        dwarves with mattocks (d12), hill orcs, hobbits, were-creatures in animal form -- so below
        ELBERETH_REST_BELOW the bot fought them on to death instead of resting on Elbereth (all of these respect it,
        monmove.c onscary). With the flag the exemption holds only while the monster's own melee
        (nhmodel.prayer.monster_turn_damage: mhitu.c to-hit against our AC, its attacks and speed) leaves P(it deals
        >= our HP within LONE_WEAK_TURNS turns) below LONE_WEAK_PDIE.
        Evidence (tr0/p0/v2a/b3/wz0, 1023 games): 60 shallow deaths end with a single attacker kind of mlevel <= 2 in
        their last 8 turns, 34 of them with no Elbereth rest in their last 40 turns -- v2a wiz-hum-neu-mal s201 and
        wiz-gno-neu-mal s201 (a lone rothe after the force bolts ran out, 10-11 HP to dead, no engraving), p0
        wiz-hum-neu-mal s202 (a hobbit from 13 HP), tr0 bar-hum-cha-mal s211 (a rothe at 7-9 HP)."""
        if not jf_config.LONE_WEAK_THREAT:
            return False
        try:
            import math
            from .nhmodel.prayer import _phi, monster_turn_damage
            bl = self.agent.blstats
            name = getattr(monster[3], 'mname', 'unknown')
            m1, v1, spd = monster_turn_damage(name, int(bl.armor_class), int(bl.depth), int(bl.experience_level))
            turns = jf_config.LONE_WEAK_TURNS
            mean, var = m1 * spd * turns, v1 * spd * turns
            if mean <= 0:
                return False
            p_die = 1.0 - _phi((bl.hitpoints - 0.5 - mean) / math.sqrt(max(var, 1.0)))
            return p_die >= jf_config.LONE_WEAK_PDIE
        except Exception:
            return False

    @Strategy.wrap
    @_hold_loop
    def elbereth_rest(self):
        agent = self.agent
        bl = agent.blstats
        if self.shot_recently():
            self._elbereth_resting = False
            yield False
        resting = self._elbereth_resting
        threshold = ELBERETH_REST_UNTIL if resting else ELBERETH_REST_BELOW
        # a fast hitter (a leocrotta took a dive from 100 to 14 HP in 6 turns) can't be outrun: hide
        # behind Elbereth as soon as HP falls fast, not only below 40%
        falling = not resting and self._fast_hp_loss()
        if (bl.hitpoints >= threshold * bl.max_hitpoints and not falling) or \
                agent.current_level().dungeon_number == GEHENNOM:
            self._elbereth_resting = False
            yield False
        if DIG_ESCAPE and self._dig_escape_action() is not None:
            # a digger digs on its Elbereth instead of resting on it: the hole leaves this level's monsters
            # behind (base-public s0 rested among Medusa-4's snakes, then fought them from the square)
            self._elbereth_resting = False
            yield False
        near = self._near_hostiles()
        # a lone weak monster is better killed than hidden from (engraving gives it a free hit)
        if len(near) == 1 and getattr(near[0][3], 'mlevel', 99) <= 2 and bl.hitpoints >= 6 and \
                not self._lone_weak_deadly(near[0]):
            self._elbereth_resting = False
            yield False
        # REST_FIGHT_WEAK: ... at any HP when one blow kills it (makemon difficulty <= 2, not faster than us): the
        # engraving is a sure free attack and a dust typo (1 in 25 letters, 28% per word) another; our swing kills
        # ~3 times in 4. With HOLD_LOOP the rest really holds, and an XL2 at 5/20 HP died engraving twice next to a
        # goblin that the leaked attack used to kill (gc-h3smoke public s4 vs base3)
        if jf_config.REST_FIGHT_WEAK and len(near) == 1 and getattr(near[0][3], 'difficulty', 99) <= 2 and \
                getattr(near[0][3], 'mmove', 99) <= 12 and \
                getattr(near[0][3], 'mname', '') not in _only_ranged_monsters():
            self._elbereth_resting = False
            yield False
        if not near or any(self._ignores_elbereth(m[3]) or
                           (jf_config.LMINION_ELBERETH and self._lawful_minion(m[3])) for m in near) or \
                agent.character.prop.blind or agent.character.prop.polymorph:
            self._elbereth_resting = False
            yield False
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth' and not agent.can_engrave():
            self._elbereth_resting = False
            yield False
        yield True
        if not self._elbereth_resting:
            agent.log(f'ELBERETH rest start: {[m[3].mname for m in near]}')
        self._elbereth_resting = True
        self._hold_squares.add((agent.current_level().key(), bl.y, bl.x))
        if engraving != 'elbereth':
            agent.engrave('Elbereth')
            return
        agent.search()

    WATER_DEMON = None
    RAVEN = None
    TITAN = None
    MINOTAUR = None

    @Strategy.wrap
    @_hold_loop
    def water_demon_vigil(self):
        """A water demon released by a fountain dip summons other demons whenever it attacks in melee
        (mhitu.c, 1 in 13): one fetched Yeenoghu within two turns. Demons respect Elbereth and a scared
        monster doesn't melee, so stand on Elbereth (never attacking from it) while one is close."""
        agent = self.agent
        turn = agent.blstats.time
        if 'You unleash a water demon' in agent.message:
            self._demon_vigil_until = turn + (jf_config.DEMON_VIGIL_TURNS if jf_config.DEMON_FIX else 150)
        if self.shot_recently():
            yield False
        if turn > self._demon_vigil_until or agent.current_level().dungeon_number == GEHENNOM:
            yield False
        if DiveLogic.WATER_DEMON is None:
            DiveLogic.WATER_DEMON = MON.from_name('water demon')
        y0, x0 = agent.blstats.y, agent.blstats.x
        # DEMON_FIX: hold while a demon is anywhere close, not just within 2 -- at 3+ fight2 threw daggers
        # at it from the Elbereth square, which erases the engraving (jf14 s11: 'Olbereth', then melee)
        radius = jf_config.DEMON_VIGIL_RADIUS if jf_config.DEMON_FIX else 2
        all_demons = list(zip(*utils.isin(agent.glyphs, [DiveLogic.WATER_DEMON]).nonzero()))
        demons = [p for p in all_demons if max(abs(p[0] - y0), abs(p[1] - x0)) <= radius]
        if not demons:
            yield False
        engraving = (agent.inventory.engraving_below_me or '').lower()
        level = agent.current_level()
        # DEMON_FIX: the dip leaves us standing on the fountain, where 'You can't write on the fountain!':
        # the old vigil gave up there and fought -- the demon summoned three more in three turns and an
        # XL7 died at T14764 (jf16 s8). Step off to the free square farthest from the demons first.
        if jf_config.DEMON_FIX and engraving != 'elbereth' and \
                (level.objects[y0, x0] in G.FOUNTAIN or not agent.can_engrave()):
            spot = self._demon_step_off(all_demons)
            if spot is None:
                yield False
            yield True
            agent.log(f'DEMON vigil: stepping off the fountain to {spot} to engrave')
            agent.move(*spot)
            return
        if engraving != 'elbereth' and not agent.can_engrave():
            yield False
        yield True
        if engraving != 'elbereth':
            agent.log('DEMON vigil: Elbereth against the water demon')
            agent.engrave('Elbereth')
            return
        agent.search(1)

    def _demon_step_off(self, demons):
        """An adjacent square we can walk to and engrave on: known floor/corridor (not a fountain, trap,
        door or water), no monster on it; the one farthest from the demons."""
        agent = self.agent
        level = agent.current_level()
        y0, x0 = agent.blstats.y, agent.blstats.x
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = y0 + dy, x0 + dx
                if (dy, dx) == (0, 0) or not (0 <= y < level.walkable.shape[0] and 0 <= x < level.walkable.shape[1]):
                    continue
                terrain = level.objects[y, x]
                if not level.walkable[y, x] or not (terrain in PLAIN_FLOOR or terrain == -1) or \
                        agent.monster_tracker.monster_mask[y, x] or level.intact_doors[y0, x0] and dy and dx:
                    continue
                far = min((max(abs(p[0] - y), abs(p[1] - x)) for p in demons), default=9)
                if best is None or far > best[0]:
                    best = (far, (y, x))
        return None if best is None else best[1]

    @Strategy.wrap
    def leave_minetown_hallucinating(self):
        """Hallucinating in Minetown every monster looks random, so any fight may hit a peaceful and bring
        the Watch (killing even an angry watchman is murder: Luck -2). Take the nearest stairs out."""
        agent = self.agent
        level = agent.current_level()
        if not agent.character.prop.hallu or level.key() != agent.global_logic.minetown_level:
            yield False
        dis = agent.bfs()
        stairs = [(dis[y, x], y, x) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP, G.STAIR_DOWN).nonzero())
                  if dis[y, x] != -1]
        if not stairs:
            yield False
        yield True
        _, y, x = min(stairs)
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            agent.go_to(y, x)
            return
        agent.log('HALLU leaving Minetown')
        agent.move('<' if level.objects[y, x] in G.STAIR_UP else '>')

    @Strategy.wrap
    @_hold_loop
    def faint_shelter(self):
        """Starving (Weak or worse), no food carried, prayer not yet safe: wait on Elbereth."""
        agent = self.agent
        bl = agent.blstats
        if self.shot_recently():
            yield False
        # after a failed prayer no prayer is coming (the god stays angry): waiting only starves, and it
        # blocked the rescue dive (two replays sat on Elbereth until 'died of starvation')
        # a vault guard (we were teleported into a vault) must be answered and followed out, and guards
        # ignore Elbereth: a sheltering XL7 ignored "Please drop that gold and follow me" and was killed
        if utils.isin(agent.glyphs, G.GUARD).any():
            yield False
        if not jf_config.FAINT_SHELTER or bl.hunger_state < Hunger.WEAK or agent.prayer_failed or \
                agent.current_level().dungeon_number == GEHENNOM or agent.character.prop.blind or \
                agent.edible_carried_food():
            yield False   # (carried food the bot won't eat, e.g. wolfsbane or a sacrifice corpse, doesn't count)
        if bl.hunger_state >= Hunger.FAINTING:
            prayer_due = agent.fainting_prayer_due()
        else:
            prayer_due = agent.is_safe_to_pray(agent.SAFE_HUNGER_PRAYER_GAP)
        near = self._near_hostiles()
        # the prayer comes next; a monster Elbereth can't stop is the fight's business
        if prayer_due or any(self._ignores_elbereth(m[3]) for m in near):
            yield False
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth' and not agent.can_engrave():
            yield False
        # This runs above fight2: fighting while Fainting is what kills (each faint is several helpless
        # turns; a pony killed a fainting XL8 that kept trading blows). With everything near respecting
        # Elbereth, engrave even next to a monster (one free hit) and then just wait on it.
        yield True
        if engraving != 'elbereth':
            agent.log(f'FAINT shelter: Elbereth while starving {[m[3].mname for m in near]}')
            agent.engrave('Elbereth')
            return
        agent.search(1 if near else 5)

    _PASSIVE_SESSILE = frozenset(('brown mold', 'yellow mold', 'green mold', 'red mold', 'shrieker', 'floating eye',
                                  'acid blob', 'gas spore', 'lichen'))

    @Strategy.wrap
    @_hold_loop
    def faint_guard(self):
        """Weak or Fainting, no prayer due yet, and a monster close: stand on Elbereth instead of fighting.

        eat.c newuhs(): the Weak->Fainting transition always faints at once (u.uhs <= WEAK), and every
        later turn faints with rn2(20 - uhunger/10) >= 19 (10%/turn at uhunger -10, 37% at -100), each
        faint lasting 10 - uhunger/10 turns. With no prayer before the 1100-turn gap, 8 of the 18 Dlvl-1
        grind deaths in the HEAD baselines were fainted melee deaths at 63-84 HP (hill orcs public s3/s7,
        a single bat jf14 s6, rothes jf14 s12/jf26 s9, werejackal pack jf25 s14). Scared monsters don't
        melee (monmove.c distfleeck/onscary), so a faint on an Elbereth square is safe from them. Unlike
        FAINT_SHELTER this only holds while a monster is close, so the grind still hunts and eats the rest
        of the time (a lone newt/rat/jackal is still fought: it is food and can't kill us)."""
        agent = self.agent
        bl = agent.blstats
        # FAINT_GUARD_RESCUE: after a failed prayer (the rescue dive) no prayer is coming, but a faint next to a
        # monster still kills before starvation does (base2-jf25 s0/s3: fainted on Dlvl 2-3 of the rescue among
        # a wererat pack / a dwarf zombie and kobold mummy); hold only while the threat is close (no idle hold)
        rescue_guard = jf_config.FAINT_GUARD_RESCUE and agent.prayer_failed and self.rescue
        # CAMP_GUARD: a tool-less planned dive (the Mines camp) is guarded like the grind, idle hold included
        camp_guard = jf_config.CAMP_GUARD and self.diving and not self.rescue and not agent.prayer_failed and \
            self.digging_tool() is None
        # CAMP_HUNGER_GAP: the tool-less dive's later hunger cycles wait for the long gap -- on Elbereth, as the tour does,
        # but only once Fainting when idle, and never with a dwarf to hunt in view (CAMP_GUARD held Weak on Elbereth
        # and let a pick-carrying dwarf go: t35 arm-jf16 s11)
        camp_only = not camp_guard and agent.camp_hunger()
        if camp_only and self._hunt_targets():
            yield False
        camp_guard = camp_guard or camp_only
        if not jf_config.FAINT_GUARD or (self.diving and not rescue_guard and not camp_guard) or \
                bl.hunger_state < Hunger.WEAK or \
                (agent.prayer_failed and not rescue_guard) or \
                agent.current_level().dungeon_number == GEHENNOM or agent.character.prop.blind or \
                agent.edible_carried_food():
            yield False
        # a vault guard must be answered and followed (see faint_shelter)
        if utils.isin(agent.glyphs, G.GUARD).any():
            yield False
        if self.shot_recently():
            yield False
        fainting = bl.hunger_state >= Hunger.FAINTING
        # emergency_strategy (above us) prays when the prayer is due; don't hold when it is about to
        if fainting:
            if agent.fainting_prayer_due():
                yield False
        elif agent.is_safe_to_pray(agent._hunger_prayer_gap()) and not agent._eat_before_praying():
            yield False
        y0, x0 = bl.y, bl.x

        def relevant(m):
            # what can't come to us is no threat unless already adjacent, and a passive one not even then
            # (the first version held against a red mold and a floating eye: public s8)
            if getattr(m[3], 'mmove', 12) <= 3:
                return max(abs(m[1] - y0), abs(m[2] - x0)) <= 1 and \
                    getattr(m[3], 'mname', '') not in self._PASSIVE_SESSILE
            return True

        near = [m for m in self._near_hostiles(radius=jf_config.FAINT_GUARD_RADIUS) if relevant(m)]
        # FAINT_GUARD_IDLE: wait out the faints on Elbereth even with nothing in view -- a bat arrived during a
        # faint and killed a 60-HP XL5 before the bot had a conscious turn to react (jf14 s6: 'You stop
        # searching.  The bat misses!  You faint from lack of food.'). From ~30 turns into Weak (the Weak->
        # Fainting faint comes at nutrition 0, ~35-50 turns after Weak begins) until the prayer is due.
        weak_long = self._guard_weak_since is not None and \
            bl.time - self._guard_weak_since >= jf_config.FAINT_GUARD_IDLE_WEAK and not camp_only
        # ... and for a while after holding against a monster: the reactive guard let go when a werejackal
        # fled out of range, the bot walked off its Elbereth and fainted 10 turns later (gc-lf2 public s4)
        idle = jf_config.FAINT_GUARD_IDLE and (fainting or weak_long or bl.time < self._guard_hold_until) and \
            not rescue_guard
        if idle:
            level = agent.current_level()
            here = level.corpses_to_eat.get((y0, x0), {})
            if any(agent._is_corpse_editable(mid, age) for mid, age in here.items()):
                idle = False   # a meal underfoot: let the eating strategy (below us) have it
        if (not near and not idle) or any(self._ignores_elbereth(m[3]) for m in near):
            yield False

        def difficulty(m):
            return getattr(m[3], 'difficulty', 99)

        def trivial(m):
            # can't hurt a fainted 50+ HP Valkyrie much and is a meal: newt, jackal, sewer rat, goblin, kobold
            # (makemon difficulty 1, not faster than us)
            return difficulty(m) <= 1 and getattr(m[3], 'mmove', 99) <= 12

        if idle:
            threat = True
        elif fainting:
            # a rescue's faints only get longer (no prayer is coming): a lone newt bit a fainted 44-HP XL5 to death
            # in ~100 turns (rescue agent's SIM jf14 s8), so there a lone trivial monster is fought only above 70%
            fight_above = 0.7 if rescue_guard else 0.5
            threat = not (len(near) == 1 and trivial(near[0]) and bl.hitpoints >= fight_above * bl.max_hitpoints)
        else:
            # Weak: the faint is still up to ~50 turns away; hold only against what a single faint can't
            # afford: a real fighter (difficulty >= 4: hill orc, rothe, giant ant, dwarf, werejackal), a
            # fast biter (bat, giant bat, little dog), or two monsters above difficulty 1. The first version
            # (monster level >= 2) also held against lone iguanas and kobold lords, which a faint survives.
            threat = any(difficulty(m) >= 4 or (difficulty(m) >= 2 and getattr(m[3], 'mmove', 0) > 12)
                         for m in near) or sum(1 for m in near if not trivial(m)) >= 2
        if not threat:
            yield False
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth' and not agent.can_engrave():
            yield False
        yield True
        if near:
            self._guard_hold_until = bl.time + 20
        if jf_config.DURABLE_ELBERETH:
            target = self._durable_target()
            if target is not None:
                # this level's engraved Elbereth is a few steps away: hold there (it doesn't smudge while we faint)
                if getattr(self, '_durable_walk_logged', None) != (bl.time // 50, target):
                    self._durable_walk_logged = (bl.time // 50, target)
                    agent.log(f'DURABLE walking to our engraved Elbereth at {target} for the hold')
                try:
                    agent.go_to(*target, max_steps=1)
                except AgentPanic:
                    self._durable_walk_blocked_until = bl.time + 20
                return
        key = agent.current_level().key()
        self._hold_squares.add((key, bl.y, bl.x))
        if engraving != 'elbereth':
            if jf_config.DURABLE_ELBERETH:
                here = (int(bl.y), int(bl.x))
                if self._durable_sq.get(key) == here:
                    agent.log(f'DURABLE our engraved Elbereth at {here} reads {engraving!r}: given up')
                    self._durable_sq.pop(key, None)
                tool = agent.durable_engrave_tool() if self._durable_here_ok() else None
                if tool is not None:
                    if agent.engrave_durable(tool):
                        self._durable_sq[key] = here
                    return
            agent.log(f'FAINT guard ({"Fainting" if fainting else "Weak"}{", idle" if idle else ""}): Elbereth vs '
                      f'{[m[3].mname for m in near]} hp={bl.hitpoints}/{bl.max_hitpoints}')
            agent.engrave('Elbereth')
            return
        # one turn at a time while Fainting: a faint interrupting a counted search is read as a longer faint by
        # the faint-length hunger estimate (dive.update), ~30 nutrition too low at XL 7 (grind-food)
        agent.search(1 if near or fainting else 3)

    def _durable_target(self):
        """DURABLE_ELBERETH: this level's engraved Elbereth (y, x) when it is within DURABLE_WALK steps and nothing is
        next to us, else None."""
        agent = self.agent
        bl = agent.blstats
        sq = self._durable_sq.get(agent.current_level().key())
        if sq is None or (int(bl.y), int(bl.x)) == sq or bl.time < self._durable_walk_blocked_until or \
                agent.character.prop.blind or self._near_hostiles(radius=1):
            return None
        dis = agent.bfs()
        return sq if 0 < dis[sq] <= jf_config.DURABLE_WALK else None

    def _durable_here_ok(self):
        """DURABLE_ELBERETH: an engraving here may take its 8 helpless turns -- plain floor, sighted, not levitating,
        nothing hostile within DURABLE_CLEAR, outside Gehennom, and no engraved Elbereth of ours on this level yet."""
        agent = self.agent
        bl = agent.blstats
        level = agent.current_level()
        prop = agent.character.prop
        if prop.blind or prop.hallu or prop.polymorph or self.levitating() or not agent.can_engrave() or \
                level.dungeon_number == GEHENNOM or level.key() in self._durable_sq:
            return False
        if level.objects[bl.y, bl.x] not in G.FLOOR or (bl.y, bl.x) in level.stair_destination:
            return False
        return not self._near_hostiles(radius=jf_config.DURABLE_CLEAR)

    def _step_off_durable(self, here):
        agent = self.agent
        level = agent.current_level()
        y0, x0 = int(here[1]), int(here[2])
        for y, x in agent.neighbors(y0, x0, shuffle=False):
            if level.walkable[y, x] and not agent.monster_tracker.monster_mask[y, x] and \
                    level.objects[y, x] in G.FLOOR:
                agent.log(f'HOLD over: stepping off our engraved Elbereth at {(y0, x0)}')
                try:
                    agent.move(y, x)
                except AgentPanic:
                    agent.search(1)
                return
        agent.search(1)

    @Strategy.wrap
    def wipe_hold_elbereth(self):
        """HOLD_LOOP: once a hold is over (fed and healthy), don't stay on its Elbereth. A scared monster never
        comes to blows and fight2 won't attack from the square (-100), so the grind's search-in-place there
        farmed nothing: grind-food's g15h-public s10 stood on one square from T20000 to T130000 at XL 6, 97
        prayers, then a failed one. Engraving 'x' over it wipes the dust (1 turn); the tour only."""
        agent = self.agent
        bl = agent.blstats
        if not jf_config.HOLD_LOOP or self.diving or not self._hold_squares:
            yield False
        here = (agent.current_level().key(), bl.y, bl.x)
        if here not in self._hold_squares:
            yield False
        durable = jf_config.DURABLE_ELBERETH and self._durable_sq.get(here[0]) == (int(bl.y), int(bl.x))
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth':
            self._hold_squares.discard(here)
            yield False
        # the holds themselves run above us; don't undo a shelter that is about to be needed again
        if bl.hunger_state >= Hunger.WEAK or bl.hitpoints < ELBERETH_REST_UNTIL * bl.max_hitpoints or \
                agent.character.prop.blind or (not durable and not agent.can_engrave()):
            yield False
        yield True
        if durable:
            # DURABLE_ELBERETH: dust can't wipe engraved text ('You cannot wipe out the message ...'), and the square
            # is kept for the next hold: step off it (it stays a hold square, so this runs whenever we stop on it)
            self._step_off_durable(here)
            return
        self._hold_squares.discard(here)
        agent.log(f'HOLD over: wiping our Elbereth at {here[1:]}')
        agent.engrave('x')

    def _fast_hp_loss(self):
        bl = self.agent.blstats
        recent = [hp for t, hp in self._hp_history if t >= bl.time - RETREAT_WINDOW]
        return bool(recent) and max(recent) - bl.hitpoints >= RETREAT_FAST_LOSS * bl.max_hitpoints

    def _crowded_arrival(self):
        """Just walked down into a crowd or next to a boss: (level key, stairs above) to retreat to."""
        agent = self.agent
        if self._arrived is None:
            return None
        key, turn, above = self._arrived
        if agent.current_level().key() != key or agent.blstats.time - turn > ARRIVAL_WATCH_TURNS:
            return None
        near = self._near_hostiles(radius=CROWD_RADIUS)
        if ARRIVAL_FIX:
            # once the retreat works it must not loop: arm64 jf16 s3 came back down onto four giant ants and a
            # leprechaun at the foot of the Dlvl 3 stairs 14 times (every ~160 turns, T19785-T21991); and trivial
            # monsters (newts) don't make a crowd. One retreat per staircase; count monsters of level >= 2.
            if self._crowd_retreats.get(above, 0) >= 1:
                return None
            near = [m for m in near if getattr(m[3], 'mlevel', 99) >= 2 or
                    getattr(m[3], 'mname', '') in BOSS_MONSTERS]
        if len(near) >= CROWD_SIZE or any(getattr(m[3], 'mname', '') in BOSS_MONSTERS for m in near):
            return above
        return None

    def _up_leads_to_medusa(self):
        """MEDUSA_NO_RETREAT: this level's '<' may open onto Medusa's '>' -- on her island, next to her, whose gaze
        stones anyone who sees her (medusa.des: she sits on or beside the down stairs). True for the level right
        below a known Medusa level, and below any unvisited Dungeons-of-Doom level deep enough to be hers (a hole
        can drop us past her level unseen)."""
        if self.agent.character.prop.blind:
            return False
        level = self.agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM:
            return False
        above = (level.dungeon_number, level.level_number - 1)
        if self.medusa_level is not None:
            return tuple(int(v) for v in self.medusa_level) == above
        return above[1] >= MEDUSA_MIN_DEPTH and above not in self.agent.levels

    def _raven_up_stairs(self):
        """The raven island's '<' (y, x) within RAVEN_CYCLE_STEPS, else None."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        ups = {(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())}
        # the '<' we came down by shows us, not the stairs: the stair memory knows it
        ups |= {(int(p[0]), int(p[1])) for p, dest in level.stair_destination.items()
                if dest[0][0] == level.dungeon_number and dest[0][1] < level.level_number}
        reachable = [(dis[p], p) for p in ups if 0 <= dis[p] <= RAVEN_CYCLE_STEPS]
        return min(reachable)[1] if reachable else None

    def _stairs_lead_to_raven_level(self):
        """Standing on a '>' that leads down onto Medusa-3's raven island."""
        agent = self.agent
        dest = agent.current_level().stair_destination.get((agent.blstats.y, agent.blstats.x))
        return dest is not None and tuple(int(v) for v in dest[0]) in \
            {tuple(int(v) for v in k) for k in self._raven_levels}

    def _raven_level_below(self):
        """The level right below this one is Medusa-3's raven island (RAVEN_CYCLE: never dig down onto it)."""
        if not (RAVEN_CYCLE and self.medusa_level is not None and self.medusa_level in self._raven_levels):
            return False
        key = self.agent.current_level().key()
        return int(key[0]) == int(self.medusa_level[0]) and int(key[1]) + 1 == int(self.medusa_level[1])

    @Strategy.wrap
    def raven_cycle(self):
        """RAVEN_CYCLE (see there) as a stair-dance on the raven island: with hostiles about (or blind and bleeding)
        walk to the '<' and hold it -- hit what is adjacent, wait for the rest -- and climb when hurt; the ravens
        can't follow, rest_if_hurt heals us upstairs, and the stairs bring us back onto the '<'. The dig plan runs
        only once nothing has come at us for RAVEN_HOLD_IDLE turns."""
        agent = self.agent
        bl = agent.blstats
        if not (RAVEN_CYCLE and self.diving and self.on_medusa_level() and self.medusa_level in self._raven_levels):
            yield False
        if self._raven_cycles >= RAVEN_CYCLE_MAX:
            yield False
        up = self._raven_up_stairs()
        if up is None:
            yield False
        blind = agent.character.prop.blind
        low = bl.hitpoints < RAVEN_CYCLE_HP * bl.max_hitpoints
        hurt = self._hurt_since(bl.time - 2)
        near = self._near_hostiles(radius=RAVEN_DANCE_RADIUS)
        adjacent = [m for m in near if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1]
        if adjacent or hurt:
            self._raven_contact_turn = bl.time
        idle = bl.time - self._raven_contact_turn >= RAVEN_HOLD_IDLE
        if not low and (idle or not (near or hurt)):
            yield False   # quiet: the dig plan goes on
        if self._in_own_pit() and not low and not (blind and hurt):
            yield False   # the pit roll came good: the hole is 1-2 dig turns away
        yield True
        key = agent.current_level().key()
        if (bl.y, bl.x) != up:
            agent.log(f'RAVEN to the < at {up} (hp {bl.hitpoints}/{bl.max_hitpoints}, blind {blind}, '
                      f'near {[m[3].mname for m in near[:3]]})')
            self._raven_step_toward(up)
            return
        if low or (blind and bl.hitpoints < RAVEN_BLIND_HP * bl.max_hitpoints):
            agent.log(f'RAVEN cycle {self._raven_cycles + 1}: up the stairs to heal '
                      f'(hp {bl.hitpoints}/{bl.max_hitpoints}, blind {blind})')
            agent.move('<')
            if agent.current_level().key() != key:
                self._raven_cycles += 1
            return
        # hold the '<': hit what is next to us, wait for the rest to come
        if adjacent:
            m = min(adjacent, key=lambda m: m[0])
            agent.log(f'RAVEN holding the <: hitting {m[3].mname} at {(int(m[1]), int(m[2]))}')
            agent.step(A.Command.FIGHT)
            agent.direction(int(m[1]), int(m[2]))
            return
        agent.search(1)

    def _raven_step_toward(self, target):
        """One step towards `target`; a monster in the way (seen or not) gets hit instead."""
        agent = self.agent
        start = (agent.blstats.y, agent.blstats.x)
        try:
            agent.go_to(*target, max_steps=1)
            return
        except AgentPanic as e:
            m = re.search(r'Monster on a next tile when moving: \((\d+), ?(\d+)\)', str(e))
            if m is None or (agent.blstats.y, agent.blstats.x) != start:
                raise
            y, x = int(m.group(1)), int(m.group(2))
        agent.log(f'RAVEN hitting the monster in the way at {(y, x)}')
        agent.step(A.Command.FIGHT)
        agent.direction(y, x)

    @Strategy.wrap
    def retreat_upstairs(self):
        """Low or fast-falling HP (or a crowded arrival): take the up stairs if they are close."""
        agent = self.agent
        bl = agent.blstats
        # a retreat that can't move (a Grey-elf in the way) must not keep pre-empting the fight: an
        # s4 dive 'retreated' 9 times in 2 turns without fighting back and died
        if bl.time < self._retreat_blocked_until:
            yield False
        # the Valley's '<' leads to the castle's east edge, outside the moat, with no way back down but the
        # castle's trap doors
        if self.in_valley():
            yield False
        if self.xorn_buffer():
            yield False   # (VALLEY_XORN: up is the Valley's dead-end corner; the form's HP is only a buffer)
        # the castle's '<' leads onto the level above -- Medusa's, next to her, when she is castle-1 (2 of 61 real
        # castle kits 'petrified by Medusa' in pwc-dp1); the castle depth is banked, the passage needs us here
        if jf_config.CASTLE_NO_RETREAT and self.castle.castle_key is not None and \
                agent.current_level().key() == self.castle.castle_key:
            yield False   # (also after the castle mode gave up: pwc-dp3 jf27-s0 retreated onto Medusa's level)
        if jf_config.MEDUSA_NO_RETREAT and self._up_leads_to_medusa():
            yield False
        crowd = self._crowded_arrival()
        in_trouble = bl.hitpoints < RETREAT_BELOW * bl.max_hitpoints or self._fast_hp_loss()
        if crowd is None and (RETREAT_BELOW <= 0 or not in_trouble or not self._near_hostiles(radius=3)):
            yield False
        if DIG_ESCAPE and crowd is None and self._dig_escape_action() is not None:
            # digging out under Elbereth beats walking to a '<' with the monsters following (base-jf14 s3 ping-
            # ponged between Medusa-3's ravens and a fire giant upstairs until it died)
            yield False
        if crowd is not None:
            self._avoid_stairs_until[crowd] = bl.time + ARRIVAL_RETREAT_REST
            self._crowd_retreats[crowd] = self._crowd_retreats.get(crowd, 0) + 1
            self._arrived = None
            agent.log(f'RETREAT crowded arrival: {[m[3].mname for m in self._near_hostiles(CROWD_RADIUS)]}')
        level = agent.current_level()
        if level.dungeon_number == Level.SOKOBAN or bl.depth <= 1:
            yield False
        dis = agent.bfs()
        # fleeing across the level from a faster monster only hands it free hits: on fast HP loss
        # (without low HP) take the stairs only if they are a step or two away
        reach = RETREAT_MAX_DISTANCE if (crowd is not None or bl.hitpoints < RETREAT_BELOW * bl.max_hitpoints) else 2
        if ARRIVAL_FIX and crowd is not None and bl.hitpoints >= RETREAT_BELOW * bl.max_hitpoints:
            # a crowd retreat only from (or next to) the '<' we came down: walking back to it through the
            # crowd is what kills (dive-safety, Medusa-3 harness: walking among the ravens)
            reach = 1
        ups = [p for p in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())
               if 0 <= dis[p] <= reach]
        if ARRIVAL_FIX and self._arrival_square is not None and self._arrival_square[0] == level.key():
            # the '<' we came down onto: the map shows us there, not the stairs (see ARRIVAL_FIX)
            p = self._arrival_square[1]
            if 0 <= dis[p] <= reach and p not in ups:
                ups.append(p)
        if not ups:
            yield False
        yield True
        y, x = min(ups, key=lambda p: dis[p])
        agent.log(f'RETREAT upstairs at hp {bl.hitpoints}/{bl.max_hitpoints}')
        # don't come straight back down the same staircase into the same fight
        if self._arrived is not None and self._arrived[0] == level.key():
            self._avoid_stairs_until[self._arrived[2]] = bl.time + ARRIVAL_RETREAT_REST
        start = (bl.y, bl.x, level.key())
        try:
            if (bl.y, bl.x) != (y, x):
                agent.go_to(y, x, max_steps=1)
            if (agent.blstats.y, agent.blstats.x) == (y, x):
                agent.move('<')
        finally:
            if (agent.blstats.y, agent.blstats.x, agent.current_level().key()) == start:
                self._retreat_blocked_until = agent.blstats.time + 15

    # ---------------------------------------------------------------- mines

    def use_mines(self):
        # with a pick-axe, digging the main dungeon beats banking Mines' End
        race_ok = self.agent.character.race in (Character.DWARF, Character.GNOME) or \
            (MINES_TOOL_TRIP and self.diving and not self.rescue)
        return MINES_ROUTE and not self.mines_done and race_ok and \
            (not self.diving or self.digging_tool() is None)

    def _stairs_down(self, level):
        return list(zip(*utils.isin(level.objects, G.STAIR_DOWN).nonzero()))

    def _mines_branch_target(self):
        """(level key, (y, x)) of the Mines branch '>': a known one, else an untried '>' on a level with two."""
        target = None
        for key, level in self.agent.levels.items():
            if level.dungeon_number != Level.DUNGEONS_OF_DOOM:
                continue
            downs = self._stairs_down(level)
            for p in downs:
                dest = level.stair_destination.get(p)
                if dest is not None and dest[0][0] == Level.GNOMISH_MINES:
                    target = (key, p)
            if target is None and len(downs) >= 2:
                untried = [p for p in downs if level.stair_destination.get(p) is None and
                           (key, (int(p[0]), int(p[1]))) not in self._branch_skip]
                if untried:
                    target = (key, untried[0])
            if target is not None:
                break
        return target

    def _search_branch(self):
        """FAST_BRANCH: on Dlvl 2-4 explore only until the branch '>' shows up (a second '>'), instead of the
        XP gate's full exploration of each level first (jf16/4: 1599 turns on Dlvl 2, 2633 on Dlvl 3, before
        even looking for the branch). An exhausted level without it: on down the main '>'. When Dlvl 4 is
        exhausted too, the branch room hides behind a secret door or corridor, or on a level we fell past
        (hunt-v2 jf14/0 zapped a digging wand past its landing on Dlvl 3 and missed the branch base took by
        luck): search walls for BRANCH_HIDDEN_TURNS per level, climbing from Dlvl 4 to 2, then give up."""
        agent = self.agent
        level = agent.current_level()
        depth = agent.blstats.depth
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or not 2 <= depth <= MINES_BRANCH_MAX_DEPTH:
            return False
        found = lambda: self._mines_branch_target() is not None
        budget = BRANCH_SEARCH_TURNS if BRANCH_FIX2 else FULL_EXPLORE_TURNS
        if level.key() not in self.fully_explored and self.turns_on_level() <= budget:
            self._task('search mines branch')
            # BRANCH_FIX2: replanned every EXPLORE_REPLAN_TURNS (EXPLORE_BUDGETS): base3-jf14 s0 oscillated between
            # an item pickup and an exploration target on Dlvl 4 for 2500 turns inside this run
            cond = self._budgeted(found) if BRANCH_FIX2 else found
            if self.exploration(0).until(agent, cond).run(return_condition=True):
                return True
            if found():
                return False
            # AutoAscend's BFS treats known traps as walls: part of the level can lie beyond one (hunt-v2
            # jf16/4 took a teleport trap on Dlvl 4 and never saw the branch '>' base reached in 80 turns)
            if level.key() not in self._branch_trap_walk and utils.isin(level.objects, G.TRAPS).any():
                self._branch_trap_walk.add(level.key())
                agent.log(f'DIVE no Mines branch on {level.key()} yet: exploring past the known traps')
                agent._allow_walking_through_traps_turn = agent._last_turn
                agent.last_bfs_step = -1
                return True
            agent.log(f'DIVE no Mines branch on {level.key()} (explored)')
            self.fully_explored.add(level.key())
        if not BRANCH_HIDDEN_TURNS or (depth < MINES_BRANCH_MAX_DEPTH and not self._branch_hidden):
            return False   # on down the main '>'
        if BRANCH_FIX2 and not self._branch_hidden and \
                any(level.stair_destination.get(p) is None for p in self._stairs_down(level)):
            # the only '>' seen here may be the branch itself (the main one unseen): take it before searching walls
            # -- base3arm-jf25 s9 explored Dlvl 4, saw one '>' (the branch, base2 took it) and searched walls for
            # 1500 turns instead; a main '>' leads to Dlvl 5, and _climb_back_for_branch brings us back
            agent.log(f'DIVE the untried down stairs on {level.key()} may be the branch: taking them')
            return False
        # every candidate level is explored and none showed the branch
        if not self._branch_hidden:
            agent.log('DIVE Mines branch hidden: searching Dlvl 4-2 for it')
            self._branch_hidden = True
        t0 = self._branch_search_start.setdefault(level.key(), agent.blstats.time)
        budget = HIDDEN_SEARCH2_TURNS if HIDDEN_SEARCH2 else BRANCH_HIDDEN_TURNS
        if agent.blstats.time - t0 < budget:
            self._task('search hidden mines branch')
            if HIDDEN_SEARCH2 and self._search_toward_void(found, t0, budget):
                return True
            self.exploration(None).until(agent, lambda: found() or
                                         agent.blstats.time - t0 >= budget).run()
            return True
        if depth > 2:
            ups = list(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))
            if ups and self._take_stairs(ups, '<'):
                return True
            self._task('search hidden mines branch: up stairs')
            if self.exploration(None).until(agent, lambda: any(
                    agent.bfs()[p] != -1 for p in zip(*utils.isin(agent.current_level().objects,
                                                                   G.STAIR_UP).nonzero()))).run(return_condition=True):
                return True
        agent.log('DIVE hidden Mines branch not found on Dlvl 2-4; giving the Mines route up')
        self.mines_done = True
        return False

    def _search_toward_void(self, found, t0, budget):
        """HIDDEN_SEARCH2: the branch room of an 'explored' Dlvl 2-4 hides behind a secret door or corridor.
        Rooms fill the 80 columns, so a wide never-seen region is where it lies (arm jf14 s0: Dlvl 3 blank in
        columns 0-22, Dlvl 2 in 0-28 and 70-79, while AutoAscend's wall search went nearest-first around the
        known rooms and ran out of budget). Search from the reachable square next to the most unseen ground,
        each square once (search_count), 15 turns a time; False when there is no such square."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        unseen = (~level.seen).astype(np.float32)
        unseen[0, :] = 0
        unseen[-1, :] = 0
        void = ndimage.uniform_filter(unseen, size=(7, 15), mode='constant')
        # a search covers the 8 squares around us: skip spots next to one already searched
        covered = ndimage.maximum_filter(level.search_count, size=3, mode='constant') >= 15
        cand = (dis >= 0) & level.walkable & ~covered
        if not cand.any():
            return False
        # stand next to the void: the best void score within 2 squares of the spot
        near_void = ndimage.maximum_filter(void, size=5, mode='constant')
        score = near_void - 0.004 * np.maximum(dis, 0)
        score[~cand] = -np.inf
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        if near_void[y, x] < 0.35:
            return False   # no big blank region left next to anything we can reach
        if (agent.blstats.y, agent.blstats.x) != (int(y), int(x)):
            agent.go_to(int(y), int(x), max_steps=20)
            return True
        agent.log(f'HIDDEN branch search at {(int(y), int(x))} toward the void ({near_void[y, x]:.2f})')
        agent.search(15)
        return True

    @Strategy.wrap
    def trip_branch_strategy(self):
        """FAST_BRANCH for the tour-mode pick trip (jf_config.PICK_TRIP_XL, dive.pick_trip): the tour's
        go_to_level_strategy explores Dlvl 2-4 round-robin with wall searching until the branch turns up (old
        trips: only 16 of 55 reached the Mines within 2500 turns, median 1775 turns). Instead: take a known
        branch '>' (go_to_mines), else explore this level only until a second '>' shows up, else go down the
        main '>'. Yields once the trip is in the Mines: the tour goes on to its target level from there."""
        agent = self.agent
        if not FAST_BRANCH or not self.pick_trip or self.diving:
            yield False
        level = agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or agent.blstats.depth > MINES_BRANCH_MAX_DEPTH:
            yield False
        yield True
        if self._mines_branch_target() is not None:
            if self.go_to_mines():
                return
        elif self._search_branch():
            return
        self._trip_descend()

    def _trip_descend(self):
        """The trip goes down the main '>' of this level, with the pet if it is close (the tour's stairs
        routine waits for it: 0 deaths in 145k old trip turns)."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        downs = [p for p in self._stairs_down(level) if dis[p] != -1 and self._stairs_ok(level, *p)]
        if not downs:
            reachable = lambda: any(agent.bfs()[p] != -1 for p in self._stairs_down(agent.current_level()))
            self._task('trip: look for the stairs down')
            if not self.exploration(None).until(agent, reachable).run(return_condition=True):
                agent.search()
            return
        y, x = min(downs, key=lambda p: dis[p])
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            self._task('trip: to the stairs down')
            agent.go_to(y, x)
            return
        if self.rest_if_hurt():
            return
        waited = self.__dict__.setdefault('_trip_pet_wait', {})
        k = (level.key(), (y, x))
        if agent.has_pet and waited.get(k, 0) < 20 and not utils.any_in(
                agent.glyphs[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2], G.PETS):
            waited[k] = waited.get(k, 0) + 1
            agent.search()   # let the pet catch up
            return
        agent.log(f'TRIP going down the main stairs at {(y, x)}')
        agent.move('>')

    def go_to_mines(self):
        """Head for the Mines branch '>': a known one, else an untried '>' on a level with two."""
        agent = self.agent
        target = self._mines_branch_target()
        if target is None:
            return FAST_BRANCH and (self.diving or self.pick_trip) and self._search_branch()
        key, (y, x) = target
        if self._last_task != 'go to mines branch':
            lv = agent.levels[key]
            agent.log(f'DIVE mines branch candidate {(int(y), int(x))} on {key}; down stairs there: '
                      f'{[((int(a), int(b)), lv.stair_destination.get((a, b))) for a, b in self._stairs_down(lv)]}')
        self._task('go to mines branch')
        if agent.current_level().key() != key:
            path = agent.exploration.get_path_to_level(*key)
            if path is None:
                here = agent.current_level()
                if CLIMB_FIX and self.diving and here.dungeon_number == Level.DUNGEONS_OF_DOOM and \
                        key[1] < here.level_number:
                    # the branch '>' is on a level above and no stairs we know lead there (we came down a hole or
                    # a trap door): climb, don't go down the main '>' (see CLIMB_FIX)
                    return self._climb_back_for_branch()
                return False
            if jf_config.ROBUST_FIXES2:
                return self._branch_path_step(key, (int(y), int(x)), path)
            agent.exploration.follow_level_path_strategy(path, agent.exploration.go_to_strategy).run()
            return True
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            if agent.bfs()[y, x] == -1:
                if not FAST_BRANCH:
                    return False
                # seen but cut off: returning False sent plan_step down the main '>' and the next plan_step
                # back up to it -- hunt-v2 public/13 ping-ponged Dlvl 3 <-> 4 for thousands of turns. Work at
                # it (neighbours, known traps, then exploring) for a while, then skip that '>'.
                k = (key, (int(y), int(x)))
                t0 = self._branch_reach.setdefault(k, agent.blstats.time)
                if agent.blstats.time - t0 > 600:
                    agent.log(f'DIVE candidate branch stairs at {(y, x)} on {key} unreachable: skipped')
                    self._branch_skip.add(k)
                    return True   # replan
                if self._take_stairs([(y, x)], '>'):
                    return True
                self.exploration(None).until(agent, self._budgeted(
                    lambda: agent.bfs()[y, x] != -1 or agent.blstats.time - t0 > 600)).run()
                return True
            agent.go_to(y, x)
            return True
        if self.rest_if_hurt():
            return True
        agent.move('>')
        return True

    def _branch_path_step(self, key, target, path):
        """ROBUST_FIXES2: one staircase of the stair path to the Mines branch level per plan step.
        follow_level_path_strategy walked the whole path in one go and asserted when its first staircase was cut off;
        the assert came back every few steps (s23 cmp-main-jf45 s2: a trap door dropped us into an unexplored corner
        of a known Dlvl 4, 1950 panics and ~1000 turns before a random walk got through). _take_stairs waits out a
        peaceful on the stairs and breaks a boulder there; cut off, we explore for a way to the staircase. After
        BRANCH_PATH_TURNS cut off: an untried candidate '>' is skipped, a known branch staircase gives the Mines route
        up (plan_step then descends the main dungeon)."""
        agent = self.agent
        (sy, sx), _, direction = path[0]
        sy, sx = int(sy), int(sx)
        if self._take_stairs([(sy, sx)], direction):
            return True
        here = agent.current_level().key()
        t0 = self._branch_reach.setdefault((key, target, here, (sy, sx)), agent.blstats.time)
        if agent.blstats.time - t0 > BRANCH_PATH_TURNS:
            dest = agent.levels[key].stair_destination.get(target) if key in agent.levels else None
            if dest is not None and dest[0][0] == Level.GNOMISH_MINES:
                agent.log(f'DIVE staircase {(sy, sx)} on {here} towards the Mines branch {target} on {key} cut off '
                          f'for {BRANCH_PATH_TURNS} turns: giving the Mines route up')
                self.mines_done = True
            else:
                agent.log(f'DIVE staircase {(sy, sx)} on {here} towards the branch candidate {target} on {key} cut '
                          f'off for {BRANCH_PATH_TURNS} turns: candidate skipped')
                self._branch_skip.add((key, target))
            return True   # replan
        self.exploration(None).until(agent, self._budgeted(
            lambda: agent.bfs()[sy, sx] != -1 or agent.blstats.time - t0 > BRANCH_PATH_TURNS)).run()
        return True

    def _branch_levels_pending(self):
        """Some of Dlvl 2-4 still owes the branch search: never seen, not explored, a candidate '>' untried, or the
        hidden-branch search has budget left."""
        if self._mines_branch_target() is not None or self._branch_search_open():
            return True
        if self._branch_hidden:
            return False
        # an unexplored candidate level, or all explored but the hidden-branch search not yet begun
        return BRANCH_HIDDEN_TURNS > 0 or any((Level.DUNGEONS_OF_DOOM, d) not in self.fully_explored
                                              for d in range(2, MINES_BRANCH_MAX_DEPTH + 1))

    def _climb_back_for_branch(self):
        """Below Dlvl 4 with the branch search unfinished: climb back (BRANCH_CLIMB_MAX levels at most).
        base3-public s10 fell through a trap door 3 turns after reaching Dlvl 4 and gave the Mines route up on
        Dlvl 5 (no pick all game); a candidate '>' that was the main one did the same (smoke public/0). The '<' we
        arrived on isn't on our map until we step off it; its recorded destination finds it."""
        agent = self.agent
        level = agent.current_level()
        depth = agent.blstats.depth
        if depth > MINES_BRANCH_MAX_DEPTH + BRANCH_CLIMB_MAX or not self._branch_levels_pending():
            return False
        t0 = self._climb_start.setdefault(level.key(), agent.blstats.time)
        if agent.blstats.time - t0 > BRANCH_CLIMB_TURNS:
            return False
        if CLIMB_FIX:
            ups, capped = self._climb_ups(level)
            if capped and not ups:
                agent.log(f'DIVE climb back: the stairs up at {capped} on {level.key()} taken {BRANCH_STAIRS_CAP} '
                          f'times already; no more climbing from here')
                return False
            here = (int(agent.blstats.y), int(agent.blstats.x))
            if here in ups:   # _take_stairs climbs them now
                self._climb_count[(level.key(), here)] = self._climb_count.get((level.key(), here), 0) + 1
        else:
            ups = set(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))
            ups |= {p for p, d in level.stair_destination.items()
                    if d is not None and d[0] == (Level.DUNGEONS_OF_DOOM, level.level_number - 1)}
        if t0 == agent.blstats.time:
            agent.log(f'DIVE below the Mines branch levels with their search unfinished: climbing back from {depth}')
        self._task('climb back for the Mines branch')
        if ups and self._take_stairs([(int(y), int(x)) for y, x in ups], '<'):
            return True
        if CLIMB_FIX:
            reachable = lambda: any(agent.bfs()[p] != -1 for p in self._climb_ups(agent.current_level())[0])
        else:
            reachable = lambda: any(agent.bfs()[p] != -1 for p in zip(*utils.isin(agent.current_level().objects,
                                                                               G.STAIR_UP).nonzero()))
        if not self.exploration(None).until(agent, self._budgeted(reachable)).run(return_condition=True):
            agent.search()
        return True

    def _climb_ups(self, level):
        """CLIMB_FIX: the '<' of a main-dungeon level that lead, or may lead, to the level above -- not one known to
        go elsewhere (Sokoban's, on the level below the Oracle), nor one the branch climb took BRANCH_STAIRS_CAP
        times already. Returns (usable, capped)."""
        above = (Level.DUNGEONS_OF_DOOM, level.level_number - 1)
        dest_key = lambda d: (int(d[0][0]), int(d[0][1]))
        ups = {(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())}
        ups |= {(int(p[0]), int(p[1])) for p, d in level.stair_destination.items()
                if d is not None and dest_key(d) == above}
        usable, capped = [], []
        for p in sorted(ups):
            d = level.stair_destination.get(p)
            if d is not None and dest_key(d) != above:
                continue
            (capped if self._climb_count.get((level.key(), p), 0) >= BRANCH_STAIRS_CAP else usable).append(p)
        return usable, capped

    def _wand_waits(self):
        """WAND_WAITS2: WAND_FIRST's zap at any hostile in view waits while the Mines route (a dwarf's pick) is
        pending: on Dlvl 1-4, or climbing back to them. base4-jf14 s1 zapped its wand of digging at a goblin 4
        squares away on Dlvl 2, then at a giant rat, a giant beetle and three homunculi, and landed tool-less on
        Dlvl 7; s14 zapped at a floating eye on Dlvl 2 and at a leprechaun on Dlvl 3, and never reached the
        branch it had seen (A027 without WAND_FIRST: pick-axe from Mines 1 at T11667, same seed)."""
        if not WAND_WAITS2 or not self.use_mines():
            return False
        limit = MINES_BRANCH_MAX_DEPTH + (BRANCH_CLIMB_MAX if BRANCH_FIX2 else 0)
        return self.agent.current_level().dungeon_number == Level.DUNGEONS_OF_DOOM and \
            self.agent.blstats.depth <= limit

    def _branch_search_open(self):
        """FAST_BRANCH's hidden-branch search still has budget on some of Dlvl 2-4."""
        if not self._branch_hidden:
            return False
        t = self.agent.blstats.time
        for depth in range(2, MINES_BRANCH_MAX_DEPTH + 1):
            start = self._branch_search_start.get((Level.DUNGEONS_OF_DOOM, depth))
            if start is None or t - start < BRANCH_HIDDEN_TURNS:
                return True
        return False

    def mines_step(self):
        """Stairs-first descent to Mines' End; the bottom is the level without a '>'."""
        agent = self.agent
        level = agent.current_level()
        need_xl = MINES_REQUIRED_XL.get(level.level_number + 1, 0)
        if agent.blstats.experience_level < need_xl and level.key() not in self.fully_explored and \
                self.turns_on_level() < FULL_EXPLORE_TURNS:
            self._task('mines level-up')
            if not self.exploration(0).run(return_condition=True):
                self.fully_explored.add(level.key())
            return
        dis = agent.bfs()
        seen = self._stairs_down(level)
        downs = [p for p in seen if dis[p] != -1]
        if not downs:
            # a '>' in view but cut off by known traps: walk through (the safe kinds of) them, as descend() does in
            # the main dungeon -- a robustness replay (public s6) fell through a trap door into a Mines level 4
            # pocket whose only way out was a trap and searched it for 4000 turns with the '>' in view
            if seen and jf_config.TRAP_LAST_RESORT > 0 and self._walk_through_traps():
                return
            reachable = lambda: any(agent.bfs()[p] != -1 for p in self._stairs_down(agent.current_level()))
            if self.exploration(0).until(agent, reachable).run(return_condition=True):
                return
            if not seen and level.level_number >= MINES_MIN_LEVELS:
                agent.log(f'DIVE Mines bottom reached at depth {agent.blstats.depth}')
                self.mines_done = True
                return
            # MINES_UNSTUCK: a tool-less dive gives the route up sooner when the '>' is cut off (base2-public s8
            # and s14 stood 4000-6000 turns on Mines 1 first; the levels above are searched already)
            stuck = MINES_STUCK_TURNS if (MINES_UNSTUCK and self.digging_tool() is None) else STUCK_EXPLORE_TURNS
            # a '>' is known but cut off: search for another way, within a budget
            if self.turns_on_level() > stuck:
                agent.log('DIVE Mines stairs unreachable; giving the Mines route up')
                self.mines_done = True
                return
            self.exploration(None).until(agent, self._budgeted(
                reachable, lambda: self.turns_on_level() > stuck)).run()
            return
        y, x = min(downs, key=lambda p: dis[p])
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            agent.go_to(y, x)
            return
        if self.rest_if_hurt():
            return
        agent.move('>')

    def rest_if_hurt(self):
        """Never take a way down while hurt. With nothing around, rest; with monsters around, wait a
        turn and let the fight logic deal with them (b4 stair-danced into a gargoyle: retreat up,
        the gargoyle followed, 'nothing to rest from' was false, so it went straight back down)."""
        agent = self.agent
        if self.xorn_buffer() or self._mino_alert():
            return False
        digger = DIVE_REST and self.diving and self.digging_tool() is not None
        # a digger takes stairs like a hole: a deep rest to 95% at XL 8 (1 HP per 5 turns) lets the level's
        # monsters come (base-jf25 s13 rested 180 turns at a Dlvl 14 '>' and died there)
        threshold = DIG_REST_BELOW if digger else REST_BEFORE_DESCEND
        # RAVEN_CYCLE: these stairs lead back onto the raven island -- only at full strength and seeing
        raven = RAVEN_CYCLE and self._stairs_lead_to_raven_level()
        if raven:
            threshold = RAVEN_CYCLE_REST
        if agent.blstats.hitpoints >= threshold * agent.blstats.max_hitpoints and \
                not (raven and agent.character.prop.blind):
            return False
        if raven and agent._hurt_recently(3):
            # something up here is hurting us: fight it here (fight2 preempts), not among the ravens
            self._task('hold before the raven island')
            agent.search(1)
            return True
        if digger and agent._hurt_recently(3):
            # something is hurting us right here (base-jf16 s7 rested on a '>' Elbereth while a rock troll's
            # partisan reached it from two squares away, 17 -> 10 HP): the stairs are the escape
            return False
        # starving with no prayer left: resting only faints the time away (rescue dives rested at the stairs
        # of Dlvl 1-2 until they starved); go on unless badly hurt
        if self.starving() and agent.blstats.hitpoints >= 0.4 * agent.blstats.max_hitpoints:
            return False
        self._task('rest before descending')
        if digger and self._rest_elbereth():
            return True
        agent.search(1 if agent.get_visible_monsters() else 20)
        return True

    def _digger_here(self):
        """Diving with a usable digging tool on a level we can still dig through."""
        agent = self.agent
        level = agent.current_level()
        return self.diving and level.dungeon_number in MAIN_LINE and level.dungeon_number != GEHENNOM and \
            level.key() not in self.undiggable and self.digging_tool() is not None

    def _hunger_hold(self):
        """HUNGER_DEEP: Weak or Fainting while diving, nothing edible carried, the god not angry, and the hunger
        prayer not due yet (agent's gap for this hunger state and depth): don't dig deeper for now. Not on Medusa's
        level (its islands are no place to wait) and not where no prayer comes (Gehennom)."""
        agent = self.agent
        bl = agent.blstats
        if not agent.hunger_deep() or bl.hunger_state < Hunger.WEAK or agent.prayer_failed or \
                self.in_gehennom() or self.on_medusa_level():
            return False
        if agent.edible_carried_food():
            return False   # agent.eat_deep eats it
        gap = agent._faint_prayer_gap() if bl.hunger_state >= Hunger.FAINTING else agent._hunger_prayer_gap()
        return not agent.is_safe_to_pray(gap)

    def _rest_elbereth(self):
        """DIVE_REST: engrave Elbereth before resting, so that what arrives meanwhile can't melee us (the
        deep rests that killed diggers were on bare floor). Capped per square; not while blind (unreadable)."""
        agent = self.agent
        if agent.current_level().dungeon_number == GEHENNOM or agent.character.prop.blind or \
                agent.character.prop.polymorph or not agent.can_engrave() or \
                (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
            return False
        spot = (agent.current_level().key(), agent.blstats.y, agent.blstats.x, 'rest')
        tries = self.__dict__.setdefault('_elbereth_tries', {})
        if tries.get(spot, 0) >= ELBERETH_TRIES_ESCAPE:
            return False
        tries[spot] = tries.get(spot, 0) + 1
        agent.log('DIVE Elbereth before resting')
        agent.engrave('Elbereth')
        return True

    # ------------------------------------------------------------- descend

    def starving(self):
        """Weak or worse after a failed prayer (the god stays angry) with nothing edible carried."""
        agent = self.agent
        return agent.prayer_failed and agent.blstats.hunger_state >= Hunger.WEAK and not agent.edible_carried_food()

    # ------------------------------------------------------------- PREP track (prep lane)

    def excalibur_candidate(self):
        """The long sword a fountain dip could turn into Excalibur (global_logic.dip_for_excalibur's rule), None once
        Excalibur is carried (its dmg_bonus is known) or with no long sword."""
        candidate = None
        for item in flatten_items(self.agent.inventory.items):
            if item.is_unambiguous() and item.object == O.from_name('long sword'):
                if item.dmg_bonus is not None:
                    return None
                candidate = item
        return candidate

    def _prep_fountain_target(self):
        """PREP_EXCAL_DIVE: the nearest reachable fountain within PREP_EXCAL_DIST steps on a Dungeons of Doom level
        while a dip is safe (HP >= 70%, not Weak, no hostile within PREP_EXCAL_CALM squares, no released water demon
        about, not levitating), or None."""
        agent = self.agent
        bl = agent.blstats
        level = agent.current_level()
        if not jf_config.PREP_EXCAL_DIVE or not self.diving or self.rescue:
            return None
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or agent.character.alignment != Character.LAWFUL:
            return None
        if bl.experience_level < 5 or bl.hunger_state >= Hunger.WEAK or bl.hitpoints < 0.7 * bl.max_hitpoints:
            return None
        if bl.time <= max(self._demon_vigil_until, self._prep_dip_block_until) or self.levitating():
            return None
        fountains = utils.isin(level.objects, G.FOUNTAIN)
        if not fountains.any() or self.excalibur_candidate() is None:
            return None
        dis = agent.bfs()
        mask = fountains & (dis != -1) & (dis <= jf_config.PREP_EXCAL_DIST)
        if not mask.any():
            return None
        for m in agent.get_visible_monsters():
            if max(abs(int(m[1]) - bl.y), abs(int(m[2]) - bl.x)) <= jf_config.PREP_EXCAL_CALM:
                return None
        ys, xs = mask.nonzero()
        i = int(np.argmin(dis[ys, xs]))
        return int(ys[i]), int(xs[i])

    def prep_fountain_dip(self):
        """PREP_EXCAL_DIVE (prep lane): the dig-dive dips for Excalibur at the fountains it passes. The grind dips only
        on its own levels and NO_DIP_WITH_TOOL stops every dip once a pick is carried, so cand-g made Excalibur in 22 of
        270 fresh games. That rule's reason is a dip's silent curse (fountain.c rnd(30) == 16) welding the WIELDED
        sword so the pick can't be applied (eg-glh public/6); here the digging tool (or bare hands) goes in hand first, so a cursed
        sword stays in the pack -- and the starting sword's BUC is known, so the curse shows and get_best_melee_weapon
        never wields it. Each dip: Excalibur 1/6 (lawful, XL >= 5), dry fountain 1/3 -> ~3/8 per fountain; a water
        demon 1/30 (DEMON_FIX's vigil; a wish (20 - depth)% of the time). Returns True when it acted."""
        target = self._prep_fountain_target()
        if target is None:
            return False
        agent = self.agent
        y, x = target
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            self._task('walk to a fountain (Excalibur)')
            agent.go_to(y, x)
            return True
        sword = self.excalibur_candidate()
        main = agent.inventory.items.main_hand
        tool = self.digging_tool()
        if main is not None and main is sword:
            # never dip the sword in hand: the digging tool (or bare hands) first
            if main.status == Item.CURSED or not agent.inventory.wield(tool):
                agent.log('PREP dip: cannot take the sword out of our hand; no dive dips for 500 turns')
                self._prep_dip_block_until = agent.blstats.time + 500
            return True
        self._task('dip for Excalibur (dive)')
        agent.log(f'PREP dip for Excalibur at {(y, x)} (XL {agent.blstats.experience_level}, '
                  f'depth {agent.blstats.depth})')
        with agent.atom_operation():
            sword = agent.inventory.move_to_inventory(sword)
            agent.step(A.Command.DIP)
            agent.type_text(agent.inventory.items.get_letter(sword))
            if ('What do you want to dip ' in agent.message and 'into?' in agent.message) or \
                    "You don't have anything to dip " in agent.message:
                self._prep_dip_block_until = agent.blstats.time + 50
                raise AgentPanic('no fountain here')
        return True

    _PREP_PICKUP_GLYPHS = None

    @classmethod
    def _prep_pickup_glyphs(cls):
        if cls._PREP_PICKUP_GLYPHS is None:
            classes = {nh.RING_CLASS, nh.WAND_CLASS, nh.AMULET_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS}
            cls._PREP_PICKUP_GLYPHS = frozenset(
                g for g in G.NORMAL_OBJECTS if ord(nh.objclass(nh.glyph_to_obj(g)).oc_class) in classes)
        return cls._PREP_PICKUP_GLYPHS

    def prep_dive_pickup(self):
        """PREP_DIVE_PICKUP (prep lane): the dig-dive picks up the rings, wands, amulets, potions and scrolls it sees
        within PREP_PICKUP_DIST steps (Dungeons/Mines levels above PREP_PICKUP_MAX_DEPTH, no hostile within 7, HP >= 60%,
        not Weak, never in a shop); the item priority (arrange_items) decides what stays. The dive never ran item
        pickup (ARMOR_UP's finding), so the castle kit is ~all from the Dlvl 1-3 grind: 0.9 rings, 4.6 potions, 5.4
        scrolls, 1.8 wands per cand-g castle arrival (87 of 270 fresh games), 5 with a known lift. mklev puts ~0.4
        objects in a room, ~40% of them of these classes; each unknown potion is levitation 4.2% of the time, each
        ring levitation / teleport control 1/28, each wand cold 4% (the castle lift plan tries unknown potions and
        rings). Returns True when it acted."""
        if not jf_config.PREP_DIVE_PICKUP or not self.diving or self.rescue:
            return False
        agent = self.agent
        level = agent.current_level()
        bl = agent.blstats
        if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES) or \
                bl.depth > jf_config.PREP_PICKUP_MAX_DEPTH or self.castle.active() or self.on_medusa_level():
            return False
        if bl.hunger_state >= Hunger.WEAK or bl.hitpoints < 0.6 * bl.max_hitpoints or self.levitating():
            return False
        for m in agent.get_visible_monsters():
            if max(abs(int(m[1]) - bl.y), abs(int(m[2]) - bl.x)) <= 7:
                return False
        done = self._prep_picked.setdefault(level.key(), set())
        here = (int(bl.y), int(bl.x))
        if level.shop_interior[here] or level.shop[here]:
            return False
        below = agent.inventory.items_below_me
        if below and here not in done and \
                any(i.category in (nh.RING_CLASS, nh.WAND_CLASS, nh.AMULET_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS)
                    for i in below):
            done.add(here)
            self._task('pick up (prep)')
            agent.log(f'PREP pickup at {here}: {[i.text for i in below][:6]}')
            agent.inventory.pickup_and_drop_items().run()
            return True
        mask = utils.isin(agent.glyphs, self._prep_pickup_glyphs())
        if not mask.any():
            return False
        dis = agent.bfs()
        if level.key() not in self._prep_seen_levels:
            self._prep_seen_levels.add(level.key())
            ds = sorted(int(dis[p]) for p in zip(*mask.nonzero()))
            agent.log(f'PREP pickup candidates on {level.key()}: {len(ds)} squares, BFS distances {ds[:8]}')
        mask &= (dis > 0) & (dis <= jf_config.PREP_PICKUP_DIST) & ~level.shop_interior & ~level.shop
        for y, x in done:
            mask[y, x] = False
        if not mask.any():
            return False
        ys, xs = mask.nonzero()
        i = int(np.argmin(dis[ys, xs]))
        y, x = int(ys[i]), int(xs[i])
        if self._prep_pickup_target == (level.key(), (y, x)) and \
                bl.time - self._prep_pickup_since > jf_config.PREP_PICKUP_DIST * 3:
            done.add((y, x))   # couldn't get there in time (a boulder, a closed door, a peaceful in the way)
            return False
        if self._prep_pickup_target != (level.key(), (y, x)):
            self._prep_pickup_target = (level.key(), (y, x))
            self._prep_pickup_since = bl.time
        self._task('walk to an item (prep)')
        agent.go_to(y, x)
        return True

    def prep_mid_level(self):
        """PREP_MID (prep lane): explore this main-line level fully before digging on -- the middle game's XP, items,
        fountains and armour -- while XL < PREP_MID_XL, the depth is PREP_MID_MIN_DEPTH..PREP_MID_MAX_DEPTH, the turn is
        below PREP_MID_TURN_CAP and this level has had fewer than PREP_MID_TURNS turns. Only with a digging tool in the
        pack: a hole out (DIG_ESCAPE) is the way off a level turned bad, and a danger (_prep_calm) ends the level at
        once. The dig-dive otherwise reaches the castle at XL 7 (median, 360 s23 games, ledger F084): monsters there
        are made for (depth + XL) / 2, and XL 14 needs 80,000 XP -- the grind's Dlvl 1-3 spawns give ~100-160 XP per
        1000 turns (F042), a Dlvl 5-12 level several times that. DIVE_SUPPLY (power-route, R081) tried this for items
        with a strict danger rule (leave at difficulty >= XL-1) and left most levels at once."""
        if not jf_config.PREP_MID or self.rescue or not self.diving:
            return False
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or level.key() in self.fully_explored:
            return False
        bl = agent.blstats
        if not (jf_config.PREP_MID_MIN_DEPTH <= bl.depth <= jf_config.PREP_MID_MAX_DEPTH):
            return False
        if bl.experience_level >= jf_config.PREP_MID_XL or bl.time > jf_config.PREP_MID_TURN_CAP:
            return False
        if self.turns_on_level() > jf_config.PREP_MID_TURNS or self.digging_tool() is None:
            return False
        if level.key() in self.prep_mid_left:
            return False
        if bl.hunger_state >= Hunger.WEAK:
            return False
        monsters = agent.get_visible_monsters()
        if monsters and not self._prep_calm(monsters):
            self.prep_mid_left.add(level.key())
            agent.log(f'PREP mid: leaving {level.key()} at XL {bl.experience_level} HP {bl.hitpoints}/'
                      f'{bl.max_hitpoints}: {[(m[3].mname, getattr(m[3], "difficulty", -1)) for m in monsters[:4]]}')
            return False
        return True

    def _prep_calm(self, monsters):
        """PREP_MID: an ordinary fight -- HP >= PREP_MID_ESCAPE_HP, fewer than PREP_MID_CROWD hostiles within 6, no
        boss kind, nothing that fights on through Elbereth (@ humans and elves, minotaurs: our fallback when a fight
        goes wrong) and no monster of makemon difficulty above XL + PREP_MID_DIFF."""
        bl = self.agent.blstats
        xl = bl.experience_level
        if bl.hitpoints < jf_config.PREP_MID_ESCAPE_HP * bl.max_hitpoints:
            return False

        def dist(m):
            return max(abs(int(m[1]) - bl.y), abs(int(m[2]) - bl.x))

        # a monster of speed <= 3 (ghost, trapper, lurker, molds, jellies, piercers) is no danger beyond 2 squares:
        # pmid1 jf43 s0 left Dlvl 5 and 6 for a ghost (difficulty 12) and Dlvl 7 for a trapper (14)
        live = [m for m in monsters if getattr(m[3], 'mmove', 12) > 3 or dist(m) <= 2]
        # (a crowd counts monsters of difficulty 3+: a pack of jackals or sewer rats is XP, not a danger)
        near = [m for m in live if dist(m) <= 6 and getattr(m[3], 'difficulty', 99) >= 3]
        if len(near) >= jf_config.PREP_MID_CROWD:
            return False
        ignorers = [m for m in live if self._melee_ignores_elbereth(m[3])]
        for m in live:
            name = getattr(m[3], 'mname', '')
            diff = getattr(m[3], 'difficulty', 99)
            if name in BOSS_MONSTERS or diff > xl + jf_config.PREP_MID_DIFF:
                return False
            # an @ or a minotaur fights on through Elbereth, our fallback: only a lone one of difficulty <= XL is
            # fought (digging out next to one costs ~5 turns of free hits: pmid1 jf43 s0 lost 34 of 68 HP leaving
            # Dlvl 9 at a lone Woodland-elf, difficulty 6 at XL 7)
            if m in ignorers and (name == 'minotaur' or len(ignorers) > 1 or diff > xl):
                return False
        return True

    def required_xl(self, depth):
        return REQUIRED_XL.get(depth, REQUIRED_XL[max(REQUIRED_XL)])

    def should_explore_fully(self):
        agent = self.agent
        key = agent.current_level().key()
        if key in self.fully_explored:
            return False
        if self.prep_mid_level():
            return True
        # the XP gate is for the stairs dive: a digger spends a few turns per level (an XL9 s7 dig-dive
        # explored Dlvl 16 'to level up' and met an umber hulk); a rescue has no time for it
        if self.digging_tool() is not None or self.rescue:
            return False
        # no XP gate in Gehennom: every turn spent there brings a spawn from its nastier pool (1 in 50 per turn)
        if self.in_gehennom():
            return False
        if FAST_BRANCH and self.use_mines() and agent.current_level().dungeon_number == Level.DUNGEONS_OF_DOOM \
                and agent.blstats.depth <= MINES_BRANCH_MAX_DEPTH:
            return False  # the Mines route needs only the branch '>' here: go_to_mines looks for it
        if agent.blstats.hunger_state >= Hunger.HUNGRY and agent.inventory.items.total_nutrition() == 0:
            return False  # corpses and food are deeper
        if self.turns_on_level() > FULL_EXPLORE_TURNS:
            return False
        return agent.blstats.experience_level < self.required_xl(agent.blstats.depth + 1)

    def should_farm(self):
        agent = self.agent
        if FARM_TURNS <= 0 or agent.current_level().key() not in self.fully_explored:
            return False
        if agent.blstats.hunger_state >= Hunger.WEAK and not agent.is_safe_to_pray(1000):
            return False  # nothing left to fix hunger here
        if self.turns_on_level() > FULL_EXPLORE_TURNS + FARM_TURNS:
            return False
        return agent.blstats.experience_level < self.required_xl(agent.blstats.depth + 1)

    def farm_xp(self):
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        ups = [p for p in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()) if dis[p] != -1]
        if ups:
            y, x = min(ups, key=lambda p: dis[p])
            if (agent.blstats.y, agent.blstats.x) != (y, x):
                agent.go_to(y, x)
                return
        agent.search(20)

    def digging_wand(self):
        for item in flatten_items(self.agent.inventory.items):
            if item.is_wand() and item.is_unambiguous() and \
                    item.object == O.from_name('digging', nh.WAND_CLASS) and \
                    not self.agent.inventory.is_known_empty(item):
                return item
        return None

    @staticmethod
    def is_digging_tool(item, shield_stuck):
        """A pick-axe, or a dwarvish mattock (both hands: the shield comes off first, so not with a
        cursed shield). A third of the dwarves' digging tools are mattocks."""
        if not item.is_unambiguous():
            return False
        if item.status == Item.CURSED and not (CURSED_PICK_OK and (item.equipped or
                                                                   item.object == O.from_name('pick-axe'))):
            return False
        if item.object == O.from_name('pick-axe'):
            return True
        return item.object == O.from_name('dwarvish mattock') and not shield_stuck

    def _shield_stuck(self):
        shield = self.agent.inventory.items.off_hand
        return shield is not None and shield.status == Item.CURSED

    def best_digging_tool(self, items):
        shield_stuck = self._shield_stuck()
        tools = [i for i in flatten_items(items) if self.is_digging_tool(i, shield_stuck)]
        # the pick-axe first: it is lighter and one-handed
        tools.sort(key=lambda i: i.object != O.from_name('pick-axe'))
        return tools[0] if tools else None

    @Strategy.wrap
    def dig_first(self):
        """Preempts fight2 while diving with a digging tool: finish the hole rather than walk to a fight."""
        agent = self.agent
        if not DIG_FIRST or not self.diving:
            yield False
        level = agent.current_level()
        if level.key() in self.undiggable or level.dungeon_number not in MAIN_LINE or \
                agent.blstats.time < self._dig_blocked_until:
            yield False
        # hardfloor: a pick-axe only digs a pit in the Valley (and a pit holds us for 2-7 turns)
        if self.in_valley() or self._raven_level_below():
            yield False
        monsters = agent.get_visible_monsters()
        # MEDUSA_BLIND_DIG: blind with nothing in view the plan below never digs (wait_out_unexpected_state rests
        # first): dig on from here -- only actions at our own square, no blind walk
        blind_dig = not monsters and self._blind_dig_due()
        if not monsters and not blind_dig:
            yield False   # nothing to run from: the dive plan digs as usual
        if monsters and self.prep_mid_level() and self._prep_calm(monsters):
            yield False   # PREP_MID: an ordinary fight is fought (fight2) -- dig out only when it isn't
        if DIG_ESCAPE and level.dungeon_number != GEHENNOM:
            action = self._escape_next()
            if action is None or (blind_dig and action[0] not in self._blind_actions()):
                yield False
            yield True
            if blind_dig:
                agent.log(f'DIVE blind on Medusa\'s level, nothing in view -> {action[0]} '
                          f'({"MEDUSA_BLIND_DIG" if MEDUSA_BLIND_DIG else "MEDUSA_LIFT float"})')
            # Keep acting while the escape still applies. Returning after one action let one step of the
            # strategy underneath run before this hook fired again (agent.preempt re-runs the chain and checks
            # the hooks only on the next update): fight2's swing from our fresh Elbereth wipes it (uhitm.c
            # u_wipe_engr), its 'wait' wastes a turn (found by grind-combat, HOLD_LOOP). Higher-priority hooks
            # (prayer, retreat, engulfed) still interrupt us on any step.
            key = level.key()
            if not ESCAPE_V2:
                self._escape_act(action)
                return
            for _ in range(DIG_ESCAPE_ACTIONS):
                before = agent.step_count
                self._escape_act(action)
                seen = agent.get_visible_monsters()
                if agent.step_count == before or agent.current_level().key() != key or \
                        not (seen or self._blind_dig_due()):
                    return
                action = self._escape_next()
                if action is None or (not seen and action[0] not in self._blind_actions()):
                    return
            return
        bl = agent.blstats
        near = [m for m in monsters if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= DIG_FIRST_RADIUS]
        if bl.hitpoints < DIG_FIRST_MIN_HP * bl.max_hitpoints:
            yield False
        if near and not self._sheltered_dig_ok(near):
            yield False
        tool = self.digging_tool()
        if tool is None or not self._diggable_spot(bl.y, bl.x) or \
                (tool.object == O.from_name('dwarvish mattock') and agent.inventory.items.off_hand is not None):
            yield False
        # a '>' a few steps away is the plan's business, and so is the portal level
        if self.should_sweep_portal():
            yield False
        yield True
        agent.log(f'DIVE digging out, hostiles at {[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
        self.dig_with_tool(tool)

    def _escape_next(self):
        """dig_first's next DIG_ESCAPE action: ('eel',) first, then _dig_escape_action(); None: not ours."""
        if ESCAPE_V2 and self._eel_hold_escape_due():
            # an eel has wrapped itself around us: its next hit drowns us (mhitu.c AD_WRAP with u.ustuck). An
            # Elbereth makes it flee, and monflee() releases the hero -- the one action that saves us (harness
            # ds-med3 seed 9: crawled out of a flooded hole, grabbed, swung its sword, drowned)
            return ('eel', None)
        action = self._dig_escape_action()
        # the portal level is the plan's business
        if action is None or self.should_sweep_portal():
            return None
        return action

    def _escape_act(self, action):
        agent = self.agent
        level = agent.current_level()
        what, arg = action
        monsters = agent.get_visible_monsters()
        if what == 'eel':
            agent.log('DIVE held by an eel: Elbereth to make it let go')
            self._eel_engraved = agent.blstats.time
            agent.engrave('Elbereth')
            return
        if what == 'scare':
            if MEDUSA_BLIND_DIG and self._sight_back:
                # sight is back: the Elbereth under us may be intact -- look first (no game time; the next
                # escape step re-decides with what the look read)
                self._look_after_sight()
                if (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
                    return
            tries = self.__dict__.setdefault('_elbereth_tries', {})
            tries[arg] = tries.get(arg, 0) + 1
            agent.log(f'DIVE Elbereth to scare {[m[3].mname for m in monsters[:3]]} before walking to a dig square')
            agent.engrave('Elbereth')
            return
        if what == 'wait':
            # DROWN_GUARD: monsters stand on the squares a flooded digger would crawl out to; on Elbereth they flee
            if not agent.character.prop.blind and (agent.inventory.engraving_below_me or '').lower() != 'elbereth' \
                    and self._elbereth_before_digging_escape():
                return
            agent.log(f'DIVE waiting for a way out of a possible flood: {[m[3].mname for m in monsters[:3]]}')
            agent.search(1)
            return
        if what == 'reroll':
            self._medusa_reroll(arg)
            return
        if what == 'cycle_up':
            self._medusa_cycle_climb(arg)
            return
        if what == 'hole_walk':
            self._raven_step_toward(arg)
            return
        if what == 'plunge':
            self._medusa_plunge(arg)
            return
        if what == 'cycle_rest':
            self._task('rest before the plunge')
            if not self._rest_elbereth():
                agent.search(1 if monsters else 10)
            return
        if what == 'blind_engrave':
            tries = self.__dict__.setdefault('_elbereth_tries', {})
            tries[arg] = tries.get(arg, 0) + 1
            self.__dict__.setdefault('_engrave_turn', {})[arg] = agent.blstats.time
            agent.log(f"DIVE blind on Medusa's level: Elbereth ({tries[arg]}), holding here until sight returns")
            agent.engrave('Elbereth')
            return
        if what == 'wait_sight':
            agent.search(1)
            return
        if what == 'raven_gap':
            waits = self.__dict__.setdefault('_raven_gap_waits', {})
            waits[arg] = waits.get(arg, 0) + 1
            if waits[arg] == 1:
                agent.log(f'DIVE RAVEN_GAP: on our Elbereth until no raven is within {jf_config.RAVEN_GAP_RADIUS} '
                          f'(at {arg[1:]}), hostiles at {[(m[3].mname, int(m[0])) for m in monsters[:4]]}')
            agent.search(1)
            return
        if what == 'lift_on':
            self._lift_put_on(arg)
            return
        if what == 'float':
            self._lift_step(arg)
            return
        if what == 'lift_off':
            self._lift_take_off(arg)
            return
        if what == 'hover':
            self._lift_hover()
            return
        if what == 'potion_elbereth':
            tries = self.__dict__.setdefault('_elbereth_tries', {})
            tries[arg] = tries.get(arg, 0) + 1
            agent.log(f'DIVE MEDUSA_POTION_LIFT: Elbereth before a quaff ({tries[arg]})')
            agent.engrave('Elbereth')
            return
        if what == 'potion_wait':
            self._potion_waits[arg] = self._potion_waits.get(arg, 0) + 1
            agent.log(f'DIVE MEDUSA_POTION_LIFT: on Elbereth, waiting for {[m[3].mname for m in monsters[:3]]} to move off')
            agent.search(1)
            return
        if what == 'potion_lift':
            self._potion_quaff(arg)
            return
        if what == 'potion_hold':
            agent.search(1)
            return
        if what == 'freeze':
            cold, d, delta = arg
            key = level.key()
            self._freeze_zaps[key] = self._freeze_zaps.get(key, 0) + 1
            self.__dict__.setdefault('_frozen_dirs', {}).setdefault(
                (key, agent.blstats.y, agent.blstats.x), set()).add(delta)
            agent.log(f'DIVE MEDUSA_FREEZE: {cold.text!r} {d} at a moat neighbour of the dig square '
                      f'({self._freeze_zaps[key]}), hostiles at {[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
            if cold.category == nh.WAND_CLASS:
                agent.zap(cold, d)
            elif jf_config.DEEP_ITEMS:
                # a horn is an instrument: music.c asks 'Improvise?' before the direction (the plain APPLY below
                # left that prompt to the default handler); an empty one only makes a sound (a tooled horn's)
                from . import opp_items
                res = opp_items.play(agent, cold, d)
                if not res.get('direction'):
                    agent.inventory.empty_wands.add(cold.text)
            else:
                with agent.atom_operation():
                    agent.step(A.Command.APPLY)
                    agent.type_text(agent.inventory.items.get_letter(cold))
                    if 'In what direction?' in agent.single_message:
                        agent.direction(d)
            return
        if what == 'zap':
            key = level.key()
            spot = (agent.blstats.y, agent.blstats.x)
            zone = WAND_RESERVE and self._wand_zone()
            if zone:
                self._reserve_zaps += 1
            agent.log(f'DIVE zapping {arg.text!r} down to escape'
                      f'{f" (WAND_RESERVE zone zap {self._reserve_zaps})" if zone else ""}, hostiles at '
                      f'{[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
            agent.zap(arg, '>')
            if agent.current_level().key() != key:
                self._dug_holes[key] = spot
            if agent.current_level().key() == key:
                if 'here is too hard to dig' in agent.message:
                    self.undiggable.add(key)
                elif 'too hard to' in agent.message or 'fills with' in agent.message:
                    self._bad_dig_spots.add((key, spot))
                    if 'fills with' in agent.message and key == self.medusa_level:
                        self._medusa_floods += 1
                elif WAND_RESERVE and 'dig a pit in the' in agent.message:
                    # dig.c dighole: a zap down digs only a pit where Can_dig_down is false -- the Dungeons'
                    # bottom level, the castle (a maze landing at bot x <= 9 that passed as a filler level)
                    agent.log(f'DIVE the wand dug only a pit: undiggable floor at depth {agent.blstats.depth}')
                    self.undiggable.add(key)
                    if level.dungeon_number == Level.DUNGEONS_OF_DOOM and agent.blstats.depth >= 25:
                        self.castle.on_bottom(key)
            return
        if what == 'step':
            agent.log(f'DIVE walking to dig at {arg}, hostiles at {[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
            start = (agent.blstats.y, agent.blstats.x)
            try:
                agent.go_to(*arg, max_steps=1)
            finally:
                # a monster in the way (the move panics): let the fight clear it for a few turns, as
                # retreat_upstairs does
                if (agent.blstats.y, agent.blstats.x) == start:
                    self._dig_walk_blocked_until = agent.blstats.time + 5
            return
        agent.log(f'DIVE digging out, hostiles at {[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
        self.dig_with_tool(arg)

    def _dig_escape_action(self):
        """DIG_ESCAPE: what digging out would do right now -- ('dig', tool), ('step', (y, x)) onto a diggable
        neighbour -- or None (no tool, level not diggable, or an adjacent monster that melees through
        Elbereth: that one is fought). No HP floor: prayer and the other emergencies preempt digging."""
        agent = self.agent
        if not (DIG_ESCAPE and DIG_FIRST and self.diving) or self._raven_level_below():
            return None
        if self.weld_hold():
            return None   # WELD_HOLD: no hole out of this level until WELD_PRAY frees the hands (fight2 fights)
        level = agent.current_level()
        if level.dungeon_number == GEHENNOM or level.key() in self.undiggable or \
                level.dungeon_number not in MAIN_LINE or agent.blstats.time < self._dig_blocked_until or \
                utils.any_in(agent.glyphs, G.SWALLOW):
            return None   # (engulfed: engulfed_fight's business, and an engrave attempt in there forbids the square)
        maze = self._maze_zap()
        if maze is not None:
            return maze   # WAND_RESERVE: a maze below Medusa -- out before its minotaurs arrive
        if self._potion_hold_due():
            return ('potion_hold', None)   # MEDUSA_POTION_LIFT: a quaff's confusion/stun/hallucination is held out
        lift = self._medusa_lift_action()
        if lift is not None:
            return lift   # MEDUSA_LIFT: float to a dry square first (levitating nothing below can dig or engrave)
        bl = agent.blstats
        monsters = agent.get_visible_monsters()
        adjacent = [m for m in monsters if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1]
        # a known wand of digging zapped down holes the floor at once (zap.c zap_dig -> dighole): the escape
        # where the pick-axe can't work, and one flood roll instead of the pick's two on Medusa's islands
        # (top-level only: a wand inside a bag has no letter to zap it by)
        wand = next((i for i in agent.inventory.items if i.is_wand() and i.is_unambiguous() and
                     i.object == O.from_name('digging', nh.WAND_CLASS) and
                     not agent.inventory.is_known_empty(i)), None) if DIG_WAND_ESCAPE else None
        # AT_ELBERETH_FIX + AT_DIG_RADIUS 2: one step further out too -- it is next to us after its move, before the
        # dig's first turn (at3 harness seed 0: a Grey-elf pair and a soldier 2 steps away; four digs started, each
        # stopped by their first attack, with a wield of Excalibur in between each time: 72 -> 11 HP, no pit)
        radius = jf_config.AT_DIG_RADIUS if jf_config.AT_ELBERETH_FIX else 1
        close = adjacent + [m for m in monsters if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) == 2 and 0 < m[0] <= 2] \
            if radius >= 2 else adjacent
        if any(self._melee_ignores_elbereth(m[3]) for m in close):
            # an @ next to us: fight it. Digging beside it is futile: any attack, hit or miss, stops the dig
            # occupation before its next turn (mhitu.c hitmsg/missmu -> stop_occupation; allmain.c runs the
            # occupation only after the monsters' move), so a monster attacking every turn blocks all progress.
            # (dsafe-A jf16 s11 dug on in its pit beside a Grey-elf and a werewolf: 90 -> 12 HP, no hole.)
            if jf_config.STALKER_ZAP_FIX and wand is not None and self._wand_reserved() and \
                    self._followers_only([m for m in close if self._melee_ignores_elbereth(m[3])]):
                return None   # (see jf_config.STALKER_ZAP_FIX) it falls through the hole with us: fight it here
            return self._wand_escape(wand)
        cycle = self._medusa_cycle_action()
        if cycle is not None:
            # MEDUSA_HOLE_CYCLE: no dig on her level; the hole above is the way down ('cycle_hold': fight first)
            return None if cycle[0] == 'cycle_hold' else cycle
        blind_hold = self._blind_holding()
        if adjacent and agent._hurt_recently(2) and not self._elbereth_possible() and not blind_hold:
            # bitten while digging with no Elbereth under us and none to be had here (engrave cap, forbidden
            # square): every attack stops the dig, so a hole takes ~12 turns of free hits -- fight instead
            # (dsafe-t2 jf25 s11 dug on under a soldier ant and a large dog, 58 -> 16 HP in 5 turns)
            return self._wand_escape(wand)
        if jf_config.AT_THREAT_AVOID:
            holding, act = self._at_hold()
            if holding:
                self._at_count(level.key())   # (once per turn: this also runs as other strategies' condition)
                return act   # a zap down, or None: fight2 fights the @ on level ground and try_dig_down waits
        pit_ok = not self._in_own_pit() or (WAND_RESERVE and self._reserve_emergency())
        if WAND_FIRST and wand is not None and pit_ok and not self._wand_waits() and not self._wand_reserved():
            # with hostiles in view the wand's instant hole beats the pick's ~8 turns under attack (and a pit
            # that wipes our Elbereth); it is also the only escape of a dive without a pick-axe (base3arm-jf16
            # s4 zapped its wand of digging on Dlvl 9, then fought the Big Room's crowd on Dlvl 10 until a
            # Green-elf's wand of fire killed it)
            zap = self._wand_escape(wand)
            if zap is not None and self._wet_neighbours(bl.y, bl.x) == 0:
                return zap
        tool = self.digging_tool()
        if tool is None:
            return None
        max_wet = self._dig_max_wet()
        up = self._medusa_reroll_stairs(max_wet)
        if up is not None:
            return ('reroll', up)
        if max_wet is None:
            max_wet = 0   # stranded: nothing to dig here but our own pit, if any
        if WAND_RESERVE and wand is not None and self.on_medusa_level():
            # the kept wand holes the floor at once, even on a dry square (one action instead of ~8 among her
            # snakes and ravens), from the best square by _zap_key -- walking there scared-first as below
            plan = self._zap_plan(wand)
            if plan is not None and plan[0] == 'step' and blind_hold:
                # blind: no walk among unseen monsters -- zap from here if it can hole the floor, else hold
                plan = ('zap', wand) if self._zap_key(bl.y, bl.x) is not None else (self._blind_hold() or plan)
            if plan is not None:
                if plan[0] == 'step' and ESCAPE_V2 and adjacent and not agent.character.prop.blind and \
                        (agent.inventory.engraving_below_me or '').lower() != 'elbereth' and \
                        not agent.character.prop.polymorph and agent.can_engrave():
                    spot = (level.key(), bl.y, bl.x, 'scare')
                    if self.__dict__.get('_elbereth_tries', {}).get(spot, 0) < 2:
                        return ('scare', spot)
                if plan[0] == 'zap':
                    return self._freeze_action() or plan   # MEDUSA_FREEZE: ice first, then the zap is sure
                return plan
        # our own pit is the best square of all (4 dig turns to the hole), even when the map shows the pit
        # there (#terrain or invisibility write S_pit at our square). A staircase we just took shows us, not
        # the stairs, so the map doesn't know it (base-public s6 wasted its first turn on Medusa-3 on "The
        # stairs are too hard to dig in")
        on_stairs = (bl.y, bl.x) in level.stair_destination
        if self._in_own_pit() or (not on_stairs and self._diggable_spot(bl.y, bl.x, max_wet) and
                                  self._dig_order_here_ok(max_wet)) or \
                (level.objects[bl.y, bl.x] in (SS.S_pit, SS.S_spiked_pit) and
                 self._wet_neighbours(bl.y, bl.x) <= max_wet and
                 not (DIG_BOULDER_FIX and (level.key(), (bl.y, bl.x)) in self._bad_dig_spots)):
            freeze = self._freeze_action()
            if freeze is not None:
                return freeze   # MEDUSA_FREEZE: no moat neighbour left -> no flood roll can come up wet
            if max_wet > 0 and wand is not None:
                return ('zap', wand)
            if DROWN_GUARD and self._wet_neighbours(bl.y, bl.x) > 0:
                # a flood here drowns us unless a free land square is next to us (drown -> crawl_destination):
                # base3 lost 7 Medusa digs that way ('You sink like a rock', no crawl). Wait on Elbereth for the
                # monsters standing on our way out to flee; if the ground itself is the problem, dig elsewhere.
                if self._crawl_exits(bl.y, bl.x, count_monsters=False) < DROWN_GUARD_EXITS:
                    target = self._dig_walk_target(max_wet, min_exits=DROWN_GUARD_EXITS)
                    if target is not None and agent.blstats.time >= self._dig_walk_blocked_until:
                        return ('step', target)
                elif self._crawl_exits(bl.y, bl.x) < DROWN_GUARD_EXITS and \
                        agent.blstats.time - self._drown_wait_since(bl.y, bl.x) < DROWN_GUARD_WAIT:
                    return ('wait', None)
            gap = self._raven_gap_wait()
            if gap is not None:
                return gap   # RAVEN_GAP: the ravens scatter from our Elbereth first
            return ('dig', tool)
        # standing on stairs, in a doorway, on a wetter square (Medusa-3's island): walk to the nearest
        # square we can dig, a few steps at most
        hold = self._blind_hold() if blind_hold else None
        if hold is not None:
            return hold   # MEDUSA_BLIND_HOLD: no walk among unseen ravens; sight first
        if agent.blstats.time < self._dig_walk_blocked_until:
            return None
        target = self._dig_walk_target(max_wet)
        if target is None:
            return None
        # monsters next to us would get free hits all the way (dsafe-A2 jf14 s11 walked 3 squares of Medusa-4's
        # north islet among snakes, a cobra and black naga hatchlings: 86 -> 14 HP). Scare them first: an
        # Elbereth makes them flee for rnd(10) turns (monflee), and a fleeing monster moves away, not attacks.
        if ESCAPE_V2 and adjacent and not agent.character.prop.blind and \
                (agent.inventory.engraving_below_me or '').lower() != 'elbereth' and \
                not agent.character.prop.polymorph and agent.can_engrave():
            spot = (level.key(), bl.y, bl.x, 'scare')
            if self.__dict__.get('_elbereth_tries', {}).get(spot, 0) < 2:
                return ('scare', spot)   # (counted when done: this also runs as a mere condition check)
        return ('step', target)

    _EEL_GRAB = re.compile(r'(eel|kraken) swings itself around you|You cannot escape from the [a-z ]*(eel|kraken)')
    _PYTHON_GRAB = re.compile(r'python swings itself around you|You cannot escape from the [a-z ]*python')   # PYTHON_HOLD

    def _raven_gap_wait(self):
        """RAVEN_GAP (jf_config): on Medusa-3, before a fresh dig on the chosen square, stay on our intact Elbereth while
        a raven is within RAVEN_GAP_RADIUS (at most RAVEN_GAP_MAX turns per square): ('raven_gap', spot) or None."""
        if not (jf_config.RAVEN_GAP and self.on_medusa_level() and self.medusa_level in self._raven_levels):
            return None
        agent = self.agent
        if agent.character.prop.blind or self._in_own_pit():
            return None
        if (agent.inventory.engraving_below_me or '').lower() != 'elbereth':
            return None   # the dig's own Elbereth first (dig_with_tool -> _elbereth_before_digging)
        bl = agent.blstats
        r = jf_config.RAVEN_GAP_RADIUS
        if not any(getattr(m[3], 'mname', '') == 'raven' and
                   max(abs(int(m[1]) - int(bl.y)), abs(int(m[2]) - int(bl.x))) <= r
                   for m in agent.get_visible_monsters()):
            return None
        spot = (agent.current_level().key(), int(bl.y), int(bl.x))
        if self.__dict__.get('_raven_gap_waits', {}).get(spot, 0) >= jf_config.RAVEN_GAP_MAX:
            return None
        return ('raven_gap', spot)

    def _eel_hold_escape_due(self):
        """Held by an eel (or kraken) and not yet on an Elbereth written since the grab (a typo'd one is
        rewritten: the drowning comes with the eel's next hit)."""
        agent = self.agent
        turn = agent.blstats.time
        if self._EEL_GRAB.search(agent.message) or (jf_config.PYTHON_HOLD and self._PYTHON_GRAB.search(agent.message)):
            self._eel_hold_turn = turn
        if 'You get released' in agent.message:
            self._eel_hold_turn = -10
        if turn - self._eel_hold_turn > 3 or agent.character.prop.polymorph or not agent.can_engrave():
            return False
        if (agent.inventory.engraving_below_me or '').lower() == 'elbereth' and not agent.character.prop.blind \
                and self._eel_engraved >= self._eel_hold_turn:
            return False
        return self._eel_engraved < turn   # at most one engraving per turn

    def _followers_only(self, monsters):
        """STALKER_ZAP_FIX: every one of `monsters` is next to us and M2_STALK -- dog.c keepdogs() takes an adjacent,
        unfleeing stalker along through our hole (levl_follower), so a zap down doesn't get us away from any of them."""
        bl = self.agent.blstats
        return bool(monsters) and all(
            max(abs(int(m[1]) - int(bl.y)), abs(int(m[2]) - int(bl.x))) <= 1 and
            bool(getattr(m[3], 'mflags2', 0) & MON.M2_STALK) for m in monsters)

    def _dig_wand(self):
        """A known wand of digging not known to be empty, top-level only (a wand inside a bag has no letter to zap
        it by)."""
        agent = self.agent
        return next((i for i in agent.inventory.items if i.is_wand() and i.is_unambiguous() and
                     i.object == O.from_name('digging', nh.WAND_CLASS) and
                     not agent.inventory.is_known_empty(i)), None)

    def _wand_zone(self):
        """WAND_RESERVE: Medusa's level or a Dungeons-of-Doom level below it -- where the kept wand is spent."""
        return self.on_medusa_level() or bool(self.below_medusa())

    def _reserve_emergency(self):
        """WAND_RESERVE: below WAND_RESERVE_HP with a hostile within 2 -- the kept wand's zap is due anywhere, even
        from our own pit (jf41 s13: pit dug, a titan's boulder in it, killed by the gremlin with the wand kept)."""
        bl = self.agent.blstats
        return bl.hitpoints < WAND_RESERVE_HP * bl.max_hitpoints and bool(self._near_hostiles(radius=2))

    def _wand_reserved(self):
        """WAND_RESERVE: above Medusa's level (or before it is known) the wand of digging waits while the pick-axe
        works -- unless in an emergency (_reserve_emergency)."""
        if not WAND_RESERVE or self._wand_zone() or self.digging_tool() is None:
            return False
        return not self._reserve_emergency()

    def _maze_below_medusa(self):
        """WAND_RESERVE: a level below Medusa's that can't be the castle (see WAND_CASTLE_WAIT): a zap holes it."""
        if not self.below_medusa():
            return False
        agent = self.agent
        key = agent.current_level().key()
        if key in self.undiggable or self.castle.castle_key == key or key in self._door_heard:
            return False
        castle = self.castle.castle_key
        if castle is not None and castle[0] == key[0] and key[1] < castle[1]:
            return True   # above the known castle
        vkey, vpos = self._visit_pos
        if vkey != key or vpos is None:
            return False
        if vpos[1] >= 10:
            return True   # we arrived outside the castle's fall region (bot x 0-9)
        return agent.blstats.time - self._visit_start[1] >= WAND_CASTLE_WAIT

    def _cold_item(self):
        """A known wand of cold or frost horn with charges (top-level: it is used by its letter), or None."""
        agent = self.agent
        for item in agent.inventory.items:
            if jf_config.DEEP_ITEMS and item.category == nh.TOOL_CLASS and item.glyphs and \
                    item.glyphs[0] in self._deep_frost and not agent.inventory.is_known_empty(item):
                return item   # DEEP_ITEMS: an unknown horn whose ray froze the moat (a frost horn)
            if not item.is_unambiguous() or agent.inventory.is_known_empty(item) or item.comment == 'EMPT':
                continue
            name = item.object.name
            if (item.category == nh.WAND_CLASS and name == 'cold') or \
                    (item.category == nh.TOOL_CLASS and name == 'frost horn'):
                return item
        return None

    def cold_reserved(self, item):
        """COLD_RESERVE (see there): True when fight2 must not zap this item at a monster."""
        if not COLD_RESERVE or item.category != nh.WAND_CLASS or not item.is_unambiguous() or \
                item.object.name != 'cold' or self.in_gehennom():
            return False
        return not self._reserve_emergency()

    def _freeze_action(self):
        """MEDUSA_FREEZE (see there): ('freeze', (item, direction, delta)) toward a moat neighbour of our square while
        one is left, else None. Blind (a raven's claw), the map keeps the water it saw and we can't see the ice: each
        direction is zapped once per square (_frozen_dirs), so the remembered water doesn't draw a second zap. Harness
        cold kit: 6 of 13 Medusa-3 deaths came blinded before the first freeze when this waited for sight."""
        if not (MEDUSA_FREEZE and self.on_medusa_level()):
            return None
        agent = self.agent
        key = agent.current_level().key()
        if (agent.character.prop.blind and not MEDUSA_FREEZE_BLIND) or \
                self._freeze_zaps.get(key, 0) >= MEDUSA_FREEZE_MAX:
            return None
        cold = self._cold_item()
        if cold is None:
            return None
        bl = agent.blstats
        level = agent.current_level()
        h, w = level.objects.shape
        # (a direction is zapped once per square: a monster standing on the new ice hides it from the map)
        done = self.__dict__.setdefault('_frozen_dirs', {}).get((key, bl.y, bl.x), set())
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            y, x = bl.y + dy, bl.x + dx
            if not (0 <= y < h and 0 <= x < w) or (dy, dx) in done:
                continue
            # never-seen terrain counts as water: on Medusa's islands a square next to us that we never saw is one a
            # raven or an eel has covered since we arrived -- open water (g6 jf14 s2: three freezes from the raven
            # island, then the wand of digging's hole flooded from a neighbour a raven had hidden); a ray over land
            # does no harm
            if agent.glyphs[y, x] in WET or level.objects[y, x] in WET or level.objects[y, x] == -1:
                return ('freeze', (cold, agent.calc_direction(bl.y, bl.x, y, x), (dy, dx)))
        return None

    # MEDUSA_BLIND_DIG: what dig_first may do blind with nothing in view -- all at our own square, no walk
    _BLIND_DIG_ACTIONS = frozenset(('dig', 'zap', 'freeze', 'blind_engrave', 'wait_sight', 'eel'))
    _LIFT_ACTIONS = frozenset(('lift_on', 'float', 'lift_off', 'hover'))

    def _blind_dig_due(self):
        """Blind on Medusa's level while diving, with nothing in view, dig_first still acts: MEDUSA_BLIND_DIG, or a
        MEDUSA_LIFT float under way (global wait_out rested '.' mid-air while the snakes bit: lift jf14 s85, blinded
        by a cobra 2 steps into a 12-step float, 80 -> 1 HP hovering)."""
        if not (DIG_ESCAPE and self.diving and self.agent.character.prop.blind and self.on_medusa_level()):
            return False
        return MEDUSA_BLIND_DIG or (MEDUSA_LIFT and self._lift is not None)

    def _blind_actions(self):
        """The escape actions dig_first may take blind with nothing in view (see _blind_dig_due)."""
        acts = self._BLIND_DIG_ACTIONS if MEDUSA_BLIND_DIG else frozenset()
        return acts | self._LIFT_ACTIONS if MEDUSA_LIFT else acts

    def _look_after_sight(self):
        """MEDUSA_BLIND_DIG: sight came back on a square last looked at blind, which reads no engraving ('You try
        to feel what is lying here'): look again (':' takes no time) before writing an Elbereth over one that may
        be intact (mm-m4p-bh14 s24 rewrote it at each of 3 sight returns and was spat blind again each time)."""
        if not (MEDUSA_BLIND_DIG and self._sight_back) or self.agent.character.prop.blind:
            return
        self._sight_back = False
        inv = self.agent.inventory
        if inv.engraving_below_me == '':
            inv.engraving_below_me = None
            inv.update()

    def _blind_holding(self):
        """MEDUSA_BLIND_HOLD applies: blind on Medusa's level, for at most MEDUSA_BLIND_HOLD_MAX turns of the spell."""
        if not (MEDUSA_BLIND_HOLD and self.agent.character.prop.blind and self.on_medusa_level()):
            return False
        return self._blind_since is None or self.agent.blstats.time - self._blind_since <= MEDUSA_BLIND_HOLD_MAX

    def _blind_hold(self):
        """MEDUSA_BLIND_HOLD's action while blind on Medusa's level: ('blind_engrave', spot) when no Elbereth was
        written on this square yet in this blind spell or we were hurt since the last one (it may have come out
        wrong), else ('wait_sight', None); None when engraving is impossible here (the usual logic decides)."""
        agent = self.agent
        if agent.character.prop.polymorph or not agent.can_engrave() or utils.any_in(agent.glyphs, G.SWALLOW):
            return None
        bl = agent.blstats
        spot = (agent.current_level().key(), bl.y, bl.x, 'blind')
        tries = self.__dict__.setdefault('_elbereth_tries', {})
        last = self.__dict__.setdefault('_engrave_turn', {}).get(spot)
        if last is None or self._hurt_since(last):
            if tries.get(spot, 0) >= MEDUSA_BLIND_TRIES:
                return None
            return ('blind_engrave', spot)
        return ('wait_sight', None)

    # ------------------------------------------------------------------ MEDUSA_LIFT

    def _lift_ring(self, glyph=None):
        """MEDUSA_LIFT: a known ring of levitation not known to be cursed (the one of that glyph if given), worn or with a
        free ring finger; None if there is none."""
        items = list(self.agent.inventory.items)
        worn = [i for i in items if i.category == nh.RING_CLASS and i.equipped]
        for it in items:
            if it.category != nh.RING_CLASS or not it.is_unambiguous() or it.object.name != 'levitation':
                continue
            if glyph is not None and it.glyphs[0] != glyph:
                continue
            if it.equipped or (it.status != Item.CURSED and len(worn) < 2):
                return it
        return None

    def _floatable(self, level, y, x):
        """A square a levitating hero can float onto: water, or walkable ground without a boulder (no leverage to
        push one while levitating)."""
        agent = self.agent
        if agent.glyphs[y, x] in G.BOULDER or level.objects[y, x] in G.BOULDER:
            return False
        if level.objects[y, x] == -1:
            mc = self._map_char(y, x)   # MEDUSA_MAP: never seen -- the level's fixed map decides
            if mc is not None:
                return mc in '.}'
        return agent.glyphs[y, x] in WET or level.objects[y, x] in WET or bool(level.walkable[y, x])

    _WRAPPERS = frozenset(('giant eel', 'electric eel', 'kraken'))

    def _eel_reach(self):
        """Squares next to a visible eel or kraken: its wrap drowns a levitating hero too (mhitu.c AD_WRAP)."""
        out = set()
        for m in self.agent.get_visible_monsters():
            if getattr(m[3], 'mname', '') in self._WRAPPERS:
                out.update((int(m[1]) + dy, int(m[2]) + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        return out

    def _float_dist(self, level, start, goal=None, limit=MEDUSA_LIFT_MAX_STEPS, avoid=frozenset()):
        """8-way BFS distances over floatable squares from start (no diagonal step into or out of a doorway), not
        through the avoided squares (goal and start excepted)."""
        h, w = level.objects.shape
        dist = {start: 0}
        todo = [start]
        for p in todo:
            if p == goal or dist[p] >= limit:
                continue
            door_p = level.objects[p] in G.DOORS
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (p[0] + dy, p[1] + dx)
                    if (dy, dx) == (0, 0) or n in dist or not (0 <= n[0] < h and 0 <= n[1] < w):
                        continue
                    if not self._floatable(level, *n) or (n in avoid and n != goal):
                        continue
                    if dy and dx and (door_p or level.objects[n] in G.DOORS):
                        continue
                    dist[n] = dist[p] + 1
                    todo.append(n)
        return dist

    def _lift_dry(self, level, y, x):
        """A dry diggable square as far as we know: no moat in its 3x3. Without the level's fixed map (MEDUSA_MAP)
        every neighbour must have been seen; with it, never-seen squares are what the map says (floor we never had
        in view counts too)."""
        h, w = level.objects.shape
        key = level.key()
        if (key, (int(y), int(x))) in self._bad_dig_spots:
            return False
        mc = self._map_char(y, x)
        if level.objects[y, x] == -1 and mc == '.':
            return not (level.shop[y, x] or level.shop_interior[y, x] or (y, x) in level.stair_destination) and \
                self._wet_neighbours(y, x) == 0
        if not self._diggable_spot(y, x, 0):
            return False
        if mc is None:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and level.objects[ny, nx] == -1 and (dy or dx):
                        return False
        return True

    def _known_dry_on_foot(self):
        """MEDUSA_LIFT: a square the map knows to be dry (_lift_dry) is reachable on foot."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        ys, xs = (dis >= 0).nonzero()
        return any(self._lift_dry(level, int(y), int(x)) for y, x in zip(ys, xs))

    def _lift_target(self):
        """MEDUSA_LIFT: (float steps, (y, x)) of the nearest dry diggable square, or None."""
        agent = self.agent
        level = agent.current_level()
        start = (int(agent.blstats.y), int(agent.blstats.x))
        eels = self._eel_reach()
        for avoid in ((eels, frozenset()) if eels else (frozenset(),)):
            dist = self._float_dist(level, start, avoid=avoid)
            best = None
            for p, d in dist.items():
                if d == 0 or agent.monster_tracker.monster_mask[p]:
                    continue
                if (best is None or d < best[0]) and self._lift_dry(level, *p):
                    best = (d, p)
            if best is not None:
                return best
        return None

    def _medusa_lift_action(self):
        """MEDUSA_LIFT (see there): ('lift_on', ring) / ('float', (y, x)) / ('lift_off', ring), or None.
        MEDUSA_POTION_LIFT: also its quaff steps (_potion_lift_plan) and ('hover', None) over the dry square."""
        if not (MEDUSA_LIFT and DIG_ESCAPE and self.diving and self.on_medusa_level()):
            return None
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        bl = agent.blstats
        pos = (int(bl.y), int(bl.x))
        lev = self.levitating()
        st = self._lift
        if st is not None and st['key'] != key:
            st = self._lift = None
        if st is None:
            if lev or key in self._lift_done or agent.character.prop.blind or self.digging_tool() is None or \
                    utils.any_in(agent.glyphs, G.SWALLOW):
                return None
            m3 = key in self._raven_levels or self._medusa_variant_name() == 'medusa-3'
            if m3 and not (MEDUSA_POTION_LIFT and MEDUSA_POTION_LIFT_M3):
                # not on Medusa-3: its dry land is 7-12 floats from the raven island, and levitating we can't write
                # Elbereth -- harness (mm-lv-*-m3-*, 49 catalog seeds) 10/49 passed with the float vs 15/49 digging on
                # the island under Elbereth; the 30 ravens killed 37 of the 39 floaters within 9-29 turns
                return None
            max_wet = self._dig_max_wet()
            if max_wet == 0 and self._known_dry_on_foot():
                # a dry square is reachable on foot (_dig_max_wet alone also counts squares beside never-seen ones:
                # on Medusa's level those are the water squares an eel or a raven covered at arrival -- smoke jf14
                # s24 dug beside one and flooded)
                return None
            ring = None if m3 else self._lift_ring()
            if ring is None:
                return self._potion_lift_plan(max_wet) if MEDUSA_POTION_LIFT else None
            target = self._lift_target()
            if target is None:
                return None
            self._lift = st = {'key': key, 'target': target[1], 'glyph': ring.glyphs[0], 'turn': bl.time,
                               'steps': target[0], 'on': False}
            agent.log(f'DIVE MEDUSA_LIFT: no dry square on foot (max_wet {max_wet}); floating to {target[1]} '
                      f'({target[0]} steps) with {ring.text!r}')
        potion = bool(st.get('potion'))   # MEDUSA_POTION_LIFT: a potion's lift (no ring; it ends on its own)
        ring = None if potion else self._lift_ring(st['glyph'])
        if not lev:
            if st['on']:
                # we were up and are down again: at the target (or the ring came off / was stolen): the float is over
                agent.log(f'DIVE MEDUSA_LIFT: down at {pos} (target {st["target"]})')
                self._lift_done.add(key)
                self._lift = None
                return None
            if ring is None or bl.time - st['turn'] > 10:
                agent.log('DIVE MEDUSA_LIFT: the ring is gone or would not lift us: given up')
                self._lift_done.add(key)
                self._lift = None
                return None
            adjacent = [m for m in agent.get_visible_monsters()
                        if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1 and not self._melee_ignores_elbereth(m[3])]
            if adjacent and not st.get('scared') and agent.can_engrave() and not agent.character.prop.polymorph and \
                    (agent.inventory.engraving_below_me or '').lower() != 'elbereth':
                # up there no Elbereth can be written: scare what stands next to us first, so it flees (monflee,
                # rnd(10) turns) instead of following the float with its bites (lift jf14 s99: a giant spider bit
                # 80 -> 47 HP in the first 3 turns of the float)
                st['scared'] = True
                return ('scare', (key, int(bl.y), int(bl.x), 'lift'))
            return ('lift_on', ring)
        st['on'] = True
        over_water = agent.glyphs[pos] in WET or level.objects[pos] in WET
        if potion:
            # a potion's lift can't be ended (a blessed one aside: 'hover' tries '>'), so no 'late': over the dry
            # square we hover until it ends. Its last turns are announced (timeout.c levitation_dialogue, at 5 and 3
            # turns left): then any land within reach beats a dry square further off
            if self._LEV_ENDING.search(agent.message or ''):
                st['ending'] = True
            late = bool(st.get('ending')) and pos != st['target'] and \
                st['target'] not in self._float_dist(level, pos, goal=st['target'], limit=3)
        else:
            late = bl.time - st['turn'] > 2 * st['steps'] + 15
        if pos == st['target'] or (late and not over_water):
            if potion:
                return ('hover', None)
            return ('lift_off', ring) if ring is not None and ring.equipped else None
        target = st['target']
        if late or agent.monster_tracker.monster_mask[target] or not self._lift_dry(level, *target):
            # re-plan: the nearest dry square again, or (late) any land to come down on
            new = self._lift_target()
            if new is not None and not late:
                st['target'] = target = new[1]
            else:
                dist = self._float_dist(level, pos)
                land = [(d, p) for p, d in dist.items()
                        if d > 0 and not (agent.glyphs[p] in WET or level.objects[p] in WET) and
                        not agent.monster_tracker.monster_mask[p] and level.walkable[p]]
                if not land:
                    return ('hover', None) if potion else None
                st['target'] = target = min(land)[1]
        # round the monsters in the way rather than through them (each blow at a blocker is a turn of bites with no
        # Elbereth to be had up here: smoke jf14 s38 fought a black naga hatchling mid-float, was spat blind and bitten
        # from 76 to 9 HP), and clear of the eels' wrap
        eels = self._eel_reach()
        mons = {(int(y), int(x)) for y, x in zip(*agent.monster_tracker.monster_mask.nonzero())} - {pos, target}
        dist = {}
        for avoid in (eels | mons, mons, eels, frozenset()):
            dist = self._float_dist(level, target, goal=pos, limit=MEDUSA_LIFT_MAX_STEPS * 2, avoid=avoid)
            if pos in dist:
                break
        d0 = dist.get(pos)
        if d0 is None:
            return ('hover', None) if potion else None
        nexts = [(pos[0] + dy, pos[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        nexts = [n for n in nexts if dist.get(n) == d0 - 1]
        if not nexts:
            return ('hover', None) if potion else None
        # a free square first; else the one with a monster on it (it gets attacked)
        nexts.sort(key=lambda n: (bool(agent.monster_tracker.monster_mask[n]), n))
        return ('float', nexts[0])

    # ------------------------------------------------------------------ MEDUSA_POTION_LIFT

    _LEV_ENDING = re.compile(r'You float slightly lower|You wobble unsteadily')

    def _potion_lift_candidates(self):
        """MEDUSA_POTION_LIFT: [(rank, potion)] of the top-level potions that may be levitation and weren't quaffed for a
        lift yet, best first: a known potion of levitation (rank 0), then unknown ones not known cursed, by the chance
        that they are it (objects.c probabilities of what each may still be)."""
        out = []
        for it in self.agent.inventory.items:
            if it.category != nh.POTION_CLASS or not it.glyphs or it.glyphs[0] in self._potion_tried or \
                    'unpaid' in (it.text or ''):
                continue
            objs = list(it.objs)
            if not any(getattr(o, 'name', '') == 'levitation' for o in objs):
                continue
            if it.is_unambiguous():
                out.append((0.0, it))
            elif it.status != Item.CURSED:
                total = sum(getattr(o, 'prob', 0) or 0 for o in objs) or 1
                lev = sum(getattr(o, 'prob', 0) or 0 for o in objs if getattr(o, 'name', '') == 'levitation')
                out.append((1.0 - lev / total, it))
        out.sort(key=lambda t: t[0])
        return out

    def _potion_lift_plan(self, max_wet):
        """MEDUSA_POTION_LIFT (see there), stranded with no ring: ('potion_elbereth', spot) / ('potion_wait', spot) /
        ('potion_lift', potion), or None: nothing to quaff, not now, or a flood reducer is still in hand. No agent step
        here (also a condition check)."""
        agent = self.agent
        prop = agent.character.prop
        if prop.confusion or prop.stun or prop.hallu or prop.polymorph:
            return None
        if self._dig_wand() is not None or self._cold_item() is not None or \
                (jf_config.DEEP_ITEMS and self._deep_charge_items() is not None):
            return None   # the zero-cost flood reducers first: a zap is one flood roll, ice leaves no moat beside
        if self._medusa_reroll_stairs(max_wet) is not None:
            return None   # the '<' reroll first, also free (pl1-m4-on jf14 s42 quaffed sleeping instead of climbing)
        cands = self._potion_lift_candidates()
        if not cands:
            return None
        bl = agent.blstats
        if bl.hitpoints < MEDUSA_POTION_LIFT_HP * bl.max_hitpoints or self._lift_target() is None:
            return None
        threats = self._deep_threats(radius=2)
        if any(self._melee_ignores_elbereth(m[3]) for _, m in threats):
            return None
        spot = (agent.current_level().key(), int(bl.y), int(bl.x), 'potion')
        if (agent.inventory.engraving_below_me or '').lower() != 'elbereth':
            # sleeping/paralysis leave us 25-34 turns helpless (potion.c): on an intact Elbereth only
            if not agent.can_engrave() or self.__dict__.get('_elbereth_tries', {}).get(spot, 0) >= 4:
                return None
            return ('potion_elbereth', spot)
        if any(d <= 1 for d, _ in threats) and self._potion_waits.get(spot, 0) < 5:
            return ('potion_wait', spot)   # scared by the Elbereth, they move off (monflee)
        return ('potion_lift', cands[0][1])

    def _potion_quaff(self, potion):
        """MEDUSA_POTION_LIFT: quaff one potion; a lift becomes MEDUSA_LIFT's float (a potion state: no ring), a
        confusion/stun/hallucination starts the hold (_potion_hold_due)."""
        agent = self.agent
        key = agent.current_level().key()
        text = potion.text
        self._potion_tried.add(potion.glyphs[0])
        before = self._lift_target()
        agent.log(f'DIVE MEDUSA_POTION_LIFT: stranded on a wet islet (max_wet {self._dig_max_wet()}), dry square '
                  f'{before}: quaffing {text!r}')
        self._potion_level = key
        try:
            agent.inventory.quaff(potion)
        finally:
            # (also when the step raised: a hit taken while asleep hands control to another strategy mid-step --
            # pl2-m4-on jf14 s43 -- and a lift must still become the float)
            try:
                agent.log(f'DIVE MEDUSA_POTION_LIFT: quaffed {text!r}: {agent.message!r}')
                bl = agent.blstats
                if self.levitating() and self._lift is None:
                    target = self._lift_target() or before or (0, (int(bl.y), int(bl.x)))
                    self._lift = {'key': key, 'target': target[1], 'glyph': None, 'turn': bl.time,
                                  'steps': target[0], 'on': True, 'potion': True}
                    agent.log(f'DIVE MEDUSA_POTION_LIFT: afloat: to {target[1]} ({target[0]} steps)')
            except Exception:  # never mask the quaff's own exception
                pass

    def _potion_hold_due(self):
        """MEDUSA_POTION_LIFT: confused, stunned or hallucinating on the Medusa level where it quaffed -- hold the square
        (the Elbereth written before the quaff) until it clears, at most MEDUSA_POTION_HOLD_MAX turns and while HP >=
        50%: a confused or stunned apply digs a random way (dig.c use_pick_axe -> confdir), and hallucinated '@'s and
        minotaurs draw the fights Elbereth would spare us. The hold starts when the status shows, not at the quaff:
        booze's confusion shows only after its 'Call a ... potion:' prompt (pl1-m4-on jf14 s24: walked off confused)."""
        if not MEDUSA_POTION_LIFT or self._potion_level is None:
            return False
        agent = self.agent
        prop = agent.character.prop
        bl = agent.blstats
        if not (prop.confusion or prop.stun or prop.hallu) or agent.current_level().key() != self._potion_level:
            self._potion_hold_since = None
            return False
        if self._potion_hold_since is None:
            self._potion_hold_since = bl.time
            agent.log('DIVE MEDUSA_POTION_LIFT: confused/stunned/hallucinating: holding the square until it clears')
        return bl.time - self._potion_hold_since <= MEDUSA_POTION_HOLD_MAX and \
            bl.hitpoints >= 0.5 * bl.max_hitpoints

    def _lift_hover(self):
        """MEDUSA_POTION_LIFT: over the dry square until the potion's lift ends: '>' once (a blessed potion's lift ends
        at will: do.c dodown, I_SPECIAL), then strike what stands next to us (nothing else can be done up here: no
        Elbereth, no dig), else search."""
        agent = self.agent
        st = self._lift
        bl = agent.blstats
        if st is not None and not st.get('down_tried'):
            st['down_tried'] = True
            agent.direction('>')
            agent.log(f'DIVE MEDUSA_POTION_LIFT: over {(int(bl.y), int(bl.x))}, trying to come down: {agent.message!r}')
            return
        peaceful = agent.monster_tracker.peaceful_monster_mask
        for m in agent.get_visible_monsters():
            if max(abs(int(m[1]) - int(bl.y)), abs(int(m[2]) - int(bl.x))) == 1 and not peaceful[m[1], m[2]]:
                self._lift_step((int(m[1]), int(m[2])))
                return
        agent.search(1)

    def _lift_put_on(self, ring):
        agent = self.agent
        letter = agent.inventory.items.get_letter(ring)

        def gen():
            if 'What do you want to put on?' not in agent.single_message:
                return
            yield letter
            if 'Which ring-finger' in agent.single_message:
                yield 'r'

        with agent.atom_operation():
            agent.step(A.Command.PUTON, gen())
        agent.log(f'DIVE MEDUSA_LIFT: put on {ring.text!r}: {agent.message!r}')
        agent.inventory.items.update(force=True)

    def _lift_take_off(self, ring):
        agent = self.agent
        letter = agent.inventory.items.get_letter(ring)

        def gen():
            if 'What do you want to remove?' in agent.single_message:
                yield letter

        with agent.atom_operation():
            agent.step(A.Command.REMOVE, gen())
        agent.log(f'DIVE MEDUSA_LIFT: remove {ring.text!r} at {(agent.blstats.y, agent.blstats.x)}: '
                  f'{agent.message!r}')
        agent.inventory.items.update(force=True)
        if 'cursed' in agent.message and self.levitating():
            # stuck up there: a cursed levitation ring is major trouble (pray.c TROUBLE_CURSED_LEVITATION)
            if not agent.prayer_failed and agent.is_safe_to_pray(500):
                agent.log('DIVE MEDUSA_LIFT: the ring is cursed: praying')
                agent.pray()
            else:
                agent.log('DIVE MEDUSA_LIFT: the ring is cursed and no prayer is safe: stuck floating')
                self._lift_done.add(agent.current_level().key())
                self._lift = None

    def _lift_step(self, sq):
        agent = self.agent
        bl = agent.blstats
        y, x = sq
        d = agent.calc_direction(bl.y, bl.x, y, x)
        if agent.monster_tracker.monster_mask[y, x]:
            with agent.atom_operation():
                agent.step(A.Command.FIGHT)
                agent.direction(d)
            return
        agent.direction(d)

    def _zap_key(self, y, x):
        """WAND_RESERVE's rank of a zap square on Medusa's level, lower is better: None where a zap can't hole the
        floor (stairs, shop, a refused square), else (1 if a flood there would drown us, moat neighbours). A flood
        (fillholetyp: k/(k+1) for k moat neighbours) drops us in the water and we crawl out only to a free adjacent
        land square (trap.c drown -> crawl_destination: no monster on it, no diagonal squeeze); with none we drown
        on the spot -- harness medusa-4 with a wand (mm-wr1-m4-wod) drowned that way, s45 on its first zap. A wetter
        square with a way out beats a drier one without: a flood there only moves us."""
        if not self._diggable_spot(y, x, max_wet=8):
            return None
        wet = self._wet_neighbours(y, x)
        return (int(wet > 0 and self._crawl_exits(y, x, count_monsters=False) == 0), wet)

    def _zap_plan(self, wand):
        """WAND_RESERVE on Medusa's level: ('zap', wand) where we stand when no reachable square ranks better
        (_zap_key), else ('step', the best one, nearest first); ('wait', None) while monsters stand on every way out
        of a flood here (DROWN_GUARD's wait: Elbereth first, a scared monster moves off); None: no zap square."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        bl = agent.blstats
        here = self._zap_key(bl.y, bl.x)
        dis = agent.bfs()
        best = None
        ys, xs = ((dis > 0) & (dis <= MEDUSA_WALK_RADIUS)).nonzero()
        for y, x in zip(ys, xs):
            if agent.monster_tracker.monster_mask[y, x] or level.objects[y, x] in FALL_TRAPS or \
                    (key, (int(y), int(x))) in self._bad_dig_spots:
                continue
            k = self._zap_key(y, x)
            if k is not None:
                cand = (k, int(dis[y, x]), (int(y), int(x)))
                if best is None or cand < best:
                    best = cand
        if here is not None and (best is None or here <= best[0]):
            if here[1] > 0 and self._crawl_exits(bl.y, bl.x) == 0 and \
                    bl.time - self._drown_wait_since(bl.y, bl.x) < DROWN_GUARD_WAIT:
                return ('wait', None)
            return ('zap', wand)
        if best is not None and bl.time >= self._dig_walk_blocked_until:
            return ('step', best[2])
        return ('zap', wand) if here is not None else None

    def _maze_zap(self):
        """WAND_RESERVE on a maze below Medusa: ('zap', wand) right where we stand (a minotaur may be on its way,
        unseen in the dark maze), else None."""
        if not (WAND_RESERVE and DIG_WAND_ESCAPE and self.diving and self._maze_below_medusa()):
            return None
        wand = self._dig_wand()
        return self._wand_escape(wand) if wand is not None else None

    def _wand_escape(self, wand):
        """('zap', wand) when a known wand of digging can hole the floor right here, else None (fight)."""
        if wand is None:
            return None
        agent = self.agent
        level = agent.current_level()
        y, x = agent.blstats.y, agent.blstats.x
        if (y, x) in level.stair_destination or level.shop[y, x] or level.shop_interior[y, x] or \
                (level.key(), (y, x)) in self._bad_dig_spots:
            return None
        terrain = level.objects[y, x]
        if not (terrain in PLAIN_FLOOR or terrain == -1 or terrain in (SS.S_pit, SS.S_spiked_pit)):
            return None
        return ('zap', wand)

    def _elbereth_possible(self):
        """An intact Elbereth under us, or one we may still engrave on this square in this dig phase."""
        agent = self.agent
        blind = agent.character.prop.blind
        if not blind and (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
            return True
        if agent.character.prop.polymorph or not agent.can_engrave():
            return False
        spot = (agent.current_level().key(), agent.blstats.y, agent.blstats.x, self._in_own_pit())
        # blind: fighting blind is no better than engraving again (ravens)
        cap = ELBERETH_TRIES_BLIND if blind else ELBERETH_TRIES_ESCAPE
        return self.__dict__.get('_elbereth_tries', {}).get(spot, 0) < cap

    def _hurt_since(self, turn):
        """HP fell at some point since `turn` (within the last 12 turns of history)."""
        hist = [hp for t, hp in self._hp_history if t >= turn]
        return bool(hist) and max(hist) > self.agent.blstats.hitpoints

    def _medusa_terrain(self):
        """MEDUSA_HOP: (land, water) bool arrays of Medusa's level from the variant's fixed map, overridden by what the
        bot has seen (a flooded dig square is water now). land: floor and doorways (walkable), water: moat/pool."""
        from . import medusa_maps
        agent = self.agent
        level = agent.current_level()
        name = self._medusa_variant_name()
        h, w = level.objects.shape
        land = np.zeros((h, w), dtype=bool)
        water = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                seen = level.objects[y, x]
                if seen in WET or agent.glyphs[y, x] in WET:
                    water[y, x] = True
                elif seen != -1:
                    land[y, x] = bool(level.walkable[y, x]) and seen != SS.S_tree
                else:
                    c = medusa_maps.char_at(name, y, x)
                    water[y, x] = c == '}'
                    land[y, x] = c in '.+'
        land[agent.blstats.y, agent.blstats.x] = True
        water[agent.blstats.y, agent.blstats.x] = False
        return land, water

    def _medusa_hop_plan(self):
        """MEDUSA_HOP: ('walk', square next to the channel) / ('hop', channel square) / None. See MEDUSA_HOP."""
        agent = self.agent
        if not (self.diving and self.on_medusa_level()) or self.levitating() or \
                self._medusa_hops >= MEDUSA_HOP_MAX or agent.blstats.carrying_capacity > 1 or \
                self._medusa_variant_name() is None:
            return None
        level = agent.current_level()
        land, water = self._medusa_terrain()
        labels, _ = ndimage.label(land, structure=np.ones((3, 3), dtype=bool))
        here = labels[agent.blstats.y, agent.blstats.x]
        floor = land & utils.isin(level.objects, PLAIN_FLOOR) | land & (level.objects == -1)
        floor &= ~utils.isin(level.objects, G.DOORS) & ~level.shop & ~level.shop_interior
        wet_count = ndimage.convolve(water.astype(int), np.ones((3, 3), dtype=int), mode='constant')
        dry = floor & (wet_count == 0)
        for (k, (by, bx)) in self._bad_dig_spots:
            if k == level.key():
                dry[by, bx] = False
        if (dry & (labels == here)).any():
            return None   # a dry square on our own land: the ordinary walk-and-dig
        good = set(int(v) for v in np.unique(labels[dry]) if v) - {int(here)}
        if not good:
            return None
        dis = agent.bfs()
        mons = agent.monster_tracker.monster_mask
        h, w = land.shape
        best = None
        for wy, wx in zip(*water.nonzero()):
            if mons[wy, wx]:
                continue   # an eel or a jellyfish in the channel: stepping there attacks it
            ours, far, free = [], 0, 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = wy + dy, wx + dx
                    if (dy, dx) == (0, 0) or not (0 <= y < h and 0 <= x < w) or not land[y, x]:
                        continue
                    if labels[y, x] == here:
                        if dis[y, x] >= 0:
                            ours.append((int(dis[y, x]), (int(y), int(x))))
                        free += 1
                    elif not mons[y, x]:
                        free += 1
                        far += int(labels[y, x]) in good
            if not ours or not far:
                continue
            d, origin = min(ours)
            key = (-far / max(free, 1), d)
            if best is None or key < best[0]:
                best = (key, origin, (int(wy), int(wx)))
        if best is None:
            return None
        _, origin, channel = best
        if (agent.blstats.y, agent.blstats.x) != origin and \
                not utils.adjacent((agent.blstats.y, agent.blstats.x), channel):
            return ('walk', origin)
        return ('hop', channel)

    def _medusa_hop(self, plan):
        agent = self.agent
        kind, (y, x) = plan
        if kind == 'walk':
            self._task('medusa hop: to the channel')
            agent.log(f'DIVE MEDUSA_HOP: walking to {(y, x)} beside a moat channel toward dry land')
            agent.go_to(y, x)
            return
        y0, x0 = agent.blstats.y, agent.blstats.x
        self._medusa_hops += 1
        self._task('medusa hop')
        agent.log(f'DIVE MEDUSA_HOP {self._medusa_hops}: stepping into the moat at {(y, x)} from {(y0, x0)}')
        with agent.atom_operation():
            agent.direction(agent.calc_direction(y0, x0, y, x))
        agent.log(f'DIVE MEDUSA_HOP: now at {(agent.blstats.y, agent.blstats.x)} ({agent.message[-100:]!r})')

    def _above_medusa(self):
        """This is the level right above Medusa's."""
        if self.medusa_level is None:
            return False
        key = self.agent.current_level().key()
        return int(key[0]) == int(self.medusa_level[0]) and int(key[1]) + 1 == int(self.medusa_level[1])

    def _medusa_cycle_action(self):
        """MEDUSA_HOLE_CYCLE (see there): ('cycle_up', '<') on her level, ('plunge', hole) / ('hole_walk', square) /
        ('cycle_rest', None) on the level above once we have climbed off hers, else None."""
        if not (MEDUSA_HOLE_CYCLE and DIG_ESCAPE and self.diving) or self.medusa_level is None or self.levitating():
            return None
        agent = self.agent
        bl = agent.blstats
        level = agent.current_level()
        if self.on_medusa_level():
            if self._medusa_cycles >= MEDUSA_HOLE_CYCLE_MAX or self._in_own_pit() or \
                    bl.time < self._medusa_reroll_blocked_until:
                return None
            above = (self.medusa_level[0], self.medusa_level[1] - 1)
            if not any(int(k[0]) == int(above[0]) and int(k[1]) == int(above[1]) for k in self._dug_holes) and \
                    self.digging_tool() is None and self.digging_wand() is None:
                return None   # no hole up there and nothing to dig one with
            dis = agent.bfs()
            ups = {(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())}
            # the '<' we came down by shows us, not the stairs: the stair memory knows it
            ups |= {(int(p[0]), int(p[1])) for p, dest in level.stair_destination.items()
                    if dest[0][0] == level.dungeon_number and dest[0][1] < level.level_number}
            reachable = [(dis[p], p) for p in ups if 0 <= dis[p] <= MEDUSA_HOLE_CYCLE_STEPS]
            return ('cycle_up', min(reachable)[1]) if reachable else None
        if self._medusa_cycles == 0 or not self._above_medusa() or bl.time < self._medusa_reroll_blocked_until:
            return None
        hole = self._dug_holes.get(level.key())
        if hole is not None:
            hy, hx = hole
            if max(abs(hy - bl.y), abs(hx - bl.x)) > 1:
                dis = agent.bfs()
                near = [(dis[hy + dy, hx + dx], (hy + dy, hx + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (dy or dx) and 0 <= hy + dy < dis.shape[0] and 0 <= hx + dx < dis.shape[1] and
                        0 <= dis[hy + dy, hx + dx] <= MEDUSA_HOLE_CYCLE_HOLE_STEPS]
                if near:
                    return ('hole_walk', min(near)[1])
                hole = None   # too far: the dive digs a new one
        # every landing down there costs HP (bc-smoke: 53 of 81 in 6 turns among Medusa-3's ravens): rest up here
        # first -- beside the hole (the walk above comes first), or before digging one; fight what is close
        if bl.hitpoints < MEDUSA_HOLE_CYCLE_REST * bl.max_hitpoints and not self._in_own_pit():
            return ('cycle_hold', None) if self._near_hostiles(radius=2) else ('cycle_rest', None)
        if hole is None:
            return None   # the dive digs one (beside the '>' we arrived on; the '>' itself is avoided)
        return ('plunge', hole)

    def _medusa_cycle_climb(self, up):
        agent = self.agent
        if (agent.blstats.y, agent.blstats.x) != up:
            if getattr(self, '_cycle_walk_logged', None) != (self._medusa_cycles, up):
                self._cycle_walk_logged = (self._medusa_cycles, up)
                agent.log(f'MEDUSA_HOLE_CYCLE: to her < at {up} (climb {self._medusa_cycles + 1}, '
                          f'hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints})')
            start = (agent.blstats.y, agent.blstats.x)
            turn = agent.blstats.time
            self._raven_step_toward(up)
            if (agent.blstats.y, agent.blstats.x) == start and agent.blstats.time == turn:
                self._medusa_reroll_blocked_until = turn + 3   # no move and no turn: let the dig plan act
            return
        key = agent.current_level().key()
        agent.log(f'MEDUSA_HOLE_CYCLE: climbing off her level (climb {self._medusa_cycles + 1})')
        agent.move('<')
        if agent.current_level().key() != key:
            self._medusa_cycles += 1
            # the '>' we stand on leads straight back onto her '<': our hole is the way down
            self._avoid_stairs_until[(agent.current_level().key(), (agent.blstats.y, agent.blstats.x))] = 10 ** 9

    def _medusa_plunge(self, hole):
        agent = self.agent
        key = agent.current_level().key()
        hy, hx = hole
        if agent.monster_tracker.monster_mask[hy, hx]:
            agent.log(f'MEDUSA_HOLE_CYCLE: hitting the monster on our hole at {hole}')
            agent.step(A.Command.FIGHT)
            agent.direction(hy, hx)
            return
        agent.log(f'MEDUSA_HOLE_CYCLE: into our hole at {hole} (after climb {self._medusa_cycles}, '
                  f'hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints})')
        turn = agent.blstats.time
        agent.direction(hy, hx)
        if agent.current_level().key() != key:
            return
        if (agent.blstats.y, agent.blstats.x) == (hy, hx):
            agent.log('MEDUSA_HOLE_CYCLE: no fall -- the hole is gone')
            self._dug_holes.pop(key, None)
        elif agent.blstats.time == turn:
            self._medusa_reroll_blocked_until = turn + 3

    def _medusa_reroll_stairs(self, max_wet):
        """DIG_ESCAPE on Medusa's level, stranded where every reachable square has >= MEDUSA_REROLL_WET moat
        neighbours: a hole there floods with odds n/(n+1) at the pit and again at the hole (dig.c fillholetyp:
        1 success in 9 at n = 2), and each flood makes the islet wetter until we drown (dsafe-t1 public s9
        took the stairs onto a Medusa-4 islet and drowned on its 7th hole). The '<' on the islet re-rolls the
        landing: climb, dig down beside the '>' above, and fall at a random spot of the level's arrival region
        (u_on_rndspot; Medusa-4's holds dry squares in its north part). Returns the '<' (y, x) or None."""
        # (max_wet None: every reachable square flooded or refused -- base-jf27 s5 then sat 3000 turns on the '<'
        # of a Medusa-3 island, searching, until an invisible stalker killed it)
        if not (DIG_ESCAPE and self.on_medusa_level()):
            return None
        stranded = jf_config.MEDUSA_STRANDED_REROLL and max_wet is None
        if jf_config.MEDUSA_STRANDED_REROLL:
            # (see MEDUSA_STRANDED_REROLL) remembered for a climb some other strategy makes from here (update())
            bl = self.agent.blstats
            self._medusa_stranded = (self.agent.current_level().key(), (int(bl.y), int(bl.x))) if stranded else None
        if self.agent.blstats.time < self._medusa_reroll_blocked_until:
            return None
        if MEDUSA_SKIP:
            # A hole or trap door drops an extra level 1 time in 4 (trap.c fall_through: newlevel++ while
            # !rn2(4)), so every hole dug from the level above skips Medusa's level outright a quarter of the time,
            # and the 3 in 4 that land here come down on a fresh random square of the arrival region. That beats
            # a pick-axe hole beside n moat squares (1/(n+1)^2: 11% at n = 2) and doesn't flood the island any
            # further. So climb once a dig here has flooded or the best square is that wet -- on Medusa-3 too
            # (base3: 10 of 17 Medusa deaths there, after 2-6 floods and no reroll).
            wet_enough = max_wet is None or max_wet >= MEDUSA_REROLL_WET or \
                (max_wet >= 1 and self._medusa_floods >= MEDUSA_SKIP_FLOODS)
            # Medusa-3: every island square is wet anyway, and each dig leaves us exposed to 30 fast ravens when the
            # pit wipes our Elbereth (a raven's claw blinds; a blind dust Elbereth holds only ~1 time in 3): go
            # straight back up while the '<' is close, before they gather
            skip_first = MEDUSA_SKIP_FIRST and self.medusa_level in self._raven_levels and \
                max_wet is not None and max_wet >= 1
            if not (wet_enough or skip_first) or self._medusa_rerolls >= MEDUSA_SKIP_REROLLS:
                return None
        elif stranded:
            # (see MEDUSA_STRANDED_REROLL) nothing here can be dug at all: climb, Medusa-3 included -- searching on an
            # isolated '<' only waits for the ravens
            if self._medusa_rerolls >= jf_config.MEDUSA_STRANDED_REROLLS:
                return None
        else:
            if (max_wet is not None and max_wet < MEDUSA_REROLL_WET) or self._medusa_rerolls >= MEDUSA_REROLLS:
                return None
            # not on Medusa-3 (its 30 ravens): its '<' and its fall region are the same island, so a reroll only
            # ping-pongs between the ravens and the level above (dsafe-A jf14 s3 met a leocrotta up there, blind)
            if self.medusa_level in self._raven_levels:
                return None
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        ups = {(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())}
        # the '<' we came down by shows us, not the stairs: the stair memory knows it
        ups |= {(int(p[0]), int(p[1])) for p, dest in level.stair_destination.items()
                if dest[0][0] == level.dungeon_number and dest[0][1] < level.level_number}
        reachable = [(dis[p], p) for p in ups if dis[p] != -1]
        if MEDUSA_SKIP and MEDUSA_SKIP_FIRST and self.medusa_level in self._raven_levels and \
                not (max_wet is None or max_wet >= MEDUSA_REROLL_WET or self._medusa_floods >= MEDUSA_SKIP_FLOODS):
            reachable = [r for r in reachable if r[0] <= MEDUSA_SKIP_FIRST_STEPS]   # only while it's close
        return min(reachable)[1] if reachable else None

    def _medusa_reroll(self, up):
        agent = self.agent
        if (agent.blstats.y, agent.blstats.x) != up:
            agent.log(f'DIVE Medusa islet too wet (max_wet {self._dig_max_wet()}): walking to the < at {up}')
            start = (agent.blstats.y, agent.blstats.x)
            try:
                agent.go_to(*up, max_steps=1)
            finally:
                if (agent.blstats.y, agent.blstats.x) == start:
                    self._medusa_reroll_blocked_until = agent.blstats.time + 5
            return
        self._medusa_rerolls += 1
        agent.log(f'DIVE Medusa islet too wet: up the stairs to dig down again (reroll {self._medusa_rerolls})')
        agent.move('<')
        # the '>' we stand on now would lead straight back to the islet
        self._avoid_stairs_until[(agent.current_level().key(), (agent.blstats.y, agent.blstats.x))] = 10 ** 9

    def _drown_wait_since(self, y, x):
        """First turn of the current DROWN_GUARD wait on this square."""
        waits = self.__dict__.setdefault('_drown_wait', {})
        k = (self.agent.current_level().key(), int(y), int(x))
        return waits.setdefault(k, self.agent.blstats.time)

    def _dig_order_values(self, max_wet):
        """MEDUSA_DIG_ORDER: {(y, x): flood-only expectimax value} of the reachable diggable squares with max_wet moat
        neighbours (the fewest there are), {} if it doesn't apply. The model: a pick-axe hole at k moat neighbours comes
        out dry with 1/(k+1)^2 (fillholetyp for the pit, then the hole); otherwise the square turns to moat and we crawl
        to a random free land neighbour (none: drowned) and dig on, always from a square with the fewest moat
        neighbours, the best of them. Cached until the islet's map changes."""
        if not (MEDUSA_DIG_ORDER and max_wet and self.on_medusa_level()):
            return {}
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        dis = agent.bfs()
        walk = [(int(y), int(x)) for y, x in zip(*(dis >= 0).nonzero())
                if not (agent.glyphs[y, x] in WET or level.objects[y, x] in WET)]
        if not walk or len(walk) > MEDUSA_DIG_ORDER_MAX:
            return {}
        base = tuple(self._wet_neighbours(*p) for p in walk)
        ok = tuple(self._diggable_spot(*p, max_wet=8) and (key, p) not in self._bad_dig_spots for p in walk)
        sig = (key, tuple(walk), base, ok)
        cache = getattr(self, '_dig_order_cache', None)
        if cache is not None and cache[0] == sig:
            return cache[1]
        idx = {p: i for i, p in enumerate(walk)}
        nb = [[idx[(p[0] + dy, p[1] + dx)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (p[0] + dy, p[1] + dx) in idx] for p in walk]
        memo = {}

        def component(fl, start):
            seen = 1 << start
            todo = [start]
            while todo:
                i = todo.pop()
                for j in nb[i]:
                    b = 1 << j
                    if not (fl & b) and not (seen & b):
                        seen |= b
                        todo.append(j)
            return seen

        def dig_values(fl, comp, depth):
            ks = {i: base[i] + sum(1 for j in nb[i] if fl >> j & 1)
                  for i in range(len(walk)) if comp >> i & 1 and ok[i]}
            if not ks:
                return {}
            lo = min(ks.values())
            out = {}
            for i, k in ks.items():
                if k != lo:
                    continue
                ps = 1.0 / (k + 1) ** 2
                fl2 = fl | (1 << i)
                exits = [j for j in nb[i] if not (fl2 >> j & 1)]
                cont = sum(value(fl2, component(fl2, j), depth + 1) for j in exits) / len(exits) if exits else 0.0
                out[i] = ps + (1 - ps) * cont
            return out

        def value(fl, comp, depth):
            if depth >= MEDUSA_DIG_ORDER_DEPTH:
                return 0.0
            m = (fl, comp)
            if m not in memo:
                vals = dig_values(fl, comp, depth)
                memo[m] = max(vals.values()) if vals else 0.0
            return memo[m]

        here = (int(agent.blstats.y), int(agent.blstats.x))
        start = idx.get(here)
        if start is None:
            return {}
        root = dig_values(0, component(0, start), 0)
        out = {walk[i]: v for i, v in root.items() if base[i] == max_wet}
        self._dig_order_cache = (sig, out)
        return out

    def _dig_order_best(self, max_wet):
        """MEDUSA_DIG_ORDER: the best dig square nothing stands on (ties: the nearest), or None. (A raven or snake on
        the best square made the walk fall back to the nearest square -- often the worst one: mm-pk-ord-m3-16.)"""
        vals = self._dig_order_values(max_wet)
        agent = self.agent
        dis = agent.bfs()
        here = (int(agent.blstats.y), int(agent.blstats.x))
        free = {p: v for p, v in vals.items() if (p == here or not agent.monster_tracker.monster_mask[p]) and dis[p] >= 0}
        if not free:
            return None
        top = max(free.values())
        return min((int(dis[p]), p) for p, v in free.items() if v >= top - 0.005)[1]

    def _dig_order_here_ok(self, max_wet):
        """MEDUSA_DIG_ORDER: our own square is (one of) the best free squares to dig from -- True when the order
        doesn't apply."""
        vals = self._dig_order_values(max_wet)
        if not vals:
            return True
        best = self._dig_order_best(max_wet)
        here = (int(self.agent.blstats.y), int(self.agent.blstats.x))
        return best is None or (here in vals and vals[here] >= vals[best] - 0.005)

    def _dig_walk_target(self, max_wet, min_exits=0):
        """The nearest free diggable square within DIG_WALK_RADIUS steps (DIG_ESCAPE), the driest first among
        equally near ones (with min_exits: only squares a flooded digger could crawl out of, DROWN_GUARD)."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        best = None
        order = self._dig_order_best(max_wet) if not min_exits else None
        if order is not None and dis[order] > 0:
            return order   # MEDUSA_DIG_ORDER
        # on Medusa's level a dry square is worth a long walk: the alternative is a flood gamble (Medusa-4's
        # dry squares are in the NE hut, often 10+ steps from where the fall put us)
        radius = MEDUSA_WALK_RADIUS if (ESCAPE_V2 and self.on_medusa_level()) else DIG_WALK_RADIUS
        ys, xs = ((dis > 0) & (dis <= radius)).nonzero()
        for y, x in zip(ys, xs):
            if agent.monster_tracker.monster_mask[y, x] or level.objects[y, x] in FALL_TRAPS or \
                    not self._diggable_spot(y, x, max_wet) or \
                    (level.key(), (int(y), int(x))) in self._bad_dig_spots:
                continue
            if min_exits and self._wet_neighbours(y, x) > 0 and \
                    self._crawl_exits(y, x, count_monsters=False) < min_exits:
                continue
            k = (int(dis[y, x]), self._wet_neighbours(y, x))
            if best is None or k < best[0]:
                best = (k, (int(y), int(x)))
        return None if best is None else best[1]

    def _sheltered_dig_ok(self, near):
        """SHELTERED_DIG: keep digging with monsters close by when they can't hurt us in melee -- all of
        them respect Elbereth and we stand on it (or will engrave it first) -- or when the dive has been
        stuck on this level for long (a sleeping zoo held a public-seed dig-diver for 6700 turns: fight2
        danced beside it while dig_first refused to dig with monsters within 2)."""
        if not SHELTERED_DIG:
            return False
        agent = self.agent
        if self.turns_on_level() >= SHELTERED_DIG_STUCK and not agent._hurt_recently(20):
            return True
        if self.on_scare_scroll():
            return True   # everything near is scared (GEHENNOM_SCARE)
        if agent.current_level().dungeon_number == GEHENNOM or any(self._ignores_elbereth(m[3]) for m in near):
            return False
        on_elbereth = (agent.inventory.engraving_below_me or '').lower() == 'elbereth'
        return on_elbereth or (agent.can_engrave() and not agent.character.prop.blind)

    def mattock_digger(self):
        """Diving with a mattock and no pick-axe: shields are left behind (the mattock needs both hands)."""
        if not self.diving:
            return False
        tool = self.best_digging_tool(self.agent.inventory.items)
        return tool is not None and tool.object == O.from_name('dwarvish mattock')

    def digging_tool(self):
        inv = self.agent.inventory.items
        tool = self.best_digging_tool(inv)
        if tool is None:
            return None
        # a cursed weapon welded to the hand can't be swapped for the pick-axe
        main = inv.main_hand
        if main is not None and main is not tool and main.status == Item.CURSED:
            # CURSED_PICK_OK: unless the welded weapon is itself a digging tool: dig with that one
            if CURSED_PICK_OK and self.is_digging_tool(main, False):
                return main
            return None
        return tool

    def _known_digging_tools(self):
        """Pick-axes lying where we have seen them (the tour picks them up and drops them again)."""
        found = []
        shield_worn = self._shield_stuck()
        for key, level in self.agent.levels.items():
            if key[0] not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES):
                continue
            for y, x in zip(*(level.item_count > 0).nonzero()):
                if (key, (y, x)) in self._fetch_given_up or level.shop_interior[y, x]:
                    continue
                if any(self.is_digging_tool(i, shield_worn) for i in flatten_items(level.items[y, x])):
                    found.append((key, (int(y), int(x))))
        for key, (y, x) in self.tool_spots:
            if jf_config.SHOP_GUARD and key in self.agent.levels and self.agent.levels[key].shop[y, x]:
                continue   # a dwarf dying in a shop leaves its pick to the shopkeeper (and pickups skip shop floors)
            if (key, (y, x)) not in self._fetch_given_up and (key, (int(y), int(x))) not in found and \
                    (key[0] in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES)):
                found.append((key, (int(y), int(x))))
        return found

    def should_fetch_digging_tool(self):
        # Gehennom has no way back up to the Dungeons of Doom except the Valley's '<' to the castle's east edge
        if self.in_gehennom():
            return False
        if self.digging_tool() is not None:
            self._fetch = None
            return False
        if self._fetch is not None:
            return True
        agent = self.agent
        if agent.blstats.time - self._fetch_scan_turn < 50:
            return False
        self._fetch_scan_turn = agent.blstats.time
        here = agent.current_level().key()
        best = None
        for key, pos in self._known_digging_tools():
            if key == here:
                dis = agent.bfs()
                if dis[pos] == -1 and self._neighbour_distance(dis, *pos) is None:
                    continue
                cost = 0
            else:
                path = agent.exploration.get_path_to_level(*key)
                if path is None:
                    continue
                cost = len(path)
            if best is None or cost < best[0]:
                best = (cost, key, pos)
        if best is None:
            return False
        _, key, pos = best
        agent.log(f'DIVE fetching the digging tool at {pos} on {key}')
        self._fetch = (key, pos, agent.blstats.time)
        return True

    def fetch_digging_tool(self):
        agent = self.agent
        key, (y, x), started = self._fetch
        if agent.blstats.time - started > FETCH_TOOL_TURNS:
            agent.log(f'DIVE digging tool at {(y, x)} on {key}: out of time')
            self._fetch_given_up.add((key, (y, x)))
            self._fetch = None
            return
        if agent.current_level().key() != key:
            # one staircase at a time: follow_level_path_strategy asserts when a peaceful blocks the
            # stairs (an s7 dive panicked on a Mines '<' for turns); _take_stairs waits it out
            path = agent.exploration.get_path_to_level(*key)
            if not path:
                agent.log(f'DIVE digging tool at {(y, x)} on {key}: no way there')
                self._fetch_given_up.add((key, (y, x)))
                self._fetch = None
                return
            (sy, sx), _, direction = path[0]
            if not self._take_stairs([(sy, sx)], direction):
                # cut off (peacefuls, boulders): explore for a way to it, within FETCH_TOOL_TURNS
                self.exploration(None).until(agent, self._budgeted(
                    lambda: agent.bfs()[sy, sx] != -1,
                    lambda: agent.blstats.time - started > FETCH_TOOL_TURNS)).run()
            return
        pos = (agent.blstats.y, agent.blstats.x)
        if pos != (y, x):
            dis = agent.bfs()
            if dis[y, x] != -1:
                agent.go_to(y, x)
                return
            # a dwarf's fresh tunnel isn't on our map: reach a neighbour, then step in
            tries = self._spot_visits.get((key, (y, x)), 0) + 1
            self._spot_visits[(key, (y, x))] = tries
            if self._neighbour_distance(dis, y, x) is None or tries > 6:
                agent.log(f'DIVE digging tool at {(y, x)} on {key}: unreachable')
                self._fetch_given_up.add((key, (y, x)))
                self._fetch = None
                return
            if not utils.adjacent(pos, (y, x)):
                agent.go_to(y, x, stop_one_before=True)
                return
            agent.move(agent.calc_direction(pos[0], pos[1], y, x))
            return
        agent.inventory.pickup_and_drop_items().run()
        if self.digging_tool() is None:
            agent.log(f'DIVE no usable digging tool at {(y, x)} on {key}')
        else:
            agent.log(f'DIVE picked up {self.digging_tool().text!r}')
        self._fetch_given_up.add((key, (y, x)))
        self.tool_spots.discard((key, (y, x)))
        self._fetch = None

    def _peaceful_dwarves(self):
        agent = self.agent
        glyphs = [MON.from_name(n) for n in DWARF_NAMES]
        mask = agent.monster_tracker.peaceful_monster_mask & utils.isin(agent.glyphs, glyphs)
        if not mask.any():
            return []
        dis = agent.bfs()
        found = []
        for y, x in zip(*mask.nonzero()):
            d = self._neighbour_distance(dis, y, x)
            if d is not None:
                found.append((d, int(y), int(x)))
        if DIGGER_FIRST:
            # a dwarf seen digging carries a pick-axe or mattock for sure: go for it first
            key = agent.current_level().key()
            return sorted(found, key=lambda f: (not self._is_digger(key, (f[1], f[2])), f[0]))
        return sorted(found)

    def keep_digging_tool(self):
        return self.diving or KEEP_TOOL_IN_TOUR or bool(jf_config.PICK_TRIP_XL) or bool(GRIND_HUNT_XL) or \
            self._arc_early_dig() or self._rog_early_dig()

    def _arc_early_dig(self):
        """ARC_DIG_DIVE_XL applies: the character is an Archeologist."""
        try:
            return ARC_DIG_DIVE_XL is not None and self.agent.character.role == Character.ARCHEOLOGIST
        except Exception:
            return False

    def _rog_early_dig(self):
        """ROG_DIG_DIVE_XL applies: the character is a Rogue."""
        try:
            return ROG_DIG_DIVE_XL is not None and self.agent.character.role == Character.ROGUE
        except Exception:
            return False

    def _dig_dive_xl(self):
        """The XL from which a digging-tool holder leaves the tour for the dig-dive."""
        if self._arc_early_dig():
            return min(ARC_DIG_DIVE_XL, self._min_xl(DIG_DIVE_XL))
        if self._rog_early_dig():
            return min(ROG_DIG_DIVE_XL, self._min_xl(DIG_DIVE_XL))
        return self._min_xl(DIG_DIVE_XL)

    def _grind_hunting(self):
        """GRIND_HUNT_XL: the tour hunts dwarves too (the Dlvl-1 grind meets them from XL 7)."""
        return bool(GRIND_HUNT_XL) and not self.diving and self.agent.blstats.experience_level >= GRIND_HUNT_XL

    @Strategy.wrap
    def hunt_strategy(self):
        """The tour's Mines visit hunts too, and checks the piles a killed dwarf left (the dive does
        both from plan_step)."""
        if self.diving or not DWARF_HUNT or not (HUNT_IN_TOUR or self.pick_trip or self._grind_hunting()):
            yield False
        spot = self._local_tool_spot() if self.digging_tool() is None else None
        hunt = spot is None and self.should_hunt_dwarf()
        approach = spot is None and not hunt and self.should_approach_dwarf()
        if spot is None and not hunt and not approach:
            yield False
        yield True
        if spot is not None:
            self._visit_tool_spot(spot)
        elif hunt:
            self.hunt_dwarf()
        elif not self._approach_dwarf():
            if self.should_hunt_dwarf():   # the line to it is open now
                self.hunt_dwarf()
            else:
                self.agent.search()        # must act: a preempting strategy that doesn't loops

    def _local_tool_spot(self):
        agent = self.agent
        here = agent.current_level().key()
        dis = agent.bfs()
        spots = [(dis[p], p) for k, p in self.tool_spots
                 if k == here and (k, p) not in self._fetch_given_up and dis[p] != -1]
        return min(spots)[1] if spots else None

    def _visit_tool_spot(self, spot):
        agent = self.agent
        key = agent.current_level().key()
        if (agent.blstats.y, agent.blstats.x) != spot:
            tries = self._spot_visits.get((key, spot), 0) + 1
            self._spot_visits[(key, spot)] = tries
            if tries > 20:
                self._fetch_given_up.add((key, spot))
                return
            agent.go_to(*spot)
            return
        agent.inventory.pickup_and_drop_items().run()
        agent.log(f'DIVE checked the pile at {spot}: tool {self.digging_tool()!r}')
        self._fetch_given_up.add((key, spot))
        self.tool_spots.discard((key, spot))

    def first_level_done(self):
        """The tour's Dlvl 1 grind ends at XL 8 (DT6A), or earlier for a tool run or a Samurai (SAM_DIVE_XL)."""
        xl = self.agent.blstats.experience_level
        # DIVE_FED / DIVE_PRAYER_GAP hold the grind (the milestone would otherwise move on to the Mines tour)
        return (xl >= 8 or self._sam_grind_over(xl) or (TOOL_RUN_XL is not None and xl >= TOOL_RUN_XL)) and \
            self.fed_for_dive() and self.prayer_ready_for_dive() and self.hp_ready_for_dive()

    def _sam_grind_over(self, xl):
        """SAM_DIVE_XL: a Samurai's Dlvl-1 grind is over at this XL (False for every other role, or on any error)."""
        try:
            return SAM_DIVE_XL is not None and xl >= SAM_DIVE_XL and \
                self.agent.character.role == Character.SAMURAI
        except Exception:
            return False

    def hp_ready_for_dive(self):
        """DIVE_START_HP: at least that fraction of max HP, or DIVE_START_HP_WAIT turns spent waiting. Logs the wait."""
        if not DIVE_START_HP:
            return True
        bl = self.agent.blstats
        if bl.hitpoints >= DIVE_START_HP * bl.max_hitpoints:
            if self._hp_wait_start is not None:
                self.agent.log(f'DIVE_START_HP waited {bl.time - self._hp_wait_start} turns '
                               f'(hp {bl.hitpoints}/{bl.max_hitpoints})')
                self._hp_wait_start = None
            return True
        if self._hp_wait_start is None:
            self._hp_wait_start = bl.time
            self.agent.log(f'DIVE_START_HP waiting: hp {bl.hitpoints}/{bl.max_hitpoints}')
        # past the cap the wait stays over (both callers, first_level_done and should_dive, see the same answer)
        if bl.time - self._hp_wait_start > DIVE_START_HP_WAIT:
            if not self._hp_wait_over:
                self._hp_wait_over = True
                self.agent.log(f'DIVE_START_HP gave up waiting after {bl.time - self._hp_wait_start} turns')
            return True
        return False

    def prayer_ready_for_dive(self):
        """DIVE_PRAYER_GAP: at least that many turns since the last prayer (see the flag). Logs the wait."""
        if not DIVE_PRAYER_GAP:
            return True
        if DIVE_PRAYER_TOOL_ONLY and self.digging_tool() is None:
            return True
        agent = self.agent
        bl = agent.blstats
        last = agent.last_prayer_turn
        gap = None if last is None else bl.time - last
        ready = gap is None or gap >= DIVE_PRAYER_GAP or agent.prayer_failed
        if ready and DIVE_PRAYER_FOOD and not agent.prayer_failed:
            ready = bl.hunger_state < Hunger.HUNGRY or agent.carried_food_nutrition() >= DIVE_PRAYER_FOOD_MIN
        if not ready and self._prayer_wait_start is None:
            self._prayer_wait_start = bl.time
            agent.log(f'DIVE_PRAYER_GAP waiting: {gap} turns since the last prayer, hunger {bl.hunger_state}, '
                      f'food {agent.carried_food_nutrition()}')
        if not ready and bl.time - self._prayer_wait_start > DIVE_PRAYER_MAX_WAIT:
            agent.log(f'DIVE_PRAYER_GAP gave up waiting after {bl.time - self._prayer_wait_start} turns')
            ready = True
        if ready and self._prayer_wait_start is not None:
            agent.log(f'DIVE_PRAYER_GAP waited {bl.time - self._prayer_wait_start} turns (gap {gap}, '
                      f'hunger {bl.hunger_state})')
            self._prayer_wait_start = None
        return ready

    def weld_hold(self):
        """jf_config.WELD_HOLD (with WELD_PRAY): welded to a cursed two-hander (or a weapon beside a cursed shield: no free
        hand, so no Elbereth) while diving at depth <= WELD_HOLD_MAX_DEPTH -- stay on this level until WELD_PRAY's prayer
        (emergency_strategy, once WELD_PRAY_GAP turns have passed since the last prayer) frees the hands. Over once the
        hands are free, a prayer has failed, a prayer came and left the weld (a Weak/Fainting prayer fixes hunger first),
        or after WELD_HOLD_MAX_WAIT turns -- then the dive digs on welded, as before.
        cand-g: jf48 s10 and jf50 s7 welded a mattock at prayer gaps 864 and 805, dug on and died 65 and 60 turns later on
        Dlvl 8 and 11 (a quasit, a troll: no Elbereth, 2 hits in 10 unskilled)."""
        if not (jf_config.WELD_HOLD and jf_config.WELD_PRAY) or not self.diving:
            return False
        agent = self.agent
        if agent.prayer_failed or agent.character.prop.polymorph or not agent.no_free_hand() or \
                agent.blstats.depth > jf_config.WELD_HOLD_MAX_DEPTH or self.in_gehennom():
            if self._weld_hold_start is not None and not agent.no_free_hand():
                agent.log(f'WELD_HOLD over: hands free after {agent.blstats.time - self._weld_hold_start} turns')
                self._weld_hold_start = None
            return False
        turn = agent.blstats.time
        if self._weld_hold_start is None:
            self._weld_hold_start = turn
            self._weld_hold_prayer = agent.last_prayer_turn
            last = agent.last_prayer_turn
            agent.log(f'WELD_HOLD start: welded to {agent.inventory.items.main_hand.text!r}, '
                      f'{None if last is None else turn - last} turns since the last prayer (WELD_PRAY_GAP '
                      f'{jf_config.WELD_PRAY_GAP}), depth {agent.blstats.depth}')
        if self._weld_hold_gave_up:
            return False
        if agent.last_prayer_turn != self._weld_hold_prayer:
            # a prayer came and the hands are still welded: a Weak/Fainting prayer fixed the worse trouble (pray.c
            # fix_worst_trouble: TROUBLE_STARVING ranks above TROUBLE_UNUSEABLE_HANDS; a Luck-0 prayer fixes both only
            # half the time) and restarted the timeout -- another ~1100 turns welded on a shallow level is no better
            # than digging on (da-wh1: jf50 s7 and jf48 s10 were killed on Dlvl 2 that way, 300 and 160 turns into it)
            self._weld_hold_gave_up = True
            agent.log(f'WELD_HOLD over: a prayer at T{agent.last_prayer_turn} left the hands welded '
                      f'({turn - self._weld_hold_start} turns held)')
            return False
        if turn - self._weld_hold_start > jf_config.WELD_HOLD_MAX_WAIT:
            self._weld_hold_gave_up = True
            agent.log(f'WELD_HOLD gave up after {turn - self._weld_hold_start} turns')
            return False
        return True

    def fed_for_dive(self):
        """jf_config.DIVE_FED: the grind ends fed -- Not Hungry within DIVE_FED_GAP turns of the last hunger prayer
        (>= ~400 nutrition left), or >= DIVE_FED_FOOD nutrition of food carried; otherwise it goes on on Dlvl 1
        until the next hunger prayer (at most DIVE_FED_MAX_WAIT turns). Most base dives started 300-1000 turns
        after the last prayer with no food: Weak/Fainting came 100-700 turns in, on Dlvl 10-26 (4 of 48 tool
        dives died fainted), or during the dwarf kill's prayer hold (jf16 s13: fainting at Dlvl 9, dead)."""
        if not jf_config.DIVE_FED:
            return True
        agent = self.agent
        bl = agent.blstats
        if self._fed_wait_start is None:
            self._fed_wait_start = bl.time
        if bl.time - self._fed_wait_start > jf_config.DIVE_FED_MAX_WAIT or agent.prayer_failed:
            return True
        if agent.carried_food_nutrition() >= jf_config.DIVE_FED_FOOD:
            return True
        fed = bl.hunger_state == Hunger.SATIATED or \
            (bl.hunger_state == Hunger.NOT_HUNGRY and agent.last_prayer_turn is not None and
             bl.time - agent.last_prayer_turn <= jf_config.DIVE_FED_GAP)
        if not fed and not self._fed_wait_logged:
            self._fed_wait_logged = True
            agent.log(f'DIVE_FED waiting: hunger={bl.hunger_state} gap={None if agent.last_prayer_turn is None else bl.time - agent.last_prayer_turn} '
                      f'food={agent.carried_food_nutrition()}')
        return fed

    def _min_xl(self, default):
        return default if TOOL_RUN_XL is None else min(default, TOOL_RUN_XL)

    def _ditch_pet_role(self):
        """Knights and Samurai ditch their starting pet (DITCH_PET_ROLES); DITCH_PET turns it on for everyone."""
        try:
            return self.agent.character.role in DITCH_PET_ROLES
        except Exception:
            return False

    def _ditch_pet_check(self, level, first):
        """The ditch's entry condition (no agent steps): starts an attempt from state 0, ends a late one."""
        agent = self.agent
        bl = agent.blstats
        if self._ditch_state == 0:
            if bl.time < getattr(self, '_ditch_retry_after', -1):
                return False
            if level.key() != first or not agent.has_pet or agent.get_visible_monsters() or \
                    bl.hitpoints < 0.8 * bl.max_hitpoints:
                return False
            if bl.time < (0 if self._ditch_pet_role() else DITCH_PET_AFTER):
                return False
            if self._pet_hunger_turn is not None and bl.time - self._pet_hunger_turn <= PET_HUNGER_WINDOW:
                return False   # a hunger-confused pet bites an adjacent hero: don't herd it to the stairs
            if self._ditch_tries >= DITCH_PET_TRIES:
                self._ditch_state = 3
                return False
            dis = agent.bfs()
            if not any(dis[p] != -1 for p in self._stairs_down(level)):
                return False
            self._ditch_tries += 1
            agent.log(f'DITCH pet: taking it down to Dlvl 2 (try {self._ditch_tries})')
            self._ditch_state = 1
            self._ditch_started = bl.time
            self._ditch_pet_came = False
        if bl.time - self._ditch_started > DITCH_PET_BUDGET:
            agent.log(f'DITCH pet: out of time (state {self._ditch_state})')
            if jf_config.DITCH_RETRY and self._ditch_tries < DITCH_PET_TRIES:
                # DITCH_RETRY: one timed-out walk to the stairs (the pet wandered off) ended the ditch for good, and a
                # Knight fed the pony its apples and carrots until they ran out; the starving horse then kicked it to
                # death (all_v1 kni s1, T4494). Try again later.
                self._ditch_state = 0
                self._ditch_retry_after = bl.time + DITCH_RETRY_WAIT
                return False
            self._ditch_state = 3
            return False
        return True

    @Strategy.wrap
    def ditch_pet_strategy(self):
        agent = self.agent
        from .global_logic import Milestone
        if self._ditch_state == 3 or self.diving or not (DITCH_PET or self._ditch_pet_role()) or \
                agent.global_logic.milestone != Milestone.BE_ON_FIRST_LEVEL:
            yield False
        bl = agent.blstats
        level = agent.current_level()
        first = (Level.DUNGEONS_OF_DOOM, 1)
        try:
            go = self._ditch_pet_check(level, first)
        except Exception as e:   # fail safe: no ditch, the grind goes on with the pet
            agent.log(f'DITCH pet: check failed, off: {type(e).__name__} {str(e)[:150]}')
            self._ditch_state = 3
            go = False
        if not go:
            yield False
        yield True
        pos = (bl.y, bl.x)
        pets = utils.isin(agent.glyphs, G.PETS)
        pet_adjacent = any(utils.adjacent(pos, (int(y), int(x))) for y, x in zip(*pets.nonzero()))
        key = level.key()
        if key == (Level.DUNGEONS_OF_DOOM, 2) and pets.any():
            self._ditch_pet_came = True   # keepdogs brought it along (dog.c:590 monnear + levl_follower)
        if self._ditch_state == 1:
            if key == (Level.DUNGEONS_OF_DOOM, 2):
                self._ditch_state = 2
                return
            if key != first:
                self._ditch_state = 3
                return
            dis = agent.bfs()
            downs = [p for p in self._stairs_down(level) if dis[p] != -1]
            if not downs:
                self._ditch_state = 3
                return
            y, x = min(downs, key=lambda p: dis[p])
            if pos != (y, x):
                agent.go_to(y, x)
            elif pet_adjacent:
                agent.move('>')
            else:
                agent.search()   # wait for the pet to come close enough to follow
            return
        # state 2: on Dlvl 2 with the pet; climb back when it is not adjacent
        if key == first:
            if self._ditch_pet_came or not self._ditch_pet_role():
                agent.log(f'DITCH pet: back on Dlvl 1, pet left on Dlvl 2: {self._ditch_pet_came}')
                self._ditch_state = 3
            else:
                # it wasn't next to us when we went down (or was eating): it is still up here
                agent.log('DITCH pet: it did not follow us down, trying again')
                self._ditch_state = 0
            return
        # the '<' we arrived on is under us, so the map's objects don't show it until we step off: the
        # arrival itself recorded it as the way back to Dlvl 1 (Agent.move stair_destination)
        ups = [tuple(p) for p, dest in level.stair_destination.items() if dest is not None and dest[0] == first]
        ups += [p for p in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()) if p not in ups]
        if not ups:
            self._ditch_state = 3
            return
        y, x = ups[0]
        if pos != (y, x):
            agent.go_to(y, x)
        elif not pet_adjacent:
            agent.move('<')
        else:
            agent.search()   # let it wander off

    def should_hunt_dwarf(self):
        return self._hunt_allowed() and bool(self._hunt_targets())

    def should_approach_dwarf(self):
        if not APPROACH_DWARVES or not self._hunt_allowed():
            return False
        key = self.agent.current_level().key()
        start, n = self._approach_count.get(key, (0, 0))
        # APPROACH_STEPS moves per APPROACH_WINDOW turns: a grind level is home for thousands of turns
        # (eg-glh public: a Dlvl-2 digger seen in rock 15+ times over 700 turns, never attacked, once a flat
        # per-level budget ran out)
        if n >= APPROACH_STEPS and self.agent.blstats.time - start < APPROACH_WINDOW:
            return False
        return bool(self._unreachable_dwarves())

    def _unreachable_dwarves(self):
        """Peaceful dwarves in view with no reachable neighbour square (a digger inside its tunnel), diggers
        first, then the nearest."""
        agent = self.agent
        if agent.character.prop.blind or agent.character.prop.hallu:
            return []
        mask = agent.monster_tracker.peaceful_monster_mask & utils.isin(agent.glyphs, self._dwarf_glyphs())
        if not mask.any():
            return []
        dis = agent.bfs()
        out = [(int(y), int(x)) for y, x in zip(*mask.nonzero()) if self._neighbour_distance(dis, y, x) is None]
        if not out:
            return out
        if ALIGN_BUDGET and not self._may_kill_dwarf(witnesses=max(int(mask.sum()) - 1, 0)):
            return []
        key = agent.current_level().key()
        y0, x0 = agent.blstats.y, agent.blstats.x
        return sorted(out, key=lambda p: (not self._is_digger(key, p), max(abs(p[0] - y0), abs(p[1] - x0))))

    def _lure(self, ty, tx):
        """LURE_DWARVES: a peaceful dwarf we see along a straight line (8 directions) within LURE_RANGE but can't
        reach, a digger in its tunnel above all: throw a missile at it. hmon wakes and angers it before any damage
        (uhitm.c hmon_hitmon, also for thrown objects), and a hostile dwarf digs its own way to us; fight2 finishes
        it next to us and DWARF_PILES checks its pile. The same alignment cost as a melee kill, no Luck.
        base3arm-jf14 s2 saw diggers 41 times on the dive's first levels and never got to attack one."""
        agent = self.agent
        y0, x0 = agent.blstats.y, agent.blstats.x
        dy, dx = ty - y0, tx - x0
        n = max(abs(dy), abs(dx))
        if n < 2 or n > LURE_RANGE or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            return False
        if agent.character.prop.polymorph or agent.character.prop.blind or agent.character.prop.hallu:
            return False
        # every peaceful humanoid that sees the hit turns hostile (mon.c setmangry): gnome lords shoot crossbows
        # (F020); only lure a dwarf with no other peaceful in view
        if int(agent.monster_tracker.peaceful_monster_mask.sum()) > 1:
            return False
        sy, sx = int(np.sign(dy)), int(np.sign(dx))
        for i in range(1, n):
            p = (y0 + sy * i, x0 + sx * i)
            if agent.monster_tracker.monster_mask[p] or agent.glyphs[p] in G.BOULDER or \
                    agent.glyphs[p] in G.PETS:
                return False   # the missile would hit something else first
        ammo = self._lure_ammo()
        if ammo is None:
            return False
        dir = agent.calc_direction(y0, x0, ty, tx, allow_nonunit_distance=True)
        agent.log(f'LURE throwing {ammo.text!r} at the peaceful dwarf at {(ty, tx)} '
                  f'(digger={self._is_digger(agent.current_level().key(), (ty, tx))}, align~{self._align_est})')
        self._hunting = True
        agent.fire(ammo, dir)
        agent.monster_tracker.on_panic()
        return True

    def _lure_ammo(self):
        """Something to throw by hand: a thrown weapon (daggers, darts...), else a spare weapon or a gem/glass --
        any object that hits angers it (hmon), and the smoke games had thrown their daggers away by the dive.
        Never the wielded or best melee weapon, anything worn or quivered in a launcher, nor a gray stone (a
        cursed loadstone can't leave the hand)."""
        inv = self.agent.inventory
        launcher, ammo = inv.get_best_ranged_set(throwing=True)
        if ammo is not None and launcher is None:
            return ammo
        best = inv.get_best_melee_weapon()
        main = inv.items.main_hand
        spare = []
        for item in flatten_items(inv.items):
            if item is best or item is main or item.equipped or item.shop_status == Item.UNPAID:
                continue
            if item.status == Item.CURSED:
                continue
            names = {getattr(o, 'name', '') for o in item.objs}
            if item.category == nh.WEAPON_CLASS and not item.is_launcher() and item.unit_weight() <= 60:
                spare.append((0, item.unit_weight(), item))
            elif item.category == nh.GEM_CLASS and not any('gray stone' in n or n in ('loadstone', 'luckstone',
                                                                                     'touchstone', 'flint')
                                                            for n in names):
                spare.append((1, item.unit_weight(), item))
        return min(spare, key=lambda t: t[:2])[2] if spare else None

    def _open_line_to(self, ty, tx):
        """The dwarf is in view along a straight line: every square between is clear of rock (vision.c
        does_block), i.e. a tunnel it dug, which we haven't walked (dark corridors show only next to us).
        Mark those squares walkable so the BFS reaches it. Water doesn't block the view: never across it."""
        agent = self.agent
        level = agent.current_level()
        y0, x0 = agent.blstats.y, agent.blstats.x
        dy, dx = ty - y0, tx - x0
        n = max(abs(dy), abs(dx))
        if n < 2 or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            return False
        sy, sx = int(np.sign(dy)), int(np.sign(dx))
        line = [(y0 + sy * i, x0 + sx * i) for i in range(1, n)]
        if any(agent.glyphs[p] in WET or level.objects[p] in WET or agent.glyphs[p] in G.BOULDER or
               agent.glyphs[p] in G.DOOR_CLOSED or level.objects[p] in G.DOOR_CLOSED for p in line):
            return False
        todo = [p for p in line if not level.walkable[p]]
        if not todo:
            return False
        for p in todo:
            level.walkable[p] = True
            if level.objects[p] in G.STONE or level.objects[p] in G.WALL:
                level.objects[p] = SS.S_corr   # dug: a corridor now
        agent.last_bfs_step = -1   # the BFS cache ignores map edits
        agent.log(f'APPROACH line to the dwarf at {(ty, tx)} is open: {todo}')
        return True

    def _approach_dwarf(self):
        """One move toward a peaceful dwarf we can't reach yet; False if nothing was done."""
        agent = self.agent
        key = agent.current_level().key()
        targets = self._unreachable_dwarves()
        if not targets:
            return False
        ty, tx = targets[0]
        start, n = self._approach_count.get(key, (0, 0))
        t = agent.blstats.time
        self._approach_count[key] = (start, n + 1) if t - start < APPROACH_WINDOW else (t, 1)
        if LURE_DWARVES and self._lure(ty, tx):
            return True
        if self._open_line_to(ty, tx):
            return False   # no move: the caller hunts it now
        dis = agent.bfs()
        ys, xs = np.nonzero(dis >= 0)
        if len(ys) == 0:
            return False
        cheb = np.maximum(np.abs(ys - ty), np.abs(xs - tx))
        inline = (ys == ty) | (xs == tx) | (np.abs(ys - ty) == np.abs(xs - tx))
        lure = LURE_DWARVES and self._lure_ammo() is not None
        if lure:
            # a throwing position first (in line, 2..LURE_RANGE away), the nearest one; else as below
            spot = inline & (cheb >= 2) & (cheb <= LURE_RANGE)
            order = np.lexsort((dis[ys, xs], cheb, ~spot))
        else:
            # closest to it, a square in line with it (to see down its tunnel) breaking ties, then the nearest
            order = np.lexsort((dis[ys, xs], ~inline, cheb))
        i = order[0]
        y0, x0 = agent.blstats.y, agent.blstats.x
        throw_spot = lure and bool(spot[i])
        if (ys[i], xs[i]) == (y0, x0) or (cheb[i] > max(abs(ty - y0), abs(tx - x0)) and not throw_spot):
            return False
        agent.log(f'APPROACH dwarf at {(ty, tx)} via {(int(ys[i]), int(xs[i]))}')
        agent.go_to(int(ys[i]), int(xs[i]), max_steps=3)
        return True

    def _hunt_fit(self):
        """The health gate for hunting and searching dwarves (the one place for phase exceptions, e.g. a rescue)."""
        bl = self.agent.blstats
        return bl.hitpoints >= 0.7 * bl.max_hitpoints and bl.hunger_state < Hunger.WEAK

    def _tool_blocked(self):
        """A digging tool is carried but can't be used: the wielded weapon is cursed (welded). Another dwarf's pick
        would be just as unusable (early-game eg-glh public/6: an Excalibur dip cursed the long sword, and the hunt
        killed two more dwarves for nothing)."""
        inv = self.agent.inventory.items
        tool = self.best_digging_tool(inv)
        main = inv.main_hand
        return tool is not None and main is not None and main is not tool and main.status == Item.CURSED

    def _hunt_allowed(self):
        """Everything but a target: the phase, the level, health, the kill budget."""
        agent = self.agent
        if not DWARF_HUNT or not self._may_kill_dwarf():
            return False
        if (DWARF_PILES or ALIGN_BUDGET) and self._tool_blocked():
            return False
        grind = self._grind_hunting()
        if not self.diving and not self.pick_trip and not grind and \
                (not HUNT_IN_TOUR or agent.blstats.experience_level < self._min_xl(HUNT_MIN_XL)):
            return False
        if self.digging_tool() is not None or self._fetch is not None:
            return False
        level = agent.current_level()
        # dwarves wander the main dungeon too; never in Minetown (the Watch defends peacefuls)
        if level.dungeon_number not in (Level.GNOMISH_MINES, Level.DUNGEONS_OF_DOOM) or \
                not self._hunt_level_ok(level):
            return False
        if level.dungeon_number == Level.DUNGEONS_OF_DOOM and not self.diving and not grind:
            return False
        bl = agent.blstats
        if not self._hunt_fit():
            return False
        started = self._hunt_started.setdefault(level.key(), bl.time)
        # the grind meets dwarves over thousands of turns, and a tour-mode pick trip explores a Mines level for
        # 1000-1500 turns (the clock starts at the first check on the level): no per-level hunt budget there.
        # Nor in the Mines for hunt-v2 (the search budget decides how long we stay): a hunt-v2 replay of jf27/14
        # saw four digging dwarves on Mines 2 after the 400 turns and attacked none.
        v2 = DWARF_PILES or DIGGER_FIRST or ALIGN_BUDGET
        if bl.time - started > DWARF_HUNT_TURNS and not grind and not (self.pick_trip and v2) and \
                not (v2 and level.dungeon_number == Level.GNOMISH_MINES):
            return False
        return True

    def _hunt_targets(self):
        """Peaceful dwarves worth attacking now: diggers only in a GRIND_HUNT_DIGGERS_ONLY grind, and (ALIGN_BUDGET)
        only while the budget also covers the other peaceful dwarves in view, which turn hostile at the attack."""
        targets = self._peaceful_dwarves()
        if not targets:
            return targets
        key = self.agent.current_level().key()
        if self._grind_hunting() and GRIND_HUNT_DIGGERS_ONLY:
            targets = [t for t in targets if self._is_digger(key, (t[1], t[2]))]
        if ALIGN_BUDGET and targets:
            glyphs = self._dwarf_glyphs()
            n_peaceful = int((self.agent.monster_tracker.peaceful_monster_mask &
                              utils.isin(self.agent.glyphs, glyphs)).sum())
            if not self._may_kill_dwarf(witnesses=max(n_peaceful - 1, 0)):
                return []
        return targets

    def should_search_dwarves(self):
        agent = self.agent
        if not DWARF_HUNT or not self.diving or not self._may_kill_dwarf():
            return False
        if self.digging_tool() is not None or self._fetch is not None:
            return False
        level = agent.current_level()
        if level.dungeon_number != Level.GNOMISH_MINES or level.key() == agent.global_logic.minetown_level or \
                (MINETOWN_GUARD and level.key() in self._town_levels):
            return False
        bl = agent.blstats
        if not self._hunt_fit():
            return False
        started = self._search_started.setdefault(level.key(), bl.time)
        budget = DWARF_SEARCH_TURNS
        key = level.key()
        if DIGGER_FIRST and key in self._crash_turn:
            budget += CRASH_SEARCH_BONUS   # a dwarf with a pick digs somewhere on this level
            # ...and while it keeps digging (heard or seen lately), stay on, up to CRASH_SEARCH_MAX
            marks = self._diggers.get(key, {})
            last = max([self._crash_turn[key]] + list(marks.values()))
            if bl.time - last <= 150:
                budget = max(budget, min(bl.time - started + 1, CRASH_SEARCH_MAX))
        return bl.time - started <= budget

    _BLOCKER_NAMES = frozenset(('gnome', 'gnome lord', 'gnomish wizard', 'gnome king'))

    def _clear_blocker(self):
        """MINES_UNSTUCK, Mines only (not Minetown): boxed in by peaceful gnomes for 30+ turns (the BFS takes
        peacefuls for walls; a Valkyrie is stealthy, so monmove.c disturb() never wakes sleeping ones): attack
        an adjacent one. A gnome's malign is 0 for us: -1 alignment for the anger (and each witness), nothing
        for the kill. base2-public s8 stood 4000+ turns on Mines 1 ringed by gnomes in a narrow passage."""
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number != Level.GNOMISH_MINES or not self._hunt_level_ok(level) or \
                agent.character.prop.hallu or agent.character.prop.blind:
            self._boxed_since = None
            return False
        dis = agent.bfs()
        if int((dis != -1).sum()) >= 40:
            self._boxed_since = None
            return False
        t = agent.blstats.time
        if self._boxed_since is None or self._boxed_since[0] != level.key():
            self._boxed_since = (level.key(), t)
            return False
        if t - self._boxed_since[1] < 30:
            return False
        y0, x0 = agent.blstats.y, agent.blstats.x
        for y, x in agent.neighbors(y0, x0, shuffle=False):
            if not agent.monster_tracker.peaceful_monster_mask[y, x] or not MON.is_monster(agent.glyphs[y, x]):
                continue
            name = MON.permonst(agent.glyphs[y, x]).mname
            if name not in self._BLOCKER_NAMES:
                continue
            if y != y0 and x != x0 and (level.intact_doors[y0, x0] or level.intact_doors[y, x]):
                continue
            agent.log(f'BLOCKER boxed in {t - self._boxed_since[1]} turns: attacking the peaceful {name} at {(y, x)}')
            self._boxed_since = (level.key(), t)
            with agent.atom_operation():
                agent.step(A.Command.FIGHT)
                agent.direction(agent.calc_direction(y0, x0, y, x))
            agent.monster_tracker.on_panic()
            return True
        return False

    def should_search_home_digger(self):
        agent = self.agent
        if not HOME_DIGGER_TURNS or not self.diving or self.rescue or self.digging_tool() is not None:
            return False
        level = agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or not self._may_kill_dwarf():
            return False
        key = level.key()
        t = agent.blstats.time
        seen = [self._crash_turn.get(key, -10 ** 9)] + list(self._diggers.get(key, {}).values())
        if t - max(seen) > HOME_DIGGER_WINDOW:
            return False
        started = self._home_search.setdefault(key, t)
        if started == t:
            agent.log(f'HOME digger heard/seen here {t - max(seen)} turns ago: searching {key} for it first')
        bl = agent.blstats
        return t - started < HOME_DIGGER_TURNS and bl.hitpoints >= 0.7 * bl.max_hitpoints and \
            bl.hunger_state < Hunger.WEAK

    def search_home_digger(self):
        agent = self.agent
        key = agent.current_level().key()
        started = self._home_search[key]
        stop = lambda: bool(self._hunt_targets()) or (APPROACH_DWARVES and bool(self._unreachable_dwarves())) or \
            self.digging_tool() is not None or self._fetch_scan_turn < 0 or \
            agent.blstats.time - started >= HOME_DIGGER_TURNS
        self._task('home level: search for its digger')
        if self.exploration(0).until(agent, stop).run(return_condition=True):
            return
        self._task('home level: patrol for its digger')
        if not agent.exploration.patrol().until(agent, stop).run(return_condition=True):
            agent.search(5)

    _VI_KEYS = {'y': (-1, -1), 'k': (-1, 0), 'u': (-1, 1), 'h': (0, -1), 'l': (0, 1), 'b': (1, -1), 'j': (1, 0),
                'n': (1, 1)}

    def _dig_out(self):
        """RETURN_FIX, dive only: boxed in (the BFS reaches fewer than DIG_OUT_REACH squares for DIG_OUT_TURNS turns)
        with a usable digging tool: dig sideways toward the nearest known square outside the pocket. base3-jf16 s14
        picked up a pick-axe in a Mines-3 pocket and sat there ~3000 turns (reachable=2): the pick's 100 weight put
        the pack over 600, and NetHack refuses a diagonal squeeze with more. Never in Minetown (digging walls there
        brings the Watch), never a shop wall; a wall 'too hard to dig' is not tried again."""
        agent = self.agent
        if not self.diving or agent.blstats.time < self._dig_out_block:
            return False
        level = agent.current_level()
        key = level.key()
        if level.dungeon_number not in (Level.GNOMISH_MINES, Level.DUNGEONS_OF_DOOM) or \
                key == agent.global_logic.minetown_level or key in self._town_levels:
            return False
        dis = agent.bfs()
        reach = dis != -1
        n = int(reach.sum())
        t = agent.blstats.time
        if n >= DIG_OUT_REACH:
            self._boxed_dig = None
            return False
        tool = self.digging_tool()
        if tool is None or agent.character.prop.polymorph:
            return False
        if self._boxed_dig is None or self._boxed_dig[0] != key:
            self._boxed_dig = (key, t)
            return False
        if t - self._boxed_dig[1] < DIG_OUT_TURNS:
            return False
        y0, x0 = agent.blstats.y, agent.blstats.x
        outside = level.walkable & ~reach & ~level.shop & ~level.shop_interior
        if outside.any():
            ys, xs = outside.nonzero()
            i = int(np.argmin(np.maximum(np.abs(ys - y0), np.abs(xs - x0))))
            ty, tx = int(ys[i]), int(xs[i])
        else:
            ty, tx = C.SIZE_Y // 2, C.SIZE_X // 2
        # dig from the square of the pocket nearest the target
        ry, rx = reach.nonzero()
        j = int(np.argmin(np.maximum(np.abs(ry - ty), np.abs(rx - tx)) * 100 + dis[ry, rx]))
        sy, sx = int(ry[j]), int(rx[j])
        if (sy, sx) != (y0, x0) and max(abs(sy - ty), abs(sx - tx)) < max(abs(y0 - ty), abs(x0 - tx)):
            agent.go_to(sy, sx)
            return True
        with agent.atom_operation():
            tool = agent.inventory.move_to_inventory(tool)
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(tool))
            msg = agent.single_message
            m = re.search(r'In what direction do you want to dig\? \[([^\]]*)\]', msg)
            best = None
            for c in (m.group(1) if m else ''):
                if c not in self._VI_KEYS:
                    continue
                dy, dx = self._VI_KEYS[c]
                py, px = y0 + dy, x0 + dx
                if not (0 <= py < C.SIZE_Y and 0 <= px < C.SIZE_X) or (key, (py, px)) in self._dig_out_bad or \
                        level.shop[py, px] or level.shop_interior[py, px]:
                    continue
                score = max(abs(ty - py), abs(tx - px))
                if best is None or score < best[0]:
                    best = (score, py, px)
            if best is None:
                if msg.startswith('In what direction'):
                    agent.step(A.Command.ESC)
                agent.log(f'DIG OUT boxed ({n} squares) but no way to dig: {msg!r}')
                self._dig_out_block = t + 200
                return True
            _, py, px = best
            agent.direction(py, px)
        msg = agent.message
        if 'too hard to' in msg:
            self._dig_out_bad.add((key, (py, px)))
        agent.log(f'DIG OUT boxed ({n} squares reachable) toward {(ty, tx)}: dug at {(py, px)}: {msg[:90]!r}')
        agent.last_bfs_step = -1
        return True

    def should_camp(self):
        agent = self.agent
        if not MINES_CAMP or self._camp_over or not self.diving or self.rescue or self._tool_blocked():
            return False
        if self.digging_tool() is not None or agent.prayer_failed:
            return False
        if agent.current_level().dungeon_number != Level.GNOMISH_MINES:
            return False
        t = agent.blstats.time
        if self._camp_start is None:
            self._camp_start = t
            agent.log(f'CAMP: the tool-less dive stays in the Mines (levels 1-{MINES_CAMP_MAX_LEVEL}, '
                      f'{MINES_CAMP_TURNS} turns)')
        if t - self._camp_start > MINES_CAMP_TURNS:
            self._camp_over = True
            agent.log('CAMP: budget spent, on down the Mines')
            return False
        return True

    def _camp_stairs(self, level, direction):
        if direction == '>':
            return self._stairs_down(level)
        return list(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))

    def camp_step(self):
        """MINES_CAMP: search this camp level for dwarves (hunts, approaches and piles come first in plan_step),
        then move on to the next level of the sweep."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        t = agent.blstats.time
        if self._camp_visit is None or self._camp_visit[0] != key:
            self._camp_visit = (key, t)
            if CAMP_FIX:
                # a fresh visit: the timers for leaving this level start over (a stale one from an earlier visit
                # turned the sweep around at once: base3-public s13 ping-ponged Mines 2 <-> 3 every turn)
                for d in ('<', '>'):
                    self._camp_moves.pop((key, d), None)
        started = self._camp_visit[1]
        budget = CAMP_VISIT_TURNS
        if key in self._crash_turn and t - self._crash_turn[key] < 2000:
            budget += CRASH_SEARCH_BONUS   # a digger was heard here lately
        ok = self._hunt_level_ok(level) and level.level_number <= MINES_CAMP_MAX_LEVEL
        bl = agent.blstats
        if ok and t - started < budget and bl.hitpoints >= 0.5 * bl.max_hitpoints:
            stop = lambda: bool(self._hunt_targets()) or (APPROACH_DWARVES and bool(self._unreachable_dwarves())) or \
                self.digging_tool() is not None or self._fetch_scan_turn < 0 or agent.blstats.time - started >= budget
            # the unexplored parts first; an explored level is patrolled (wandering dwarves come into view; the
            # exploration's wall searching would stand still)
            self._task('camp: search for dwarves')
            if self.exploration(0).until(agent, stop).run(return_condition=True):
                return
            self._task('camp: patrol')
            if not agent.exploration.patrol().until(agent, stop).run(return_condition=True):
                agent.search(5)
            return
        # the next level of the sweep: down to MINES_CAMP_MAX_LEVEL, then back up to Mines 1, and so on
        n = level.level_number
        below = (Level.GNOMISH_MINES, n + 1)
        town_below = agent.global_logic.minetown_level == below or below in self._town_levels
        if n >= MINES_CAMP_MAX_LEVEL or (town_below and n + 1 >= MINES_CAMP_MAX_LEVEL):
            self._camp_dir = -1   # (no point walking into a Minetown that ends the sweep)
        elif n <= 1:
            self._camp_dir = 1
        direction = '>' if self._camp_dir > 0 else '<'
        k = (key, direction)
        self._camp_moves.setdefault(k, t)
        if t - self._camp_moves[k] > CAMP_STAIRS_TURNS:
            # that way is cut off (a '>' we can't find or reach): turn around
            self._camp_moves.pop(k, None)
            self._camp_dir = -self._camp_dir
            self._camp_visit = (key, t)
            agent.log(f'CAMP: no way {direction} from {key}, turning around')
            return
        self._task(f'camp: next level {direction}')
        stairs = self._camp_stairs(level, direction)
        if stairs and self._take_stairs(stairs, direction):
            if agent.current_level().key() != key:
                self._camp_moves.pop(k, None)
            return
        reachable = lambda: any(agent.bfs()[p] != -1 for p in self._camp_stairs(agent.current_level(), direction))
        if not self.exploration(None).until(agent, self._budgeted(reachable)).run(return_condition=True):
            agent.search()

    def should_tool_quest(self):
        agent = self.agent
        if not DWARF_HUNT or self._quest_over or not self.diving:
            return False
        if self.digging_tool() is not None:
            if self._quest_started is not None:
                agent.log('DIVE tool quest done: got a digging tool')
                self._quest_started = None
            return False
        bl = agent.blstats
        if self._quest_started is not None:
            budget = EARLY_DETOUR_TURNS if self._quest_early else TOOL_QUEST_TURNS
            if bl.time - self._quest_started > budget or self._quest_target() is None:
                agent.log('DIVE tool quest given up' + (' (early detour)' if self._quest_early else ''))
                if self._quest_early:
                    self._early_done = True
                else:
                    self._quest_over = True
                self._quest_started = None
                self._quest_early = False
                return False
            return True
        level = agent.current_level()
        if EARLY_DETOUR and not self._early_done and level.dungeon_number == Level.DUNGEONS_OF_DOOM and \
                bl.depth <= EARLY_DETOUR_DEPTH:
            self._quest_early = True
            if self._quest_target() is not None:
                agent.log(f'DIVE no digging tool at depth {bl.depth}: early detour to the Mines')
                self._quest_started = bl.time
                return True
            self._quest_early = False
            self._early_done = True
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or bl.depth < TOOL_QUEST_DEPTH or \
                self.turns_on_level() < TOOL_QUEST_STUCK_TURNS or self.down_targets() or \
                self._quest_target() is None:
            return False
        agent.log(f'DIVE stuck on {level.key()} with no way down: tool quest to the Mines')
        self._quest_started = bl.time
        return True

    def _quest_target(self):
        """The shallowest known Mines level (not Minetown) whose dwarf search isn't used up."""
        agent = self.agent
        now = agent.blstats.time
        minetown = agent.global_logic.minetown_level
        keys = sorted(k for k in agent.levels if k[0] == Level.GNOMISH_MINES and k != minetown and
                      (not self._quest_early or k[1] <= EARLY_DETOUR_MINES_LEVELS))
        for key in keys:
            started = self._search_started.get(key)
            if started is None or now - started <= DWARF_SEARCH_TURNS:
                return key
        return None

    def tool_quest(self):
        agent = self.agent
        target = self._quest_target()
        if agent.current_level().key() == target:
            # should_search_dwarves / should_hunt_dwarf run first; nothing left here but to wait a turn
            agent.search()
            return
        path = agent.exploration.get_path_to_level(*target)
        if not path:
            # levels we fell into have no known stairs: climb one level at a time until the levels the
            # tour walked (and their stairs) connect us to the Mines
            level = agent.current_level()
            if level.dungeon_number != Level.DUNGEONS_OF_DOOM or agent.blstats.depth <= 1:
                agent.log(f'DIVE tool quest: no way to {target}')
                self._search_started[target] = -10 ** 9   # skip it
                return
            ups = list(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))
            if ups and self._take_stairs(ups, '<'):
                return
            self.exploration(None).until(agent, self._budgeted(lambda: any(
                agent.bfs()[p] != -1 for p in zip(*utils.isin(agent.current_level().objects,
                                                               G.STAIR_UP).nonzero())))).run()
            return
        (sy, sx), _, direction = path[0]
        if not self._take_stairs([(sy, sx)], direction):
            self.exploration(None).until(agent, self._budgeted(lambda: agent.bfs()[sy, sx] != -1)).run()

    def hunt_dwarf(self):
        agent = self.agent
        targets = self._hunt_targets()
        if not targets:
            return
        _, y, x = targets[0]
        pos = (agent.blstats.y, agent.blstats.x)
        if not utils.adjacent(pos, (y, x)):
            agent.go_to(y, x, stop_one_before=True, max_steps=2)
            return
        digger = self._is_digger(agent.current_level().key(), (y, x)) if DIGGER_FIRST else None
        agent.log(f'DIVE attacking a peaceful dwarf at {(y, x)} for its digging tool'
                  + (f' (digger={digger}, align~{self._align_est})' if (DIGGER_FIRST or ALIGN_BUDGET) else ''))
        self._hunting = True
        with agent.atom_operation():
            agent.step(A.Command.FIGHT)
            agent.direction(agent.calc_direction(pos[0], pos[1], y, x))
        # it is hostile now: re-list the monsters so the fight logic takes over
        agent.monster_tracker.on_panic()

    def _crawl_exits(self, py, px, count_monsters=True):
        """Adjacent squares a flooded digger could crawl out to (hack.c crawl_destination: goodpos -- land, no
        monster -- and no diagonal squeeze between two rock squares). With none, a hole that fills with water
        drowns us on the spot: 'You fall into the water! You sink like a rock.' and no 'try to crawl out'."""
        agent = self.agent
        level = agent.current_level()
        h, w = level.objects.shape
        n = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = py + dy, px + dx
                if (dy, dx) == (0, 0) or not (0 <= y < h and 0 <= x < w):
                    continue
                terrain = level.objects[y, x]
                if agent.glyphs[y, x] in WET or terrain in WET or terrain == SS.S_tree or \
                        not level.walkable[y, x] or terrain in G.DOORS:
                    continue
                if count_monsters and agent.monster_tracker.monster_mask[y, x]:
                    continue
                if dy and dx:
                    def rock(yy, xx):
                        return not level.walkable[yy, xx] and level.objects[yy, xx] not in WET
                    if rock(py + dy, px) and rock(py, px + dx):
                        continue
                n += 1
        return n

    def _medusa_variant_name(self):
        """MEDUSA_MAP: the medusa_maps variant of the current level (Medusa's), once enough seen squares agree."""
        if not MEDUSA_MAP or not self.on_medusa_level():
            return None
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        name = self._medusa_variant.get(key)
        if name is not None:
            return name
        turn = agent.blstats.time
        if getattr(self, '_medusa_variant_tried', None) == (key, turn):
            return None
        self._medusa_variant_tried = (key, turn)
        from . import medusa_maps
        water = list(zip(*utils.isin(level.objects, WET).nonzero()))
        dry = list(zip(*utils.isin(level.objects, PLAIN_FLOOR).nonzero()))
        best = medusa_maps.match(water, dry)
        if best is not None and best[2] >= MEDUSA_MAP_MIN_SEEN and best[1] <= MEDUSA_MAP_MAX_BAD * best[2]:
            self._medusa_variant[key] = best[0]
            agent.log(f'DIVE MEDUSA_MAP: this is {best[0]} ({best[1]} of {best[2]} seen squares disagree)')
            return best[0]
        return None

    def _medusa_moat_mask(self, level):
        """MEDUSA_MAP: a bool array (level shape) of the variant's moat squares, cached per variant."""
        name = self._medusa_variant_name()
        if name is None:
            return None
        cache = self.__dict__.setdefault('_moat_masks', {})
        if name not in cache:
            from . import medusa_maps
            h, w = level.objects.shape
            mask = np.zeros((h, w), dtype=bool)
            for y in range(h):
                for x in range(w):
                    mask[y, x] = medusa_maps.char_at(name, y, x) == '}'
            cache[name] = mask
        return cache[name]

    def _map_char(self, y, x):
        """MEDUSA_MAP: the fixed map's character at screen (y, x), or None if the variant isn't known."""
        name = self._medusa_variant_name()
        if name is None:
            return None
        from . import medusa_maps
        return medusa_maps.char_at(name, int(y), int(x))

    def _wet_neighbours(self, py, px):
        agent = self.agent
        level = agent.current_level()
        around = agent.glyphs[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
        around_known = level.objects[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
        wet = utils.isin(around, WET) | utils.isin(around_known, WET)
        if MEDUSA_MAP and self.on_medusa_level() and (around_known == -1).any() and \
                self._medusa_variant_name() is not None:
            # (see MEDUSA_MAP) a never-seen neighbour is moat if the level's fixed map says so
            y0, x0 = max(py - 1, 0), max(px - 1, 0)
            for (dy, dx) in zip(*(around_known == -1).nonzero()):
                if self._map_char(y0 + dy, x0 + dx) == '}':
                    wet[dy, dx] = True
        elif MEDUSA_UNKNOWN_WET and self.on_medusa_level():
            # (see MEDUSA_UNKNOWN_WET) never-seen terrain next to the square, not our own square
            unknown = around_known == -1
            cy, cx = py - max(py - 1, 0), px - max(px - 1, 0)
            unknown[cy, cx] = False
            wet |= unknown
        return int(wet.sum())

    def _diggable_spot(self, py, px, max_wet=0):
        agent = self.agent
        level = agent.current_level()
        # a square always covered by objects (a leprechaun hall is gold wall to wall) never shows its
        # floor: an s12 digger found no 'floor' there and explored the hall until it starved
        terrain = level.objects[py, px]
        if not (terrain in PLAIN_FLOOR or (terrain == -1 and level.walkable[py, px]) or
                (DIG_IN_PITS and terrain in PITS)) or \
                ((level.shop[py, px] or level.shop_interior[py, px]) and not self._shop_dig_ok(py, px)) or \
                (level.key(), (py, px)) in self._bad_dig_spots:
            return False
        # the square we arrived on by stairs was never seen ('@' covers it: terrain -1), but it is a staircase
        # (rescue agent, 78a30e1: 'The beam bounces off the stairs' emptied wands of digging in 6 of 90 games)
        if jf_config.WAND_STAIRS_FIX and (py, px) in level.stair_destination:
            return False
        # a hole next to water or lava fills with it (dig.c fillholetyp: n moat squares around fill it
        # with probability n/(n+1)); only islands with no dry square (Medusa variants) accept the risk
        return self._wet_neighbours(py, px) <= max_wet

    def _shop_dig_ok(self, py, px):
        """An empty shop floor square an Archeologist may dig through.
        hypothesis: an Archeologist's dug hole often drops it into a shop, where the shopkeeper sits in the
        door and never lets a pick-axe carrier out (shk.c shk_move: badinv); with the dive's digging restricted
        to non-shop floor the bot then never leaves -- seeds 5, 6 and 14 fought a small mimic among the stock
        until they starved. shk.c shopdig() grabs the pack of a customer falling through only if they owe
        something (billct or debit), and dokick.c impact_drop bills only for goods falling with us, so a
        bare interior square, dug with nothing unpaid in the pack, takes us down with everything we carry.
        Gated to Archeologists (the early dig-divers that land in shops)."""
        agent = self.agent
        level = agent.current_level()
        if not self._arc_early_dig() or not level.shop_interior[py, px] or \
                level.objects[py, px] not in PLAIN_FLOOR:
            return False
        if level.item_count[py, px] or len(level.items[py, px]):
            return False
        if (py, px) == (agent.blstats.y, agent.blstats.x):
            if agent.inventory.items_below_me:
                return False
        elif nh.glyph_is_object(int(agent.glyphs[py, px])) or agent.monster_tracker.monster_mask[py, px]:
            return False
        return not any(i.shop_status == Item.UNPAID for i in flatten_items(agent.inventory.items))

    def try_dig_down(self):
        """Dig down with a pick-axe (or zap a wand of digging down): one level per hole, and on
        Medusa's level it skips the water. Dig from plain floor with no water around (a hole beside
        water floods) and outside shops (the shopkeeper grabs the pack of a customer falling through)."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        if key in self.undiggable or level.dungeon_number not in MAIN_LINE:
            return False
        if self.in_valley():
            self.undiggable.add(key)   # hardfloor (gehennom.des): the pick-axe would only dig a pit
            return False
        if self._raven_level_below():
            return False   # RAVEN_CYCLE: down the stairs onto the raven island's '<', never a hole into the ravens
        if self.in_gehennom() and self.levitating():
            # "You can't reach the floor": every such try would count towards DIG_MAX_TRIES and could mark a
            # diggable level undiggable; a potion's levitation (the castle crossing) ends within ~150 turns
            self._task('wait for levitation to end')
            agent.search(5)
            return True
        if jf_config.HUNGER_DEEP and jf_config.HUNGER_HOLD and self._hunger_hold():
            # HUNGER_DEEP: Weak with no food and no hunger prayer due yet -- a voluntary hole now lands a fainting
            # hero among deeper monsters (cmp-main jf42 s6 dug Dlvl 7 -> 25 while Weak after an HP prayer had
            # reset the timeout and died fainted there). Hold on an Elbereth here until the prayer is due; the
            # escape digs (dig_first / DIG_ESCAPE) still run when something comes for us.
            self._task('hunger hold')
            if not self._rest_elbereth():
                agent.search(10)
            return True
        cycle = self._medusa_cycle_action()
        if cycle is not None:
            if cycle[0] == 'cycle_hold':
                return False   # MEDUSA_HOLE_CYCLE: hurt with a hostile close -- no plunge, no dig (fight2 fights)
            self._escape_act(cycle)   # MEDUSA_HOLE_CYCLE
            return True
        tool = self.digging_tool() if agent.blstats.time >= self._dig_blocked_until else None
        wand = self.digging_wand() if tool is None else None
        # WAND_RESERVE: on Medusa's level the kept wand holes the floor instead of the pick-axe (one flood roll and
        # one action instead of two and ~8); the mazes below her are zapped on arrival (plan_step)
        spend = WAND_RESERVE and DIG_WAND_ESCAPE and tool is not None and self.diving and self.on_medusa_level()
        if spend:
            wand = self._dig_wand()
        if WAND_WAITS and wand is not None and self.use_mines() and \
                level.dungeon_number == Level.DUNGEONS_OF_DOOM and agent.blstats.depth <= MINES_BRANCH_MAX_DEPTH:
            wand = None   # the Mines route (a pick-axe) first; the wand's 4-8 holes are the fallback
        if tool is None and wand is None:
            return False
        if jf_config.SHOP_DIG and tool is not None and self._shop_dig(tool):
            return True
        # a '>' a few steps away is as fast, and arriving on the up stairs keeps a way back -- but not from the
        # pit we dug: climbing out takes turns, the hole only 4 more (dsafe-A jf14 s5 thrashed between its pit
        # and a '>' 3 squares away under a centaur's crossbow bolts: 'You are still in a pit' x6)
        dis = agent.bfs()
        if not (DIG_ESCAPE and tool is not None and self._in_own_pit()):
            for d, _, _, kind in self.down_targets():
                if kind == 'stairs' and d <= DIG_STAIRS_RADIUS:
                    return False
        if jf_config.AT_THREAT_AVOID and tool is not None:
            holding, act = self._at_hold()
            if holding:
                # an @ comes for us (or was just in sight): no pit to be caught in -- fight2 meets it on level ground
                self._at_count(key)
                if act is not None:
                    self._escape_act(act)
                else:
                    self._task('AT_THREAT hold')
                    agent.search(1)
                return True
        y, x = agent.blstats.y, agent.blstats.x
        candidates = utils.isin(level.objects, PLAIN_FLOOR) | ((level.objects == -1) & level.walkable)
        floor = [p for p in zip(*candidates.nonzero()) if dis[p] >= 0]
        max_wet = 0
        if not any(self._diggable_spot(*p) for p in floor):
            # all reachable floor borders water: first a moat channel toward dry land (MEDUSA_HOP)
            hop = self._medusa_hop_plan() if MEDUSA_HOP and tool is not None else None
            if hop is not None:
                self._medusa_hop(hop)
                return True
            # ... else take the square with the fewest wet neighbours
            wet = [self._wet_neighbours(*p) for p in floor if self._diggable_spot(*p, max_wet=8)]
            up = self._medusa_reroll_stairs(min(wet) if wet else None) if tool is not None else None
            if up is not None:
                self._medusa_reroll(up)
                return True
            if not wet:
                return False
            max_wet = min(wet)
        blind_hold = self._blind_holding()
        if spend and wand is not None:
            # the best zap square by _zap_key (the flood / castle-pit bookkeeping lives in _escape_act)
            plan = self._zap_plan(wand)
            if plan is not None and plan[0] == 'step' and blind_hold:
                plan = ('zap', wand) if self._zap_key(y, x) is not None else (self._blind_hold() or plan)
            if plan is not None:
                if plan[0] == 'zap':
                    plan = self._freeze_action() or plan   # MEDUSA_FREEZE
                self._escape_act(plan)
                return True
        # DIG_ESCAPE: the staircase we just came down by shows us, not the stairs (see _dig_escape_action)
        on_stairs = DIG_ESCAPE and (y, x) in level.stair_destination
        if DROWN_GUARD and max_wet > 0 and not self._in_own_pit() and \
                self._crawl_exits(y, x, count_monsters=False) < DROWN_GUARD_EXITS:
            # (see DROWN_GUARD) a square a flooded digger can crawl out of, if there is one
            safe = [(dis[p], p) for p in floor if dis[p] > 0 and self._diggable_spot(*p, max_wet) and
                    self._crawl_exits(*p, count_monsters=False) >= DROWN_GUARD_EXITS]
            if safe:
                agent.go_to(*min(safe)[1])
                return True
        if (on_stairs or not self._diggable_spot(y, x, max_wet) or not self._dig_order_here_ok(max_wet)) and \
                not (DIG_ESCAPE and self._in_own_pit()):
            spots = [(dis[p], p) for p in floor if dis[p] > 0 and self._diggable_spot(*p, max_wet)]
            if not spots:
                return False
            hold = self._blind_hold() if blind_hold else None
            if hold is not None:
                self._escape_act(hold)   # MEDUSA_BLIND_HOLD: sight first, then the walk
                return True
            order = self._dig_order_best(max_wet)   # MEDUSA_DIG_ORDER
            agent.go_to(*(order if order is not None and dis[order] > 0 else min(spots)[1]))
            return True
        if tool is not None:
            rest_below = GEHENNOM_DIG_REST_BELOW if self.in_gehennom() else DIG_REST_BELOW
            if agent.blstats.hitpoints < rest_below * agent.blstats.max_hitpoints and \
                    not (DIVE_REST and self._in_own_pit()) and not self.xorn_buffer() and not self._mino_alert():
                self._task('rest before digging')
                if DIVE_REST and self._rest_elbereth():
                    return True
                agent.search(1 if agent.get_visible_monsters() else 20)
                return True
            freeze = self._freeze_action()
            if freeze is not None:
                self._escape_act(freeze)   # MEDUSA_FREEZE: ice on the moat neighbours before the pick's flood rolls
                return True
            self.dig_with_tool(tool)
            return True
        if self.rest_if_hurt():
            return True
        agent.log(f'DIVE zapping {wand.text!r} down')
        agent.zap(wand, '>')
        if agent.current_level().key() != key:
            self._dug_holes[key] = (y, x)
        if agent.current_level().key() == key and ('too hard to dig' in agent.message or
                                                    'here is too hard' in agent.message):
            self.undiggable.add(key)
        if jf_config.WAND_STAIRS_FIX and agent.current_level().key() == key and \
                'beam bounces off the' in agent.message:
            # zap.c zap_dig: on stairs or a ladder the beam hits the ceiling instead (a charge and a rock on
            # the head); whatever the map thought this square was, never zap here again
            agent.log(f'DIVE the digging beam bounced off stairs at {(y, x)}')
            self._bad_dig_spots.add((key, (y, x)))
        return True

    def _elbereth_before_digging(self):
        agent = self.agent
        if DIG_ESCAPE and agent.current_level().dungeon_number != GEHENNOM:
            return self._elbereth_before_digging_escape()
        if not (ELBERETH_DIG or SHELTERED_DIG) or agent.current_level().dungeon_number == GEHENNOM or \
                agent.character.prop.blind or agent.character.prop.polymorph:
            return False
        if (agent.inventory.engraving_below_me or '').lower() == 'elbereth' or not agent.can_engrave():
            return False
        bl = agent.blstats
        near = [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= ELBERETH_DIG_RADIUS]
        if SHELTERED_DIG:
            near = [m for m in near if not self._ignores_elbereth(m[3])]
            if not near and not ELBERETH_ALWAYS:
                return False
        elif not near or any(self._ignores_elbereth(m[3]) for m in near):
            return False
        spot = (agent.current_level().key(), bl.y, bl.x)
        tries = self.__dict__.setdefault('_elbereth_tries', {})
        if tries.get(spot, 0) >= 3:   # an engraving that won't take (CleverShovel: 4 per square)
            return False
        tries[spot] = tries.get(spot, 0) + 1
        agent.log(f'DIVE Elbereth before digging: {[m[3].mname for m in near]}')
        agent.engrave('Elbereth')
        return True

    def _elbereth_before_digging_escape(self):
        """DIG_ESCAPE's engraving before a dig step: against every monster whose melee Elbereth stops (breathers
        and casters included), always on Medusa's level, capped per square and dig phase (the pit erases it).
        Blind (a raven's claw), the engraving can't be read back: engrave once per phase (a blind dust
        Elbereth survives its letters only ~1 time in 3, engrave.c) and dig on."""
        agent = self.agent
        if agent.character.prop.polymorph or not agent.can_engrave() or utils.any_in(agent.glyphs, G.SWALLOW):
            return False
        if jf_config.DEEP_ITEMS and self._deep_on_scare():
            return False   # on our scroll of scare monster: nothing melees us, and the pit doesn't remove it
        blind = agent.character.prop.blind
        self._look_after_sight()   # MEDUSA_BLIND_DIG
        bl = agent.blstats
        tries = self.__dict__.setdefault('_elbereth_tries', {})
        # ELBERETH_REWRITE_FIX: the writes up to the last whole read-back don't count against the cap
        held = self.__dict__.setdefault('_elbereth_held', {})
        if not blind and (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
            if jf_config.ELBERETH_REWRITE_FIX:
                spot = (agent.current_level().key(), bl.y, bl.x, self._in_own_pit())
                held[spot] = tries.get(spot, 0)
            return False
        near = [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= ELBERETH_DIG_RADIUS and
                not self._melee_ignores_elbereth(m[3])]
        # Weak or Fainting: a faint leaves us helpless for turns in the middle of a dig (base-jf14 s8: a troll,
        # base-jf16 s13: a giant ant, base-jf16 s9: a panther -- all Elbereth-respecting -- killed fainted
        # diggers); the engraving holds them off while we're out
        spot = (agent.current_level().key(), bl.y, bl.x, self._in_own_pit())
        # the pit erased the Elbereth we needed before it: the monsters that made us engrave are still around
        # even if out of sight now (base-public s14: an invisible ogre king hit us in the fresh pit; s1: a
        # chameleon)
        rewrite = spot[3] and tries.get(spot[:3] + (False,), 0) > 0
        if not near and not (ELBERETH_ALWAYS or self.on_medusa_level() or bl.hunger_state >= Hunger.WEAK or
                             rewrite):
            return False
        if blind:
            # can't read it back: the only sign it didn't take (a blind dust Elbereth keeps all 8 letters ~1 time
            # in 3) is being hurt since we wrote it -- then write it again. Digging under constant attack makes
            # no progress at all: every hit stops the occupation before its next turn (mhitu stop_occupation),
            # which is how Medusa-3's ravens killed every harness digger (ds-med3-A: blinded in the pit,
            # 'It bites!', 7 applies without a single dig turn)
            last = self.__dict__.setdefault('_engrave_turn', {}).get(spot)
            if tries.get(spot, 0) >= ELBERETH_TRIES_BLIND or (last is not None and not self._hurt_since(last)):
                return False
        elif tries.get(spot, 0) - (held.get(spot, 0) if jf_config.ELBERETH_REWRITE_FIX else 0) >= ELBERETH_TRIES_ESCAPE:
            return False
        tries[spot] = tries.get(spot, 0) + 1
        self.__dict__.setdefault('_engrave_turn', {})[spot] = bl.time
        agent.log(f'DIVE Elbereth before digging{" (blind)" if blind else ""}'
                  f'{" in the pit" if spot[3] else ""}: {[m[3].mname for m in near]}')
        agent.engrave('Elbereth')
        return True

    def _boulder_blocks_dig(self, key, spot):
        """DIG_BOULDER_FIX: 'There isn't enough room to dig in here' since our last apply on this square (it arrives a
        step after the apply): a boulder lies on it -- give the square up (see DIG_BOULDER_FIX)."""
        mark = getattr(self, '_dig_mark', None)
        if mark is None or mark[:2] != (key, spot):
            return False
        agent = self.agent
        hist = agent._message_history[mark[2]:] + [agent.message]
        if not any("isn't enough room to dig" in m for m in hist):
            return False
        agent.log(f'DIVE a boulder blocks the dig at {spot}: digging elsewhere')
        self._bad_dig_spots.add((key, spot))
        self._pit_at = None
        self._dig_mark = None
        return True

    def _shop_dig(self, tool):
        """jf_config.SHOP_DIG: inside a shop with a digging tool (its shopkeeper holds the door against the pick) and no
        diggable square outside the shop within reach -> dig down from an empty shop square: shk.c shopdig(1) grabs the
        pack only when we owe something (billct or debit), and dokick.c impact_drop makes the goods on the hole's
        square fall with us (theft: the shopkeeper chases us), so nothing unpaid in the pack and nothing on the square.
        cand-g: jf46 s13 sat 9324 turns in Matagami's general store on Dlvl 14 and starved; jf50 s6 (4210 turns, Dlvl
        8) and jf60 s6 (3135 turns, Dlvl 9) died in theirs. True when it acted."""
        agent = self.agent
        level = agent.current_level()
        y, x = int(agent.blstats.y), int(agent.blstats.x)
        if not (level.shop[y, x] or level.shop_interior[y, x]):
            self._shop_since = None
            return False
        # only a stall: most shop landings walk out within 30-230 turns (60 of the 64 in cand-g), and digging at once
        # there only reshuffled those games (da-all1: 17 firings, 14 of them such visits, net -0.12)
        turn = agent.blstats.time
        if self._shop_since is None or self._shop_since[0] != level.key():
            self._shop_since = (level.key(), turn)
        if turn - self._shop_since[1] < jf_config.SHOP_DIG_AFTER:
            return False
        dis = agent.bfs()
        reach = dis >= 0
        outside = reach & ~level.shop & ~level.shop_interior
        if any(self._diggable_spot(int(py), int(px)) for py, px in zip(*outside.nonzero())):
            return False   # the plain search walks out to it
        unpaid = [it.text for it in flatten_items(agent.inventory.items) if it.shop_status == Item.UNPAID]
        if unpaid:
            if self._shop_dig_logged != (level.key(), 'unpaid'):
                self._shop_dig_logged = (level.key(), 'unpaid')
                agent.log(f'SHOP_DIG: not digging with unpaid items in the pack: {unpaid[:3]}')
            return False
        cands = []
        for py, px in zip(*(reach & level.shop_interior).nonzero()):
            py, px = int(py), int(px)
            if (py, px) in level.stair_destination or self._wet_neighbours(py, px) > 0 or \
                    (level.key(), (py, px)) in self._bad_dig_spots:
                continue
            if (py, px) == (y, x):
                empty = not agent.inventory.items_below_me and \
                    (level.objects[y, x] in ROOM_FLOOR or level.objects[y, x] == -1)
            else:
                empty = agent.glyphs[py, px] in ROOM_FLOOR   # in view showing bare floor: no object, no monster
            if empty:
                cands.append((int(dis[py, px]), (py, px)))
        if not cands:
            if self._shop_dig_logged != (level.key(), 'none'):
                self._shop_dig_logged = (level.key(), 'none')
                agent.log('SHOP_DIG: no empty shop square in reach')
            return False
        _, (ty, tx) = min(cands)
        if (ty, tx) != (y, x):
            self._task('shop dig: walk to an empty square')
            agent.log(f'SHOP_DIG: walking to the empty shop square {(ty, tx)}')
            agent.go_to(ty, tx)
            return True
        shield = agent.inventory.items.off_hand
        if tool.object == O.from_name('dwarvish mattock') and shield is not None:
            # dig_with_tool would drop it here: a shop buys what is dropped, and a bought item on the square falls
            # with us as stolen goods -- take it off and keep it
            agent.log(f'SHOP_DIG: taking off {shield.text!r} (a mattock needs both hands) and keeping it')
            agent.inventory.takeoff(shield)
            return True
        self._task('shop dig')
        agent.log(f'SHOP_DIG: digging down from the empty shop square {(y, x)} (nothing unpaid)')
        self.dig_with_tool(tool)
        return True

    def dig_with_tool(self, tool):
        agent = self.agent
        key = agent.current_level().key()
        spot = (agent.blstats.y, agent.blstats.x)
        if DIG_BOULDER_FIX and self._boulder_blocks_dig(key, spot):
            return
        if self._elbereth_before_digging():
            return
        shield = agent.inventory.items.off_hand
        if tool.object == O.from_name('dwarvish mattock') and shield is not None:
            # a mattock needs both hands; a digger falls through the floor anyway, so leave the shield
            agent.log(f'DIVE dropping {shield.text!r} to dig with a mattock')
            agent.inventory.takeoff(shield)
            shield = next((i for i in agent.inventory.items if i.is_armor() and not i.equipped and
                           i.text.split(' (')[0] == shield.text.split(' (')[0]), None)
            if shield is not None:
                agent.inventory.drop(shield)
            return
        for _ in range(3):
            tries = self._dig_tries.get(key, 0) + 1
            self._dig_tries[key] = tries
            agent.log(f'DIVE digging down with {tool.text!r} (try {tries})')
            if DIG_BOULDER_FIX:
                self._dig_mark = (key, spot, len(agent._message_history))
            with agent.atom_operation():
                tool = agent.inventory.move_to_inventory(tool)
                agent.step(A.Command.APPLY)
                agent.type_text(agent.inventory.items.get_letter(tool))
                prompted = 'In what direction do you want to dig?' in agent.single_message
                if prompted:
                    agent.direction('>')
                elif agent.single_message.startswith('In what direction'):
                    agent.step(A.Command.ESC)
            msg = agent.message
            if agent.current_level().key() != key:
                self._dug_holes[key] = spot   # (MEDUSA_HOLE_CYCLE) it stays open behind us
                return
            # Waking from a faint: the deafness that came with it ends right after, and its 'You can hear
            # again' stops the new dig before any progress (a starving rescue dive spent ~300 turns and
            # 15 tries on one level, close to DIG_MAX_TRIES marking it undiggable). Not the floor's
            # fault: don't count it, dig on at once.
            if prompted and 'You stop digging' in msg and \
                    any(s in msg for s in ('You can hear again', 'You regain consciousness', 'You faint')):
                self._dig_tries[key] = tries - 1
                if agent.blstats.hunger_state <= Hunger.FAINTING:   # not passed out again
                    if jf_config.ELBERETH_REWRITE_FIX and not agent.character.prop.blind and \
                            (agent.inventory.engraving_below_me or '').lower() != 'elbereth':
                        return   # the faint's wipes: the next call writes it again first (_elbereth_before_digging)
                    continue
            break
        if prompted and 'dig a pit in the' in msg:
            self._pit_at = (key, spot)
        if 'hole fills with' in msg and key == self.medusa_level:
            self._medusa_floods += 1
        if DIG_TRY_FIX and prompted:
            applies = self._dig_applies.get(key, 0) + 1
            self._dig_applies[key] = applies
            if 'You stop digging' in msg and 'dig a pit in the' not in msg:
                # a monster's attack or arrival cut the dig short: no evidence against the floor
                self._dig_tries[key] = tries = tries - 1
            if applies >= DIG_MAX_APPLIES:
                tries = DIG_MAX_TRIES
        if not prompted:
            # can't swap weapons (welded), stuck in a web, ...: try again later
            agent.log(f'DIVE could not dig: {msg!r}')
            self._dig_blocked_until = agent.blstats.time + 100
        elif "isn't enough room to dig" in msg or 'hole fills with' in msg or \
                ('too hard to' in msg and 'here is too hard to dig' not in msg):
            # a flooded hole (we crawled out elsewhere), a boulder, or stairs/altar/throne under objects
            self._bad_dig_spots.add((key, spot))
            self._dig_tries[key] = tries - 1
        elif 'here is too hard to dig' in msg or tries >= DIG_MAX_TRIES:
            agent.log(f'DIVE floor here cannot be dug through ({msg!r})')
            self.undiggable.add(key)
            if self.medusa_level == key:
                # a moat level that can't be dug is the castle (we fell past Medusa), not Medusa's level
                self.medusa_level = None
            if agent.blstats.depth >= 20:
                # the Castle (or another bottom level): what could still get us further?
                inv = '; '.join(f'{agent.inventory.items.get_letter(i)} - {i.text}'
                                for i in agent.inventory.items.all_items)
                agent.log(f'DIVE bottom reached at depth {agent.blstats.depth}; inventory: {inv}')
                try:
                    plan = '; '.join(f'{a} {i.text} ({why})' for a, i, why in power.passage_plan(agent))
                    agent.log(f'POWER passage plan at depth {agent.blstats.depth} AC {agent.blstats.armor_class}: '
                              f'{plan or "nothing"}')
                except Exception as e:   # diagnostics only
                    agent.log(f'POWER passage plan failed: {e!r}')
            if 'here is too hard to dig' in msg and agent.blstats.depth >= 25 and \
                    agent.current_level().dungeon_number == Level.DUNGEONS_OF_DOOM:
                self.castle.on_bottom(key)

    def descend(self):
        agent = self.agent
        if self.try_dig_down():
            return
        targets = self.down_targets()
        if not targets and utils.isin(agent.current_level().objects, G.STAIR_DOWN).any() and \
                self._walk_through_traps():
            targets = self.down_targets()
        if not targets:
            if self._stall_descend():
                return
            if self._read_magic_mapping():
                return
            self.exploration(None).until(agent, self._budgeted(lambda: bool(self.down_targets()))).run()
            return

        _, y, x, kind = targets[0]
        if kind == 'stairs':
            if (agent.blstats.y, agent.blstats.x) != (y, x):
                agent.go_to(y, x)
                return
            if self.rest_if_hurt():
                return
            agent.log(f'DIVE going down stairs at {(y, x)}')
            above = (agent.current_level().key(), (y, x))
            if ARRIVAL_FIX:
                self._arrival_pending = (above[0], above[1], agent.blstats.time)
            agent.move('>')
            self._arrived = (agent.current_level().key(), agent.blstats.time, above)
            return

        # trap door / hole: stand next to it, then step in
        if TRAPDOOR_PLUNGE and (agent.blstats.y, agent.blstats.x) == (y, x):
            # standing on it ('You escape a trap door.'): go_to(stop_one_before) asserted on our own square, 25
            # times at the end of a jf26 s10 dive and 10 in jf27 s0; '>' plunges in (do.c dodown: uescaped_shaft)
            if self.rest_if_hurt():
                return
            key = agent.current_level().key()
            agent.log(f'DIVE plunging into the trap door under us at {(y, x)}')
            with agent.atom_operation():
                agent.direction('>')
            if agent.current_level().key() == key:
                self._dead_traps.add((key, (y, x)))   # can't go down here (levitating...): don't retry it
            return
        if not utils.adjacent((agent.blstats.y, agent.blstats.x), (y, x)):
            agent.go_to(y, x, stop_one_before=True)
            return
        if self.rest_if_hurt():
            return
        self.step_onto(y, x, 'trap door')

    MAGIC_MAPPING_MIN_DEPTH = 3

    def _read_magic_mapping(self):
        """No '>' known: read a known scroll of magic mapping instead of exploring (a Tourist starts with
        four and never read one). From Dlvl 3, where exploring gets dangerous; once per level."""
        agent = self.agent
        level = agent.current_level()
        mapped = self.__dict__.setdefault('_magic_mapped', set())
        prop = agent.character.prop
        if level.key() in mapped or agent.blstats.depth < self.MAGIC_MAPPING_MIN_DEPTH or \
                prop.blind or prop.confusion or prop.hallu:
            return False
        scrolls = [i for i in flatten_items(agent.inventory.items)
                   if i.category == nh.SCROLL_CLASS and i.is_unambiguous() and i.object.name == 'magic mapping'
                   and i.status != Item.CURSED and i.shop_status == Item.NOT_SHOP]
        if not scrolls:
            return False
        mapped.add(level.key())
        letter = agent.inventory.items.get_letter(scrolls[0])
        agent.log(f'DIVE reading magic mapping ({letter}) at depth {agent.blstats.depth}')
        with agent.atom_operation():
            agent.step(A.Command.READ)
            agent.type_text(letter)
        agent.inventory.items.update(force=True)
        return True

    def step_onto(self, y, x, what):
        agent = self.agent
        key = agent.current_level().key()
        level = agent.current_level()
        y0, x0 = agent.blstats.y, agent.blstats.x
        if y != y0 and x != x0 and level.intact_doors[y0, x0]:
            # no diagonal step out of a doorway: sidestep to a square orthogonal to both first
            for my, mx in ((y0, x), (y, x0)):
                if level.walkable[my, mx] and not level.intact_doors[my, mx] and \
                        level.objects[my, mx] not in G.DOORS and level.objects[my, mx] not in FALL_TRAPS:
                    agent.move(my, mx)
                    return
        agent.log(f'DIVE stepping onto {what} at {(y, x)}')
        with agent.atom_operation():
            agent.direction(agent.calc_direction(y0, x0, y, x))
        if 'diagonally out of an intact doorway' in agent.message:
            # the portal sweep looped on this 51 times without the clock moving
            level.intact_doors[y0, x0] = True
            raise AgentPanic('diagonal step out of a doorway')
        if agent.current_level().key() == key and (agent.blstats.y, agent.blstats.x) == (y, x):
            # didn't fall (e.g. "You escape a trap door."): step off so the next try steps on again
            agent.log(f'DIVE {what} did not trigger')
            raise AgentPanic(f'{what} did not trigger')

    # ------------------------------------------------------------- gehennom

    def in_gehennom(self):
        """GEHENNOM_DIVE on and on a Gehennom level (dnum 1: the Valley of the Dead and below)."""
        return jf_config.GEHENNOM_DIVE and self.agent.current_level().dungeon_number == GEHENNOM

    def in_valley(self):
        level = self.agent.current_level()
        return jf_config.GEHENNOM_DIVE and level.dungeon_number == GEHENNOM and level.level_number == 1

    def levitating(self):
        return bool(int(self.agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) &
                    getattr(nh, 'BL_MASK_LEV', 0))

    def _mino_alert(self):
        """MINO_GUARD (mino_guard.py): a minotaur was in view on this level lately. It hunts us (monmove.c
        set_apparxy knows where we are, and it ignores Elbereth): dig or descend, never rest here."""
        guard = getattr(self, 'mino_guard', None)
        return guard is not None and guard.alert()

    def _gehennom_digger(self):
        """Below the Valley with a way to dig down: the dig-dive's own rest rule (GEHENNOM_DIG_REST_BELOW)."""
        return self.in_gehennom() and not self.in_valley() and \
            self.agent.current_level().key() not in self.undiggable and \
            (self.digging_tool() is not None or self.digging_wand() is not None)

    def _valley_know_map(self, level):
        """The Valley is dark and fixed: unseen squares that are floor in some variant of the level file count
        as walkable, so the BFS plans the whole walk; whatever we actually see overrides it (update_level)."""
        floor = getattr(level, 'valley_floor', None)
        if floor is None:
            floor = level.valley_floor = valley.floor_mask(level.walkable.shape)
        unseen = floor & ~level.seen
        if (unseen & ~level.walkable).any():
            level.walkable[unseen] = True
            self.agent.last_bfs_step = -1   # the BFS cache doesn't know about the change

    def _valley_next_door(self, level):
        """The first door along the route that isn't open yet: the doors are in a chain (each one's far side is
        only reachable through the one before), so it is the one to work on. None: all three are open."""
        return next(((door, stands) for door, stands in valley.DOORS if not level.walkable[door]), None)

    def _valley_route_open(self):
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        pos = (agent.blstats.y, agent.blstats.x)
        if dis[valley.DOWN_STAIRS] != -1 or pos == valley.DOWN_STAIRS:
            return True
        nxt = self._valley_next_door(level)
        return nxt is not None and any(s == pos or dis[s] != -1 for s in nxt[1])

    def valley_step(self):
        """The Valley of the Dead: hardfloor, so walk to its '>' in the far north-west (valley.py). The way
        crosses Moloch's temple (peaceful priest) and three locked secret doors, which a pick-axe breaks
        through in ~3 turns each (dig.c may_dig() ignores NON_DIGGABLE for SDOOR); without one, search and
        kick. Retreats up the Valley's '<' are off (they lead to the castle's east edge)."""
        agent = self.agent
        level = agent.current_level()
        bl = agent.blstats
        pos = (bl.y, bl.x)
        self.undiggable.add(level.key())
        self._valley_know_map(level)
        if self._valley_arrival is None:
            self._valley_arrival = bl.time
            inv = '; '.join(f'{agent.inventory.items.get_letter(i)} - {i.text}' for i in agent.inventory.items.all_items)
            agent.log(f'VALLEY arrived at {pos} turn {bl.time} xl {bl.experience_level} hp {bl.hitpoints}/'
                      f'{bl.max_hitpoints} ac {bl.armor_class} hunger {bl.hunger_state}; inventory: {inv}')
            # VALLEY_SPRINT: the dig-dive lands with the pick-axe in hand; the first blow otherwise costs a turn
            # of wielding while the landing crowd hits (vr-base: 'What do you want to wield?' opened most fights)
            if jf_config.VALLEY_SPRINT and agent.wield_best_melee_weapon():
                return
        prop = agent.character.prop
        if prop.stun or prop.confusion:
            # a stunned or confused dig swings in a random direction (dig.c confdir) and steps stagger
            self._task('valley: wait out stun or confusion')
            agent.search(3)
            return
        down = valley.DOWN_STAIRS
        if level.seen[down] and level.objects[down] != -1 and level.objects[down] not in G.STAIR_DOWN and \
                pos != down:
            agent.log(f'VALLEY no down stairs at {down} (glyph {int(level.objects[down])}): searching the usual way')
            self._valley_misplaced = True
            return
        dis = agent.bfs()
        if pos == down or dis[down] != -1:
            return self._valley_descend(down)
        # hurt: rest, on the '<' if it is close (valley_retreat climbs from there as soon as a fight goes
        # badly), until VALLEY_REST_UNTIL -- the walk west leaves that lifeline behind
        threshold = VALLEY_REST_UNTIL if self._valley_resting else \
            (VALLEY_SPRINT_REST_BELOW if jf_config.VALLEY_SPRINT else REST_BELOW)
        self._valley_resting = bl.hitpoints < threshold * bl.max_hitpoints and bl.hunger_state < Hunger.WEAK
        if self._valley_resting:
            up = valley.UP_STAIRS
            if pos != up and 0 <= dis[up] <= VALLEY_REST_REACH:
                self._task('valley: to the up stairs to rest')
                agent.go_to(*up)
                return
            self._task('valley: rest')
            agent.search(1 if agent.get_visible_monsters() else 20)
            return
        if VALLEY_CAMP and not self._valley_camp_done:
            up = valley.UP_STAIRS
            if self._valley_camp_start is None:
                self._valley_camp_start = bl.time
            quiet = bl.time - max(self._hostile_seen_turn, self._valley_camp_start)
            if (quiet >= VALLEY_CAMP_QUIET and bl.hitpoints >= VALLEY_REST_UNTIL * bl.max_hitpoints) or \
                    bl.time - self._valley_camp_start >= VALLEY_CAMP_MAX or dis[up] == -1 and pos != up:
                self._valley_camp_done = True
                agent.log(f'VALLEY camp over after {bl.time - self._valley_camp_start} turns (quiet {quiet})')
            elif pos != up:
                self._task('valley: camp on the up stairs')
                agent.go_to(*up)
                return
            else:
                self._task('valley: camp on the up stairs')
                agent.search(1 if agent.get_visible_monsters() else 10)
                return
        nxt = self._valley_next_door(level)
        if nxt is not None:
            door, stands = nxt
            here = pos in stands
            free = [s for s in stands if s != pos and dis[s] != -1]
            # the priest of Moloch wanders his temple and may stand on a square we want
            taken = [s for s in stands if s != pos and agent.monster_tracker.monster_mask[s] and
                     self._neighbour_distance(dis, *s) is not None]
            if (here or free or taken) and door == valley.DOORS[0][0] and prop.hallu and not self._in_temple(pos):
                # hallucinating, the peaceful priest of Moloch looks like any monster (farlook drops
                # 'peaceful'), and fight2 hits back at whatever is next to us once hurt: stay out of his temple
                self._task('valley: wait out hallucination before the temple')
                agent.search(5)
                return
            if here:
                return self._valley_open_door(door)
            if free:
                self._task('valley: to a secret door')
                agent.go_to(*min(free, key=lambda s: dis[s]))
                return
            if taken:
                self._task('valley: wait for the door square to clear')
                s = taken[0]
                if not utils.adjacent(pos, s):
                    agent.go_to(*s, stop_one_before=True)
                    return
                agent.search(2)
                return
        # the next door (or the '>') isn't reachable: a known trap (the fixed spiked pits) may be what walls us
        # off, else walls of the level's variant we haven't seen yet
        if self._walk_through_traps() and self._valley_route_open():
            return
        self._task('valley: explore')
        self.exploration(None).until(agent, self._valley_route_open).run()

    @staticmethod
    def _in_temple(pos):
        (y1, x1), (y2, x2) = valley.TEMPLE
        return y1 <= pos[0] <= y2 and x1 <= pos[1] <= x2

    def _valley_target(self, level, dis):
        """Where valley_step is walking to: the '>' once reachable, else the nearest free square to work the
        next door from. None when standing there already or nothing is reachable."""
        pos = (self.agent.blstats.y, self.agent.blstats.x)
        if dis[valley.DOWN_STAIRS] != -1:
            return valley.DOWN_STAIRS if pos != valley.DOWN_STAIRS else None
        up = valley.UP_STAIRS
        if (self._valley_resting and 0 <= dis[up] <= VALLEY_REST_REACH) or \
                (VALLEY_CAMP and not self._valley_camp_done and dis[up] != -1):
            return up if pos != up else None
        nxt = self._valley_next_door(level)
        if nxt is None or pos in nxt[1]:
            return None
        free = [s for s in nxt[1] if dis[s] != -1]
        return min(free, key=lambda s: dis[s]) if free else None

    # monsters never to melee on the way (paralysis, passive acid/stun/fire, explosions): fight2 decides
    VALLEY_NO_MELEE = frozenset(('floating eye', 'gelatinous cube', 'acid blob', 'yellow mold', 'green mold',
                                 'brown mold', 'red mold', 'lichen', 'spotted jelly', 'blue jelly', 'ochre jelly',
                                 'cockatrice', 'chickatrice', 'gas spore', 'unknown'))

    @Strategy.wrap
    def valley_sneak(self):
        """Preempts fight2 in the Valley: its three graveyards hold a sleeping undead on every square (mkroom.c
        fill_zoo, MM_ASLEEP) and a Valkyrie's intrinsic stealth keeps them asleep as we pass (monmove.c
        disturb()); fight2 would take on each one that comes into view. While nothing hurts us, only attack
        what stands on the next square of the way and walk past the rest (graveyard monsters and zombies
        are slow; whatever does bite brings fight2 back)."""
        agent = self.agent
        if not VALLEY_SNEAK or not self.in_valley() or self._valley_misplaced:
            yield False
        if jf_config.VALLEY_SPRINT:
            yield from self._valley_sprint()
            return
        bl = agent.blstats
        prop = agent.character.prop
        if prop.hallu or prop.stun or prop.confusion or prop.blind or \
                (bl.hitpoints < 0.5 * bl.max_hitpoints and not VALLEY_GRAVE_FILTER):
            yield False
        monsters = agent.get_visible_monsters()
        if not monsters:
            yield False   # the plan walks on by itself
        if VALLEY_GRAVE_FILTER:
            # graveyard sleepers never count as a fight: awake monsters next to us go to fight2
            awake = [m for m in monsters if not self._valley_sleeper(m)]
            if any(utils.adjacent((m[1], m[2]), (bl.y, bl.x)) for m in awake):
                yield False
            hurt = bool(awake) and agent._hurt_recently(3)
        else:
            if agent._hurt_recently(3):
                yield False
            hurt = False
        level = agent.current_level()
        dis = agent.bfs()
        target = self._valley_target(level, dis)
        if target is None:
            yield False
        path = agent.path(bl.y, bl.x, *target, dis=dis)
        if len(path) < 2:
            yield False
        ny, nx = path[1]
        blocker = next((m for m in monsters if (m[1], m[2]) == (ny, nx)), None)
        if blocker is not None and getattr(blocker[3], 'mname', '') in self.VALLEY_NO_MELEE:
            yield False
        # (a sleeper on the next square is attacked even while something else hurts us from afar, or at low HP on
        # the way to rest: fight2 leaves sleepers alone, and the plan's move into it would only panic)
        if blocker is None and (hurt or bl.hitpoints < 0.5 * bl.max_hitpoints):
            yield False   # something awake hurts us (fight2), or too weak to walk on (the plan rests)
        if blocker is None and agent.monster_tracker.monster_mask[ny, nx]:
            yield False   # a peaceful or unlisted monster there: fight2 / the plan sort it out
        yield True
        if blocker is None:
            agent.move(ny, nx)
            return
        if agent.wield_best_melee_weapon():
            return
        agent.log(f'VALLEY attacking the {blocker[3].mname} on the way at {(ny, nx)}')
        agent.melee_attack(ny, nx)

    # VALLEY_SPRINT router: the extra cost of stepping into a square held by a visible monster (a graveyard sleeper
    # or a resident in the way). vr-sprint1 (plain shortest path) died cutting through them: a giant mummy on the
    # way at the landing (73 -> 28 HP in 3 turns), a column of zombies and a ghoul (paralysis) in the graveyards
    _SPRINT_DANGER = 30
    _SPRINT_DANGEROUS = frozenset(('giant mummy', 'ettin mummy', 'human mummy', 'elf mummy', 'ghoul',
                                   'giant zombie', 'ettin zombie', 'unknown'))

    def _sprint_block_cost(self, mon):
        mlet = getattr(mon, 'mlet', '')
        cls = ord(mlet) if isinstance(mlet, str) and len(mlet) == 1 else -1
        name = getattr(mon, 'mname', '')
        if cls == MON.S_BAT:
            return 2          # fought anyway (VALLEY_SPRINT_OUTRUN)
        if name in self._SPRINT_DANGEROUS or cls in (MON.S_VAMPIRE, MON.S_DEMON, MON.S_LICH) or \
                getattr(mon, 'mlevel', 0) >= 8:
            return self._SPRINT_DANGER
        if cls == MON.S_GHOST:
            return 8          # harmless but AC -5 and ~45 HP: slow to cut through
        return 4

    def _valley_route(self, target, monsters, dis):
        """Dijkstra from target over the squares reachable now (agent.bfs): cost[u] to step into u -- 1, plus
        VALLEY_SPRINT_GRAVE_COST on graveyard squares not yet seen empty (each holds a sleeper until killed), plus
        _sprint_block_cost for a visible monster. Returns (dist, cost): dist[s] = cost of the squares after s."""
        import heapq
        agent = self.agent
        reach = dis >= 0
        pos = (int(agent.blstats.y), int(agent.blstats.x))
        reach[pos] = True
        h, w = reach.shape
        cost = np.ones((h, w))
        for sq in valley.GRAVEYARD:
            if sq not in self._valley_grave_clear:
                cost[sq] += VALLEY_SPRINT_GRAVE_COST
        for m in monsters:
            cost[int(m[1]), int(m[2])] += self._sprint_block_cost(m[3])
        squeeze = agent.inventory.items.total_weight <= 600
        dist = np.full((h, w), np.inf)
        dist[target] = 0
        heap = [(0.0, target)]
        while heap:
            d, (y, x) = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            nd = d + cost[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < h and 0 <= nx < w and reach[ny, nx] and nd < dist[ny, nx] and \
                            (not (dy and dx) or squeeze or reach[ny, x] or reach[y, nx]):
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, (ny, nx)))
        return dist, cost

    def _valley_route_step(self, target, monsters, dis):
        """The first square of the cheapest way to target (None: no way), and its entry cost."""
        dist, cost = self._valley_route(target, monsters, dis)
        agent = self.agent
        pos = (int(agent.blstats.y), int(agent.blstats.x))
        reach = dis >= 0
        squeeze = agent.inventory.items.total_weight <= 600
        best, best_v = None, np.inf
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = pos[0] + dy, pos[1] + dx
                if not (dy or dx) or not (0 <= ny < reach.shape[0] and 0 <= nx < reach.shape[1]) or \
                        not reach[ny, nx]:
                    continue
                if dy and dx and not (squeeze or reach[ny, pos[1]] or reach[pos[0], nx]):
                    continue
                v = cost[ny, nx] + dist[ny, nx]
                if v < best_v:
                    best, best_v = (ny, nx), v
        return best, (cost[best] if best is not None else np.inf)

    def valley_note_graves(self):
        """Every Valley step: graveyard squares next to us seen without a monster are empty for good (the fill's
        sleepers don't move until woken)."""
        agent = self.agent
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        mask = agent.monster_tracker.monster_mask
        for y in range(y0 - 1, y0 + 2):
            for x in range(x0 - 1, x0 + 2):
                if (y, x) in valley.GRAVEYARD:
                    if mask[y, x]:
                        self._valley_grave_clear.discard((y, x))
                    else:
                        self._valley_grave_clear.add((y, x))

    def _valley_sprint(self):
        """VALLEY_SPRINT (valley_sneak's body when on) drives every Valley move toward the next door square or the
        '>': a Valkyrie is Fast (speed ~16) and nearly every Valley monster is slower (zombies 6-8, mummies 10-14,
        vampires 12-14, ghosts 3), and one that must move to close in can't hit in the same move (monmove.c
        dochug: m_move() == 1 returns before mattacku), so walking on takes few blows while standing to fight a
        crowd took 10-30 HP a turn (vr-base: 61 real castle kits, 0 out, 37 dead within 100 turns). The way is the
        cheapest by _valley_route (graveyard sleepers and dangerous monsters cost extra, so it goes round them);
        attack only what can't be outrun when it is next to us (mmove > VALLEY_SPRINT_OUTRUN: bats) and what holds
        the next square of that way. When the cheapest way leads through a dangerous monster and the '<' is close,
        go up (mummies and bats never follow, monst.c M2_STALK) and come back through the castle trap door to a new
        landing square, at most VALLEY_SPRINT_REROLLS times. At a door square with hostiles adjacent fight2 clears
        them (digging stops for any adjacent hostile); with none, dig. On the '>' go down at once."""
        agent = self.agent
        bl = agent.blstats
        prop = agent.character.prop
        if prop.hallu or prop.stun or prop.confusion or prop.blind:
            yield False
        monsters = agent.get_visible_monsters()
        if not monsters and (self._valley_resting or bl.hitpoints < VALLEY_SPRINT_REST_BELOW * bl.max_hitpoints):
            yield False   # valley_step rests
        level = agent.current_level()
        pos = (int(bl.y), int(bl.x))
        dis = agent.bfs()
        down, up = valley.DOWN_STAIRS, valley.UP_STAIRS
        near = [m for m in monsters if utils.adjacent((m[1], m[2]), pos)]
        if pos == down:
            if not monsters or self.levitating():
                yield False   # valley_step: rest first / wait out levitation
            yield True
            agent.log(f'VALLEY sprint: down the stairs at hp {bl.hitpoints}/{bl.max_hitpoints} with '
                      f'{[m[3].mname for m in monsters]} in view')
            self._valley_descend(down)
            return
        if self._valley_resting and 0 <= dis[up] <= VALLEY_REST_REACH:
            if pos == up:
                yield False   # on the '<': valley_retreat / valley_step decide
            target = up
        elif dis[down] != -1:
            target = down
        else:
            nxt = self._valley_next_door(level)
            if nxt is None:
                yield False
            door, stands = nxt
            if pos in stands:
                if not monsters or near or door in self._valley_undiggable_doors or \
                        bl.time < self._dig_blocked_until:
                    yield False   # valley_step digs / fight2 clears the neighbours / search and kick
                yield True
                self._task('valley: sprint: dig through a secret door')
                self._valley_open_door(door)
                return
            free = [s for s in stands if dis[s] != -1]
            if not free:
                yield False   # valley_step explores
            target = min(free, key=lambda s: dis[s])
        if target is None or dis[target] == -1:
            yield False
        fast = [m for m in near if getattr(m[3], 'mmove', 12) > VALLEY_SPRINT_OUTRUN and
                getattr(m[3], 'mname', '') not in self.VALLEY_NO_MELEE]
        at = {(int(m[1]), int(m[2])): m for m in monsters}
        step, step_cost = (None, np.inf) if fast else self._valley_route_step(target, monsters, dis)
        if not fast and step is None:
            yield False
        blocker = at.get(step) if step is not None else None
        if blocker is not None and step_cost >= self._SPRINT_DANGER and \
                self._valley_rerolls < VALLEY_SPRINT_REROLLS and 0 <= dis[up] <= VALLEY_RETREAT_REACH:
            # the cheapest way on is through a dangerous monster: up the '<' (it stays unless it's a stalker) and
            # back in through the trap door to another landing square
            ustep = None
            if pos != up:
                ustep, ucost = self._valley_route_step(up, monsters, dis)
                if ustep is None or ustep in at:
                    ustep = False
            if ustep is not False:
                yield True
                if pos == up:
                    self._valley_rerolls += 1
                    self.valley_retreats += 1
                    self.valley_retreat_turn = bl.time
                    agent.log(f'VALLEY sprint: reroll {self._valley_rerolls}: the way on is through the '
                              f'{blocker[3].mname}; up to the castle at hp {bl.hitpoints}/{bl.max_hitpoints}, near '
                              f'{[m[3].mname for m in near]}')
                    agent.move('<')
                else:
                    self._task('valley: sprint: to the up stairs (reroll)')
                    agent.move(*ustep)
                return
        if blocker is not None and getattr(blocker[3], 'mname', '') in self.VALLEY_NO_MELEE:
            yield False
        if blocker is None and not fast and agent.monster_tracker.monster_mask[step]:
            yield False   # a peaceful or unlisted monster there: fight2 / the plan sort it out
        yield True
        victim = fast[0] if fast else blocker
        if victim is None:
            self._task('valley: sprint')
            agent.move(*step)
            return
        if agent.wield_best_melee_weapon():
            return
        agent.log(f'VALLEY sprint: attacking the {victim[3].mname} at {(int(victim[1]), int(victim[2]))} '
                  f'({"fast" if fast else "on the way"}) hp {bl.hitpoints}/{bl.max_hitpoints}')
        agent.melee_attack(int(victim[1]), int(victim[2]))

    @Strategy.wrap
    def valley_retreat(self):
        """Up the Valley's '<' when a fight goes badly (VALLEY_RETREAT): the castle's east edge has Elbereth and
        prayer, and whatever was adjacent follows us up to fight there; the castle mode rests and drops back in
        through the trap door behind the castle's back door, to a new random landing spot."""
        agent = self.agent
        if not VALLEY_RETREAT or not self.in_valley():
            yield False
        bl = agent.blstats
        near = self._near_hostiles(radius=4)
        if not near:
            yield False
        low = bl.hitpoints < VALLEY_RETREAT_BELOW * bl.max_hitpoints
        crowded = bl.hitpoints < VALLEY_CROWD_BELOW * bl.max_hitpoints and \
            len(self._near_hostiles(radius=CROWD_RADIUS)) >= CROWD_SIZE
        if not (low or crowded) or bl.time < self._retreat_blocked_until:
            yield False
        up = valley.UP_STAIRS
        pos = (bl.y, bl.x)
        dis = agent.bfs()
        if pos != up and not 0 <= dis[up] <= VALLEY_RETREAT_REACH:
            yield False
        yield True
        if pos != up:
            self._task('valley: retreat to the up stairs')
            start = (pos, agent.current_level().key())
            try:
                agent.go_to(*up, max_steps=1)
            finally:
                if ((agent.blstats.y, agent.blstats.x), agent.current_level().key()) == start:
                    self._retreat_blocked_until = agent.blstats.time + 5   # blocked: fight a few turns
            return
        self.valley_retreats += 1
        self.valley_retreat_turn = bl.time
        agent.log(f'VALLEY retreat {self.valley_retreats} up to the castle at hp {bl.hitpoints}/{bl.max_hitpoints}, '
                  f'turn {bl.time}, near {[m[3].mname for m in near]}')
        agent.move('<')

    # --------------------------------------------------------- valley fort (jf_config.VALLEY_FORT)

    # mhitu.c hitmsg/missmu/wildmiss: a melee attack on us reads 'The X hits!', '... bites!', '... touches you!',
    # 'The X misses!', 'The X just misses!', 'The X swings wildly and misses!' -- none of them from a scared monster
    # (monster-vs-monster lines, mhitm.c, name the victim: 'The X hits the Y.')
    _FORT_MELEE = re.compile(r"(?:\b[Tt]he [a-z][a-z' -]*?|\bIt|[A-Z][a-z]+'s ghost) (?:hits!|bites!|kicks!|stings!|"
                             r"butts!|touches you!|misses!|just misses!|swings wildly|engulfs you)")

    def _fort_on_pile(self):
        """On the Valley '<' where our scroll pile lies, and nothing has meleed us there."""
        agent = self.agent
        return self._fort_pile == 'scare' and self.in_valley() and \
            (agent.blstats.y, agent.blstats.x) == valley.UP_STAIRS

    def _fort_watch(self):
        """Every Valley step (update hook): a melee attack on us while we stand on the pile means none of the
        dropped scrolls is scare monster (the gamble failed): the fort is over, the usual Valley play takes it."""
        agent = self.agent
        if not self._fort_on_pile():
            return
        m = self._FORT_MELEE.search(agent.message or '')
        if m is None:
            return
        near = [mon for mon in agent.get_visible_monsters()
                if utils.adjacent((mon[1], mon[2]), (agent.blstats.y, agent.blstats.x))]
        agent.log(f'VALLEY fort: meleed on the pile ({m.group(0)!r}; adjacent {[x[3].mname for x in near]}): '
                  f'no scare monster in it')
        self._fort_pile = 'dead'
        self._fort_done = True

    @Strategy.wrap
    @_hold_loop
    def valley_fort(self):
        """VALLEY_FORT: every Valley resident heads for us (monmove.c set_apparxy), and a scroll of scare monster
        under us makes each one that comes next to us flee without attacking (dochug: distfleeck sets 'scared',
        mattacku needs !scared; onscary has no Gehennom exception for the scroll). So: walk to the '<' on landing,
        drop the scroll(s) there, and strike from that square whatever stands next to us -- a cornered scared
        monster stays and takes it; our blows don't spoil the scroll (unlike Elbereth). The '<' under us is the
        way out when something hurts us from range. Leave for the walk once the Valley has been quiet for
        VALLEY_FORT_QUIET turns and HP is back; within VALLEY_FORT_REACH steps of the pile, go back to it as soon
        as the walk meets something (_fort_pull_reason): it follows and is struck down from the pile (the pull);
        beyond that reach walk on (_fort_walk_action)."""
        agent = self.agent
        if not jf_config.VALLEY_FORT or not self.in_valley() or self._valley_misplaced:
            yield False
        bl = agent.blstats
        turn = bl.time
        pos = (bl.y, bl.x)
        up = valley.UP_STAIRS
        if self._fort_start is None:
            self._fort_start = turn
            self._fort_hostile_turn = turn
        if self._fort_dropped and self._fort_pile != 'scare':
            yield False   # no pile, or it failed: the usual Valley play
        dis = agent.bfs()
        monsters = agent.get_visible_monsters()
        if self._fort_done:
            # the walk: back to the pile when it meets something (and the pile is in reach), else on
            in_reach = pos == up or 0 <= dis[up] <= VALLEY_FORT_REACH
            why = self._fort_pull_reason(monsters, pos) \
                if in_reach and self._fort_returns < VALLEY_FORT_MAX_RETURNS else None
            if why is None:
                act = self._fort_walk_action(dis, monsters) if VALLEY_FORT_WALK else None
                if act is None:
                    yield False
                yield True
                act()
                return
            self._fort_done = False
            self._fort_hostile_turn = turn
            self._fort_returns += 1
            agent.log(f'VALLEY fort: back to the pile at hp {bl.hitpoints}/{bl.max_hitpoints} ({dis[up]} steps, '
                      f'return {self._fort_returns}: {why})')
        if monsters:
            self._fort_hostile_turn = turn
        prop = agent.character.prop
        adjacent = [m for m in monsters if utils.adjacent((m[1], m[2]), pos)]
        if self._fort_meal is not None and pos != up:
            meal = self._fort_meal_here(pos)
            if meal is not None and not adjacent:
                yield True
                self._task('valley fort: eat a corpse beside the pile')
                agent.log(f'VALLEY fort: eating {meal.text!r} beside the pile at hp {bl.hitpoints}/{bl.max_hitpoints} '
                          f'hunger {bl.hunger_state}')
                self._fort_meal = None
                agent.inventory.eat(meal)
                return
            self._fort_meal = None   # gone, or something came: back on the pile
        if pos != up:
            if dis[up] == -1 or prop.stun or prop.confusion:
                yield False   # (the usual play; a stunned step staggers anywhere)
            yield True
            path = agent.path(bl.y, bl.x, *up, dis=dis)
            ny, nx = path[1]
            # a bat (speed 20) can't be outrun on a long way back: strike it when it is next to us
            fast = [m for m in adjacent if getattr(m[3], 'mmove', 12) > VALLEY_SPRINT_OUTRUN and
                    getattr(m[3], 'mname', '') not in self.VALLEY_NO_MELEE]
            if agent.monster_tracker.monster_mask[ny, nx] or (fast and dis[up] > 3):
                if agent.monster_tracker.peaceful_monster_mask[ny, nx]:
                    self._task('valley fort: wait for the way to the up stairs')
                    agent.search(1)
                    return
                self._task('valley fort: clear the way to the up stairs')
                if agent.wield_best_melee_weapon():
                    return
                ty, tx = (ny, nx) if agent.monster_tracker.monster_mask[ny, nx] else (int(fast[0][1]), int(fast[0][2]))
                agent.melee_attack(ty, tx)
                return
            self._task('valley fort: to the up stairs')
            agent.move(ny, nx)
            return
        if not self._fort_dropped:
            self._fort_dropped = True
            known, candidates = power.scare_scrolls(agent)
            drop = known[:1]
            if not drop and VALLEY_FORT_GAMBLE:
                drop = [it for it, p in candidates if p > 0 and agent.inventory._scroll_key(it) not in self._not_scare_keys]
            if not drop:
                agent.log('VALLEY fort: no scroll that may be scare monster')
                self._fort_pile = None
                yield False
            yield True
            self._fort_pile = 'scare'
            agent.log(f'VALLEY fort: dropping {[it.text for it in drop]} on the up stairs at turn {turn}, hp '
                      f'{bl.hitpoints}/{bl.max_hitpoints}, adjacent {[m[3].mname for m in adjacent]}')
            # one of each stack; never picked up again (a second pickup turns scare monster to dust, pickup.c)
            agent.inventory.drop(drop, [1] * len(drop))
            agent.inventory._note_dropped(drop, [1] * len(drop), force=True)
            return
        food = agent.edible_carried_food() if bl.hunger_state >= Hunger.HUNGRY and not adjacent else []
        # Weak with nothing to eat: no prayer in Gehennom, but the castle side is one step up (pray.c: Weak is major
        # trouble), and the pile waits here (vx-fort1: jf14-s8 and jf25-s6 fainted on the pile after ~500 turns) --
        # only when that prayer would be safe: else the castle mode drops us straight back, and v4 climbed and fell
        # five times in 50 turns (vxs-strong-fort seed 6)
        starving = self._fort_starving()
        if bl.hitpoints < VALLEY_FORT_CLIMB_BELOW * bl.max_hitpoints or starving:
            # only missiles, rays, breath or spells get through the scroll: up to the castle side to heal (the
            # castle mode rests and drops back through the trap door; the pile waits on this '<')
            yield True
            self.valley_retreats += 1
            self.valley_retreat_turn = turn
            agent.log(f'VALLEY fort: climbing at hp {bl.hitpoints}/{bl.max_hitpoints} hunger {bl.hunger_state} '
                      f'(retreat {self.valley_retreats}), turn {turn}, adjacent {[m[3].mname for m in adjacent]}')
            agent.move('<')
            return
        if food:
            yield True
            self._task('valley fort: eat')
            agent.inventory.eat(food[0])
            return
        meal = self._fort_meal_square(monsters)
        if meal is not None:
            yield True
            self._fort_meal = meal
            self._task('valley fort: step off to a corpse')
            agent.log(f'VALLEY fort: stepping off to eat a {MON.permonst(meal[2]).mname} corpse at {meal[:2]}, hunger '
                      f'{bl.hunger_state}')
            agent.move(meal[0], meal[1])
            return
        quiet = turn - self._fort_hostile_turn
        # the first hold lasts VALLEY_FORT_MIN turns at least: the residents stuck behind walls arrive late, and
        # every kill is experience (vx-fort1 jf14-s4 held 626 turns: XL 8 -> 11); jf14-s1 and jf16-s1 left after
        # 40-78 quiet turns and met them on the way
        if (quiet >= VALLEY_FORT_QUIET and bl.hitpoints >= VALLEY_FORT_LEAVE * bl.max_hitpoints and
                (self._fort_returns or turn - self._fort_start >= VALLEY_FORT_MIN)) or \
                (not self._fort_returns and turn - self._fort_start >= VALLEY_FORT_MAX and
                 bl.hitpoints >= 0.75 * bl.max_hitpoints and not adjacent):
            self._fort_done = True
            agent.log(f'VALLEY fort: leaving for the walk at turn {turn} ({turn - self._fort_start} turns, quiet '
                      f'{quiet}), hp {bl.hitpoints}/{bl.max_hitpoints} xl {bl.experience_level}')
            yield False
        yield True
        if agent.wield_best_melee_weapon():
            return
        target = next((m for m in adjacent if getattr(m[3], 'mname', '') not in self.VALLEY_NO_MELEE), None)
        if target is not None:
            self._task('valley fort: strike from the pile')
            agent.melee_attack(target[1], target[2])
            return
        self._task('valley fort: hold')
        agent.search(1 if monsters else 5)

    WRAITH_ID = None

    def _fort_meal_square(self, monsters):
        """(y, x, monster id) of a corpse next to the pile worth a step off it now, or None: nothing hostile within 2,
        and a wraith corpse (eat.c cpostfx: PM_WRAITH -> pluslvl(), +1 XL; its corpse is ours 1 kill in 6 in the
        Valley: 50% corpse_chance x the graveyard level's 1/3) or, Hungry, any fresh edible one (agent's rules:
        no poisonous/acidic/were/cannibal corpses, none older than CORPSE_MAX_AGE) -- a meal off the floor instead
        of a castle trip for a prayer (v4-v5: ~40% of castle trips died there)."""
        agent = self.agent
        y0, x0 = valley.UP_STAIRS
        if any(max(abs(int(m[1]) - y0), abs(int(m[2]) - x0)) <= 2 for m in monsters):
            return None
        if DiveLogic.WRAITH_ID is None:
            DiveLogic.WRAITH_ID = MON.from_name('wraith') - nh.GLYPH_MON_OFF
        hungry = agent.blstats.hunger_state >= Hunger.HUNGRY
        level = agent.current_level()
        best = None
        for (y, x), mapping in level.corpses_to_eat.items():
            if max(abs(y - y0), abs(x - x0)) != 1 or not level.walkable[y, x] or \
                    agent.monster_tracker.monster_mask[y, x]:
                continue
            for mid, age in mapping.items():
                if not agent._is_corpse_editable(mid, age):
                    continue
                if mid == DiveLogic.WRAITH_ID:
                    return (y, x, mid)
                if hungry and best is None:
                    best = (y, x, mid)
        return best

    def _fort_meal_here(self, pos):
        """The corpse item below us that the meal trip came for, or None."""
        agent = self.agent
        y, x, mid = self._fort_meal
        if pos != (y, x):
            return None
        for item in agent.inventory.items_below_me or []:
            if item.is_corpse() and item.monster_id == mid and item.count == 1:
                return item
        return None

    def _fort_pull_reason(self, monsters, pos):
        """Why the walk should go back to the pile now (None: walk on): HP below VALLEY_FORT_PULL_HP; an awake
        hostile of makemon difficulty >= VALLEY_FORT_DANGER within VALLEY_FORT_DANGER_RADIUS (not a ghost:
        harmless and speed 3); two awake hostiles within 2; Weak with nothing to eat. Graveyard sleepers don't
        count (they stay put) -- one we woke by striking it does, once it has struck back."""
        agent = self.agent
        bl = agent.blstats
        if bl.hitpoints < VALLEY_FORT_PULL_HP * bl.max_hitpoints:
            return 'hurt'
        if self._fort_starving():
            return 'weak'
        turn = bl.time
        close = []
        for m in monsters:
            if self._valley_sleeper(m):
                continue
            d = max(abs(int(m[1]) - pos[0]), abs(int(m[2]) - pos[1]))
            name = getattr(m[3], 'mname', '?')
            if name.endswith('ghost') or turn - self._fort_pulled.get(name, -10 ** 9) < VALLEY_FORT_PULL_COOLDOWN:
                continue
            if d <= VALLEY_FORT_DANGER_RADIUS and getattr(m[3], 'difficulty', 0) >= VALLEY_FORT_DANGER:
                self._fort_pulled[name] = turn
                return name
            if d <= 2:
                close.append(name)
        if VALLEY_FORT_CROWD and len(close) >= 2:
            for name in close:
                self._fort_pulled[name] = turn
            return 'crowd ' + '+'.join(close)
        return None

    def _fort_starving(self):
        """Weak with nothing to eat, and a hunger prayer on the castle side would be safe (the gap the dive's own
        hunger prayers wait for, no failed prayer): worth the climb."""
        agent = self.agent
        if agent.blstats.hunger_state < Hunger.WEAK or agent.edible_carried_food() or agent.prayer_failed:
            return False
        last = agent.last_prayer_turn
        return last is None or agent.blstats.time - last > agent.SAFE_HUNGER_PRAYER_GAP

    def _fort_walk_action(self, dis, monsters):
        """VALLEY_FORT_WALK: one step of the walk from the pile to the '>' -- the way VALLEY_SPRINT's router picks
        (graveyard squares not seen empty and monsters on the way cost extra, so it goes round the sleepers),
        striking only a bat next to us (it can't be outrun) and whatever holds the next square; a secret door
        is dug when nothing is next to us; eat when hungry and nothing is near. None: let the usual Valley play
        act (a fight at a door square, the '>' itself, stun/confusion/hallucination, low HP out of the pile's
        reach)."""
        agent = self.agent
        bl = agent.blstats
        prop = agent.character.prop
        if prop.hallu or prop.stun or prop.confusion or prop.blind:
            return None
        if bl.hitpoints < VALLEY_SPRINT_REST_BELOW * bl.max_hitpoints:
            return None   # out of the pile's reach: valley_step rests / fight2 fights
        level = agent.current_level()
        pos = (int(bl.y), int(bl.x))
        down = valley.DOWN_STAIRS
        if pos == down or dis[down] != -1:
            return None   # valley_step: to the '>' and down (it rests first when hurt)
        near = [m for m in monsters if utils.adjacent((m[1], m[2]), pos)]
        if bl.hunger_state >= Hunger.HUNGRY and \
                not any(max(abs(int(m[1]) - pos[0]), abs(int(m[2]) - pos[1])) <= 3 for m in monsters):
            food = agent.edible_carried_food()
            if food:
                return lambda: (self._task('valley fort walk: eat'), agent.inventory.eat(food[0]))
        nxt = self._valley_next_door(level)
        if nxt is None:
            return None
        door, stands = nxt
        if pos in stands:
            if near or door in self._valley_undiggable_doors or bl.time < self._dig_blocked_until:
                return None   # fight2 clears the neighbours (a dig stops for any adjacent hostile) / search, kick
            return lambda: (self._task('valley fort walk: dig through a secret door'), self._valley_open_door(door))
        free = [s for s in stands if dis[s] != -1]
        if not free:
            return None   # valley_step explores (a wall variant we haven't seen yet)
        target = min(free, key=lambda s: dis[s])
        fast = [m for m in near if getattr(m[3], 'mmove', 12) > VALLEY_SPRINT_OUTRUN and
                getattr(m[3], 'mname', '') not in self.VALLEY_NO_MELEE]
        if fast:
            victim = fast[0]
        else:
            step, _ = self._valley_route_step(target, monsters, dis)
            if step is None:
                return None
            victim = next((m for m in monsters if (int(m[1]), int(m[2])) == step), None)
            if victim is None:
                if agent.monster_tracker.monster_mask[step]:
                    return None   # a peaceful (the priest) or an unlisted monster: the usual play
                return lambda: (self._task('valley fort walk'), agent.move(*step))
            if getattr(victim[3], 'mname', '') in self.VALLEY_NO_MELEE:
                return None

        def strike():
            if agent.wield_best_melee_weapon():
                return
            self._task('valley fort walk: strike')
            agent.log(f'VALLEY fort walk: attacking the {victim[3].mname} at {(int(victim[1]), int(victim[2]))} '
                      f'hp {bl.hitpoints}/{bl.max_hitpoints}')
            agent.melee_attack(int(victim[1]), int(victim[2]))
        return strike

    # ------------------------------------------------------------- valley xorn (jf_config.VALLEY_XORN)

    _DIRS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    # the forms a vampire takes (mon.c pickvampshape: bat 3/4, fog cloud 1/4, a vampire lord's wolf 1/10): makemon's
    # newcham creates every vampshifter already shifted, and the Valley's graveyards hold vampires (mkroom.c morguemon)
    # -- killing the 'vampire bat' raises the vampire (lord) at full HP (vxx3 s3: two of them ended a 31-HP xorn form
    # in 4 turns, then the dwarf). A xorn walks around them, never through them.
    _VAMPSHIFTS = frozenset(('vampire bat', 'fog cloud', 'wolf'))

    def _vampshift_at(self, p):
        g = self.agent.glyphs[p]
        return nh.glyph_is_monster(g) and nh.permonst(nh.glyph_to_mon(g)).mname in self._VAMPSHIFTS

    def _valley_rock_path(self, start):
        """Dijkstra in bot coordinates over the whole level for a wall-walker from start to the Valley's '>'
        (valley.py): gehennom.des gives the Valley NON_DIGGABLE (0,0,75,19) but no NON_PASSWALL, so a xorn walks
        through its walls and rock -- and through the STONE outside the map: the map fills level rows 1-20 and columns
        3-78 (hack.c isok: y 0..ROWNO-1, x 1..COLNO-1), leaving level row 0 above its top wall and the columns beside
        its side walls, where nothing that walks can ever stand next to us (castle-first-pass F063: the same rows above
        the castle). Only another wall-walker or a ghost (slow, a 1d1 touch) reaches us inside rock, and rays and gazes
        stop at it, while anything on a floor square next to us strikes: floor squares cost VALLEY_XORN_FLOOR, rock
        next to floor VALLEY_XORN_EDGE, the graveyards and Moloch's temple more, the fixed traps (the sleeping-gas trap
        guards the only walkable way to the '>') and squares with or next to a monster most; a peaceful or a
        vampire's shape is never stepped into. From the east arrival region the way runs up the stone east of the map,
        along row 0 and down onto the '>' (~87 squares, ~80 of them out of every walker's reach; the way through the
        map is ~65 with ~45 exposed). castle-first-pass drafted the search."""
        import heapq
        agent = self.agent
        goal = valley.DOWN_STAIRS
        (t0r, t0c), (t1r, t1c) = valley.TEMPLE
        traps = set(valley.SPIKED_PITS) | {valley.SLEEP_GAS}
        # every monster in view, not agent.get_visible_monsters()'s: that keeps only those next to squares our BFS
        # reaches, and from inside the rock it reaches almost nothing
        mt = agent.monster_tracker
        mons = {(int(y), int(x)) for y, x in zip(*(mt.monster_mask & ~mt.peaceful_monster_mask).nonzero())}
        # never stepped into: a peaceful (Moloch's priest: 'Really attack?') or a vampire's shape
        blocked = {(int(y), int(x)) for y, x in zip(*(mt.monster_mask & mt.peaceful_monster_mask).nonzero())}
        blocked |= {p for p in mons if self._vampshift_at(p)}
        floor = getattr(self, '_valley_floor_all', None)
        if floor is None:
            floor = self._valley_floor_all = valley.floor_mask(agent.current_level().walkable.shape)
        rmax, cmax = floor.shape[0] - 1, floor.shape[1] - 1

        def cost(p):
            c = 1
            if floor[p]:
                c += VALLEY_XORN_FLOOR
            elif any(0 <= p[0] + dy <= rmax and 0 <= p[1] + dx <= cmax and floor[p[0] + dy, p[1] + dx]
                     for dy, dx in self._DIRS8):
                c += VALLEY_XORN_EDGE
            if p in valley.GRAVEYARD or (t0r <= p[0] <= t1r and t0c <= p[1] <= t1c):
                c += 30
            if p in traps:
                c += 50
            if p in blocked:
                c += 1000
            elif p in mons:
                c += 60
            elif any((p[0] + dy, p[1] + dx) in mons for dy, dx in self._DIRS8):
                c += 8
            return c

        start = (int(start[0]), int(start[1]))
        dist = {start: 0}
        prev = {}
        heap = [(0, start)]
        while heap:
            d, p = heapq.heappop(heap)
            if d > dist.get(p, 1 << 30):
                continue
            if p == goal:
                path = [p]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            for dy, dx in self._DIRS8:
                n = (p[0] + dy, p[1] + dx)
                if not (0 <= n[0] <= rmax and 0 <= n[1] <= cmax):
                    continue
                nd = d + cost(n)
                if nd < dist.get(n, 1 << 30):
                    dist[n] = nd
                    prev[n] = p
                    heapq.heappush(heap, (nd, n))
        return None

    def xorn_buffer(self):
        """VALLEY_XORN below the Valley: a wall-walking form's HP is only a buffer over the dwarf's own (polyself.c
        rehumanize at 0 form HP returns us to the HP we had before the zap) and heals 1 HP per 20 turns (allmain.c,
        u.mh without regeneration): no rest for it, no retreat up to the Valley's dead-end corner -- vxx2 s1 rested
        425 turns on Gehennom 2 as a 13/35 xorn and the form ran out on the spot; s0 went back up at once."""
        if not jf_config.VALLEY_XORN or not self.in_gehennom() or self.in_valley():
            return False
        agent = self.agent
        before = getattr(agent.character, 'hp_before_poly', None)
        if before is None or before[0] < 0.5 * before[1]:
            return False
        from . import castle_cross
        return castle_cross.wallwalker(agent)

    @Strategy.wrap
    def valley_xorn(self):
        """VALLEY_XORN: a wall-walking polymorph form in the Valley (castle-first-pass's CFP_XORN crosses the castle as
        a xorn; the form lasts rn1(500,500) turns and keeps its weapon: polyself.c drop_weapon only for cantwield
        forms) phases through the rock straight to the '>' and goes down: Gehennom 2 = castle+2, so a Dlvl-28 castle
        scores Dlvl 30 (0.691) as well; below it the Gehennom dig-dive digs on (a xorn has hands for the pick).
        It hands over to the Valley walk when the form ends (0 form HP: polyself.c rehumanize, back to the dwarf's own
        HP). It strikes only what stands on its next square and is no vampire's shape (see _VAMPSHIFTS): v2 struck
        every fast neighbour and lost more forms than it saved (vxx3: 4/14 exits against vxx2's 3/7)."""
        agent = self.agent
        if not jf_config.VALLEY_XORN or not self.in_valley():
            yield False
        from . import castle_cross
        if not castle_cross.wallwalker(agent):
            yield False
        yield True
        steps = 0
        while steps < 600 and castle_cross.wallwalker(agent) and self.in_valley():
            steps += 1
            bl = agent.blstats
            here = (int(bl.y), int(bl.x))
            if here == valley.DOWN_STAIRS:
                agent.log(f'VALLEY xorn: on the \'>\' at turn {bl.time}, hp {bl.hitpoints}/{bl.max_hitpoints}, '
                          f'going down')
                agent.direction('>')
                continue
            path = self._valley_rock_path(here)
            if not path or len(path) < 2:
                agent.search()
                continue
            nxt = path[1]
            d = agent.calc_direction(here[0], here[1], nxt[0], nxt[1])
            if steps == 1 or steps % 10 == 0:
                agent.log(f'VALLEY xorn: at {here} hp {bl.hitpoints}/{bl.max_hitpoints}, '
                          f'{len(path) - 1} steps to the \'>\' via {path[1:4]}')
            if agent.monster_tracker.peaceful_monster_mask[nxt] or self._vampshift_at(nxt):
                agent.search()   # (it moves on; the next path steps around it)
                continue
            if agent.monster_tracker.monster_mask[nxt] or agent.glyphs[nxt] == nh.GLYPH_INVISIBLE:
                with agent.atom_operation():
                    agent.step(A.Command.FIGHT)
                    agent.direction(d)
                continue
            agent.direction(d)
            if (int(agent.blstats.y), int(agent.blstats.x)) == here and \
                    ("It's a wall" in agent.message or "It's solid stone" in agent.message):
                # the form is over and the message saying so went by unseen: we are the dwarf again (vxx2 s2, s4,
                # s6 walked into the wall over and over while a troll / vampire bats killed them)
                agent.log(f'VALLEY xorn: the form is gone ({agent.message[:60]!r}) at {here}')
                agent._cfp_form = None
                return
        bl = agent.blstats
        agent.log(f'VALLEY xorn: done at {(int(bl.y), int(bl.x))} turn {bl.time} hp {bl.hitpoints}/{bl.max_hitpoints} '
                  f'(wall-walker: {castle_cross.wallwalker(agent)}, in the Valley: {self.in_valley()})')

    @Strategy.wrap
    def xorn_repoly(self):
        """VALLEY_XORN: in Gehennom and back in our own form (the xorn's HP ran out: polyself.c rehumanize), with a known
        wand of polymorph that may still have charges and polymorph control shown this game (agent._note_poly_control:
        the prompt came, and the rings worn then are still on): zap it at ourselves again -- agent.update answers
        'xorn' -- for the Valley's stone route or a fresh ~36 HP (d(8,8)) over ours while the dig-dive goes on.
        jf16-s0~10 lost its form at 7/38 on Gehennom 5 and died there as the dwarf with the wand in its pack."""
        agent = self.agent
        if not jf_config.VALLEY_XORN or not self.in_gehennom() or getattr(agent, '_poly_control_turn', None) is None:
            yield False
        from . import castle_cross, castle_power
        prop = agent.character.prop
        if castle_cross.wallwalker(agent) or getattr(agent, '_cfp_form', None) is not None or prop.stun or \
                prop.confusion or self._repoly_zaps >= VALLEY_XORN_REPOLY:
            yield False   # (another form than ours: not this strategy's to change)
        rings = getattr(agent, '_poly_control_rings', None)
        if rings is None:
            yield False
        worn = {agent.inventory.items.get_letter(i) for i in agent.inventory.items
                if i.category == nh.RING_CLASS and i.equipped}
        if not rings <= worn:
            yield False   # the ring that gave control may be gone: a random form, or system shock (rnd(30) HP)
        wand = castle_power._poly_wand(agent)
        if wand is None:
            yield False
        yield True
        self._repoly_zaps += 1
        bl = agent.blstats
        agent.log(f'VALLEY xorn: our own form at hp {bl.hitpoints}/{bl.max_hitpoints} turn {bl.time} depth {bl.depth}, '
                  f'zapping {wand.text!r} at ourselves (repoly {self._repoly_zaps})')
        agent.zap(wand, '.')
        agent.inventory.items.update(force=True)
        agent.log(f'VALLEY xorn: repoly -> {agent.message[:120]!r} (wall-walker: {castle_cross.wallwalker(agent)})')

    def valley_progress(self):
        """Log how far west the Valley walk got (for the scenario analyses), and each 5 steps of new progress along
        the way to the '>' (valley.route_distance: the route turns north and east, where 'west' shows nothing)."""
        bl = self.agent.blstats
        if self._valley_west is None or bl.x <= self._valley_west - 5:
            self._valley_west = bl.x
            self.agent.log(f'VALLEY at {(bl.y, bl.x)} turn {bl.time} hp {bl.hitpoints}/{bl.max_hitpoints} '
                           f'xl {bl.experience_level}')
        rd = valley.route_distance(self.agent.current_level().walkable.shape)
        d = int(rd[bl.y, bl.x])
        if d >= 0 and d <= getattr(self, '_valley_best_route', 10 ** 9) - 5:
            self._valley_best_route = d
            self.agent.log(f'VALLEY progress {d} steps to the > at {(int(bl.y), int(bl.x))} turn {bl.time} '
                           f'hp {bl.hitpoints}/{bl.max_hitpoints} xl {bl.experience_level}')

    def _valley_descend(self, down):
        agent = self.agent
        bl = agent.blstats
        if (bl.y, bl.x) != down:
            self._task('valley: to the down stairs')
            agent.go_to(*down)
            return
        if self.levitating():
            # "You are floating high above the stairs": a levitation ring comes off, a potion's effect ends soon
            self._task('valley: levitating over the stairs')
            ring = next((i for i in agent.inventory.items if i.category == nh.RING_CLASS and i.equipped and
                         'levitation' in i.text and i.status != Item.CURSED), None)
            if ring is not None:
                agent.log(f'VALLEY removing {ring.text!r} to take the stairs')
                with agent.atom_operation():
                    agent.step(A.Command.REMOVE)
                    if 'What do you want to remove?' in agent.message:
                        agent.type_text(agent.inventory.items.get_letter(ring))
                return
            agent.search(5)
            return
        if not (jf_config.VALLEY_SPRINT and agent.get_visible_monsters()) and self.rest_if_hurt():
            return
        level = agent.current_level()
        agent.log(f'VALLEY down the stairs at turn {bl.time} ({bl.time - self._valley_arrival} turns in the '
                  f'Valley), hp {bl.hitpoints}/{bl.max_hitpoints}')
        agent.move('>')
        self._arrived = (agent.current_level().key(), agent.blstats.time, (level.key(), down))

    def _valley_open_door(self, door):
        """Standing next to a Valley secret door: dig through it (pick-axe or mattock), else search for it
        and kick it open (locked; kicking wakes everything within ~14 squares, dokick.c wake_nearby)."""
        agent = self.agent
        tool = self.digging_tool()
        if tool is not None and door not in self._valley_undiggable_doors and \
                agent.blstats.time >= self._dig_blocked_until:
            self._task('valley: dig through a secret door')
            return self.dig_toward(tool, *door)
        if agent.glyphs[door] in G.DOOR_CLOSED or agent.current_level().objects[door] in G.DOOR_CLOSED:
            self._task('valley: kick a locked door')
            agent.kick(*door)
            return
        self._task('valley: search for a secret door')
        agent.search(5)

    def dig_toward(self, tool, y, x):
        """Apply the digging tool sideways into (y, x): a (secret or locked) door breaks in a few turns."""
        agent = self.agent
        shield = agent.inventory.items.off_hand
        if tool.object == O.from_name('dwarvish mattock') and shield is not None:
            agent.log(f'DIVE dropping {shield.text!r} to dig with a mattock')
            agent.inventory.takeoff(shield)
            shield = next((i for i in agent.inventory.items if i.is_armor() and not i.equipped and
                           i.text.split(' (')[0] == shield.text.split(' (')[0]), None)
            if shield is not None:
                agent.inventory.drop(shield)
            return
        direction = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
        agent.log(f'VALLEY digging {direction} into {(y, x)} with {tool.text!r}')
        with agent.atom_operation():
            tool = agent.inventory.move_to_inventory(tool)
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(tool))
            prompted = 'In what direction do you want to dig?' in agent.single_message
            if prompted:
                agent.direction(direction)
            elif agent.single_message.startswith('In what direction'):
                agent.step(A.Command.ESC)
        msg = agent.message
        if not prompted:
            # can't swap weapons (welded), stuck in a web, ...: search and kick instead for a while
            agent.log(f'VALLEY could not dig: {msg!r}')
            self._dig_blocked_until = agent.blstats.time + 100
        elif 'You swing' in msg:
            # thin air: the doorway is open already (an item on it hid the broken door from our map)
            agent.log(f'VALLEY {(y, x)} is open: {msg!r}')
            agent.current_level().walkable[y, x] = True
            agent.last_bfs_step = -1
        elif 'too hard to' in msg:
            agent.log(f'VALLEY {(y, x)} does not dig like a secret door: {msg!r}')
            self._valley_undiggable_doors.add((y, x))
        elif 'You break through' in msg:
            agent.log(f'VALLEY broke through the door at {(y, x)}')

    @Strategy.wrap
    def gehennom_escape(self):
        """Gehennom has neither prayer nor Elbereth: with a monster we can't outfight next to us (or low HP with
        anything next to us), a wand of digging zapped down is a one-turn escape that also banks a level."""
        agent = self.agent
        if not self.in_gehennom() or self.in_valley():
            yield False
        level = agent.current_level()
        wand = self.digging_wand() if level.key() not in self.undiggable else None
        if wand is None or self.levitating():
            yield False
        bl = agent.blstats
        adjacent = [m for m in agent.get_visible_monsters() if utils.adjacent((m[1], m[2]), (bl.y, bl.x))]
        if not adjacent:
            yield False
        low = bl.hitpoints < 0.5 * bl.max_hitpoints
        if not low and not any(getattr(m[3], 'mname', '') in GEHENNOM_THREATS or
                               getattr(m[3], 'mlevel', 0) >= GEHENNOM_THREAT_LEVEL for m in adjacent):
            yield False
        yield True
        agent.log(f'GEHENNOM escape: zapping {wand.text!r} down at hp {bl.hitpoints}/{bl.max_hitpoints}, '
                  f'next to {[m[3].mname for m in adjacent]}')
        key = level.key()
        agent.zap(wand, '>')
        if agent.current_level().key() == key and 'too hard to dig' in agent.message:
            self.undiggable.add(key)

    # 'The wraith touches you!', 'The vampire lord bites!', "Wilmar's ghost touches you!" (mhitu.c hitmsg/missmu)
    _ATTACK_MSG = re.compile(r"[Tt]he ([a-z][a-z -]*?) (?:hits|misses|just misses|bites|touches|kicks|claws|butts|"
                             r"stings|swings|thrusts|grabs|squeezes|engulfs|casts|breathes|spits|gazes|shoots|"
                             r"throws|lashes|slashes|smites)|([A-Z][a-z]+'s ghost) (?:touches|misses)")

    def _valley_sleeper(self, m):
        """A monster on a Valley graveyard square that hasn't attacked us lately: one of the fill's sleepers."""
        _, y, x, mon, _ = m
        if (int(y), int(x)) not in valley.GRAVEYARD:
            return False
        name = getattr(mon, 'mname', '')
        last = self._valley_attackers.get(name)
        if name == 'ghost':
            last = max([t for n, t in self._valley_attackers.items() if n.endswith('ghost')] + [-1])
        return last is None or self.agent.blstats.time - last > VALLEY_ATTACKER_TURNS

    def valley_fight_filter(self, monsters):
        """agent.fight_monsters: in the Valley (VALLEY_GRAVE_FILTER) leave the graveyard sleepers out."""
        if not (VALLEY_GRAVE_FILTER or jf_config.VALLEY_SPRINT) or not monsters or not self.in_valley():
            return monsters
        return [m for m in monsters if not self._valley_sleeper(m)]

    _MELEE_HIT = re.compile(r'\b(The [\w -]+?|It) (hits|bites|butts|claws|kicks|stings|touches)!')

    def on_scare_scroll(self):
        """Standing where we dropped a scroll of scare monster (GEHENNOM_SCARE), and it still lies here."""
        if self._scare_spot is None:
            return False
        agent = self.agent
        if self._castle_scare() and self._scare_spot == (agent.current_level().key(), (agent.blstats.y, agent.blstats.x)) \
                and self._MELEE_HIT.search(agent.message) and \
                any(utils.adjacent((m[1], m[2]), (agent.blstats.y, agent.blstats.x))
                    for m in agent.get_visible_monsters()):
            # the CASTLE_SCARE gamble dropped no scroll of scare monster: something next to us still melees us
            # (a scared monster doesn't: dochug !scared). Stop holding on it, let the castle plan act (a wand of
            # polymorph, rings) -- pwc-dp4 jf16-s14 dug on a fake pile under a minotaur's blows
            agent.log(f'SCARE no scare monster in the pile at {self._scare_spot[1]}: {agent.message[:80]!r}')
            self._scare_spot = None
            # none of those labels is scare monster: don't drop their other copies again
            self._not_scare_keys |= getattr(self, '_scare_dropped_keys', set())
            return False
        if self._scare_spot != (agent.current_level().key(), (agent.blstats.y, agent.blstats.x)):
            if getattr(self, '_scare_off_logged', None) != self._scare_spot:
                self._scare_off_logged = self._scare_spot
                agent.log(f'SCARE off the scroll: spot {self._scare_spot}, now '
                          f'{(agent.current_level().key(), (agent.blstats.y, agent.blstats.x))}')
            return False
        below = agent.inventory.items_below_me
        # at a pit's edge the floor objects aren't listed ('You cannot reach the bottom of the pit', pwc-smoke5
        # jf16-s1 walked off its pile), though the scroll still scares from the square
        pit_edge = 'cannot reach the bottom of the pit' in agent.message
        if below is None or (self._castle_scare() and (self.levitating() or pit_edge or not below)):
            # not parsed yet, or levitating (the floor's objects aren't listed then, but the scroll lies under
            # us and onscary still counts it: castle pwc-smoke3 public-s2 'dropped' twice): assume it is there
            return True
        if any(power.is_scare_candidate(i) for i in below):
            return True
        agent.log(f'SCARE scroll gone from {self._scare_spot[1]}: below {[i.text for i in below]}')
        self._scare_spot = None
        return False

    @Strategy.wrap
    def gehennom_scare(self):
        """GEHENNOM_SCARE: drop a scroll of scare monster when a Gehennom fight goes badly, then hold on it: dig
        down from it (scared monsters don't interrupt the dig), or in the Valley (hardfloor) rest on it until
        GEHENNOM_SCARE_UNTIL, striking what stays next to us without stepping off."""
        agent = self.agent
        castle = self._castle_scare()
        if not castle and (not GEHENNOM_SCARE or not self.in_gehennom()):
            yield False
        bl = agent.blstats
        level = agent.current_level()
        pos = (bl.y, bl.x)
        adjacent = [m for m in agent.get_visible_monsters() if utils.adjacent((m[1], m[2]), pos)]
        if self.on_scare_scroll():
            if self._scare_hold() is None:
                yield False   # healed: walk on (the scroll stays behind)
            yield True
            self._scare_hold_loop()
            return
        if castle:
            # CASTLE_SCARE (death is free on the castle: its depth is banked): the west maze's minotaurs killed
            # 21-27 of 61 real kits (castlebench), most within 30 turns; a scroll of scare monster under us stops
            # them and @ soldiers alike (monmove.c onscary), and a scared monster doesn't melee (dochug !scared)
            # -- so we can strike it dead from the square. Drop every scroll that may be one when an
            # Elbereth-ignorer comes within 2 (the kits carry 6-8 unknown scroll types; 12 of 61 hold one).
            from .castle_landing import scare_radius   # LANDING_GUARD: at first sight, not only at 2
            near = [m for m in agent.get_visible_monsters()
                    if max(abs(m[1] - pos[0]), abs(m[2] - pos[1])) <= scare_radius() and self._ignores_elbereth(m[3])]
            trouble = bool(near) or (bool(adjacent) and bl.hitpoints < GEHENNOM_SCARE_BELOW * bl.max_hitpoints)
        else:
            if not adjacent:
                yield False
            trouble = bl.hitpoints < GEHENNOM_SCARE_BELOW * bl.max_hitpoints or \
                (len(adjacent) >= GEHENNOM_SCARE_CROWD and bl.hitpoints < GEHENNOM_SCARE_CROWD_BELOW * bl.max_hitpoints)
        if not trouble or level.objects[pos] in G.STAIR_UP or level.objects[pos] in G.STAIR_DOWN:
            yield False   # (on stairs the retreat/descent is the better answer)
        known, candidates = power.scare_scrolls(agent)
        drop = known[:1]
        if not drop and (GEHENNOM_SCARE_GAMBLE or castle):
            drop = [it for it, p in candidates if p > 0 and agent.inventory._scroll_key(it) not in self._not_scare_keys]
        if not drop:
            yield False
        yield True
        self._scare_drops += 1
        agent.log(f'SCARE drop {self._scare_drops}: {[it.text for it in drop]} at {pos} hp {bl.hitpoints}/'
                  f'{bl.max_hitpoints}, adjacent {[m[3].mname for m in adjacent]}')
        # set before the drop: the end of its atomic step runs the preempt checks, which may switch strategy
        # (raise) before the next line (castle B005) -- then the hold would never start
        self._scare_spot = (level.key(), pos)
        self._scare_drop_turn = bl.time
        self._scare_dropped_keys = {agent.inventory._scroll_key(it) for it in drop}
        if castle:
            self._scare_threat_turn = bl.time   # the drop answers a threat: hold at least until it's gone
        agent.inventory.drop(drop, [1] * len(drop))
        # never picked up again (a second pickup turns scare monster to dust)
        agent.inventory._note_dropped(drop, [1] * len(drop), force=True)
        self._scare_hold_loop()

    def _pile_work(self):
        """BREACH_PILE: the hold on a scroll of scare monster at the castle is time to use: a wand (castle_power
        BREACH_MINO order) at a minotaur or other big Elbereth-ignorer in a straight line within 8 squares (a scared
        one hovers just out of reach), then the passage plan's unknown rings, amulets, boots and wishes right here
        (potions wait for the courtyard: a lift quaffed in the maze is spent walking). A ring that turns out to be
        levitation comes off again: we walk (dig) to the courtyard, not float through the maze. True: acted."""
        from . import castle_power
        agent = self.agent
        targets = castle_power.inline_targets(self)
        if targets:
            wand, why = castle_power.breach_wand(agent)
            if wand is not None:
                _, d, target = targets[0]
                castle_power.zap_breach_wand(agent, wand, why, target, d)
                return True
        castle = self.castle
        if not castle.active() or self.levitating() or agent.character.prop.polymorph:
            return False
        for kind, item in castle._plan():
            # (no amulets: 13.5% of unknown ones strangle, 90% of those cursed -- brx-t1-all jf25-s3~5 'strangulation';
            # castle_logic tries them last. A used wand of wishing stays in the plan: its ring is put on by the corner)
            if kind not in ('ring', 'boots', 'wish') or item.glyphs[0] in castle._tested:
                continue
            if kind != 'wish' and item.is_unambiguous():
                continue   # a known lift is put on by the corner (castle_logic), not here
            castle._set_state(f'breach pile: trying {kind} {item.text!r} on the scroll')
            castle._try(kind, item)
            if kind == 'ring' and self.levitating():
                agent.inventory.items.update(force=True)
                worn = [r for r in castle._worn_rings() if r.glyphs[0] == item.glyphs[0]]
                if worn and castle._remove_ring(worn[0]):
                    castle._lev_source = None
                    agent.log('BREACH pile: a ring of levitation -- off again until the courtyard')
            return True
        return False

    def _scare_throw(self):
        """CASTLE_SCARE hold: throw at an Elbereth-ignorer in a straight line 2-7 squares away (a scared minotaur
        hovers just out of reach; throwing from the scroll doesn't spoil it, unlike Elbereth). Never while
        levitating: the throw pushes us off the square ('You float in the opposite direction')."""
        agent = self.agent
        if self.levitating() or agent.character.prop.polymorph:
            return False
        bl = agent.blstats
        launcher, ammo = agent.inventory.get_best_ranged_set(throwing=True)
        if ammo is None or launcher is not None:
            return False
        level = agent.current_level()
        for _, my, mx, mon, _ in agent.get_visible_monsters():
            dy, dx = int(my - bl.y), int(mx - bl.x)   # (numpy bools don't subtract: TypeError, pwc-dp12 jf25-s8)
            dist = max(abs(dy), abs(dx))
            if dist < 2 or dist > 7 or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)) or \
                    not self._ignores_elbereth(mon):
                continue
            sy, sx = (dy > 0) - (dy < 0), (dx > 0) - (dx < 0)
            if any(not level.walkable[bl.y + sy * k, bl.x + sx * k] for k in range(1, dist)):
                continue
            agent.log(f'SCARE throwing {ammo.text!r} at the {getattr(mon, "mname", "?")} from the scroll')
            # (calc_direction wants a neighbour square: one step along the line)
            agent.fire(ammo, agent.calc_direction(bl.y, bl.x, bl.y + sy, bl.x + sx))
            return True
        return False

    def _castle_scare(self):
        """CASTLE_SCARE: on a Dungeons of Doom level at depth >= CASTLE_SCARE_DEPTH (the castle, or a maze just
        above it: that depth is banked), not while crossing the moat. The castle is only recognised when a dig
        fails ('too hard to dig', ~5 turns after the landing), and the west maze's minotaurs kill many arrivals
        before that (pwc-smoke3: 3 of 7 kits dead within 6 turns)."""
        if not jf_config.CASTLE_SCARE:
            return False
        level = self.agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or self.agent.blstats.depth < jf_config.CASTLE_SCARE_DEPTH:
            return False
        if jf_config.MEDUSA_NOT_CASTLE and self.on_medusa_level():
            return False   # (see MEDUSA_NOT_CASTLE) Medusa's level at depth 25+ is no castle landing
        return not self.castle.committed()

    def _scare_hold(self):
        """What to do on our scare monster scroll: 'dig' (a diggable Gehennom level with a tool), 'rest' (until
        GEHENNOM_SCARE_UNTIL), or None (healed: walk on)."""
        agent = self.agent
        bl = agent.blstats
        level = agent.current_level()
        tool = self.digging_tool()
        if self._castle_scare() and any(self._ignores_elbereth(m[3]) for m in agent.get_visible_monsters()):
            self._scare_threat_turn = bl.time
        if self._castle_scare() and (level.key() in self.undiggable or self.castle.active() or tool is None or
                                      self.levitating()):
            # the castle can't be dug: hold while anything hostile is close (it hovers, scared: strike it when it
            # comes next to us) and while an Elbereth-ignorer is still about -- a scared minotaur flees a few
            # squares and comes back the moment we step off (pwc-smoke3 jf14-s4 held 110 turns, walked on and was
            # killed 12 turns later); up to CASTLE_SCARE_HOLD turns
            y0, x0 = bl.y, bl.x
            mons = agent.get_visible_monsters()
            turn = bl.time
            held = turn - (self._scare_drop_turn or turn)
            if held < jf_config.CASTLE_SCARE_HOLD and \
                    (any(max(abs(m[1] - y0), abs(m[2] - x0)) <= 2 for m in mons) or
                     turn - getattr(self, '_scare_threat_turn', -10 ** 9) < 150):
                return 'rest'
            return 'rest' if bl.hitpoints < GEHENNOM_SCARE_UNTIL * bl.max_hitpoints else None
        if level.key() not in self.undiggable and not self.in_valley() and tool is not None and \
                not self.levitating() and self._diggable_spot(bl.y, bl.x, max_wet=8):
            return 'dig'
        if bl.hitpoints >= GEHENNOM_SCARE_UNTIL * bl.max_hitpoints:
            return None
        return 'rest'

    def _scare_hold_loop(self):
        """The hold repeats inside this strategy while it lasts: agent.preempt re-runs the lower stack after a
        preempting body returns, and a lower strategy (the Valley retreat, fight2's moves) then acted once before
        the upper hooks could fire -- every smoke-run hold stepped off its scroll within 1-13 turns (F003)."""
        agent = self.agent
        while self.on_scare_scroll():
            what = self._scare_hold()
            if what is None:
                return
            steps = agent.step_count
            if what == 'dig':
                self._task('scare monster: dig down')
                self.dig_with_tool(self.digging_tool())
            else:
                self._task('scare monster: rest')
                bl = agent.blstats
                adjacent = [m for m in agent.get_visible_monsters()
                            if utils.adjacent((m[1], m[2]), (bl.y, bl.x))]
                # a scared monster that can't get away stays next to us: strike it from the square (no penalty,
                # unlike Elbereth), but never step off
                target = next((m for m in adjacent if getattr(m[3], 'mname', '') not in self.VALLEY_NO_MELEE), None)
                if target is not None and bl.hitpoints >= 0.3 * bl.max_hitpoints:
                    if not agent.wield_best_melee_weapon():
                        agent.melee_attack(target[1], target[2])
                elif jf_config.BREACH_PILE and self._castle_scare() and self._pile_work():
                    pass
                elif self._castle_scare() and self._scare_throw():
                    pass
                else:
                    agent.search(1 if adjacent else 10)
            if agent.step_count == steps:
                return   # nothing happened (a refused action): never spin

    # ------------------------------------------------------------- DEEP_ITEMS (deep-arrivals, ledger F095)

    _UNIHORN = O.from_name('unicorn horn')
    _CHARGING = O.from_name('charging', nh.SCROLL_CLASS)
    _TAMING = O.from_name('taming', nh.SCROLL_CLASS)
    # zap.c zap_over_floor on Medusa's moat ('water': waterbody_name), seen or heard; the ray's name when it hits
    _DEEP_FROST_MSG = ('is bridged with ice', 'The water freezes', 'a crackling sound', 'The bolt of cold')
    _DEEP_DIRS = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1), 'ne': (-1, 1), 'nw': (-1, -1),
                  'se': (1, 1), 'sw': (1, -1)}

    @Strategy.wrap
    def deep_items_strategy(self):
        """DEEP_ITEMS (jf_config, see there): on Medusa's level a unicorn horn when blind, a scare instrument or an
        unknown horn when hostiles press us, our known scroll of scare monster under the dig square (rest on it, dig
        from it); anywhere in the dive a scroll of charging on an empty wand of digging. Below the emergency (a due
        prayer goes first), above dig_first, the Elbereth rest and fight2."""
        plan = self._deep_items_plan()
        if plan is None:
            yield False
            return
        yield True
        agent = self.agent
        key = agent.current_level().key()
        # keep acting while the plan lasts (a one-action return lets one step of the lower chain run first: the
        # dive's Elbereth on our scroll, fight2's swing -- HOLD_LOOP); a higher hook still interrupts any step
        for _ in range(40):
            steps = agent.step_count
            self._deep_items_act(plan)
            if agent.step_count == steps or agent.current_level().key() != key:
                return
            plan = self._deep_items_plan()
            if plan is None:
                return

    def _deep_threats(self, radius=2):
        """DEEP_ITEMS: [(Chebyshev distance, monster)] for the non-peaceful monsters in view within radius, nearest
        first."""
        agent = self.agent
        bl = agent.blstats
        peaceful = agent.monster_tracker.peaceful_monster_mask
        out = []
        for m in agent.get_visible_monsters():
            d = max(abs(int(m[1]) - int(bl.y)), abs(int(m[2]) - int(bl.x)))
            if d <= radius and not peaceful[m[1], m[2]]:
                out.append((d, m))
        out.sort(key=lambda t: t[0])
        return out

    _EYEWEAR = frozenset(O.from_name(n) for n in ('lenses', 'blindfold', 'towel'))

    def _deep_lenses_item(self):
        """DEEP_ITEMS: a pair of lenses to put on (not tried yet), when no eyewear is worn -- or None."""
        items = list(self.agent.inventory.items)
        if any(it.category == nh.TOOL_CLASS and it.equipped and it.is_unambiguous() and it.object in self._EYEWEAR
               for it in items):
            return None
        for it in items:
            if it.category == nh.TOOL_CLASS and it.is_unambiguous() and it.object == O.from_name('lenses') and \
                    not it.equipped and it.text not in self._deep_lenses_tried and 'unpaid' not in (it.text or ''):
                return it
        return None

    # apply.c use_unicorn_horn, cursed: make_sick / make_blinded ('A cloud of darkness falls upon you') / 'You suddenly
    # feel confused' / stunned ('You stagger') / an attribute down (attrib.c: weak, stupid, foolish, clumsy, fragile,
    # repulsive) / hallucinating / deaf
    _CURSED_UNIHORN = ('deathly sick', 'cloud of darkness', 'suddenly feel confused', 'suddenly feel trippy',
                       'You stagger', 'so cosmic', 'You feel weak!', 'You feel stupid!', 'You feel foolish!',
                       'You feel clumsy!', 'You feel fragile!', 'You feel repulsive!', 'deaf')

    def _deep_unihorn_item(self):
        """A unicorn horn not known to be cursed (apply.c use_unihorn: a cursed one adds troubles), or None."""
        for it in self.agent.inventory.items:
            if it.category == nh.TOOL_CLASS and it.is_unambiguous() and it.object == self._UNIHORN and \
                    it.status != Item.CURSED and 'unpaid' not in (it.text or '') and \
                    it.text not in self._deep_bad_horns:
                return it
        return None

    def _deep_wet_dir(self):
        """(direction, delta) toward a moat neighbour of our square not yet frozen from here (as _freeze_action),
        or None."""
        agent = self.agent
        bl = agent.blstats
        level = agent.current_level()
        h, w = level.objects.shape
        done = self.__dict__.setdefault('_frozen_dirs', {}).get((level.key(), bl.y, bl.x), set())
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            y, x = bl.y + dy, bl.x + dx
            if not (0 <= y < h and 0 <= x < w) or (dy, dx) in done:
                continue
            if agent.glyphs[y, x] in WET or level.objects[y, x] in WET or level.objects[y, x] == -1:
                return agent.calc_direction(bl.y, bl.x, y, x), (dy, dx)
        return None

    def _deep_ray_dir(self, threats):
        """Where an unknown horn's ray goes if it has one (a frost or fire horn, 4/11): at a moat neighbour (frost
        freezes it: a flood roll less; fire over the moat only evaporates some water), else at the nearest threat in
        a straight line, else down the longest free line."""
        wet = self._deep_wet_dir()
        if wet is not None:
            return wet[0]
        agent = self.agent
        bl = agent.blstats
        for _, m in threats:
            dy, dx = int(m[1]) - int(bl.y), int(m[2]) - int(bl.x)
            if dy == 0 or dx == 0 or abs(dy) == abs(dx):
                return agent.calc_direction(bl.y, bl.x, bl.y + int(np.sign(dy)), bl.x + int(np.sign(dx)))
        from . import opp_items
        return opp_items.longest_free_dir(agent)[0]

    def _deep_on_scare(self):
        """DEEP_ITEMS: standing where we dropped our known scroll of scare monster on this level."""
        if self._deep_scare is None:
            return False
        agent = self.agent
        key, pos, _ = self._deep_scare
        return key == agent.current_level().key() and pos == (int(agent.blstats.y), int(agent.blstats.x))

    def _deep_charge_plan(self, quiet_radius=6):
        """DEEP_ITEMS: ('deep_charge', (scroll, wand)) -- a known scroll of charging not known cursed on a known wand of
        digging known empty or down to <= 1 charge, at a quiet moment (nothing within quiet_radius, HP >= 50%) -- or
        None."""
        agent = self.agent
        prop = agent.character.prop
        if prop.blind or prop.confusion or prop.stun:
            return None   # (confused, charging recharges our energy instead: read.c)
        items = self._deep_charge_items()
        if items is None:
            return None
        bl = agent.blstats
        if bl.hitpoints < 0.5 * bl.max_hitpoints or self._deep_threats(radius=quiet_radius):
            return None
        return ('deep_charge', items)

    def _deep_charge_items(self):
        """DEEP_ITEMS: (scroll, wand) -- a known scroll of charging not known cursed and a known wand of digging known
        empty or down to <= 1 charge -- or None."""
        agent = self.agent
        scroll = wand = None
        for it in agent.inventory.items:
            if it.category == nh.SCROLL_CLASS and it.is_unambiguous() and it.object == self._CHARGING and \
                    it.status != Item.CURSED and 'unpaid' not in (it.text or ''):
                scroll = scroll or it
            elif it.is_wand() and it.is_unambiguous() and it.object == O.from_name('digging', nh.WAND_CLASS) and \
                    it.text not in self._deep_charge_refused:
                mo = re.search(r'\((-?\d+):(-?\d+)\)', it.text or '')
                if agent.inventory.is_known_empty(it) or it.comment == 'EMPT' or (mo and int(mo.group(2)) <= 1):
                    wand = wand or it
        if scroll is None or wand is None:
            return None
        return (scroll, wand)

    def _deep_items_plan(self):
        """DEEP_ITEMS: the next action as (kind, arg), or None. No agent step here (a preempt condition)."""
        if not (jf_config.DEEP_ITEMS and self.diving):
            return None
        agent = self.agent
        prop = agent.character.prop
        if prop.polymorph or prop.hallu or self.levitating() or utils.any_in(agent.glyphs, G.SWALLOW):
            return None
        if not self.on_medusa_level():
            return self._deep_charge_plan()
        if jf_config.MEDUSA_ZAP_FIRST and self._medusa_zap_ready():
            return None   # (see jf_config.MEDUSA_ZAP_FIRST) the kept wand holes the floor from here: no gamble before it
        bl = agent.blstats
        level = agent.current_level()
        key = level.key()
        pos = (int(bl.y), int(bl.x))
        blind = bool(prop.blind)
        threats = self._deep_threats()
        hurt = agent._hurt_recently(2)
        adjacent = any(d <= 1 for d, _ in threats)
        pressed = adjacent or len(threats) >= 2 or (blind and hurt)
        # 0. a pair of lenses on: mondata.c can_blnd -- any worn eyewear (ublindf), lenses included, stops a raven's
        # blinding claw and a cobra's blinding venom, and lenses leave our sight alone
        lenses = self._deep_lenses_item()
        if lenses is not None:
            return ('deep_lenses', lenses)
        # 0b. nothing pressing: a scroll of charging on an empty wand of digging here too (each zap is one flood roll)
        if not pressed and not blind:
            charge = self._deep_charge_plan(quiet_radius=2)
            if charge is not None:
                return charge
        # 1. blind (a raven's claw, a cobra's spit): a unicorn horn, a few tries per blind spell and per level, not
        # with a hostile next to us in view (a raven there blinds us again on its next claw: dp-on1 jf53 s2 cured
        # 12 times in 40 turns among the ravens)
        # (telepathic only: blind, telepathy is what shows a raven next to us -- without it the 'no hostile adjacent'
        # test is empty and the harness hero (no telepathy) cured among unseen ravens, dh-uhorn 24 -> 21 of 49)
        if blind and self._blind_since is not None and not adjacent and not self._deep_on_scare() and \
                getattr(agent.character, 'telepathic', False):
            horn = self._deep_unihorn_item()
            if horn is not None and \
                    self._deep_unihorn.get(self._blind_since, 0) < jf_config.DEEP_ITEMS_UNIHORN_TRIES and \
                    self._deep_unihorn.get(key, 0) < 2 * jf_config.DEEP_ITEMS_UNIHORN_TRIES:
                return ('deep_unihorn', horn)
        # 2. pressed: a scare instrument (again after HORN_REFRESH turns, or when hurt since the last blow), else an
        # unknown horn (it may be tooled: scare; frost/fire: a ray at the moat) -- not on our scare scroll (nothing
        # there melees us: its hold digs or rests below)
        if pressed and not self._deep_on_scare():
            from . import opp_items
            last = self._deep_blows.get(key)
            due = last is None or bl.time - last > jf_config.HORN_REFRESH or (bl.time > last and self._hurt_since(last + 1))
            if due:
                kit = {}
                for it in agent.inventory.items:
                    kind = opp_items.instrument_kind(agent, it)
                    if kind is not None:
                        kit.setdefault(kind, []).append(it)
                d2 = min(((int(m[1]) - pos[0]) ** 2 + (int(m[2]) - pos[1]) ** 2 for _, m in threats), default=2)
                scare = next((it for it in kit.get('scare', []) if d2 < opp_items.scare_radius2(agent, it)), None)
                if scare is not None:
                    return ('deep_horn', (scare, self._deep_ray_dir(threats)))
                if kit.get('horn') and d2 < 10 * int(bl.experience_level):
                    return ('deep_horn', (kit['horn'][0], self._deep_ray_dir(threats)))
            # a known scroll of taming (not known cursed) with 2+ hostiles next to us: read.c SCR_TAMING tames every
            # monster within 1 (5 confused) that fails resist(SCROLL_CLASS) -- ravens, snakes and eels (MR 0) never
            # resist -- and they fight the rest (a known scroll is read blind too: 'As you pronounce the formula')
            if sum(1 for d, _ in threats if d <= 1) >= 2:
                tame = next((it for it in agent.inventory.items if it.category == nh.SCROLL_CLASS and
                             it.is_unambiguous() and it.object == self._TAMING and it.status != Item.CURSED and
                             'unpaid' not in (it.text or '')), None)
                if tame is not None and not prop.confusion:
                    return ('deep_tame', tame)
        # 3. a known scroll of scare monster on the square we are about to dig (blind and hurt: where we stand) -- on
        # the wet variants (Medusa-1/2 pass 98%: the castle landing's minotaurs need the scroll more, CASTLE_SCARE)
        # or when it's dire there too
        wet = self._medusa_variant_name() in ('medusa-3', 'medusa-4') or key in self._raven_levels
        dire = bl.hitpoints < 0.4 * bl.max_hitpoints
        if not self._deep_on_scare() and (pressed or self._deep_threats(radius=3)) and (wet or dire):
            known, _ = power.scare_scrolls(agent)
            on_stairs = pos in level.stair_destination or level.objects[pos] in G.STAIR_UP or \
                level.objects[pos] in G.STAIR_DOWN
            if known and not on_stairs:
                if blind and hurt:
                    return ('deep_scare', known[0])
                act = self._dig_escape_action()
                if act is not None and act[0] in ('dig', 'raven_gap'):
                    return ('deep_scare', known[0])
        # 4. on our scroll nothing melees us: rest while blind or low (not while something still hurts us: a cobra's
        # spit, a naga's acid), then dig from it
        if self._deep_on_scare():
            _, _, turn = self._deep_scare
            if (blind or bl.hitpoints < jf_config.DEEP_ITEMS_REST_UNTIL * bl.max_hitpoints) and not hurt and \
                    bl.time - turn < jf_config.DEEP_ITEMS_REST_MAX:
                return ('deep_rest', None)
            tool = self.digging_tool()
            if tool is not None and bl.time >= self._dig_blocked_until and \
                    (self._in_own_pit() or self._diggable_spot(pos[0], pos[1], max_wet=8)):
                return ('deep_dig', tool)
        return None

    def _medusa_zap_ready(self):
        """MEDUSA_ZAP_FIRST: on Medusa's level the next DIG_ESCAPE action is WAND_RESERVE's wand of digging zapped down
        where we stand (or MEDUSA_FREEZE's ice right before that zap). No agent step here (a preempt condition)."""
        if not (WAND_RESERVE and DIG_WAND_ESCAPE and self.diving and self.on_medusa_level()) or \
                self._dig_wand() is None:
            return False
        act = self._dig_escape_action()
        return act is not None and act[0] in ('zap', 'freeze')

    def _deep_items_act(self, plan):
        agent = self.agent
        kind, arg = plan
        bl = agent.blstats
        key = agent.current_level().key()
        hp = f'hp {bl.hitpoints}/{bl.max_hitpoints}'
        if kind == 'deep_lenses':
            letter = agent.inventory.items.get_letter(arg)

            def gen():
                if 'What do you want to put on?' not in agent.single_message:
                    return
                yield letter

            with agent.atom_operation():
                agent.step(A.Command.PUTON, gen())
            agent.log(f'DEEP_ITEMS put on {arg.text!r} ({letter}): {(agent.message or "")[:100]!r}, {hp}')
            self._deep_lenses_tried.add(arg.text)
            agent.inventory.items.update(force=True)
            return
        if kind == 'deep_unihorn':
            n = self._deep_unihorn.get(self._blind_since, 0) + 1
            self._deep_unihorn[self._blind_since] = n
            self._deep_unihorn[key] = self._deep_unihorn.get(key, 0) + 1
            letter = agent.inventory.items.get_letter(arg)
            agent.log(f'DEEP_ITEMS applying {arg.text!r} ({letter}) against blindness (try {n}), {hp}')

            def gen():
                if 'What do you want to use or apply?' not in agent.single_message:
                    return
                yield letter

            with agent.atom_operation():
                agent.step(A.Command.APPLY, gen())
            msg = agent.message or ''
            agent.log(f'DEEP_ITEMS unicorn horn -> {msg[:120]!r}')
            if any(s in msg for s in self._CURSED_UNIHORN):
                # apply.c use_unihorn on a cursed horn: a random trouble instead (deathly sick 1 time in 6) -- never
                # again with this one (a prayer fixes the sickness: TROUBLE_SICK is major)
                agent.log(f'DEEP_ITEMS {arg.text!r} is cursed: not applied again')
                self._deep_bad_horns.add(arg.text)
            return
        if kind == 'deep_horn':
            from . import opp_items
            item, ray = arg
            self._deep_blows[key] = bl.time
            g = item.glyphs[0] if item.glyphs else None
            agent.log(f'DEEP_ITEMS blowing {item.text!r} (a ray would go {ray}), {hp}, hostiles '
                      f'{[(m[3].mname, d) for d, m in self._deep_threats()][:4]}')
            res = opp_items.play(agent, item, ray)
            msg = ' '.join(agent._message_history[-4:] + [res.get('msg') or ''])
            if res.get('direction') and g is not None and any(s in msg for s in self._DEEP_FROST_MSG):
                # a frost horn: MEDUSA_FREEZE's cold source from now on (_cold_item), and this direction is done
                self._deep_frost.add(g)
                delta = self._DEEP_DIRS.get(ray)
                if delta is not None:
                    self.__dict__.setdefault('_frozen_dirs', {}).setdefault(
                        (key, agent.blstats.y, agent.blstats.x), set()).add(delta)
                agent.log(f'DEEP_ITEMS {item.text!r} is a frost horn')
            return
        if kind == 'deep_scare':
            pos = (int(bl.y), int(bl.x))
            agent.log(f'DEEP_ITEMS dropping {arg.text!r} at {pos}{" (blind)" if agent.character.prop.blind else ""}, '
                      f'{hp}, hostiles {[(m[3].mname, d) for d, m in self._deep_threats(radius=3)][:4]}')
            # set before the drop: its atomic step ends with the preempt checks (castle B005)
            self._deep_scare = (key, pos, bl.time)
            self._scare_spot = (key, pos)
            self._scare_drop_turn = bl.time
            agent.inventory.drop(arg, 1)
            # never picked up again (a second pickup turns scare monster to dust)
            agent.inventory._note_dropped([arg], [1], force=True)
            return
        if kind == 'deep_rest':
            if getattr(self, '_deep_rest_logged', None) != self._deep_scare:
                self._deep_rest_logged = self._deep_scare
                agent.log(f'DEEP_ITEMS resting on our scroll of scare monster{" (blind)" if agent.character.prop.blind else ""}, {hp}')
            # a scared monster that can't get away stays next to us: strike it from the square (no penalty there,
            # unlike from an Elbereth: uhitm.c u_wipe_engr), never stepping off (as _scare_hold_loop)
            target = next((m for d, m in self._deep_threats(radius=1)), None)
            if target is not None and bl.hitpoints >= 0.5 * bl.max_hitpoints:
                if not agent.wield_best_melee_weapon():
                    agent.melee_attack(target[1], target[2])
                return
            agent.search(1)
            return
        if kind == 'deep_dig':
            agent.log(f'DEEP_ITEMS digging from our scroll of scare monster, {hp}')
            self.dig_with_tool(arg)
            return
        if kind == 'deep_tame':
            letter = agent.inventory.items.get_letter(arg)
            agent.log(f'DEEP_ITEMS reading {arg.text!r} ({letter}), {hp}, adjacent '
                      f'{[m[3].mname for d, m in self._deep_threats(radius=1)]}')

            def gen():
                if 'What do you want to read?' not in agent.single_message:
                    return
                yield letter

            with agent.atom_operation():
                agent.step(A.Command.READ, gen())
            agent.log(f'DEEP_ITEMS taming -> {(agent.message or "")[:140]!r}')
            agent.inventory.items.update(force=True)
            return
        if kind == 'deep_charge':
            scroll, wand = arg
            sl = agent.inventory.items.get_letter(scroll)
            wl = agent.inventory.items.get_letter(wand)
            agent.log(f'DEEP_ITEMS reading {scroll.text!r} ({sl}) on {wand.text!r} ({wl}), {hp}')

            def gen():
                if 'What do you want to read?' not in agent.single_message:
                    return
                yield sl
                for _ in range(6):
                    if 'What do you want to charge?' in agent.single_message:
                        yield wl
                        return
                    misc = agent._observation['misc']
                    if misc[2] and not misc[1]:
                        yield ' '
                        continue
                    return

            with agent.atom_operation():
                agent.step(A.Command.READ, gen())
            msg = ' '.join(agent._message_history[-4:] + [agent.message or ''])
            agent.log(f'DEEP_ITEMS charging -> {msg[-160:]!r}')
            if 'glow' in msg:
                agent.inventory.empty_wands.discard(wand.text)   # read.c recharge: 'Your wand ... glows'
            self._deep_charge_refused.add(wand.text)   # (charged or refused: once per wand)
            agent.inventory.items.update(force=True)
            return
        raise ValueError(kind)

    # ------------------------------------------------------------- branches

    def _walk_through_traps(self, climbing=False):
        """AutoAscend's BFS treats every known trap as a wall (exploration relents only after thousands
        of turns of searching). A dive whose staircase is walled off only by a trap walks through it: an
        s13 dive spent 5,000 turns on Mines level 1 whose '<' lay behind a trap in a 1-wide corridor."""
        agent = self.agent
        if agent._last_turn - agent._allow_walking_through_traps_turn <= 50:
            return False   # already allowed: the traps are not what blocks us
        if climbing:
            # a known trap door is escaped only 1 time in 5 (trap.c): climbing through one dropped that
            # dive back down the Mines twice. First try the way up without trap doors and holes; if that
            # is still cut off next time, the trap door is the only way: take the 1-in-5 chances
            level = agent.current_level()
            falls = utils.isin(level.objects, FALL_TRAPS)
            tries = self._climb_trap_tries.get(level.key(), 0)
            self._climb_trap_tries[level.key()] = tries + 1
            # CLIMB_NO_FALL: never the trap doors and holes (an 80% fall undoes the climb: base-jf26 s14 fell 42 times
            # in 10k turns climbing out of Mines' End, the later calls here unforbidding them each time)
            if tries == 0 or jf_config.CLIMB_NO_FALL:
                if tries and falls.any():
                    agent.log(f'DIVE climbing: {int(falls.sum())} known trap doors/holes stay closed (CLIMB_NO_FALL)')
                level.forbidden |= falls
            else:
                level.forbidden &= ~falls
        agent.log('DIVE stairs cut off: walking through known traps')
        agent._allow_walking_through_traps_turn = agent._last_turn
        agent.last_bfs_step = -1   # the BFS cache ignores the flag
        return True

    def _take_stairs(self, stairs, direction):
        """Reach one of `stairs` and climb it. A peaceful standing on (or next to) the staircase
        makes it 'unreachable' for AutoAscend's BFS; in the peaceful Mines that stranded a dive for
        13k turns until it starved. Approach a neighbour square, wait for the square to clear."""
        agent = self.agent
        pos = (agent.blstats.y, agent.blstats.x)
        if jf_config.STAIR_BOULDER_FIX and self._stairs_blocked:
            key = agent.current_level().key()
            stairs = [p for p in stairs
                      if self._stairs_blocked.get((key, (int(p[0]), int(p[1]))), -1) <= agent.blstats.time]
            if not stairs:
                return False
        if pos in stairs:
            if direction == '>' and self.rest_if_hurt():
                return True
            agent.move(direction)
            return True
        dis = agent.bfs()
        reachable = [p for p in stairs if dis[p] != -1]
        if reachable:
            agent.go_to(*min(reachable, key=lambda p: dis[p]))
            return True
        near = [(self._neighbour_distance(dis, *p), p) for p in stairs]
        near = [(d, p) for d, p in near if d is not None]
        if not near:
            if self._walk_through_traps(climbing=direction == '<'):
                return self._take_stairs(stairs, direction)
            # STALL_SESSILE / STALL_BOULDER: cut off by a mold or floating eye in a corridor, or by boulders
            if self._stall_unblock([(int(p[0]), int(p[1])) for p in stairs], f'stairs {direction}'):
                return True
            return False
        d, (y, x) = min(near)
        if not utils.adjacent(pos, (y, x)):
            agent.go_to(y, x, stop_one_before=True)
            return True
        if agent.monster_tracker.monster_mask[y, x]:
            # STALL_SESSILE: a mold or jelly on the stairs never leaves (cand-g jf49 s10: a yellow mold on the '<' held
            # the XL-7 dive's way to the Mines for 2500 turns until a hunger prayer failed)
            if self._stall_sessile_on_stairs((int(y), int(x))):
                return True
            agent.search()  # a peaceful is on the stairs: wait for it to move
            return True
        if jf_config.STAIR_BOULDER_FIX and int(agent.glyphs[y, x]) in G.BOULDER:
            return self._stair_boulder(pos, (int(y), int(x)))
        if jf_config.ROBUST_FIXES2 and y != pos[0] and x != pos[1] and agent._squeeze_blocked():
            lv = agent.current_level()
            if not lv.walkable[pos[0], x] and not lv.walkable[y, pos[1]]:
                # the last step onto the stairs is a diagonal squeeze the game just refused for our weight: the BFS
                # no longer plans it, but this direct step did, ~15 times a turn (gen-audit ga-e1-jf46 s3: 9374
                # refusals on Mines 1 until the camp's CAMP_STAIRS_TURNS gave the '>' up)
                return False
        agent.move(agent.calc_direction(pos[0], pos[1], y, x))
        return True

    # ------------------------------------------------ grind-audit: dives stuck on a level (STALL_*)
    # cand-g fresh sets (270 pinned games): 10 of 231 planned dives stayed 1500+ turns on Dlvl 1-4, 6 died there after
    # a dive hunger prayer failed or starved (ledger F090). Three of the causes are plain navigation limits: the BFS
    # never enters a boulder square or the square of a monster fight2 won't melee (ONLY_RANGED_SLOW_MONSTERS: molds,
    # floating eyes, gas spores, blobs, jellies -- also remembered while out of sight, SESSILE_MEMORY), so one of them
    # in a corridor or on the stairs is a wall forever; and a '>' behind a secret door is searched for nearest-first.

    # what an XL 7 Valkyrie may hit to clear a way: uhitm.c passive() -- brown mold/blue jelly cold (she resists it),
    # yellow mold stun, green mold/acid blob/spotted jelly acid d(2,4)-d(2,8), red mold fire d(2,4); lichens stick;
    # shriekers and violet fungi hit back weakly. A floating eye (AD_PLYS: a d(2,70)-turn freeze whenever it survives a
    # blow it can see) only blind, blindfolded or by a throw; a gas spore (4d6 blast on death) only at healthy HP with
    # nothing tame or peaceful next to it; a gelatinous cube (paralysis) or ochre jelly (3d6 acid) never.
    _STALL_MELEE = frozenset(('lichen', 'brown mold', 'yellow mold', 'green mold', 'red mold', 'shrieker',
                              'violet fungus', 'acid blob', 'blue jelly', 'spotted jelly'))

    def _stall_open_mask(self):
        """The squares STALL_SESSILE / STALL_BOULDER may clear: hostile sessile monsters (seen or remembered) and
        boulders, less the ones given up for now."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        now = agent.blstats.time
        mask = np.zeros(level.walkable.shape, dtype=bool)
        if jf_config.STALL_SESSILE and not agent.character.prop.hallu:
            sessile = _only_ranged_monsters()
            mons = utils.isin(agent.glyphs, G.MONS) & ~agent.monster_tracker.peaceful_monster_mask
            for y, x in zip(*mons.nonzero()):
                if MON.permonst(agent.glyphs[y, x]).mname in sessile:
                    mask[y, x] = True
            for p in level.sessile:
                mask[p] = True
        if jf_config.STALL_BOULDER and level.dungeon_number != Level.SOKOBAN:
            mask |= utils.isin(agent.glyphs, G.BOULDER)
        for (k, p), until in self._stall_skip.items():
            if k == key and until > now:
                mask[p] = False
        return mask

    def _stall_path(self, dis, target):
        """The squares from us to target along dis (a BFS from our square), walked back without the RNG (agent.path
        shuffles the neighbours: calling it here would change every later random choice of the game)."""
        agent = self.agent
        start = (int(agent.blstats.y), int(agent.blstats.x))
        path = [target]
        cur = target
        while cur != start:
            nxt = None
            for ny, nx in agent.neighbors(cur[0], cur[1], shuffle=False):
                if 0 <= dis[ny, nx] == dis[cur] - 1:
                    nxt = (int(ny), int(nx))
                    break
            if nxt is None:
                return None
            path.append(nxt)
            cur = nxt
        return path[::-1]

    def _stall_unblock(self, targets, what):
        """STALL_SESSILE / STALL_BOULDER: `targets` (stairs we need) are cut off. Open the squares of hostile sessile
        monsters and of boulders, find the first one on the shortest way and clear it: walk next to it, hit the
        monster or push the boulder. Only after the way has been cut off STALL_MIN_TURNS (exploring may still find a
        detour). True: acted this step."""
        if not (jf_config.STALL_SESSILE or jf_config.STALL_BOULDER) or not targets:
            return False
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        now = agent.blstats.time
        since = self._stall_cut.setdefault((key, what), now)
        if now - since < jf_config.STALL_MIN_TURNS:
            return False
        mask = self._stall_open_mask()
        if not mask.any():
            return False
        dis = agent.bfs(open_mask=mask)
        reach = [(int(dis[p]), p) for p in targets if dis[p] != -1]
        if not reach:
            return False
        _, target = min(reach)
        path = self._stall_path(dis, target)
        if path is None:
            return False
        dis_now = agent.bfs()
        i = next((i for i in range(1, len(path)) if mask[path[i]] and dis_now[path[i]] == -1), None)
        if i is None:
            return False
        blocker, before = path[i], path[i - 1]
        # nothing we could clear is no reason to walk there: every step off our course reshuffles the game (ga-b1
        # jf43 s14 walked to a floating eye it had nothing to throw at, and a game the base finished at Dlvl 29 ended
        # on Dlvl 7)
        why = self._stall_unclearable(blocker, before)
        if why is not None:
            self._stall_give_up(key, blocker, why)
            return False
        here = (int(agent.blstats.y), int(agent.blstats.x))
        if here != before:
            if dis_now[before] == -1:
                return False
            if (key, blocker) not in self._stall_tries:
                agent.log(f'STALL {what} on {key} cut off {now - since} turns: clearing {blocker} '
                          f'(glyph {int(agent.glyphs[blocker])}) from {before}')
                self._stall_tries[(key, blocker)] = 0
            agent.go_to(*before)
            return True
        return self._stall_clear(blocker, what)

    def _stall_unclearable(self, p, before):
        """Why the blocker at p can't be cleared from `before` (None: it may be). Checked before walking there."""
        agent = self.agent
        level = agent.current_level()
        bl = agent.blstats
        glyph = int(agent.glyphs[p])
        if glyph in G.BOULDER:
            if not jf_config.STALL_BOULDER:
                return 'boulder (STALL_BOULDER off)'
            ty, tx = 2 * p[0] - before[0], 2 * p[1] - before[1]
            if not (0 <= ty < C.SIZE_Y and 0 <= tx < C.SIZE_X and level.walkable[ty, tx] and
                    int(agent.glyphs[ty, tx]) not in G.BOULDER and not agent.monster_tracker.monster_mask[ty, tx]):
                return f'boulder: nothing free behind it from {before}'
            return None
        if not MON.is_monster(glyph):
            return None   # a remembered sessile square out of sight: go and look
        if agent.monster_tracker.peaceful_monster_mask[p]:
            return 'peaceful'
        name = MON.permonst(glyph).mname
        if name == 'floating eye':
            if agent.character.prop.blind or agent._feye_tool() is not None:
                return None
            if agent.character.prop.polymorph or agent.inventory.get_best_ranged_set()[1] is None:
                return 'a floating eye and no blindfold or missile'
            return None
        if name == 'gas spore':
            near = [(y, x) for y in range(p[0] - 1, p[0] + 2) for x in range(p[1] - 1, p[1] + 2)
                    if 0 <= y < C.SIZE_Y and 0 <= x < C.SIZE_X and
                    (agent.monster_tracker.peaceful_monster_mask[y, x] or int(agent.glyphs[y, x]) in G.PETS or
                     level.shop[y, x])]
            if near or bl.hitpoints < max(40, 0.6 * bl.max_hitpoints) or \
                    level.key() == agent.global_logic.minetown_level:
                return f'gas spore: hp {bl.hitpoints}/{bl.max_hitpoints}, near {near}'
            return None
        if name not in self._STALL_MELEE:
            return f'{name}: not one we hit'
        return None

    def _stall_give_up(self, key, p, why):
        self._stall_skip[(key, p)] = self.agent.blstats.time + 300
        self.agent.log(f'STALL leaving {p} on {key} alone for 300 turns: {why}')
        self.agent.last_bfs_step = -1

    def _stall_clear(self, p, what):
        """Next to the blocking square p: push its boulder or hit its sessile monster. True: acted."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        tries = self._stall_tries.get((key, p), 0)
        if tries >= jf_config.STALL_MAX_TRIES:
            self._stall_give_up(key, p, f'{tries} tries')
            self._stall_tries.pop((key, p), None)
            return False
        self._stall_tries[(key, p)] = tries + 1
        glyph = int(agent.glyphs[p])
        if glyph in G.BOULDER:
            if not jf_config.STALL_BOULDER:
                return False
            # hack.c moverock: the boulder moves one square on in our direction unless rock, a wall, another boulder,
            # a monster or iron bars is there, or the push is diagonal through a doorway (or squeezes past two)
            ty, tx = 2 * p[0] - y0, 2 * p[1] - x0
            beyond_ok = 0 <= ty < C.SIZE_Y and 0 <= tx < C.SIZE_X and level.walkable[ty, tx] and \
                int(agent.glyphs[ty, tx]) not in G.BOULDER and not agent.monster_tracker.monster_mask[ty, tx]
            if not beyond_ok:
                self._stall_give_up(key, p, f'nothing free behind it from {(y0, x0)}')
                return False
            agent.log(f'STALL pushing the boulder at {p} toward {(ty, tx)} ({what})')
            try:
                agent.move(agent.calc_direction(y0, x0, *p))
            except AgentPanic:
                self._stall_give_up(key, p, f'the push from {(y0, x0)} failed: {(agent.message or "")[:70]!r}')
            agent.last_bfs_step = -1
            return True
        if not jf_config.STALL_SESSILE or not MON.is_monster(glyph) or \
                agent.monster_tracker.peaceful_monster_mask[p]:
            return False   # a remembered monster no longer there: the next look forgets it (SESSILE_MEMORY)
        return self._stall_hit(p, MON.permonst(glyph).mname)

    def _stall_hit(self, p, name):
        """STALL_SESSILE: one attack on the sessile monster at p (next to us), if it is safe. True: acted."""
        agent = self.agent
        bl = agent.blstats
        key = agent.current_level().key()
        d = agent.calc_direction(bl.y, bl.x, *p)
        if name == 'floating eye':
            if not agent.character.prop.blind:
                # FEYE_BLIND's safe attack: a blindfold/towel put on for the blow, else a throw where fight2's ranged
                # rules allow one (no pet or peaceful behind it -- a shopkeeper -- and no Watch about)
                agent.log(f'STALL clearing the floating eye at {p}')
                if agent._feye_safe_attack(*p):
                    agent.last_bfs_step = -1
                    return True
                self._stall_give_up(key, p, 'a floating eye and no blindfold or safe throw')
                return False
        elif name == 'gas spore':
            level = agent.current_level()
            near = [(y, x) for y in range(p[0] - 1, p[0] + 2) for x in range(p[1] - 1, p[1] + 2)
                    if 0 <= y < C.SIZE_Y and 0 <= x < C.SIZE_X and (y, x) != (bl.y, bl.x) and
                    (agent.monster_tracker.peaceful_monster_mask[y, x] or int(agent.glyphs[y, x]) in G.PETS or
                     level.shop[y, x])]
            if near or bl.hitpoints < max(40, 0.6 * bl.max_hitpoints) or \
                    agent.current_level().key() == agent.global_logic.minetown_level:
                self._stall_give_up(key, p, f'gas spore: hp {bl.hitpoints}/{bl.max_hitpoints}, near {near}')
                return False
        elif name not in self._STALL_MELEE:
            self._stall_give_up(key, p, f'{name}: not one we hit')
            return False
        elif bl.hitpoints < 0.5 * bl.max_hitpoints:
            return False   # the rest comes first (plan_step rests below REST_BELOW with nothing in view)
        agent.log(f'STALL hitting the {name} at {p}')
        with agent.atom_operation():
            agent.step(A.Command.FIGHT)
            agent.direction(d)
        agent.last_bfs_step = -1
        return True

    def _stall_sessile_on_stairs(self, p):
        """STALL_SESSILE: the monster on the staircase next to us is a hostile sessile one: it never leaves."""
        agent = self.agent
        if not jf_config.STALL_SESSILE or agent.monster_tracker.peaceful_monster_mask[p] or \
                agent.character.prop.hallu:
            return False
        glyph = int(agent.glyphs[p])
        if not MON.is_monster(glyph) or MON.permonst(glyph).mname not in _only_ranged_monsters():
            return False
        key = agent.current_level().key()
        now = agent.blstats.time
        if self._stall_skip.get((key, p), -1) > now:
            return False
        since = self._stall_cut.setdefault((key, ('on stairs', p)), now)
        # a mold or jelly (speed 0) never leaves; a floating eye, gas spore, lichen or blob drifts off in time (ga-b2
        # jf48 s0 hit a floating eye on Dlvl 3's '>' after 20 turns and the game, which the base had waited out, ended
        # on Dlvl 6)
        wait = jf_config.STALL_STAIRS_WAIT if getattr(MON.permonst(glyph), 'mmove', 1) == 0 else \
            jf_config.STALL_STAIRS_WAIT_MOVER
        if now - since < wait:
            return False
        tries = self._stall_tries.get((key, p), 0)
        if tries == 0:
            agent.log(f'STALL a {MON.permonst(glyph).mname} on the stairs at {p} for {now - since} turns: clearing it')
        return self._stall_clear(p, 'stairs')

    def _stall_descend(self):
        """descend() found no way down. STALL_SESSILE/STALL_BOULDER: a known main-line '>' cut off -> clear the way.
        STALL_DOWNSEARCH: no '>' ever seen here after STALL_DOWNSEARCH_TURNS of this dive -> search the walls next to
        the largest unseen region (_search_toward_void, HIDDEN_SEARCH2's search) instead of AutoAscend's nearest-
        first wall search: cand-g jf55 s12 never saw Dlvl 2's '>' (the east half of the map blank) in 9000 grind
        and 6000 dive turns. True: acted."""
        agent = self.agent
        level = agent.current_level()
        downs = [(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_DOWN).nonzero())
                 if self._stairs_ok(level, y, x)]
        if downs:
            return self._stall_unblock(downs, 'stairs >')
        # shallow levels only: below Dlvl 10 the maze levels, Medusa's and the castle (no '>' at all) have their own logic
        # (ga-b1 jf43 s1 fired on Dlvl 28)
        if not jf_config.STALL_DOWNSEARCH_TURNS or level.dungeon_number != Level.DUNGEONS_OF_DOOM or \
                agent.blstats.depth > jf_config.STALL_DOWNSEARCH_MAX_DEPTH or agent.character.prop.blind:
            return False
        now = agent.blstats.time
        since = self._stall_cut.setdefault((level.key(), 'no >'), now)
        if now - since < jf_config.STALL_DOWNSEARCH_TURNS:
            return False
        if (level.key(), 'no >') not in self._stall_tries:
            self._stall_tries[(level.key(), 'no >')] = 0
            agent.log(f'STALL no down stairs seen on {level.key()} after {now - since} dive turns: searching toward '
                      f'the unseen part of the level')
        return self._search_toward_void(lambda: bool(self.down_targets()), since, 0)

    _BOULDER_STUCK = ('in vain', 'Perhaps that', 'monster behind', 'You try to move', 'carrying too much to get through',
                      'will not fit')

    def _stair_boulder(self, pos, b):
        """STAIR_BOULDER_FIX (B009): a boulder sits on the staircase we are next to. The BFS never enters a boulder
        square, so _take_stairs pushed it from here; with a wall or rock behind it the push fails without a turn
        passing ('You try to move the boulder, but in vain.', hack.c moverock) and the dive spun on it (jf14 s4,
        Mines 2: the '<' at (16,50) under a boulder walled in on three sides, ~300k steps with a pick-axe and a
        wand of digging in the pack, then a soldier ant; 0.075 where the seed reached Dlvl 28 without it). The
        first push is tried as before; once one fails from a square: break the boulder -- dig.c: a pick-axe or
        mattock applied at it ('You start hitting the boulder.' ... 'The boulder falls apart.' into rocks, a few
        turns for a dwarf), zap.c: a wand of striking fractures it -- else push from another side with open floor
        behind it, else leave these stairs alone for STAIR_BOULDER_WAIT turns. The wand of digging does nothing to
        a boulder (zap_dig digs terrain only). True: acted this step."""
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        t = agent.blstats.time
        # a watchdog for whatever else might keep us here without a turn passing
        spin = self._boulder_spin
        if spin is not None and spin[:3] == (key, b, t):
            self._boulder_spin = (key, b, t, spin[3] + 1)
            if spin[3] + 1 >= 12:
                return self._stairs_give_up(key, b, 'no game time passes')
        else:
            self._boulder_spin = (key, b, t, 1)
        if (key, b, pos) not in self._boulder_push_failed:
            try:
                agent.move(agent.calc_direction(pos[0], pos[1], *b))
            except AgentPanic:
                msg = agent.message or ''
                if any(w in msg for w in self._BOULDER_STUCK) or (agent.blstats.y, agent.blstats.x) == pos:
                    self._boulder_push_failed.add((key, b, pos))
                    agent.log(f'DIVE boulder on the stairs at {b} does not move from {pos}: {msg[:80]!r}')
                else:
                    raise
            return True
        # not in Sokoban: breaking a boulder there costs Luck (zap.c fracture_rock -> change_luck(-1)), and Luck < 0
        # fails every prayer
        soko = level.dungeon_number == Level.SOKOBAN
        tool = None if soko else self.digging_tool()
        if tool is not None and not agent.character.prop.polymorph and t >= self._dig_blocked_until:
            self._dig_at_boulder(tool, b)
            return True
        wand = None if soko else next((i for i in flatten_items(agent.inventory.items) if i.is_wand() and
                                       i.is_unambiguous() and i.object == O.from_name('striking', nh.WAND_CLASS) and
                                       not agent.inventory.is_known_empty(i)), None)
        if wand is not None:
            agent.log(f'DIVE zapping {wand.text!r} at the boulder on the stairs at {b}')
            agent.zap(wand, agent.calc_direction(pos[0], pos[1], *b))
            agent.last_bfs_step = -1
            return True
        # push it from a side with open floor behind it (hack.c moverock: not rock, a wall, iron bars, another
        # boulder, or a doorway entered diagonally)
        dis = agent.bfs()
        best = None
        for sy, sx in agent.neighbors(*b, shuffle=False):
            s_ = (int(sy), int(sx))
            if (key, b, s_) in self._boulder_push_failed or (s_ != pos and dis[s_] == -1):
                continue
            ty, tx = 2 * b[0] - s_[0], 2 * b[1] - s_[1]
            if not (0 <= ty < C.SIZE_Y and 0 <= tx < C.SIZE_X) or not level.walkable[ty, tx] or \
                    int(agent.glyphs[ty, tx]) in G.BOULDER or \
                    (ty != b[0] and tx != b[1] and utils.isin(level.objects[ty:ty + 1, tx:tx + 1], G.DOORS).any()):
                continue
            d = 0 if s_ == pos else int(dis[s_])
            if best is None or d < best[0]:
                best = (d, s_)
        if best is not None:
            s_ = best[1]
            if s_ != pos:
                agent.log(f'DIVE boulder on the stairs at {b}: pushing it from {s_} instead')
                agent.go_to(*s_)
            else:
                try:
                    agent.move(agent.calc_direction(pos[0], pos[1], *b))
                except AgentPanic:
                    self._boulder_push_failed.add((key, b, pos))
            return True
        return self._stairs_give_up(key, b, 'no digging tool, no striking wand, no side to push from')

    def _stairs_give_up(self, key, b, why):
        agent = self.agent
        self._stairs_blocked[(key, b)] = agent.blstats.time + jf_config.STAIR_BOULDER_WAIT
        agent.log(f'DIVE stairs at {b} blocked by a boulder ({why}): leaving them for '
                  f'{jf_config.STAIR_BOULDER_WAIT} turns')
        return False

    def _dig_at_boulder(self, tool, b):
        agent = self.agent
        shield = agent.inventory.items.off_hand
        if tool.object == O.from_name('dwarvish mattock') and shield is not None:
            # a mattock needs both hands: the shield comes off first (as for digging down)
            agent.log(f'DIVE taking off {shield.text!r} to break a boulder with a mattock')
            agent.inventory.takeoff(shield)
            return
        direction = agent.calc_direction(agent.blstats.y, agent.blstats.x, *b)
        agent.log(f'DIVE breaking the boulder on the stairs at {b} with {tool.text!r} ({direction})')
        with agent.atom_operation():
            tool = agent.inventory.move_to_inventory(tool)
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(tool))
            prompted = 'In what direction do you want to dig?' in agent.single_message
            if prompted:
                agent.direction(direction)
            elif agent.single_message.startswith('In what direction'):
                agent.step(A.Command.ESC)
        msg = agent.message or ''
        agent.log(f'DIVE boulder dig: {msg[:100]!r}')
        if not prompted:
            # a welded weapon, a web, ...: not again for a while (the other ways get their turn)
            self._dig_blocked_until = agent.blstats.time + 100
        agent.last_bfs_step = -1

    def return_to_main_dungeon(self):
        """Back to the main line from a side branch: up out of the Mines, down out of Sokoban."""
        agent = self.agent
        level = agent.current_level()
        stairs = G.STAIR_DOWN if level.dungeon_number == Level.SOKOBAN else G.STAIR_UP
        direction = '>' if level.dungeon_number == Level.SOKOBAN else '<'
        exits = list(zip(*utils.isin(level.objects, stairs).nonzero()))
        if RETURN_FIX and direction == '<':
            # the stairs we climbed before are recorded with their destination even when the map lost the '<'
            # glyph (base3-jf14 s7 searched Mines 2 for its '<' for 9000 turns with a pick-axe in the pack)
            here = level.level_number
            known = {(int(y), int(x)) for y, x in exits}
            for p, d in level.stair_destination.items():
                if d is not None and ((d[0][0] == level.dungeon_number and d[0][1] == here - 1) or
                                      (level.dungeon_number == Level.GNOMISH_MINES and here == 1 and
                                       d[0][0] == Level.DUNGEONS_OF_DOOM)):
                    if (int(p[0]), int(p[1])) not in known:
                        logged = self.__dict__.setdefault('_return_logged', set())
                        if (level.key(), p) not in logged:
                            logged.add((level.key(), p))
                            agent.log(f'RETURN up stairs at {p} known from their destination {d[0]}')
                        exits.append((int(p[0]), int(p[1])))
        if exits and self._take_stairs(exits, direction):
            return
        self.exploration(None).until(agent, self._budgeted(lambda: any(
            agent.bfs()[p] != -1 for p in zip(*utils.isin(agent.current_level().objects,
                                                           stairs).nonzero())))).run()

    def leave_quest(self):
        """Home 1 is banked on arrival; walk back through the portal and keep diving."""
        agent = self.agent
        level = agent.current_level()
        portals = [(y, x) for y, x in zip(*utils.isin(level.objects, PORTAL).nonzero())]
        if not portals and self.quest_arrival is not None:
            portals = [self.quest_arrival]
        if not portals:
            self.exploration(None).until(agent, lambda: utils.isin(agent.current_level().objects,
                                                                   PORTAL).any()).run()
            return
        y, x = portals[0]
        pos = (agent.blstats.y, agent.blstats.x)
        if pos == (y, x):
            # standing on the arrival portal: step off, then back on
            dis = agent.bfs()
            for ny, nx in agent.neighbors(y, x):
                if dis[ny, nx] == 1 and not agent.monster_tracker.monster_mask[ny, nx]:
                    with agent.atom_operation():
                        agent.direction(agent.calc_direction(y, x, ny, nx))
                    return
            agent.search()
            return
        if not utils.adjacent(pos, (y, x)):
            agent.go_to(y, x, stop_one_before=True)
            return
        self.step_onto(y, x, 'magic portal')

    # --------------------------------------------------------- portal sweep

    def should_sweep_portal(self):
        agent = self.agent
        key = agent.current_level().key()
        if self.visited_quest or self.portal_level != key or key in self.sweep_given_up:
            return False
        if not SWEEP_WITH_TOOL and self.digging_tool() is not None:
            return False
        if self.sweep_started is None:
            self.sweep_started = agent.blstats.time
        if agent.blstats.time - self.sweep_started > PORTAL_SWEEP_TURNS:
            agent.log('DIVE portal sweep out of budget')
            self.sweep_given_up.add(key)
            return False
        return True

    def portal_candidates(self):
        """Unvisited floor squares of rooms that can hold the portal.

        mklev's find_branch_room puts a branch portal on a free ROOM square of an
        ordinary room that holds neither staircase (when the level has > 2 rooms).
        """
        agent = self.agent
        level = agent.current_level()
        objs = level.objects
        floor = utils.isin(objs, ROOM_FLOOR)
        furniture = utils.isin(objs, G.STAIR_UP, G.STAIR_DOWN, G.ALTAR, G.FOUNTAIN, G.TRAPS)
        # squares showing an item (terrain never seen) next to room floor are room floor too
        unknown = level.walkable & (objs == -1)
        unknown &= utils.dilate(floor, radius=1, with_diagonal=False)
        roomish = floor | furniture | unknown
        labels, n = ndimage.label(roomish)
        cand = (floor | unknown) & ~level.was_on & ~level.shop
        if n > 2:
            for y, x in zip(*utils.isin(objs, G.STAIR_UP, G.STAIR_DOWN).nonzero()):
                if labels[y, x]:
                    cand &= labels != labels[y, x]
        return cand

    def portal_sweep(self):
        agent = self.agent
        level = agent.current_level()
        portals = list(zip(*utils.isin(level.objects, PORTAL).nonzero()))
        if portals:
            y, x = portals[0]
            if not utils.adjacent((agent.blstats.y, agent.blstats.x), (y, x)):
                dis = agent.bfs()
                if not any(dis[ny, nx] != -1 for ny, nx in agent.neighbors(y, x)):
                    # seen but walled off for now (a jf18 dive asserted 'no reachable neighbor' 3486
                    # times): uncover a way there, or give the sweep up on this level
                    if self.exploration(0).run(return_condition=True):
                        return
                    agent.log('DIVE portal in view but unreachable: sweep given up here')
                    self.sweep_given_up.add(level.key())
                    return
                agent.go_to(y, x, stop_one_before=True)
                return
            self.step_onto(y, x, 'magic portal')
            return

        # rooms first: the portal is on a room square, so finish uncovering the map
        if self.exploration(0).run(return_condition=True):
            return

        dis = agent.bfs()
        cand = self.portal_candidates() & (dis > 0)
        if not cand.any():
            # astra: 2 of 3 portal rooms were closed off (hidden door, locked closet). With every
            # known room swept, search for hidden passages (explore1 with searching) until new
            # candidates appear or the sweep budget runs out.
            if not getattr(self, '_sweep_searching', False):
                agent.log('DIVE portal sweep: known rooms swept, searching for hidden rooms')
                self._sweep_searching = True
            self.exploration(None).until(agent, lambda: bool(
                (self.portal_candidates() & (agent.bfs() > 0)).any() or
                utils.isin(agent.current_level().objects, PORTAL).any())).run()
            return
        ys, xs = cand.nonzero()
        i = int(np.argmin(dis[ys, xs]))
        agent.go_to(ys[i], xs[i])
