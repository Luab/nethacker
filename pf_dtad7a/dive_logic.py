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

from . import jf_config, jf_log, power, utils, valley
from .castle_logic import CastlePassage
from .character import Character
from .exceptions import AgentPanic
from .glyph import G, MON, SS, Hunger
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
MINES_BRANCH_MAX_DEPTH = 4     # the Mines branch staircase is on Dlvl 2-4
MINES_MIN_LEVELS = 8           # dungeon.def: the Mines have 8-9 levels, Mines' End is the last
# XP gate inside the Mines: before going to Mines level k, explore the current level fully while
# XL < MINES_REQUIRED_XL[k] (hostile orcs/ants there are the XP). Empty = no gate.
MINES_REQUIRED_XL = {}
# astra: retreat onto Elbereth at 45-65% HP, rest there with searches, never attack from it
# hand-over from AutoAscend's levelling tour to the dive
DIVE_XL = 8
DIVE_TURN = 10 ** 9
ELBERETH_REST_BELOW = 0.4
ELBERETH_REST_UNTIL = 0.85
# breathers, spitters and casters: Elbereth doesn't stop them hurting you from a distance
LAWFUL_MINIONS = ('Aleax', 'Angel', 'couatl', 'ki-rin', 'Archon')
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
# DIG_TRY_FIX: DIG_MAX_TRIES counts only applies that weren't interrupted. On jf27 s3 Dlvl 9 a bat interrupted
# all 20 applies ("You stop digging"), and a diggable level was marked undiggable.
# ON (train 2): dive-safety A2
DIG_TRY_FIX = True
DIG_MAX_APPLIES = 80           # hard cap on all applies on one level (interrupted ones included)
DIG_WALK_RADIUS = 6            # DIG_ESCAPE walks this far (BFS steps) to a square it can dig
MEDUSA_REROLL_WET = 2          # stranded where every square has this many moat neighbours: climb and re-fall
MEDUSA_REROLLS = 3
# DIG_WAND_ESCAPE (with DIG_ESCAPE): a known wand of digging, zapped down, holes the floor at once (zap.c
# zap_dig -> dighole: one flood roll, no pit phase). Used where the pick-axe can't work -- an adjacent @, or
# bitten with no Elbereth to be had -- and on Medusa's wet islands. (power: 4% of XL-8 castle arrivals carry
# one, 13% at XL 10; try_dig_down only ever zapped it with no pick-axe in the pack.)
# ON (train 2): dive-safety A2
DIG_WAND_ESCAPE = True
# Resting to 95% on a deep level lets its monsters come to us (an s7 dig-dive rested for 150 turns on
# Dlvl 15 until a leocrotta took it to 2 HP): with a digging tool, rest only below this.
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
# hypothesis: a Tourist starts with 4 identified scrolls of magic mapping that the bot never reads; a dive
# without a digging tool wanders each unexplored level at XL 8 looking for '>' until something kills it
# (fem s10/s11 died so on Dlvl 3-8). Reading one on arriving at a level whose '>' isn't in view (after
# MAP_STUCK_TURNS) puts '>' on the map and the descent walks straight to it, cutting the most dangerous
# exposure of the dive. Not below DIVE_XL: an early rescue dive gains the levels it needs while it searches.
# sources: https://nethackwiki.com/wiki/Tourist, https://nethackwiki.com/wiki/Scroll_of_magic_mapping,
#          /refs/top/1c4099e80253 (its stair search: descend() explores until down_targets appears)
MAP_WHEN_STUCK = True
MAP_STUCK_TURNS = 0
MAGIC_MAPPING = O.from_name('magic mapping', nh.SCROLL_CLASS)
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
DIG_DIVE_XL = 8
KEEP_TOOL_IN_TOUR = False
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
GRIND_HUNT_XL = 0
GRIND_HUNT_DIGGERS_ONLY = False
# the tool-less dive's Mines route explores Dlvl 2-4 only until the branch '>' is known (see _search_branch);
# the tour-mode pick trip (jf_config.PICK_TRIP_XL) too, through trip_branch_strategy
FAST_BRANCH = False
BRANCH_HIDDEN_TURNS = 500          # per level (0: off), see _search_branch
# Walk up to peaceful dwarves we see but can't reach: a digger inside its fresh tunnel (dark corridor squares
# aren't on our map) was dropped as unreachable. Crash-heard Mines levels gave 9 tools in 30 baseline visits,
# the rest 6 in 179, yet jf27/14 heard 53 crashes on Mines 2 in 3840 turns without one attack. Seeing the
# dwarf along a straight line proves the squares between aren't rock: they are marked walkable.
APPROACH_DWARVES = False
APPROACH_STEPS = 60                # approach moves per APPROACH_WINDOW turns on a level
APPROACH_WINDOW = 500
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
# a digger inside a shop digs through its floor when it owes nothing (see _trapped_in_shop)
SHOP_DIG = True
SHOP_DIG_WAIT = 300
MINES_CAMP = True
MINES_CAMP_TURNS = 8000
# The dive's first job on the level it starts from: a tool-less dive whose grind level held a digging dwarf
# (seen in rock, or 'You hear crashing rock.', within HOME_DIGGER_WINDOW turns) searches that level for it first,
# HOME_DIGGER_TURNS at most (0: off). Dwarves spawn on Dlvl 1 from XL 7; 51 of 90 baseline grinds heard one
# there, and base2-public s1/s8/s14 (tool-less dives) all had one on Dlvl 1 in the old base; hunt-v2 jf16/12
# watched a digger on Dlvl 1 for 2000 turns, then dove without a tool. Unlike GRIND_HUNT_XL it leaves the tour alone.
HOME_DIGGER_TURNS = 0
HOME_DIGGER_WINDOW = 6000
MINES_CAMP_MAX_LEVEL = 4
CAMP_VISIT_TURNS = 400
CAMP_STAIRS_TURNS = 1000         # looking for the way on to the next camp level before turning around
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
            if not jf_config.HOLD_LOOP or agent.step_count == steps:
                return
            gen = func(self, *args, **kwargs)
            if not next(gen):
                return
    return wrapper


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
            if not jf_config.HOLD_LOOP or agent.step_count == steps:
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
        self._mapped = set()           # level keys a scroll of magic mapping was read on (MAP_WHEN_STUCK)
        self.fully_explored = set()    # level keys explored to exhaustion
        self.sweep_started = None      # turn the current portal sweep began
        self.sweep_given_up = set()    # portal level keys whose sweep ran out of budget
        self._last_key = None
        self._last_task = None
        self.mines_done = False        # reached the bottom of the Mines, or gave the route up
        self._elbereth_resting = False
        self.diving = False
        self.rescue = False                # the dive began as a rescue from a failed Dlvl 1 grind
        self.pick_trip = False             # the grind's detour to the Mines for a pick-axe (PICK_TRIP_XL)
        self.undiggable = set()            # level keys where the floor is too hard to dig
        self._hp_history = []              # (turn, hp) of the last few turns
        self._status_logged = -1
        self._murder_turn = -1
        self._demon_vigil_until = -1       # turn until which a released water demon is kept off with Elbereth
        self._faint_start = None           # turn the current faint began (last awake observation)
        self._guard_weak_since = None      # turn the current Weak spell began (FAINT_GUARD_IDLE)
        self._guard_hold_until = -1        # keep holding the faint guard's Elbereth until this turn (IDLE)
        self._last_update_turn = 0
        self.pet_seen = {}                 # level key -> last turn a pet glyph was in view
        self._last_pos = None              # (level key, (y, x)) at the previous update
        self._arrived = None               # (level key, turn) of the last stairs arrival
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
        self._valley_resting = False       # resting in the Valley until VALLEY_REST_UNTIL
        self.castle = CastlePassage(self)  # castle_logic.py (jf_config.CASTLE_PASSAGE)
        self._dwarf_seen = (None, [])      # (level key, [(y, x)]) of dwarf glyphs at the last update
        self._diggers = {}                 # level key -> {(y, x): turn} where a dwarf was seen digging
        self._crash_turn = {}              # level key -> last turn 'You hear crashing rock.'
        self._align_est = None             # estimated alignment record (ALIGN_BUDGET)
        self._approach_count = {}          # level key -> (window start turn, approach moves in that window)
        self._town_levels = set()          # Mines level keys where the Watch, a shopkeeper or a priest was seen
        self._branch_hidden = False        # FAST_BRANCH: Dlvl 2-4 explored without the branch, searching walls
        self._branch_search_start = {}     # level key -> turn the wall search for the hidden branch began there
        self._branch_trap_walk = set()     # level keys explored once more past their known traps
        self._branch_skip = set()          # (level key, (y, x)) candidate branch '>' we could not reach
        self._branch_reach = {}            # (level key, (y, x)) -> turn we began trying to reach it
        self._camp_start = None            # MINES_CAMP: turn the camp began
        self._camp_over = False
        self._camp_dir = 1                 # sweep direction: +1 down, -1 up
        self._camp_visit = None            # (level key, turn we arrived) of the current camp visit
        self._camp_moves = {}              # (level key, direction) -> turn we began trying to leave that way
        self._boxed_since = None           # (level key, turn) we have been boxed in by peacefuls since
        self._home_search = {}             # level key -> turn the dive began searching it for its digger
        self.medusa_level = None           # level key of Medusa's level once seen (see MEDUSA_WET_SQUARES)
        self._pit_at = None                # (level key, (y, x)) of the pit we dug and still stand in
        self._dig_applies = {}             # level key -> all pick-axe applies (DIG_TRY_FIX)
        self._max_wet_cache = None         # (turn, level key, max_wet) for _dig_max_wet
        self._hurt_on_elbereth = -1        # last turn HP fell while we stood on an intact Elbereth
        self._medusa_rerolls = 0           # climbs off a wet Medusa islet to fall in again elsewhere
        self._dig_walk_blocked_until = -1  # turn until which DIG_ESCAPE doesn't walk to a dig square
        self._medusa_reroll_blocked_until = -1
        self._raven_levels = set()         # Medusa's level key once ravens were seen there (Medusa-3)
        self._fed_wait_start = None     # DIVE_FED: turn the grind first reached its end XL
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
        if key == self.medusa_level and key not in self._raven_levels:
            if DiveLogic.RAVEN is None:
                DiveLogic.RAVEN = MON.from_name('raven')
            if utils.isin(agent.glyphs, [DiveLogic.RAVEN]).any():
                self._raven_levels.add(key)   # Medusa-3 (medusa.des: 30 hostile ravens)
                agent.log('DIVE Medusa-3 (ravens)')
        if agent.blstats.hunger_state >= Hunger.FAINTING:
            if agent._fainting_since is None:
                agent._fainting_since = turn
        else:
            agent._fainting_since = None
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
        if 'You faint from lack of food' in msg and self._faint_start is None:
            self._faint_start = self._last_update_turn
        if 'You regain consciousness' in msg and self._faint_start is not None:
            # moves per turn: a Valkyrie is intrinsically Fast from XL 7 (16 speed on average, 4/3), not before
            # (counting 4/3 at XL 1-6 read hunger 1.3x too low and fired deadline prayers ~800 turns early)
            speed = 4 / 3 if agent.blstats.experience_level >= 7 else 1
            moves = (turn - self._faint_start) * speed
            agent._faint_measure = (turn, (10 - moves) * 10)
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
        if MINETOWN_GUARD and level.dungeon_number == Level.GNOMISH_MINES and key not in self._town_levels and \
                utils.isin(agent.glyphs, self._town_glyphs()).any():
            self._town_levels.add(key)
            agent.log(f'MINETOWN people in view on {key}: no dwarf hunting here')
        if key not in self.level_first_turn:
            self.level_first_turn[key] = turn
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
        if self.in_valley():
            # every step, not just when the plan runs: the retreat and fight2 need the '<' and the layout from
            # the first turn (three valley-x10 games were swarmed at the landing spot and never planned a step)
            self._valley_know_map(level)
            self.valley_progress()

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
        xl_trigger = xl >= DIVE_XL or (xl >= self._min_xl(DIG_DIVE_XL) and self.digging_tool() is not None)
        if xl_trigger and gl.milestone == Milestone.BE_ON_FIRST_LEVEL and not self.fed_for_dive():
            xl_trigger = False   # DIVE_FED: finish the hunger cycle on Dlvl 1 first
        if xl_trigger or gl.milestone >= Milestone.GO_DOWN or agent.blstats.time >= DIVE_TURN or \
                rescue or late_rescue or planned:
            tag = ', rescue' if rescue else ', late rescue' if late_rescue else ', early' if planned else ''
            agent.log(f'DIVE phase starts (milestone {gl.milestone.name}{tag})')
            self.diving = True
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

        # astra guard.py: don't walk on below 2/3 HP; rest while nothing hostile is in view
        # (not a Gehennom digger: see GEHENNOM_DIG_REST_BELOW)
        bl = agent.blstats
        digger = DIVE_REST and self._digger_here()
        rest_below = DIG_REST_BELOW if digger else REST_BELOW
        if bl.hitpoints < rest_below * bl.max_hitpoints and not agent.get_visible_monsters() and \
                bl.hunger_state < Hunger.WEAK and not (digger and self._in_own_pit()) and \
                not self._gehennom_digger():
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

        if self.should_read_mapping():
            self._task('read magic mapping')
            return self.read_mapping()

        if self.should_sweep_portal():
            self._task('portal sweep')
            return self.portal_sweep()

        if self.should_explore_fully():
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
                # a candidate '>' took us below Dlvl 4 (it was the main one): back up, to the other candidate or to
                # finish the hidden-branch search. go_to_mines found no path: the '<' we arrived on isn't on our map
                # until we step off it (smoke public/0: Dlvl 4 had two '>', the first was the main one, and the
                # Mines route was given up on arrival). Its destination is recorded, though.
                ups = set(zip(*utils.isin(level.objects, G.STAIR_UP).nonzero()))
                ups |= {p for p, d in level.stair_destination.items()
                        if d is not None and d[0] == (Level.DUNGEONS_OF_DOOM, level.level_number - 1)}
                if FAST_BRANCH and dnum == Level.DUNGEONS_OF_DOOM and \
                        agent.blstats.depth == MINES_BRANCH_MAX_DEPTH + 1 and ups and \
                        (self._branch_search_open() or self._mines_branch_target() is not None) and \
                        self._take_stairs([(int(y), int(x)) for y, x in ups], '<'):
                    return
                agent.log('DIVE mines branch not found above; giving the Mines route up')
                self.mines_done = True

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
        # lawful minions (is_lminion: Aleax, couatl, ki-rin, Archon) and Angels ignore it too (monmove.c onscary)
        return cls == MON.S_HUMAN or name in ('minotaur',) + LAWFUL_MINIONS

    def on_medusa_level(self):
        return self.medusa_level is not None and self.agent.current_level().key() == self.medusa_level

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

    @Strategy.wrap
    @_hold_loop
    def elbereth_rest(self):
        agent = self.agent
        bl = agent.blstats
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
        if len(near) == 1 and getattr(near[0][3], 'mlevel', 99) <= 2 and bl.hitpoints >= 6:
            self._elbereth_resting = False
            yield False
        if not near or any(self._ignores_elbereth(m[3]) for m in near) or \
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
        if engraving != 'elbereth':
            agent.engrave('Elbereth')
            return
        agent.search()

    WATER_DEMON = None
    RAVEN = None

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
        if not jf_config.FAINT_GUARD or (self.diving and not rescue_guard) or bl.hunger_state < Hunger.WEAK or \
                (agent.prayer_failed and not rescue_guard) or \
                agent.current_level().dungeon_number == GEHENNOM or agent.character.prop.blind or \
                agent.edible_carried_food():
            yield False
        # a vault guard must be answered and followed (see faint_shelter)
        if utils.isin(agent.glyphs, G.GUARD).any():
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
            bl.time - self._guard_weak_since >= jf_config.FAINT_GUARD_IDLE_WEAK
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
        if engraving != 'elbereth':
            agent.log(f'FAINT guard ({"Fainting" if fainting else "Weak"}{", idle" if idle else ""}): Elbereth vs '
                      f'{[m[3].mname for m in near]} hp={bl.hitpoints}/{bl.max_hitpoints}')
            agent.engrave('Elbereth')
            return
        # one turn at a time while Fainting: a faint interrupting a counted search is read as a longer faint by
        # the faint-length hunger estimate (dive.update), ~30 nutrition too low at XL 7 (grind-food)
        agent.search(1 if near or fainting else 3)

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
        if len(near) >= CROWD_SIZE or any(getattr(m[3], 'mname', '') in BOSS_MONSTERS for m in near):
            return above
        return None

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
            self._arrived = None
            agent.log(f'RETREAT crowded arrival: {[m[3].mname for m in self._near_hostiles(CROWD_RADIUS)]}')
        level = agent.current_level()
        if level.dungeon_number == Level.SOKOBAN or bl.depth <= 1:
            yield False
        dis = agent.bfs()
        # fleeing across the level from a faster monster only hands it free hits: on fast HP loss
        # (without low HP) take the stairs only if they are a step or two away
        reach = RETREAT_MAX_DISTANCE if (crowd is not None or bl.hitpoints < RETREAT_BELOW * bl.max_hitpoints) else 2
        ups = [p for p in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())
               if 0 <= dis[p] <= reach]
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
        return MINES_ROUTE and not self.mines_done and \
            self.agent.character.race in (Character.DWARF, Character.GNOME) and \
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
        if level.key() not in self.fully_explored and self.turns_on_level() <= FULL_EXPLORE_TURNS:
            self._task('search mines branch')
            if self.exploration(0).until(agent, found).run(return_condition=True):
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
        # every candidate level is explored and none showed the branch
        if not self._branch_hidden:
            agent.log('DIVE Mines branch hidden: searching Dlvl 4-2 for it')
            self._branch_hidden = True
        t0 = self._branch_search_start.setdefault(level.key(), agent.blstats.time)
        if agent.blstats.time - t0 < BRANCH_HIDDEN_TURNS:
            self._task('search hidden mines branch')
            self.exploration(None).until(agent, lambda: found() or
                                         agent.blstats.time - t0 >= BRANCH_HIDDEN_TURNS).run()
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
                return False
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
        digger = DIVE_REST and self.diving and self.digging_tool() is not None
        # a digger takes stairs like a hole: a deep rest to 95% at XL 8 (1 HP per 5 turns) lets the level's
        # monsters come (base-jf25 s13 rested 180 turns at a Dlvl 14 '>' and died there)
        threshold = DIG_REST_BELOW if digger else REST_BEFORE_DESCEND
        if agent.blstats.hitpoints >= threshold * agent.blstats.max_hitpoints:
            return False
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

    def required_xl(self, depth):
        return REQUIRED_XL.get(depth, REQUIRED_XL[max(REQUIRED_XL)])

    def should_explore_fully(self):
        agent = self.agent
        key = agent.current_level().key()
        if key in self.fully_explored:
            return False
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
        if self.in_valley():
            yield False
        monsters = agent.get_visible_monsters()
        if not monsters:
            yield False   # nothing to run from: the dive plan digs as usual
        if DIG_ESCAPE and level.dungeon_number != GEHENNOM:
            action = self._dig_escape_action()
            # the portal level is the plan's business
            if action is None or self.should_sweep_portal():
                yield False
            yield True
            what, arg = action
            if what == 'reroll':
                self._medusa_reroll(arg)
                return
            if what == 'zap':
                key = level.key()
                spot = (agent.blstats.y, agent.blstats.x)
                agent.log(f'DIVE zapping {arg.text!r} down to escape, hostiles at '
                          f'{[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
                agent.zap(arg, '>')
                if agent.current_level().key() == key:
                    if 'here is too hard to dig' in agent.message:
                        self.undiggable.add(key)
                    elif 'too hard to' in agent.message or 'fills with' in agent.message:
                        self._bad_dig_spots.add((key, spot))
                return
            if what == 'step':
                agent.log(f'DIVE walking to dig at {arg}, hostiles at '
                          f'{[(m[3].mname, int(m[0])) for m in monsters[:3]]}')
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

    def _dig_escape_action(self):
        """DIG_ESCAPE: what digging out would do right now -- ('dig', tool), ('step', (y, x)) onto a diggable
        neighbour -- or None (no tool, level not diggable, or an adjacent monster that melees through
        Elbereth: that one is fought). No HP floor: prayer and the other emergencies preempt digging."""
        agent = self.agent
        if not (DIG_ESCAPE and DIG_FIRST and self.diving):
            return None
        level = agent.current_level()
        if level.dungeon_number == GEHENNOM or level.key() in self.undiggable or \
                level.dungeon_number not in MAIN_LINE or agent.blstats.time < self._dig_blocked_until or \
                utils.any_in(agent.glyphs, G.SWALLOW):
            return None   # (engulfed: engulfed_fight's business, and an engrave attempt in there forbids the square)
        bl = agent.blstats
        monsters = agent.get_visible_monsters()
        adjacent = [m for m in monsters if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1]
        # a known wand of digging zapped down holes the floor at once (zap.c zap_dig -> dighole): the escape
        # where the pick-axe can't work, and one flood roll instead of the pick's two on Medusa's islands
        # (top-level only: a wand inside a bag has no letter to zap it by)
        wand = next((i for i in agent.inventory.items if i.is_wand() and i.is_unambiguous() and
                     i.object == O.from_name('digging', nh.WAND_CLASS) and
                     not agent.inventory.is_known_empty(i)), None) if DIG_WAND_ESCAPE else None
        if any(self._melee_ignores_elbereth(m[3]) for m in adjacent):
            # an @ next to us: fight it. Digging beside it is futile: any attack, hit or miss, stops the dig
            # occupation before its next turn (mhitu.c hitmsg/missmu -> stop_occupation; allmain.c runs the
            # occupation only after the monsters' move), so a monster attacking every turn blocks all progress.
            # (dsafe-A jf16 s11 dug on in its pit beside a Grey-elf and a werewolf: 90 -> 12 HP, no hole.)
            return self._wand_escape(wand)
        from .combat.fight_heur import distant_flash_directions
        if distant_flash_directions(agent, monsters):
            # a minotaur two squares off in line: flash it before it closes (fight_heur.distant_flash_directions)
            return self._wand_escape(wand)
        if adjacent and agent._hurt_recently(2) and not self._elbereth_possible():
            # bitten while digging with no Elbereth under us and none to be had here (engrave cap, forbidden
            # square): every attack stops the dig, so a hole takes ~12 turns of free hits -- fight instead
            # (dsafe-t2 jf25 s11 dug on under a soldier ant and a large dog, 58 -> 16 HP in 5 turns)
            return self._wand_escape(wand)
        tool = self.digging_tool()
        if tool is None:
            return None
        max_wet = self._dig_max_wet()
        up = self._medusa_reroll_stairs(max_wet)
        if up is not None:
            return ('reroll', up)
        if max_wet is None:
            max_wet = 0   # stranded: nothing to dig here but our own pit, if any
        # our own pit is the best square of all (4 dig turns to the hole), even when the map shows the pit
        # there (#terrain or invisibility write S_pit at our square). A staircase we just took shows us, not
        # the stairs, so the map doesn't know it (base-public s6 wasted its first turn on Medusa-3 on "The
        # stairs are too hard to dig in")
        on_stairs = (bl.y, bl.x) in level.stair_destination
        if self._in_own_pit() or (not on_stairs and self._diggable_spot(bl.y, bl.x, max_wet)) or \
                (level.objects[bl.y, bl.x] in (SS.S_pit, SS.S_spiked_pit) and
                 self._wet_neighbours(bl.y, bl.x) <= max_wet):
            if max_wet > 0 and wand is not None:
                return ('zap', wand)
            return ('dig', tool)
        # standing on stairs, in a doorway, on a wetter square (Medusa-3's island): walk to the nearest
        # square we can dig, a few steps at most
        if agent.blstats.time < self._dig_walk_blocked_until:
            return None
        target = self._dig_walk_target(max_wet)
        return None if target is None else ('step', target)

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
        """An intact Elbereth, or a square where a protective engraving may still be attempted."""
        agent = self.agent
        blind = agent.character.prop.blind
        if not blind and (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
            return True
        if agent.character.prop.polymorph or not agent.can_engrave():
            return False
        spot = (agent.current_level().key(), agent.blstats.y, agent.blstats.x, self._in_own_pit())
        # blind: fighting blind is no better than engraving again (ravens)
        if blind:
            return True
        cap = ELBERETH_TRIES_ESCAPE
        return self.__dict__.get('_elbereth_tries', {}).get(spot, 0) < cap

    def _hurt_since(self, turn):
        """HP fell at some point since `turn` (within the last 12 turns of history)."""
        hist = [hp for t, hp in self._hp_history if t >= turn]
        return bool(hist) and max(hist) > self.agent.blstats.hitpoints

    def _medusa_reroll_stairs(self, max_wet):
        """DIG_ESCAPE on Medusa's level, stranded where every reachable square has >= MEDUSA_REROLL_WET moat
        neighbours: a hole there floods with odds n/(n+1) at the pit and again at the hole (dig.c fillholetyp:
        1 success in 9 at n = 2), and each flood makes the islet wetter until we drown (dsafe-t1 public s9
        took the stairs onto a Medusa-4 islet and drowned on its 7th hole). The '<' on the islet re-rolls the
        landing: climb, dig down beside the '>' above, and fall at a random spot of the level's arrival region
        (u_on_rndspot; Medusa-4's holds dry squares in its north part). Returns the '<' (y, x) or None."""
        # (max_wet None: every reachable square flooded or refused -- base-jf27 s5 then sat 3000 turns on the '<'
        # of a Medusa-3 island, searching, until an invisible stalker killed it)
        if not (DIG_ESCAPE and self.on_medusa_level()) or (max_wet is not None and max_wet < MEDUSA_REROLL_WET) or \
                self._medusa_rerolls >= MEDUSA_REROLLS or self.agent.blstats.time < self._medusa_reroll_blocked_until:
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

    def _dig_walk_target(self, max_wet):
        """The nearest free diggable square within DIG_WALK_RADIUS steps (DIG_ESCAPE), the driest first among
        equally near ones."""
        agent = self.agent
        level = agent.current_level()
        dis = agent.bfs()
        best = None
        ys, xs = ((dis > 0) & (dis <= DIG_WALK_RADIUS)).nonzero()
        for y, x in zip(ys, xs):
            if agent.monster_tracker.monster_mask[y, x] or level.objects[y, x] in FALL_TRAPS or \
                    not self._diggable_spot(y, x, max_wet) or \
                    (level.key(), (int(y), int(x))) in self._bad_dig_spots:
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
        return self.diving or KEEP_TOOL_IN_TOUR or bool(jf_config.PICK_TRIP_XL) or bool(GRIND_HUNT_XL)

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
        """The tour's Dlvl 1 grind ends at XL 8 (DT6A), or earlier for a tool run."""
        xl = self.agent.blstats.experience_level
        return (xl >= 8 or (TOOL_RUN_XL is not None and xl >= TOOL_RUN_XL)) and self.fed_for_dive()

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

    @Strategy.wrap
    def ditch_pet_strategy(self):
        agent = self.agent
        from .global_logic import Milestone
        if not DITCH_PET or self._ditch_state == 3 or self.diving or \
                agent.global_logic.milestone != Milestone.BE_ON_FIRST_LEVEL:
            yield False
        bl = agent.blstats
        level = agent.current_level()
        first = (Level.DUNGEONS_OF_DOOM, 1)
        if self._ditch_state == 0:
            if level.key() != first or not agent.has_pet or bl.time < DITCH_PET_AFTER or \
                    agent.get_visible_monsters() or bl.hitpoints < 0.8 * bl.max_hitpoints:
                yield False
            dis = agent.bfs()
            if not any(dis[p] != -1 for p in self._stairs_down(level)):
                yield False
            agent.log('DITCH pet: taking it down to Dlvl 2')
            self._ditch_state = 1
            self._ditch_started = bl.time
        if bl.time - self._ditch_started > DITCH_PET_BUDGET:
            agent.log(f'DITCH pet: out of time (state {self._ditch_state})')
            self._ditch_state = 3
            yield False
        yield True
        pos = (bl.y, bl.x)
        pet_adjacent = any(utils.adjacent(pos, (int(y), int(x)))
                           for y, x in zip(*utils.isin(agent.glyphs, G.PETS).nonzero()))
        key = level.key()
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
            agent.log(f'DITCH pet: back on Dlvl 1, pet left behind: {not agent.has_pet}')
            self._ditch_state = 3
            return
        ups = [p for p in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())]
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
        if self._open_line_to(ty, tx):
            return False   # no move: the caller hunts it now
        dis = agent.bfs()
        ys, xs = np.nonzero(dis >= 0)
        if len(ys) == 0:
            return False
        cheb = np.maximum(np.abs(ys - ty), np.abs(xs - tx))
        inline = (ys == ty) | (xs == tx) | (np.abs(ys - ty) == np.abs(xs - tx))
        # closest to it, a square in line with it (to see down its tunnel) breaking ties, then the nearest
        order = np.lexsort((dis[ys, xs], ~inline, cheb))
        i = order[0]
        y0, x0 = agent.blstats.y, agent.blstats.x
        if (ys[i], xs[i]) == (y0, x0) or cheb[i] > max(abs(ty - y0), abs(tx - x0)):
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

    def _wet_neighbours(self, py, px):
        agent = self.agent
        level = agent.current_level()
        around = agent.glyphs[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
        around_known = level.objects[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
        return int((utils.isin(around, WET) | utils.isin(around_known, WET)).sum())

    def _diggable_spot(self, py, px, max_wet=0):
        agent = self.agent
        level = agent.current_level()
        # a square always covered by objects (a leprechaun hall is gold wall to wall) never shows its
        # floor: an s12 digger found no 'floor' there and explored the hall until it starved
        terrain = level.objects[py, px]
        if not (terrain in PLAIN_FLOOR or (terrain == -1 and level.walkable[py, px]) or
                (DIG_IN_PITS and terrain in PITS)) or \
                (level.shop[py, px] and not self._trapped_in_shop(py, px)) or \
                (level.key(), (py, px)) in self._bad_dig_spots:
            return False
        # the square we arrived on by stairs was never seen ('@' covers it: terrain -1), but it is a staircase
        # (rescue agent, 78a30e1: 'The beam bounces off the stairs' emptied wands of digging in 6 of 90 games)
        if jf_config.WAND_STAIRS_FIX and (py, px) in level.stair_destination:
            return False
        # a hole next to water or lava fills with it (dig.c fillholetyp: n moat squares around fill it
        # with probability n/(n+1)); only islands with no dry square (Medusa variants) accept the risk
        return self._wet_neighbours(py, px) <= max_wet

    def _trapped_in_shop(self, py, px):
        """A digger trapped in a shop may dig through its floor: the shopkeeper stands in the doorway of a
        pick-axe carrier, and shk.c shopdig(1) grabs the pack only of a hero who owes (billct or debit)."""
        # hypothesis: a digger that falls into a shop (the shopkeeper blocks the door, shop floor refused) never digs out -- s11 starved there on Dlvl 17 -- so dig through the shop interior when nothing is unpaid
        # sources: https://nethackwiki.com/wiki/Shop (the shopkeeper grabs the pack only of a customer with unpaid goods), https://nethackwiki.com/wiki/Tourist, /refs/top/429cf0108271 (exploration_logic: "a digger fell into a closed shop")
        if not SHOP_DIG:
            return False
        agent = self.agent
        level = agent.current_level()
        y, x = agent.blstats.y, agent.blstats.x
        if not (level.shop_interior[y, x] and level.shop_interior[py, px]):
            return False
        # only when trapped: no floor outside the shop reachable, for a while (the s4 dive walked out of a Dlvl 2
        # shop 140 turns after landing) -- at once when Weak or Fainting: each faint is turns lost to hunger
        hungry = agent.blstats.hunger_state >= Hunger.WEAK
        if (self.turns_on_level() < SHOP_DIG_WAIT and not hungry) or \
                ((agent.bfs() >= 0) & level.walkable & ~level.shop).any():
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
        if self.in_gehennom() and self.levitating():
            # "You can't reach the floor": every such try would count towards DIG_MAX_TRIES and could mark a
            # diggable level undiggable; a potion's levitation (the castle crossing) ends within ~150 turns
            self._task('wait for levitation to end')
            agent.search(5)
            return True
        tool = self.digging_tool() if agent.blstats.time >= self._dig_blocked_until else None
        wand = self.digging_wand() if tool is None else None
        if WAND_WAITS and wand is not None and self.use_mines() and \
                level.dungeon_number == Level.DUNGEONS_OF_DOOM and agent.blstats.depth <= MINES_BRANCH_MAX_DEPTH:
            wand = None   # the Mines route (a pick-axe) first; the wand's 4-8 holes are the fallback
        if tool is None and wand is None:
            return False
        # a '>' a few steps away is as fast, and arriving on the up stairs keeps a way back -- but not from the
        # pit we dug: climbing out takes turns, the hole only 4 more (dsafe-A jf14 s5 thrashed between its pit
        # and a '>' 3 squares away under a centaur's crossbow bolts: 'You are still in a pit' x6)
        dis = agent.bfs()
        if not (DIG_ESCAPE and tool is not None and self._in_own_pit()):
            for d, _, _, kind in self.down_targets():
                if kind == 'stairs' and d <= DIG_STAIRS_RADIUS:
                    return False
        y, x = agent.blstats.y, agent.blstats.x
        candidates = utils.isin(level.objects, PLAIN_FLOOR) | ((level.objects == -1) & level.walkable)
        floor = [p for p in zip(*candidates.nonzero()) if dis[p] >= 0]
        max_wet = 0
        if not any(self._diggable_spot(*p) for p in floor):
            # all reachable floor borders water: take the square with the fewest wet neighbours
            wet = [self._wet_neighbours(*p) for p in floor if self._diggable_spot(*p, max_wet=8)]
            up = self._medusa_reroll_stairs(min(wet) if wet else None) if tool is not None else None
            if up is not None:
                self._medusa_reroll(up)
                return True
            if not wet:
                return False
            max_wet = min(wet)
        # DIG_ESCAPE: the staircase we just came down by shows us, not the stairs (see _dig_escape_action)
        on_stairs = DIG_ESCAPE and (y, x) in level.stair_destination
        if (on_stairs or not self._diggable_spot(y, x, max_wet)) and not (DIG_ESCAPE and self._in_own_pit()):
            spots = [(dis[p], p) for p in floor if dis[p] > 0 and self._diggable_spot(*p, max_wet)]
            if not spots:
                return False
            agent.go_to(*min(spots)[1])
            return True
        if tool is not None:
            rest_below = GEHENNOM_DIG_REST_BELOW if self.in_gehennom() else DIG_REST_BELOW
            if agent.blstats.hitpoints < rest_below * agent.blstats.max_hitpoints and \
                    not (DIVE_REST and self._in_own_pit()):
                self._task('rest before digging')
                if DIVE_REST and self._rest_elbereth():
                    return True
                agent.search(1 if agent.get_visible_monsters() else 20)
                return True
            self.dig_with_tool(tool)
            return True
        if self.rest_if_hurt():
            return True
        agent.log(f'DIVE zapping {wand.text!r} down')
        agent.zap(wand, '>')
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
        and casters included), always on Medusa's level. Sighted retries are capped per square and dig
        phase (the pit erases it). Blind engravings cannot be read back, so renew them after damage
        indicates that protection may have failed; otherwise keep digging."""
        agent = self.agent
        if agent.character.prop.polymorph or not agent.can_engrave() or utils.any_in(agent.glyphs, G.SWALLOW):
            return False
        blind = agent.character.prop.blind
        if not blind and (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
            return False
        bl = agent.blstats
        near = [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= ELBERETH_DIG_RADIUS and
                not self._melee_ignores_elbereth(m[3])]
        # Weak or Fainting: a faint leaves us helpless for turns in the middle of a dig (base-jf14 s8: a troll,
        # base-jf16 s13: a giant ant, base-jf16 s9: a panther -- all Elbereth-respecting -- killed fainted
        # diggers); the engraving holds them off while we're out
        spot = (agent.current_level().key(), bl.y, bl.x, self._in_own_pit())
        tries = self.__dict__.setdefault('_elbereth_tries', {})
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
            # hypothesis: fresh damage while blind warrants renewing Elbereth even after
            # earlier attempts; a lifetime cap leaves raven-blinded diggers defenseless.
            if last is not None and not self._hurt_since(last):
                return False
        elif tries.get(spot, 0) >= ELBERETH_TRIES_ESCAPE:
            return False
        tries[spot] = tries.get(spot, 0) + 1
        self.__dict__.setdefault('_engrave_turn', {})[spot] = bl.time
        agent.log(f'DIVE Elbereth before digging{" (blind)" if blind else ""}'
                  f'{" in the pit" if spot[3] else ""}: {[m[3].mname for m in near]}')
        agent.engrave('Elbereth')
        return True

    def dig_with_tool(self, tool):
        agent = self.agent
        key = agent.current_level().key()
        spot = (agent.blstats.y, agent.blstats.x)
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
                return
            # Waking from a faint: the deafness that came with it ends right after, and its 'You can hear
            # again' stops the new dig before any progress (a starving rescue dive spent ~300 turns and
            # 15 tries on one level, close to DIG_MAX_TRIES marking it undiggable). Not the floor's
            # fault: don't count it, dig on at once.
            if prompted and 'You stop digging' in msg and \
                    any(s in msg for s in ('You can hear again', 'You regain consciousness', 'You faint')):
                self._dig_tries[key] = tries - 1
                if agent.blstats.hunger_state <= Hunger.FAINTING:   # not passed out again
                    continue
            break
        if prompted and 'dig a pit in the' in msg:
            self._pit_at = (key, spot)
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

    def _mapping_scroll(self):
        return next((i for i in self.agent.inventory.items if i.category == nh.SCROLL_CLASS and
                     i.is_unambiguous() and i.objs[0] == MAGIC_MAPPING and not i.status == Item.CURSED), None)

    def should_read_mapping(self):
        agent = self.agent
        level = agent.current_level()
        prop = agent.character.prop
        return MAP_WHEN_STUCK and level.key() not in self._mapped and \
            level.dungeon_number == Level.DUNGEONS_OF_DOOM and agent.blstats.depth >= 2 and \
            agent.blstats.experience_level >= DIVE_XL and \
            self.turns_on_level() > MAP_STUCK_TURNS and not self._stairs_down(level) and \
            self.digging_tool() is None and not (prop.blind or prop.confusion or prop.stun or prop.hallu) and \
            self._mapping_scroll() is not None

    def read_mapping(self):
        agent = self.agent
        scroll = self._mapping_scroll()
        self._mapped.add(agent.current_level().key())
        agent.log(f'DIVE reading {scroll.text!r}: no \'>\' after {self.turns_on_level()} turns')
        with agent.atom_operation():
            agent.step(A.Command.READ)
            agent.type_text(agent.inventory.items.get_letter(scroll))

    def descend(self):
        agent = self.agent
        if self.try_dig_down():
            return
        targets = self.down_targets()
        if not targets and utils.isin(agent.current_level().objects, G.STAIR_DOWN).any() and \
                self._walk_through_traps():
            targets = self.down_targets()
        if not targets:
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
        threshold = VALLEY_REST_UNTIL if self._valley_resting else REST_BELOW
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
        bl = agent.blstats
        prop = agent.character.prop
        if prop.hallu or prop.stun or prop.confusion or prop.blind or \
                bl.hitpoints < 0.5 * bl.max_hitpoints or agent._hurt_recently(3):
            yield False
        monsters = agent.get_visible_monsters()
        if not monsters:
            yield False   # the plan walks on by itself
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

    def valley_progress(self):
        """Log how far west the Valley walk got (for the scenario analyses)."""
        bl = self.agent.blstats
        if self._valley_west is None or bl.x <= self._valley_west - 5:
            self._valley_west = bl.x
            self.agent.log(f'VALLEY at {(bl.y, bl.x)} turn {bl.time} hp {bl.hitpoints}/{bl.max_hitpoints} '
                           f'xl {bl.experience_level}')

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
        if self.rest_if_hurt():
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
            return False
        d, (y, x) = min(near)
        if not utils.adjacent(pos, (y, x)):
            agent.go_to(y, x, stop_one_before=True)
            return True
        if agent.monster_tracker.monster_mask[y, x]:
            agent.search()  # a peaceful is on the stairs: wait for it to move
            return True
        agent.move(agent.calc_direction(pos[0], pos[1], y, x))
        return True

    def return_to_main_dungeon(self):
        """Back to the main line from a side branch: up out of the Mines, down out of Sokoban."""
        agent = self.agent
        level = agent.current_level()
        stairs = G.STAIR_DOWN if level.dungeon_number == Level.SOKOBAN else G.STAIR_UP
        direction = '>' if level.dungeon_number == Level.SOKOBAN else '<'
        exits = list(zip(*utils.isin(level.objects, stairs).nonzero()))
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
