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
STARVE_CLOCK = False
# give up looking for the Mines entrance after this many turns and go on to Sokoban (0: never)
MINES_SEARCH_TURNS = 3500
# buy food in shops (never while carrying a digging tool) until carrying BUY_FOOD_UNTIL nutrition
BUY_FOOD = True
BUY_FOOD_UNTIL = 2400
# give up an item for sale after 3 walks toward it failed (a shopkeeper in the path), for 2000 turns
# ON (train 3.1): grind-food 43a08fa, replay-verified loop fix (base4-jf14 s6 paced 1500 turns at a shop)
BUY_FOOD_GIVEUP = True
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
# from this XL the Dlvl 1 grind moves to Dlvl GRIND_DEEP_LEVEL (0: never)
GRIND_DEEP_XL = 0
GRIND_DEEP_LEVEL = 3
# no Excalibur dips while carrying a digging tool (a dip's silent curse welds the sword and locks the pick out)
# ON (train 3, A027): no Excalibur dips while carrying a digging tool
NO_DIP_WITH_TOOL = True
# the grind's main-dungeon level by XL, {min XL: Dlvl} (empty: Dlvl 1 throughout; overrides GRIND_DEEP_*):
# e.g. {5: 3, 7: 2} keeps the random-monster cap (depth + XL) / 2 at 4 from XL 5 (global_logic._grind_level)
# train 3 (early-game A027): XL 5-6 grind on Dlvl 3, XL 7 on Dlvl 2 (cap 4): prayers 12.2 -> 7.2/game, grind losses 10 -> 5
GRIND_LEVELS = {5: 3, 7: 2}
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
# ON (train 3): fires only on the castle level, where the score is banked (A018 neutral); power/gehennom build on it
CASTLE_PASSAGE = True
# engulfed: wield the best melee weapon before fighting out (a dig-diver is swallowed with its pick-axe)
ENGULF_WIELD = False
# a Hungry (or worse) dive walks to fresh edible corpses within DIVE_EAT_RADIUS (BFS steps) and eats them
DIVE_EAT = False
DIVE_EAT_RADIUS = 8
# LAST_RESORT: a known wand of digging is zapped down first (an escape that also banks a level)
# ON (train 3): castle A018 guard neutral (+0.004); harness 8 of 10 escapes
LAST_RESORT_DIG = True
# fight2 on an Elbereth square: its -100 on every attack and its 'wait' bonus assume the engraving scares what is
# next to us. With an @ (human/elf) or minotaur adjacent it doesn't: fight it (see combat/fight_heur.py
# at_ignorer_adjacent; base4-jf16 s4 waited 5 turns in its dig pit while two Green-elves hit it 65 -> 12 HP)
# ON (train 3.2, dive-safety A060): guard vs base5 neutral (0.4212 vs 0.4230, 38/45 identical); harness Big Room 34 -> 40/42, @ group 13 -> 14/14; base4+t31+base4arm: 18/135 games waited on Elbereth under @ attack, 15 died to that monster kind
AT_ELBERETH_FIX = True
# with AT_ELBERETH_FIX: melee priority bonus on the ignorer while an Elbereth could be (re)written here -- once it
# is dead the engraving holds everything else off again and the dig resumes (0: no focus)
AT_FOCUS = 10
# with AT_ELBERETH_FIX: the dig out gives way to the fight for an ignorer this many steps away (1: adjacent only)
AT_DIG_RADIUS = 2
# LAST_RESORT: pray at critical HP beside a hostile once this many turns have passed since the last prayer
# (0: off; the ordinary low-HP prayer waits 500)
# ON (train 3): castle A038 guard neutral (-0.002); harness 2 Grey-elves at 12/72 HP deaths 9/10 -> 3/10
DESPERATE_PRAYER_GAP = 250
# LAST_RESORT: zap each unknown wand once (one still unknown after a zap at a monster is no attack wand)
# ON (train 3): with DESPERATE_PRAYER_GAP (castle A038)
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
# a wished-for object is not formally identified (zap.c makewish: 'p - a granite ring'): name its appearance
# after the wish (power.learn_wished). Without it SPARE_WISHES wished for an amulet of life saving again and
# again and never wore one (base5-public s4, base5arm-jf16 s5), and castle_logic wished 3 rings of levitation,
# never put one on and then zapped the empty wand until a minotaur came (pwc-dp10 jf14-s14)
# ON (train 3.3, power B013): a wished-for object is recognised by its appearance (SPARE_WISHES never wore its amulet of life saving)
WISH_LEARN = True
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
# --- power: the post-castle campaign (castle_power.py, hooks in castle_logic; inert before the castle) ---
# on the castle's landing square: engrave-test unknown wands, drop every scroll that may be scare monster (the
# minotaurs of the west maze killed 27 of 61 real kits, most before any item was tried), then try the
# passage plan right there instead of after the walk to the courtyard
# ON (train 3.3, power castle set)
CASTLE_ARRIVAL_DRILL = True
# polymorph forms that fly/swim/breathe water cross the moat (P 0.2 per random polymorph); a known wand of
# polymorph is zapped at ourselves until one comes up; forms wait for their end at the door/trap door
# ON (train 3.3, power castle set)
CASTLE_POLY = True
# on the castle's west side an Elbereth-ignorer (minotaur, @) within 2: drop every scroll that may be scare monster
# and hold on the pile, striking what stays next to us (dive_logic.gehennom_scare's hold loop)
# ON (train 3.3, power castle set; acts from depth 25 on the main line, same score in the inertness replay)
CASTLE_SCARE = True
# castle: rest before/while crossing one square in from the moat, never on the courtyard's moat edge
# ON (train 3.3, power castle set)
CASTLE_EDGE_REST = True
# castle: go round by the south half when a sea monster was seen lately in the north half's west column (and
# back out of a column whose next square a monster holds); the choice is made in the west courtyard only
CASTLE_SEA_SWITCH = False
CASTLE_SCARE_HOLD = 1500   # turns at most on the pile while an Elbereth-ignorer is about
CASTLE_SCARE_DEPTH = 25    # from this depth on the main line (castle or the mazes right above it); a diggable
                           # level is dug down from the pile (scared monsters don't interrupt a dig)
# castle (castle_logic): dig straight east from the west maze into the courtyard instead of exploring for the
# maze's single join (the maze walls outside the castle map are diggable)
# ON (train 3.3, power): castle arrival kit suite 0/61 -> 3/61 crossings with the castle set; acts only on the castle level (score banked)
CASTLE_WEST_DIG = True
# castle: no retreat up the castle's '<' while the passage is on (it leads onto Medusa's level next to her
# when she is castle-1: 2 of 61 real kits 'petrified by Medusa' in pwc-dp1)
# ON (train 3.3, power castle set)
CASTLE_NO_RETREAT = True
# --- castle-breach: getting past the castle with the kits the dive brings (castle_logic, castle_power, dive_logic) ---
# Harness (castle-real-all, 61 real kits x 4 level-generation salts = 244 games, train 3.6 = 4103a5a): 8/244 pass,
# 8/84 among the 21 kits that hold a passage item or a polymorph source. Crossing attempts (courtyard -> moat) die in
# the moat ring's NW channel (map x 0 rows 1-5 + row 0 x 0-8, one square wide) or on the courtyard corner (0,6): 21 of
# ~40 attempts (brx-base-all3/all4), a shark waiting hidden at (0,5)/(1,5)/(0,4) biting 5d6 up to twice per our move
# while we trade blows over the water; every attempt that got past it reached the east courtyard.
# BREACH_PROBE: before the first lift is tried in the west courtyard, stand on the moat-ring corner of our half ON FOOT
# and search (detect.c mfind0: a search finds a hidden eel/shark next to us): fight what shows from dry land, step back
# to TEST_SPOT (no water next to it) to rest below BREACH_PROBE_HP, and only set off once BREACH_PROBE_QUIET searches
# in a row found nothing (at most BREACH_PROBE_BUDGET turns). A lasting lift found earlier (a ring put on where we
# landed) is taken off for the rests.
BREACH_PROBE = False
BREACH_PROBE_QUIET = 8
BREACH_PROBE_BUDGET = 300
BREACH_PROBE_HP = 0.6
# a probe older than this many turns is repeated (shorter) before the next item that may lift us
BREACH_PROBE_STALE = 25
# BREACH_WANDS: engrave-test the unknown wands on a bare west-courtyard square (2,8) before the passage plan is given up:
# the arrival drill tested them on the landing square, inside the pit of our castle-detecting dig, where the engrave
# test's look ('There is a pit here.  You see no objects here.') refused every wand and marked it tried -- jf14-s0 gave
# the castle up ('nothing that crosses water') holding an untested wand of cold (brx-smoke1)
BREACH_WANDS = False
# BREACH_RESUME: a passage given up for want of a way across ('nothing that crosses water', 'stranded') resumes when one
# turns up later (a wand named by a zap at a monster, a lift from a last-resort quaff, a crossing polymorph form)
BREACH_RESUME = False
# BREACH_COLD_FIRST: with a known wand of cold (or frost horn) and only untested potions/amulets left, freeze the west
# channel first (a ray freezes ~3 squares; the ice has no water beside it) and try the potions at the strip's east end
# on Elbereth: a lift then only has to cover the 13-square east channel (an uncursed potion lasts 11-150 turns, the whole
# way round takes ~56 at our speed). Kits: 4 of 61 carry a wand of cold, 2 of them with a potion of levitation
BREACH_COLD_FIRST = False
# BREACH_LEVWARN: 'You float slightly lower.' (5 turns left of a potion's lift): quaff another known potion of
# levitation, else don't start a water stretch from land, and over water make for the nearest dry square
BREACH_LEVWARN = False
# BREACH_SIDESTEP: floating into the west channel with a sea monster in its first one-square-wide square ((0,4) north:
# 42 attack actions there in the baseline, the commonest block), back onto the corner and wait up to 4 turns for it to
# come out after us onto one of the two squares off the corner, then pass it by the other (both lead into the column)
BREACH_SIDESTEP = False
# BREACH_NOWAIT: on the castle level (passage active) don't wait out blindness or hallucination (a potion of blindness:
# 250-450 turns, hallucination 600-800) before going on with the passage plan -- confusion and stun (short, and a
# confused step can land in the moat) are still waited out
BREACH_NOWAIT = False
# BREACH_PLUNGE: a flying polymorph form on the (55,08) trap door presses '>' (do.c dodown: a seen trap door takes a
# flyer through with TOOKPLUNGE; only levitation refuses) instead of waiting up to 3000 turns for the form to end
BREACH_PLUNGE = False
# BREACH_SPOT: the west courtyard's item-test square (1,7) under a boulder (giants carry and drop them) moves to the
# nearest free inner square, and a levitating hero goes round a boulder (or breaks it with the pick) instead of pushing
# it without leverage: brx all-flags jf14-s14 explored 4000 turns toward a buried (1,7); baseline jf14-s14~1 gave up
# 'courtyard square (1, 7) not reachable'; smoke jf14-s14~3 pushed the corner's boulder 80 times and gave up
BREACH_SPOT = False
# BREACH_RUSH: a lasting lift (ring, boots, a wished ring) rests in the west courtyard before the crossing only below
# BREACH_RUSH_HP (was 85%), and never with a land hostile within 4 -- the moat is the escape from the maze's monsters
# (2 squares into the channel nothing on land reaches us); the rest happens on the strip instead (rest stop below 80%,
# Elbereth). brx-t2-mino ring kits 4/24: most died at the landing or in the courtyard rest (jf16-s6~2: gnome king)
BREACH_RUSH = False
BREACH_RUSH_HP = 0.35
# BREACH_BYPASS: floating round the moat with every closer square held by a monster, step sideways to an equally close
# free square whose next step is free instead of attacking (east channel mouth: a sea monster on (61,5) is passed by
# (62,5) -> (61,6); the baseline attacked it from (62,4) 31 times in 434 harness games)
BREACH_BYPASS = False
# BREACH_MINO: an adjacent minotaur (or another Elbereth-ignorer of level 10+) at the castle depth: zap what may stop it
# at any HP, best first -- a known wand of sleep/death, teleportation (zap.c u_teleport_mon works on noteleport levels:
# rloc), polymorph (at IT: the kit's wand of polymorph was zapped at ourselves below 60% HP and minotaurs killed
# public-s0 in 3 of 4 salts), then striking/fire/cold/lightning/magic missile, then each unknown wand once (the old
# rule waited for HP < 90%: minotaurs take 30-50 HP a turn, 26 of 84 target games died to one)
BREACH_MINO = False
# BREACH_PILE: on a scroll of scare monster that works (CASTLE_SCARE hold, castle recognised), use the hold: zap wands
# (BREACH_MINO's order) at an Elbereth-ignorer in a straight line up to 8 squares away, and try the passage plan's rings,
# amulets, boots and wishes right there (the hold rested up to 1500 turns and tried nothing: 22 of 84 target games died
# 'before any try', several after 100-700 turns on or near a pile)
BREACH_PILE = False
# BREACH_DOOR: at the back door (57,08) -- the giant eels start at (57,07)/(57,09), both next to it -- stand on Elbereth
# while opening it (a scared eel lets go: F030), re-engraving after each kick (dokick.c u_wipe_engr(2), and its
# wake_nearby wakes them); 3 of ~40 crossing attempts drowned there
BREACH_DOOR = False
# a hunger prayer from this gap (instead of FAINT_PRAYER_GAP / WEAK_PRAYER_GAP) when a hostile within
# THREAT_RADIUS can reach us while Fainting, or while Weak with at most THREAT_WEAK_MARGIN nutrition left
# (0: off). 9 of 10 Dlvl-1 deaths in the base runs were fainted next to ordinary monsters at gaps 950-1110.
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 0)
THREAT_PRAYER_GAP = 1000
THREAT_RADIUS = 5
THREAT_WEAK_MARGIN = 15
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 4)
THREAT_MIN_DIFFICULTY = 99
# the difficulty trigger only from this gap (0: THREAT_PRAYER_GAP)
THREAT_DIFF_GAP = 0
# this many walking hostiles within THREAT_RADIUS count as a threat (99: only difficulty / Elbereth-ignorers / HP)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 2)
THREAT_MIN_COUNT = 99
# ... or our HP below this fraction (a faint on a smudged Elbereth took a 90-HP XL7 to 49 in one faint)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 0.5)
THREAT_HP_FRAC = 0.75
# ... or Fainting with no intact Elbereth under us and a walking hostile within THREAT_NE_RADIUS
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
THREAT_NO_ELBERETH = True
THREAT_NE_RADIUS = 3
THREAT_NE_MIN_DIFFICULTY = 3
# find the kill square of our melee/thrown kills from the attack itself, and of pack kills from the corpse
# glyph, when the glyph-disappearance test misses it (27% of kills: their corpses were never eaten)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
CORPSE_TRACK = True
# walk to fresh (<= CLAIM_MAX_AGE turns) edible corpses within CLAIM_DIST steps and eat them, before the pet
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
CLAIM_CORPSES = True
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
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 0)
TOUR_WEAK_PRAYER_GAP = 1700
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 0)
TOUR_FAINT_PRAYER_GAP = 1600
# per-XL tour gaps [[min_xl, weak_gap, faint_gap], ...] (the highest min_xl <= XL wins; overrides TOUR_*)
TOUR_GAPS_BY_XL = []
# the low-HP prayer only at pray.c's critically_low_hp (EXACT_PRAYER's HP rule without its turn-100 first prayer)
LOWHP_EXACT = False
# hunger-prayer gaps while diving at depth >= DIVE_GAP_MIN_DEPTH (0: WEAK_PRAYER_GAP / FAINT_PRAYER_GAP)
DIVE_WEAK_PRAYER_GAP = 0
DIVE_FAINT_PRAYER_GAP = 0
DIVE_GAP_MIN_DEPTH = 5
# the Weak and Fainting hunger-prayer gap while diving with a digging tool (0: the rules above); see
# agent._dive_tool_hunger_gap
# ON (train 3.3, dive-safety A061): first dive hunger cycle prays from gap 850; guard vs base6 +0.011 (flag fired in 4 games, +0.175, none worse); harness 13/14 vs 7/14
DIVE_TOOL_HUNGER_GAP = 850
# without STARVE_CLOCK: a Fainting prayer whatever the gap when the faint-length hunger estimate nears starvation
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
STARVE_DEADLINE = True
# Fainting just after a paralysis ended ('You can move again'): pray from this gap (0: off)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: 0)
STARVE_UNMEASURED_GAP = 1100
# measure a faint from its own screen (not the last update), and not at all after an interrupted counted search
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
FAINT_MEASURE_FIX = True
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
SESSILE_MEMORY = True
# go_to() re-plans to its real target after a path is blocked mid-way (its loop variables overwrote the target,
# so the next round aimed at the blocked square: 'end point is no longer accessible' and a strategy restart)
GOTO_TARGET_FIX = True
# the panic-loop breaker closes a square for 100 turns, not for the rest of the game (the failing moves are
# usually ours: a bear trap, a web; a permanent forbid boxed a Mines dive in for 5000+ turns)
TEMP_FORBID = True
# boxed in by diagonal squeezes while carrying > 600: drop to 550 for a while (a corridor bend held a grind 8000 turns)
UNSQUEEZE = True
UNSQUEEZE_TURNS = 30   # boxed in on one square this long first (the dive only)
# after 3 failed use_container attempts on a floor container, leave that square's containers alone (a take-out
# menu that never matched was retried 45,453 times in one game)
# ON (train 2): the take-out loop (robustness B004) hit jf14 s0 in the train-2 smoke: 11399 panics and 317k steps; with the fix 152 and 36k
CONTAINER_LOOP_FIX = True
# climbing out of a branch (or on the tool quest), known trap doors and holes stay closed: the stairs-cut-off walk
# stepped onto them again and again (base-jf26 s14 fell 42 times in 10k turns climbing out of Mines' End)
CLIMB_NO_FALL = True
# no fight2 moves while we are still in a pit (each try costs a turn and the attacker hits for free)
PIT_AWARE_FIGHT = True
# held by a bear trap: move-attempt diagonally (every diagonal attempt counts, 1 in 5 orthogonal ones: hack.c trapmove)
BEARTRAP_ESCAPE = True
# Lycanthropy (466 of 3158 dev games were infected; uncured Dlvl-1 lycanthropes died ~21% per 1000
# turns): detect 'You dream that you feel feverish' and were changes, never eat our were family's corpses
# (cannibalism: Luck -2..-5, the next prayer fails -- public s13, jf25 s10), treat a were form's HP as
# a buffer (no HP prayers/last resort for it: jf16 s11 and jf25 s9 spent their hunger prayer on the
# rat's HP and starved), and drop the load a rat can't carry so it can eat (public s4 starved Overloaded
# with 5 food items)
LYCAN_FIXES = True
# no lycanthropy cure prayer while Hungry without food (wait for the Weak hunger prayer; see cure_disease)
LYCAN_CURE_WAIT = False
# were_unload drops a were form's load whenever Overtaxed or worse, not only when Weak with food to eat
# t35: jf16 s6 0.602 -> 0.051 (a 9-HP were form dropped all 20 items on Dlvl 3 and never went back for them), jf14 s12
# and arm-jf25 s0 -0.040 each -- off
LYCAN_UNLOAD_ALWAYS = False
# Weak/Fainting in the tour with no prayer due and a monster within FAINT_GUARD_RADIUS: hold on Elbereth
# instead of fighting (dive_logic.faint_guard; fainted melee deaths were 8 of 18 Dlvl-1 grind deaths)
FAINT_GUARD = True
FAINT_GUARD_RADIUS = 4
# with FAINT_GUARD: also hold on Elbereth with nothing in view, from FAINT_GUARD_IDLE_WEAK turns into Weak and
# while Fainting, until the prayer is due (monsters arriving during a faint get no conscious turn to react to)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
FAINT_GUARD_IDLE = True
FAINT_GUARD_IDLE_WEAK = 30
# the reactive guard also in the rescue dive after a failed prayer (no idle hold there: the dive must go on)
FAINT_GUARD_RESCUE = False
# one-action hold strategies (Elbereth rest, water demon vigil, faint shelter/guard) repeat while their own
# condition holds, instead of letting one step of a lower strategy run between two hold actions
# (dive_logic._hold_loop)
# ON (train 3.2, grind-food A047/A058/t32): 90 games 0.4025 vs 0.3964; in-lane losses 6 vs 10, failed prayers 3 vs 7 (was: False)
HOLD_LOOP = True
# ... only after a failed prayer (the rescue dive): the tour's holds keep their old behaviour
HOLD_LOOP_RESCUE = False
# dev only (never set in the arena): the Nth tour hunger prayer is treated as failed, turning games into rescues
SIM_RESCUE_PRAYER = 0
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
LR_ELBERETH = False
# the Elbereth rest never hides from a lone monster one blow kills (difficulty <= 2, not fast), at any HP
REST_FIGHT_WEAK = False
# never kill a gas spore whose blast reaches any @ (or anything in Minetown); its melee is filtered out of fight2
SPORE_SAFE = False
# a floating eye is hit blindfolded (blindfold/towel on, F-attack, off again), else by a throw, else as before
# ON (train 3.4, grind-combat A063): blindfold/towel on before meleeing a floating eye (jf16 s12 replay: no freeze, no rock-mole death)
FEYE_BLIND = True
# a missile/wand/ray hit breaks the Elbereth holds (rest, faint guard/shelter, demon vigil) and fight2's
# wait-on-Elbereth for RANGED_BREAK_TURNS turns: fight2 then closes in on a weak shooter or leaves its line
# ON (train 3.4, A063): a hold breaks when shot/zapped from range; guard vs base7 45 amd64 0.446 vs 0.415, grind deaths 5 -> 1
RANGED_ON_ELB = True
RANGED_BREAK_TURNS = 8

# the item split keeps the digging tool right after the melee weapon, before the armor set: it came after the
# armor, so a heavy armor pickup could push it out -- base8-public s11 took a splint mail on Dlvl 3 and dropped its
# pick-axe (and a food ration), the dive never knew the spot, camped tool-less in the Mines and died there (0.075;
# its castle is at Dlvl 28)
# ON (train 3.5): t35/t36 public s11 kept its pick at that drop step (0.075 -> 0.602 / 0.466), no other game changed by it
TOOL_KEEP_FIRST = True
# the faint guard (and its idle hold) also in a tool-less dive that is not a rescue -- the Mines camp waits
# thousands of turns for a dwarf's pick like the grind waits for XP, but fainted unguarded among the Mines' hostiles:
# base8 tool-less camps fainted 2-23 times, and all 6 died there (large dog, soldier ant, gargoyle, gray unicorn)
# t35/t36: arm-jf25 s5 +0.187, but arm-jf16 s11 0.602 -> 0.126 (held Weak on Elbereth instead of eating and taking the
# dwarf's pick base8 took) and it fired in a deep tool dive in were form (digging_tool() None) -- off
CAMP_GUARD = False
# a welded two-hander (a cursed dwarvish mattock the dive applied: 13% of dwarves' weapons are cursed) leaves no free
# hand, so no Elbereth for the rest of the dive: base5-8 tool dives with a welded mattock scored 0.28-0.39 (4-5 per 90
# games) vs 0.49-0.52 without. pray.c counts it as major trouble (TROUBLE_UNUSEABLE_HANDS: welded and !freehand())
# and fix_worst_trouble uncurses the weapon, so the dive prays for it once WELD_PRAY_GAP turns have passed since the
# last prayer (a major-trouble prayer fails when rnz(350) > gap + 200: 3.9% at 1100). The base8 welds came 806-2431
# turns after the last prayer, all within 3 turns of the dive start
# t36: uncursed the mattock in 3 of 4 welded games, but jf16 s8 0.602 -> 0.445 and s10 0.353 -> 0.206 (the spent prayer),
# arm-jf14 s8 +0.015 -- off
WELD_PRAY = False
WELD_PRAY_GAP = 1100
# engraving (Elbereth, wand engrave-tests) follows engrave.c freehand(): a welded one-handed weapon (a cursed pick-axe the
# dive applied) beside an uncursed shield still leaves a hand to write with; hands_welded() counted any shield, so such a
# dive dug on without Elbereth (Valkyries start with a small shield)
# ON (train 3.6): replay base8-jf14 s6 (pick-axe welded at the dive start, small shield worn): Elbereth 8x vs 1x, d24 -> d25
FREEHAND_FIX = True
# no retreat up the stairs from the level below Medusa's (or below an unvisited level at Medusa depth): they come out on
# her '>' beside her, and her gaze stones us -- base9-public s3 dug past Medusa-26 at 17/78 HP, landed on 27 by its '<',
# retreated up and was 'petrified by Medusa' (8 such deaths in ~700 dev runs); CASTLE_NO_RETREAT covers the castle only
# once it is recognised
# ON (train 3.6): replay base9-public s3 no longer climbs onto Medusa's '>' (lived 740 more turns on the castle level)
MEDUSA_NO_RETREAT = True
# teleport control's "To what level do you want to teleport?" was answered ESC (agent.update's generic text-prompt
# branch), which cancels the teleport after the level teleporter is already used up. With control, from the Dungeons
# any level past the castle is find_hell() = the Valley (castle+1: 0.507-0.691, above anything short of a castle
# crossing) and from Gehennom level 50 exactly (the top of the progress table) or the vibrating-square level when
# shallower (teleport.c level_tele): answer 50 in those two dungeons. Only reachable with teleport control (2 of 360 recent games), so dominant and default on.
LEVELPORT_DEEP = True
# a wand of wishing -> Gehennom's bottom-1 (tele_route.py): wish 1 = a ring of teleport control (worn), wish 2 = '2 cursed
# scrolls of teleportation' (the wand is wrested for its last charge if needed), then read one anywhere in the Dungeons
# (-> the Valley, castle+1) and one in the Valley (-> the vibrating-square level, Dlvl ~44-52: 0.78-0.81). Harness
# lp-e2e: TC ring + a cursed scroll read on Dlvl 12 -> the Valley (Dlvl 26). Harness wr-unknown/wr2-unknown (XL 6 on Dlvl 3
# with an unidentified wand of wishing): 12/12 into Gehennom, 10/12 to Dlvl 44-51. ON: it only acts with a wand of wishing.
WISH_TELEPORT_ROUTE = True

# Never dig or zap digging down on a staircase (rescue agent, 78a30e1; ported by hand for train 2): the square
# under '@' is unknown on arrival, so _diggable_spot took the arrival '<' for floor, and a wand of digging
# zapped there only says 'The beam bounces off the stairs' -- the dive zapped again until the wand was empty
# (6 of 90 baseline games, up to 5 charges = 5 levels each; jf16/5, jf27/1).
WAND_STAIRS_FIX = True

# SHOP_GUARD (off, pick-hunt): BUY_FOOD picks up nothing for sale while we have teleportitis without teleport
# control -- base4-jf14 s10 ate a leprechaun on its grind ('You feel very jumpy.'), picked up a food ration in a
# Dlvl-3 shop and teleported out before paying it ('You escaped the shop without paying!'): Keystone Kops, then the
# shopkeeper and his wand of striking killed the XL-8 dive; and the dive fetches no dwarf's pile in a shop.
# ON (train 3.1): no shop pickups with uncontrolled teleportitis, no pile fetch in a shop (jf14 s10: teleported out unpaid, killed by the shopkeeper)
SHOP_GUARD = True

# --- valley-run (the Valley of the Dead with real castle-arrival kits: XL 7-10, 55-114 HP, AC -8..+10) ---
# VALLEY_SPRINT (off, REJECTED -- no signal): the Valley walk never stops to fight what it can outrun
# (dive_logic._valley_sprint): only adjacent monsters faster than us (bats) and the one on the next square of the way
# are attacked; the way is the cheapest by a router that charges unseen graveyard squares and dangerous monsters;
# up to 4 climbs of the '<' for a new landing square when the way on is through a dangerous monster; the landing
# wields the weapon; a door is dug as soon as nothing is next to us; the '>' is taken at once with hostiles in view;
# rest only below 45% HP; graveyard sleepers stay out of fight2 (the VALLEY_GRAVE_FILTER rule).
# Evidence (valley-real suite, 61 real castle kits started in the Valley, 3000 turns): past the '>' 0/61 in every
# arm -- base (train 3.6) median life 59 turns, v1 (4983dcc, vr-sprint1) 31, v2 (70412c3, vr-sprint2) 26; v2 got 4
# games out of the landing zone (1 to the temple) vs 2 for base. The landing kills: first hit ~3 turns after the
# fall (vampire bats, 44% of them vampires in bat form that rise again at full HP), then mummies.
VALLEY_SPRINT = False

# --- valley-exit ---
# VALLEY_FORT (off, testing): the Valley's residents all home in on us (monmove.c set_apparxy: a monster that can see
# knows our square, no line of sight needed) and an XL-8 kit can't trade blows with them: vx-base (valley-real, 61
# real kits) 0/61 past the '>', 41 dead in the Valley (median ~40 turns: 2-6 vampire-bat-type biters within ~3 turns
# of the fall, then giant/ettin mummies). A scroll of scare monster still works in Gehennom (onscary: only Elbereth
# has the Inhell exception) and a scared monster never melees (dochug: mattacku only if !scared). So on landing walk
# to the '<' (1-13 steps from every landing square), drop the scroll there (or, unknown, every scroll that may be one:
# 12 of the 61 kits truly carry one) and strike from that square whatever comes next to us, with the climb at hand;
# leave for the walk once no hostile has been in view for a while and HP is back, and come back to it to heal
# (dive_logic.valley_fort). Only acts in the Valley.
VALLEY_FORT = False
# LANDING_GUARD (off, testing; castle_landing.py): the castle landing -- a minotaur (or another big Elbereth-ignorer) at
# the castle depth gets the best known wand (beams and cold; other rays only with room to die out before a bounce),
# a frozen one is struck instead, known healing is quaffed at 60% HP beside it (one of its turns takes 30-50 HP), and
# CASTLE_SCARE drops its scrolls at 4 squares instead of 2. Inert before depth 25 on the main line.
LANDING_GUARD = False
# VALLEY_XORN (valley-exit; dive_logic.valley_xorn): a wall-walking polymorph form in the Valley (castle-first-pass's
# CFP_XORN crosses the castle as a xorn) phases through the Valley's rock to its '>' (gehennom.des: NON_DIGGABLE but no
# NON_PASSWALL) and goes down; the Gehennom dig-dive goes on below (a xorn keeps its weapon and hands). Acts only in
# the Valley in a wall-walking form.
# ON (e1d198e + this): real arm jf16 s0 pinned 0.691/D30 -> 0.723/D33; xornkit_salted 14: 5 Valley arrivals all exit
# (D34 x4, D31) vs all dying at D30 off; identity off/on until 2 turns after the Valley arrival; crash-clean (vxk-on6,
# id-t5-on) -- the turn-30000 DRIVER hang is a harness-only artifact (#polyself before the first attribute parse).
VALLEY_XORN = True

# --- castle-first-pass (castle_cross.py): get off the castle's west landing onto the moat before the landing kills us ---
# CFP_RUSH: on the castle's west side a LASTING lift (known lev ring/boots, then unknown rings, then unknown magic
# boots) goes on at once, above fight2/elbereth_rest/the scare hold, and a floating hero digs straight from the maze
# to (-1,2)/(-1,14) and floats onto the moat ring's west column at (0,1)/(0,15) (9 water squares to the dry strip
# instead of 14 from the courtyard corner). Over the moat no walker reaches us. Pinned-clock cfp-p2-jf16 s0: a ring
# of levitation in the kit, dead 25 turns after landing (master lich + elf-lord) with the ring never tried.
# ON in the train-4 candidate with CFP_MB/CFP_ZAP/CFP_XORN/POLY_XORN: the set of the first real castle pass (ledger
# R082/R083: arm jf16 s0 0.691, Dlvl 30); all act only on the castle level (POLY_XORN: a polymorph-control prompt at
# depth 25+).
CFP_RUSH = True
# CFP_MB (with CFP_RUSH): magical breathing walks the moat bottom (trap.c drown: 'But you aren't drowning. You touch
# bottom.'); an unknown amulet is put on in the lift phase and water-tested by the step off the dug launch square (no
# MB: we crawl back out onto it); a confirmed amulet or an amphibious/breathless polymorph form counts as floating for
# castle_logic's crossing. cfp-p2-jf25 s3's kit held an amulet of magical breathing and a skeleton key, never used.
CFP_MB = True
# CFP_ZAP: what blocks the way round the moat gets the best known wand first -- teleportation, striking (beams), then
# sleep/lightning/fire/cold/magic missile rays only where the ray can't bounce back (castle_logic zapped only striking)
CFP_ZAP = True
# CFP_XORN (with POLY_XORN, power-route): before CASTLE_POLY's self-zap put on the rings that may be polymorph control
# (the lift test took them off), stop zapping once we are a wall-walker, and as a xorn walk through the castle's walls
# (no NON_PASSWALL in castle.des) along the walled north corridor to a trap door (40..55,08): trap.c fall_through on
# the stronghold is find_hell(), the Valley. cfp-v1-arm jf16 s0 zapped its wand of polymorph 6 times into random forms
# with the ring of polymorph control in its pack (taken off by the lift test).
CFP_XORN = True
# CFP_PRUSH (with CFP_RUSH): potions that may lift us are quaffed one square west of the dug launch square (-1,2)/
# (-1,14) on Elbereth, not in the courtyard: a potion's 10-149 turns then start 2 moves from the moat with 9 water
# squares to the strip (14 from the courtyard), and no walk through the maze. Potions go before the magical-breathing
# water test (the dunk dilutes them: cfp-g1-arm cfpi-s1 turned its diluted potion of levitation into water).
# OFF in the train-4 candidate: castle-real-all 1/61 with it vs 2/61 without (t4-real-on, cfp-cra-prush): the rush to the
# launch square walks/digs through the maze on foot, and jf25-s0 met a troll there instead of quaffing its potion of
# levitation at the courtyard's TEST_SPOT (which crosses and passes without it)
CFP_PRUSH = False
# CFP_INVIS (with CFP_RUSH): on the castle, before the lifts, go permanently invisible with a KNOWN wand of make
# invisible (zapped at ourselves: zap.c 'ordinary' -> HInvis FROMOUTSIDE) or ring of invisibility (a worn mummy wrapping
# blocks it and comes off first): monmove.c set_apparxy makes a monster that can't see us guess our square each time we
# move (ours 1 time in 3, else a random accessible neighbour), so sharks, eels and minotaurs land ~40-55% of their melee
CFP_INVIS = False

# --- power-route (power_route.py): teleport control + a level-teleport trigger from what the dive carries ---
# TC_ROUTE (off): learn which ring gives teleport control from the game's prompts ('Where do you want to be
# teleported?' / 'To what level do you want to teleport?' with one unknown ring worn) and the tengu message, identify
# by reading (the identify menu picks rings, then teleport scrolls' BUC, wands, potions), and jump as soon as TC and a
# SURE trigger are in hand: a cursed scroll of teleportation read unconfused, or a level teleporter (teleport.c:
# confused answers to the level prompt are random 4 times in 5). From the Dungeons that lands in the Valley (castle+1),
# from Gehennom on Dlvl 50 or the vibrating-square level (LEVELPORT_DEEP). Evidence: the 61 real castle kits held a TC
# ring 4 times and a teleport scroll 34 times (4 cursed); jf16-s14 (Dlvl-29 castle, base3) carried the TC ring, a cursed
# teleport scroll and 4 identify scrolls, and died 13 turns after landing with all of it unused.
# ON (merge c44fe78): 45-game guard, dive-only version, pooled ~+0.025 (public 0.5112 vs base10 0.5023, jf14 0.4387 vs
# 0.4243, jf16 0.4611 vs 0.4094; ledger R079); harness designed kit pr-tca7b 4/7 into the Valley (Dlvl 30).
TC_ROUTE = True
# the castle once the lift plan has given up (castle_logic.given_up): dying there costs nothing and an uncontrolled level
# teleport only goes up, so read-identify, put rings that may give TC on, and pull every teleport trigger
TC_CASTLE_GAMBLE = True
# with TC known and only uncursed teleport scrolls: get confused (a stack that confused us before -- a cursed confuse
# monster scroll -- or a known potion of confusion/booze) and read one: controlled 1 time in 5
# (rnl(5) at Luck 0), else a random level (the score keeps the deepest). In Gehennom (the Valley kills an XL-8 hero in
# ~60 turns), on the castle, or from TC_LOTTERY_DEPTH down
TC_LOTTERY = True
TC_LOTTERY_DEPTH = 10
# with TC known: dip the teleport scrolls into known unholy water (potion.c H2Opotion_dip curses the whole stack: every
# scroll becomes a sure controlled jump); at the castle's end state also into water of unknown BUC (1 in 8 unholy,
# plain water blanks them)
TC_DIP = True
# a ring known (or proven by a TC prompt) to give teleport control goes on and stays on: a level teleporter stepped on
# anywhere in the Dungeons is then the Valley (6% of games step on one)
TC_WEAR = True
# identify by reading outside the castle at quiet moments (no hostile within 6, HP >= 60%, not Weak): known identify
# scrolls and unknown stacks of 2+ while unknown rings/wands/potions or a teleport scroll of unknown BUC are carried
TC_DIVE_ID = True
# drop carried rings/potions/scrolls/amulets of unknown BUC on a known altar of the current level (the grind's Dlvl 1-3
# hold ~90% of the kit's scrolls and potions: 25 of 90 base10 games stood on an altar there); the bot picks them up again
TC_ALTAR = True
# try on (and take off) unknown rings whose BUC is known not cursed, at a quiet moment of the dive: a levitation ring
# names itself ('You start to float in the air!') -- a certain lift for the castle and Medusa's islands
TC_RING_TEST = True
# polymorph control's 'Become what kind of monster?' (polyself.c) was answered ESC = a random form; on the castle answer
# 'xorn' instead: it walks through walls (M1_WALLWALK; castle.des has no NON_PASSWALL) to the trap doors behind the
# back door and falls through into the Valley (castle-first-pass moves the form; jf16-s7's Dlvl-29 kit holds a ring of
# polymorph control). ON in the train-4 candidate (the xorn pass set, see CFP_RUSH).
POLY_XORN = True

# STAIR_BOULDER_FIX (off, power-route; ledger B009): a boulder on the staircase dive_logic._take_stairs wants. The BFS
# never enters a boulder square, so it pushed the boulder from next to the stairs; with a wall behind it 'You try to
# move the boulder, but in vain.' takes no game time and the dive spun there (jf14 s4: ~300k steps on Mines 2 with a
# pick-axe and a wand of digging in the pack, then a soldier ant; 0.075 vs 0.602). After a failed push the boulder is
# broken (pick-axe/mattock applied at it, or a known wand of striking), else pushed from a side with floor behind it,
# else the stairs are left alone for STAIR_BOULDER_WAIT turns
# ON (R096): jf14 s4 0.075 -> 0.602 (pick-axe breaks the boulder on Mines 2's '<'); flag off byte-identical on 3 pinned
# seeds; 45-game guard fired only there, 0 unexpected exceptions / driver restarts.
STAIR_BOULDER_FIX = True
STAIR_BOULDER_WAIT = 300

# SOKOBAN_TRIP (off, power-route): when the grind would hand over to the dive, run the tour's Sokoban milestones first
# (dive_logic._sokoban_trip): 4 random rings + 4 random wands + the prize (bag of holding / amulet of reflection) --
# the ingredients the castle and teleport-control routes lack (TC ring in 4 of 61 real castle kits)
SOKOBAN_TRIP = False
SOKOBAN_TRIP_TURNS = 8000

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
