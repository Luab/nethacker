# Ring/amulet module tuning (ring_amulet_logic.py, scroll_identify.py). Ported from CleverShovel/nethacker@29ab0a7
# (the 9dc0822 module plus its later fixes). The on/off switches and the role gates are jf_config's RING_* flags
# (so JF_CFG and roles.py can A/B them); this file holds the constants.

# put on an unidentified amulet when safe, to find out what it is. do_wear.c: strangulation gives an
# immediate "It constricts your throat!" that we react to by removing the amulet at once
# (Amulet_off() cancels the countdown). A cursed one cannot be #removed ("You can't.  It is cursed."),
# so a prayer must be available up front and is the fallback (pray.c: strangulation is major trouble).
# OFF (jf_config.RING_AMULET_TRIAL): measured by CleverShovel on five leaders it was never better (-1.5% .. -19%): the
# trial rarely identifies the amulet and always takes it off again, so it costs a prayer's timeout and turns for nothing
AMULET_WEAR_MIN_HP_FRAC = 0.6
AMULET_WEAR_SAFE_RADIUS = 4

# wear increase accuracy/damage, protection rings and the reflection amulet (do_wear.c: they only
# change combat resolution) only while a hostile is within ENGAGE_RADIUS, take them off afterwards
ENGAGE_RADIUS = 6
DISENGAGE_COOLDOWN = 10

# wear the rings the character STARTS with and keep them on (jf_config.RING_STARTING_WEAR, OFF: CleverShovel measured
# it on four wizard leaders, mean -0.024). u_init.c ini_inv(): only the Wizard has any (two random rings); every
# starting item is created uncursed with its BUC known, and levitation / hunger / aggravate monster are never
# generated for them. What is left that hurts is polymorph (1 in 100 per turn) and teleportation (1 in 85 per turn).

# --- identifying rings/amulets by reading scrolls (scroll_identify.py, jf_config.RING_SCROLL_IDENTIFY) ---
# safe moment: HP at least this and at least this fraction of max (earth: up to d20; a blessed fire on
# ourselves: 15-25), nobody in sight, a prayer available, no shop in view
SCROLL_MIN_HP = 26
SCROLL_MIN_HP_FRAC = 0.9
SCROLL_READ_COOLDOWN = 25
# never read a scroll that is less likely than this to be identify (price-narrowed candidates)
SCROLL_MIN_P_IDENTIFY = 0.10
# read only when P(identify) >= SCROLL_SEVERE_COST x P(severe outcome), severe = punishment, a cursed
# create monster / genocide, amnesia weighted by what it costs this character (scroll_identify.py)
SCROLL_SEVERE_COST = 3.0
AMNESIA_WEIGHT_PLAIN = 0.1     # the level map and some discoveries
AMNESIA_WEIGHT_CASTER = 0.3    # + rn2(n + 1) of the n known spells (Priest, Healer, Monk)
AMNESIA_WEIGHT_WIZARD = 0.6    # a Wizard's only attack is the force bolt spell he starts with
# harm gate (29ab0a7): read an unknown scroll only if the share of harmful or wasteful outcomes (scroll_identify.HARMFUL)
# among its still possible types is at most this (None = no gate, as in 9dc0822). A single unpriced scroll is 0.30, a
# pair 0.15, a price-narrowed {identify, light, enchant weapon/armor, remove curse} 0.
SCROLL_MAX_HARM = 0.16
# (29ab0a7) weigh a stack of n scrolls of one appearance as n draws of the same type (prob ** n)
SCROLL_STACK_ODDS = True
# stderr diagnostics (RINGSTAT scroll_dbg) for a scroll read that identified nothing
SCROLL_DEBUG = False

# wear identified rings/amulets that help whenever they are worn (the combat-only ones have their own rule)
ALWAYS_WEAR_RINGS = ('slow digestion', 'free action', 'poison resistance', 'gain constitution', 'gain strength')
ALWAYS_WEAR_AMULETS = ('amulet of life saving', 'amulet versus poison')

# no ring/amulet logic from this depth on: the castle (depth 25-29) and Gehennom belong to the tree's
# own levitation / magical breathing / teleport-control machinery, which wears and removes rings and
# amulets on purpose -- removing a levitation ring over the moat would drown the character
MAX_DEPTH = 24

# --- port guards (not in CleverShovel's module) ---
# RINGSTAT lines on stderr for every put on / remove / read (CleverShovel's measurement log)
RING_DEBUG = False
# chargeable rings: put on only at a known positive enchantment (a found ring of a type we already know -- the
# Wizard's second ring of protection -- can be -3; the item parser calls an unknown BUC uncursed)
CHARGED_RINGS = ('adornment', 'gain strength', 'gain constitution', 'increase accuracy', 'increase damage',
                 'protection')
# a put on that fails takes no game time ('There are no more ring-fingers to fill.', gloves, a welded weapon):
# after this many failures of one glyph, leave it alone for PUTON_FAIL_COOLDOWN turns (agent.py asserts on 200 steps
# without a turn)
PUTON_FAIL_LIMIT = 2
PUTON_FAIL_COOLDOWN = 500
# never shed for hunger: eat.c gethungry() -- slow digestion cancels the 1/turn move hunger and costs 1 per 20 turns
NEVER_SHED = ('slow digestion',)
