# Changes over the parent engine (pf_s25p)

- Human-strategy preparation revision (experimental, 2026-10-27): ongoing
  Gnome Wizard food resupply through the existing paid shop/tool recovery
  transaction; safe utility spell study, divination practice and identification;
  metal armor trials remain a separate experiment;
  and one new Sokoban opportunity after an earlier supply route ends. Ordinary
  no-tool purchasing keeps its existing target. Native scenarios and separate
  component/combined ordinary games evaluate useful outcomes; no promotion or
  publication is implied by these policy additions.

- Gnome Wizard leader-safety revision (experimental, 2026-10-27): selectively
  adapts `daglar-dragomirov/nethacker@e8ff9e6` ray branch accounting, terrain
  collision and miss-continuation checks; urgent care before escape handlers;
  innate rescue-wand failure checks; minotaur sleep attribution; and observed
  horn identity. The emergency split preserves our hunger recovery and avoids
  repeating its state checks in the lower last-resort pass. New behavior is
  limited to Gnome Wizards. Existing preparation, pet care, dive thresholds,
  prayer gaps and force-bolt configuration remain in place. This is private
  development, with component and combined comparisons recorded separately;
  source correctness and regression tests do not establish a gameplay gain.

- Preparation continuity revision: brief reversible impairments pause branch supply visits within their original deadlines. Known identify scrolls can identify useful owned armor before diving; nearby altar testing, explicit BUC requirements, magic resistance retention, and safe fitting remain in place.

- Oracle preparation revision: fully funded major consultation before the minor, with one-time purchase accounting and food/gold reserves. Bounded Oracle-level fountain dipping uses inert owned stones, poison protection, a known teleport escape, and public flood-collateral checks. Any summon, flood, or wish ends further attempts. Fountain wishes prioritize protective equipment; requested blessing is never treated as observed BUC.

- Equipment identification revision: nearby altar testing for owned useful armor and scrolls; shared explicit public BUC checks before enchantment and equipment decisions; blessed remove curse for valuable recovered carried armor; retained magic resistance and interrupted armor fitting. This revision is evaluated separately from the submitted recovery parent.

- **Gnome Wizard routing-1 (experimental)**: bounded upper-Mines/Minetown and
  Sokoban preparation, observed branch connections, emergency release and
  resource collection after each solved Sokoban floor. Peaceful blockers no
  longer replace movement destinations; known stair routes use shortest paths.
  Blocked Sokoban pushes retain retry limits across preemption and never break
  puzzle boulders. Pet hunger and credit preparation remain enabled.
  Preparation food visits can temporarily leave one owned digging tool outside
  a known shop, buy and settle priced food, and recover the tool. The bounded
  visit preserves debt and tool recovery records across interruptions.
  Optional floor-item inspection remembers definite navigation failures for a
  bounded interval, while required travel and emergency food remain available.
  The combined survival candidate spends owned supplies during preparation,
  pauses briefly for observed hunger relief without renewing visit budgets,
  and handles declined stair navigation without assuming arrival. Known owned
  equipment preparation runs independently of optional loot gathering.
  Ordinary Gnome Wizard progression surveys floors despite carrying a pick,
  favors stairs, and retains combat and urgent escape behavior.
  Recovery permits free interface observations to finish before requiring a
  world-turn action, and useful protection stays on through transient hunger
  with available food. Weapon estimates apply the engine's -2 unskilled damage
  penalty instead of the inherited +2 sign error.
  Altar preparation uses bounded visits to a confirmed matching altar with
  fresh surplus corpses, preserving food, pet care, debt and emergency work.
  It observes and recovers the actual gift, retains Magicbane, and equips it
  from public noncurse evidence or a verified fresh gift transaction.
  Gift messages reset the prayer model's cooldown; later public reconciliation
  ends the optional cooldown-repair offerings. There is no altar camping or
  guarantee that a sacrifice yields Magicbane.

- **v9p (private-safe router)**: v5 plus only the specialist routes that beat nhbot on the hub's VERIFIED (private-seed)
  tier -- Gnomish Archeologists on pf_v36, neutral human Rangers and the female elven Ranger on pf_s25p8, Knights on
  pf_s25p8 -- and no daglar profile for Rangers, Rogues and Tourists (every Ranger and Rogue identity scored lower with
  it); Tourists back on nhbot. The public-seed-selected ports of v6/v7 overfit the 15 public seeds and are left out.
- **v9p2**: v9p plus three routes with one verified sample each: neutral human Archeologists and orcish Barbarians on
  pf_s25p8, neutral human Wizards on pf_vk_s25.

- **v10a**: Healers and Samurai back on nhbot: every Healer and Samurai specialist lost to nhbot on 90 dev held-out
  seeds (hea-gno pf_hg -0.050, hea-hum pf_hh -0.024, sam pf_v35 -0.051).
- **v10b**: dev-confirmed config: Rangers on nhbot without UNSEEN_PET_GUARD (+0.022 +- 0.012, 420 pairs), gnomish
  Cavemen on the Dlvl-1 grind (+0.073 +- 0.022, 120 pairs).
- **v10c**: neutral human Wizards on pf_s25p8 instead of pf_vk_s25 (dev held-out: 0.252 vs 0.153 over 90 seeds; the
  verified tier agrees).
- **v11pub** (public board only): per identity the best public-seed setup among v10c, v7, v9p2 and v4 (same nhbot
  code; tools/mkpub.py), as explicit per-identity routes and configs. Public-seed selection overfits by design: for the
  private/verified tier use v10c.
- **eL1fe's fixes** (ported from eL1fe/nethacker@cfc3828, its `dag25` engine = this same base): the spell-direction fix
  (`Agent.cast` sent the compass string as keys: every aimed cast was wasted), force bolt for Wizards (from
  CleverShovel 0d1fb22: never into shops, never flying on into a pet), healing spells for any role that knows them,
  stoning cures, a known teleport scroll as a last resort, no throwing where the pet may stand unseen, Wizards and
  Healers keep off metal body armor/gloves/heavy shields (spell failure), Archeologists dig out of shops, magic
  mapping from Dlvl 3, trees unwalkable, Sokoban/altar robustness. Wizards grind on Dlvl 1-3 as in eL1fe's engine;
  Rogues, Knights, Tourists and Priests keep the Dlvl-1-only grind (`ROLE_GRIND_LEVELS`).
  (I found the cast bug independently; eL1fe's version is used.)
- **Identity router** as in daglar-dragomirov/nethacker@e29eb82 (the verified-tier leader): Healers (pf_hg, pf_hh)
  and Samurai (pf_v35) play specialist engines; everyone else plays nhbot with DIVE_XL 8
  (nhbot is then identical to e29eb82's pf_base apart from the changes listed here).
- `roles.py`: per-identity overrides of engine settings (only for nhbot identities).
- `MEDUSA_HOP` (off pending A/B): on Medusa's level with no dry square on our islet, step into a one-square moat
  channel toward land that has one (trap.c drown(): the hero crawls out at once to a random free land square).
- `MINES_TOOL_TRIP` (on; +0.018 +- 0.007 per game over 272 paired held-out games): a tool-less planned dive takes the Mines route for any race (it was for
  dwarves and gnomes only), so humans, elves and orcs go and take a dwarf's pick-axe.
- DT6A's two-page `#enhance` fix (DT6A/nethacker@c9a42ac): tty menus restart item letters on each page; the parser's
  assert then fired on every fight start.
- CleverShovel's ring/amulet/scroll module (CleverShovel/nethacker@9dc0822 V1, @29ab0a7 V2), Wizards only, behind
  `RING_MODULE` (A/B pending): wears identified useful rings (slow digestion, free action, poison resistance...),
  combat-only rings in fights, sheds rings when Hungry, reads scrolls at safe moments to identify rings/amulets.
- Archeologists dig-dive from XL 5 instead of 3 (`ARC_DIG_DIVE_XL`): +0.024 +- 0.013 over 236 paired held-out games.
- Small fixes found by log analysis: the panic-loop breaker now also forbids a square a monster keeps blocking
  (`PANIC_TILE_FIX`; one grind sat 38k turns), divers keep fighting with the wielded pick-axe/mattock unless the best
  weapon is clearly better (`DIG_TOOL_MELEE`), a timed-out pet ditch is retried (`DITCH_RETRY`), a trapping gas spore is
  only hit when its blast can't kill (`SPORE_TRAP_FIX`).
- Known cold/fire wands are zapped only with a free run of squares behind the target (`RAY_BOUNCE_FIX`): seven logged
  deaths came from the bot's own bounced ray.
- `AT_THREAT_AVOID` (off pending A/B): while an elf, soldier or other Elbereth-ignoring meleer comes for a pick-axe
  digger above Medusa's level, no pit is started (every attack stops the dig; in the pit the fight is at -3 to-hit): a
  known wand of digging holes the floor at once, else the @ is fought on level ground first.
- **v3:** human Priests play nhbot instead of pf_pa (+0.053 +- 0.022 per game over 113 paired held-out games); bug fixes
  from a log study of the games that end above Dlvl 10 (`research` notes in the workspace): a prayer interrupted by a
  strategy switch is still recorded (`PRAYER_RECORD_FIX`), the prayer model's alignment record follows pray.c (+1 only
  for prayers without major trouble, `RECORD_MODEL_FIX`), fight2 charges a ray's bounce through us its real cost
  (`SELF_ZAP_FIX`), no pickup that leaves us Stressed and a lycanthrope's corpse weighs what its human form does
  (`HEAVY_LIFT_GUARD`); a garbled Elbereth is rewritten after a mid-dig faint (`ELBERETH_REWRITE_FIX`).
- Tried and left off (flags kept for reference): `AT_THREAT_AVOID` (no pit while an elf/soldier comes: -0.004 +- 0.003
  over 235 deterministic pairs), `MEDUSA_HOLE_CYCLE` (skip Medusa's level through our own hole from above: the skip works
  1 time in 4-6, but the extra landings cost more, -0.015 +- 0.005 per Medusa game), Wizards on the Dlvl-1 grind
  (-0.058 +- 0.026 over 101 pairs).
- **v4:** fixes from log studies of the deep game and of Wizards (`research` notes in the workspace). Deep game: lawful
  minions (Aleax, couatl, ...) ignore Elbereth (`LMINION_ELBERETH`), a safe prayer before a heal at critical HP deep in the
  dive (`DEEP_PRAY_FIRST`), no castle-only strategies on a deep Medusa level, Medusa-2 recognised by its titan's messages,
  a stranded Medusa-3 '<' rerolls, minotaur guard fixes (`MINO_*`): +0.0021 +- 0.0012 over 512 deterministic held-out
  pairs vs v3. Wizards: force bolt beats the melee bonus against Elbereth-ignorers and is cast from an Elbereth square at
  what the engraving doesn't scare (`FB_FOCUS`; 52 of 412 Wizard games died meleeing with power left), no casting while
  stunned/confused, no ray-wand zaps in the grind while the bolt is castable; a starving pet gets the corpses, a pet
  turned hostile is fought, no Elbereth on altars. Knights stay on the Dlvl-1 grind (the deep grind: -0.020 +- 0.023 over
  160 pairs); a durable (engraved) Elbereth for holds measured +0.000 over 512 pairs (flag off).

- **v5:** merged daglar-dragomirov/nethacker@6c814836 (the generalist-board leader, mean 0.350) into this v4 tree, which
  was already a strict superset of its engine-code changes (v4's deep-game and Wizard fixes are additive new flags;
  daglar made no edits to those nhbot files). Adopted from daglar: the populated per-identity `OVERRIDES` in `roles.py`
  (arc/cav/mon/pri/wiz play aggressive — `RING_MODULE` and `MEDUSA_HOP` on, the four safety flags off, `DIVE_XL` 8;
  bar/kni/ran/rog/tou/val play safe — the opposite, `ran` `DIVE_XL` 7), the `^X` attributes-screen identity probe
  (`roles.attributes_identity`, so the per-role config applies even when the welcome line is missed), and the Tourist
  engine `pf_dtad7a` (DT6A/nethacker@ad7a864) routed for `tou`. The `RING_MODULE`/`MEDUSA_HOP` module code is
  byte-identical to daglar's, so this reproduces the leader's behaviour exactly, with v4's extra fixes on top.

## Experimental zWizardEconomy
Experimental complete Public program zWizardEconomy: exact Public11 with Wizard-only two-force-bolt energy reserve above XL3 while healthy and known-shop off-path casting. Existing passive/explosive, low-HP, fast-target, ray-bounce, shop-tail, pet and peaceful guards retained. All10 Wizard routes use main engine; other63 identities retain baseline behavior. Open development mechanism evidence, no Private selection. Native73x15 incremental registration after73 smoke, sanity and exact bytes. No independent confirmation scheduled for this exploratory wave; no generalist improvement claimed. Prior best programs retained.
