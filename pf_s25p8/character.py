import re

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

from . import objects as O

ALL_SPELL_NAMES = [
    "force bolt",
    "drain life",
    "magic missile",
    "cone of cold",
    "fireball",
    "finger of death",
    "protection",
    "create monster",
    "remove curse",
    "create familiar",
    "turn undead",
    "detect monsters",
    "light",
    "detect food",
    "clairvoyance",
    "detect unseen",
    "identify",
    "detect treasure",
    "magic mapping",
    "sleep",
    "confuse monster",
    "slow monster",
    "cause fear",
    "charm monster",
    "jumping",
    "haste self",
    "invisibility",
    "levitation",
    "teleport away",
    "healing",
    "cure blindness",
    "cure sickness",
    "extra healing",
    "stone to flesh",
    "restore ability",
    "knock",
    "wizard lock",
    "dig",
    "polymorph",
    "cancellation",
]

ALL_SPELL_CATEGORIES = [
    "attack",
    "healing",
    "divination",
    "enchantment",
    "clerical",
    "escape",
    "matter",
]


class Property:
    def __init__(self, agent):
        self.agent = agent

    # The tty status line abbreviates conditions when it gets long (e.g. 'Hallu' -> 'Hl'),
    # so read them from the blstats condition bitmask instead.
    def _condition(self, mask):
        return bool(self.agent.last_observation['blstats'][nh.NLE_BL_CONDITION] & mask)

    @property
    def confusion(self):
        return self._condition(nh.BL_MASK_CONF)

    @property
    def stun(self):
        return self._condition(nh.BL_MASK_STUN)

    @property
    def hallu(self):
        return self._condition(nh.BL_MASK_HALLU)

    @property
    def blind(self):
        return self._condition(nh.BL_MASK_BLIND)

    @property
    def polymorph(self):
        if not nh.glyph_is_monster(self.agent.glyphs[self.agent.blstats.y, self.agent.blstats.x]):
            return False
        return self.agent.character.self_glyph != self.agent.glyphs[self.agent.blstats.y, self.agent.blstats.x]


class Character:
    UNKNOWN = -1  # for everything, e.g. alignment

    ARCHEOLOGIST = 0
    BARBARIAN = 1
    CAVEMAN = 2
    HEALER = 3
    KNIGHT = 4
    MONK = 5
    PRIEST = 6
    RANGER = 7
    ROGUE = 8
    SAMURAI = 9
    TOURIST = 10
    VALKYRIE = 11
    WIZARD = 12

    name_to_role = {
        'Archeologist': ARCHEOLOGIST,
        'Barbarian': BARBARIAN,
        'Caveman': CAVEMAN,
        'Cavewoman': CAVEMAN,
        'Healer': HEALER,
        'Knight': KNIGHT,
        'Monk': MONK,
        'Priest': PRIEST,
        'Priestess': PRIEST,
        'Ranger': RANGER,
        'Rogue': ROGUE,
        'Samurai': SAMURAI,
        'Tourist': TOURIST,
        'Valkyrie': VALKYRIE,
        'Wizard': WIZARD,
    }

    CHAOTIC = 0
    NEUTRAL = 1
    LAWFUL = 2
    UNALIGNED = 3

    name_to_alignment = {
        'chaotic': CHAOTIC,
        'neutral': NEUTRAL,
        'lawful': LAWFUL,
        'unaligned': UNALIGNED,
    }

    HUMAN = 0
    DWARF = 1
    ELF = 2
    GNOME = 3
    ORC = 4

    name_to_race = {
        'human': HUMAN,
        'dwarf': DWARF,
        'dwarven': DWARF,
        'elf': ELF,
        'elven': ELF,
        'gnome': GNOME,
        'gnomish': GNOME,
        'orc': ORC,
        'orcish': ORC,
    }

    MALE = 0
    FEMALE = 1

    name_to_gender = {
        'male': MALE,
        'female': FEMALE,
    }

    name_to_skill_type = {
        # "": O.P_NONE,
        "dagger": O.P_DAGGER,
        "knife": O.P_KNIFE,
        "axe": O.P_AXE,
        "pick-axe": O.P_PICK_AXE,
        "short sword": O.P_SHORT_SWORD,
        "broadsword": O.P_BROAD_SWORD,
        "long sword": O.P_LONG_SWORD,
        "two-handed sword": O.P_TWO_HANDED_SWORD,
        "scimitar": O.P_SCIMITAR,
        "saber": O.P_SABER,
        "club": O.P_CLUB,
        "mace": O.P_MACE,
        "morning star": O.P_MORNING_STAR,
        "flail": O.P_FLAIL,
        "hammer": O.P_HAMMER,
        "quarterstaff": O.P_QUARTERSTAFF,
        "polearms": O.P_POLEARMS,
        "spear": O.P_SPEAR,
        "trident": O.P_TRIDENT,
        "lance": O.P_LANCE,
        "bow": O.P_BOW,
        "sling": O.P_SLING,
        "crossbow": O.P_CROSSBOW,
        "dart": O.P_DART,
        "shuriken": O.P_SHURIKEN,
        "boomerang": O.P_BOOMERANG,
        "whip": O.P_WHIP,
        "unicorn horn": O.P_UNICORN_HORN,

        "attack spells": O.P_ATTACK_SPELL,
        "healing spells": O.P_HEALING_SPELL,
        "divination spells": O.P_DIVINATION_SPELL,
        "enchantment spells": O.P_ENCHANTMENT_SPELL,
        "clerical spells": O.P_CLERIC_SPELL,
        "escape spells": O.P_ESCAPE_SPELL,
        "matter spells": O.P_MATTER_SPELL,

        "bare handed combat": O.P_BARE_HANDED_COMBAT,
        "martial arts": O.P_BARE_HANDED_COMBAT,
        "two weapon combat": O.P_TWO_WEAPON_COMBAT,
        "riding": O.P_RIDING,
    }

    possible_skill_types = ['Fighting Skills', 'Weapon Skills', 'Spellcasting Skills']
    possible_skill_levels = ['Unskilled', 'Basic', 'Skilled', 'Expert', 'Master', 'Grand Master']

    SKILL_LEVEL_RESTRICTED = 0
    SKILL_LEVEL_UNSKILLED = 1
    SKILL_LEVEL_BASIC = 2
    SKILL_LEVEL_SKILLED = 3
    SKILL_LEVEL_EXPERT = 4
    SKILL_LEVEL_MASTER = 5
    SKILL_LEVEL_GRAND_MASTER = 6

    weapon_bonus = {
        SKILL_LEVEL_RESTRICTED: (-4, 2),
        SKILL_LEVEL_UNSKILLED: (-4, 2),
        SKILL_LEVEL_BASIC: (0, 0),
        SKILL_LEVEL_SKILLED: (2, 1),
        SKILL_LEVEL_EXPERT: (3, 2),
    }
    two_weapon_bonus = {
        SKILL_LEVEL_RESTRICTED: (-9, -3),
        SKILL_LEVEL_UNSKILLED: (-9, -3),
        SKILL_LEVEL_BASIC: (-7, -1),
        SKILL_LEVEL_SKILLED: (-5, 0),
        SKILL_LEVEL_EXPERT: (-3, 1),
    }
    riding_bonus = {
        SKILL_LEVEL_RESTRICTED: (-2, 0),
        SKILL_LEVEL_UNSKILLED: (-2, 0),
        SKILL_LEVEL_BASIC: (-1, 0),
        SKILL_LEVEL_SKILLED: (0, 1),
        SKILL_LEVEL_EXPERT: (0, 2),
    }
    unarmed_bonus = {
        SKILL_LEVEL_RESTRICTED: (1, 0),
        SKILL_LEVEL_UNSKILLED: (1, 0),
        SKILL_LEVEL_BASIC: (1, 1),
        SKILL_LEVEL_SKILLED: (2, 1),
        SKILL_LEVEL_EXPERT: (2, 2),
        SKILL_LEVEL_MASTER: (3, 2),
        SKILL_LEVEL_GRAND_MASTER: (3, 3),
    }
    martial_bonus = {
        SKILL_LEVEL_RESTRICTED: (1, 0),  # no one has it restricted
        SKILL_LEVEL_UNSKILLED: (2, 1),
        SKILL_LEVEL_BASIC: (3, 3),
        SKILL_LEVEL_SKILLED: (4, 4),
        SKILL_LEVEL_EXPERT: (5, 6),
        SKILL_LEVEL_MASTER: (6, 7),
        SKILL_LEVEL_GRAND_MASTER: (7, 9),
    }

    name_to_skill_level = {k: v for k, v in zip(['Restricted'] + possible_skill_levels,
                                                [SKILL_LEVEL_RESTRICTED,
                                                 SKILL_LEVEL_UNSKILLED, SKILL_LEVEL_BASIC, SKILL_LEVEL_SKILLED,
                                                 SKILL_LEVEL_EXPERT,
                                                 SKILL_LEVEL_MASTER, SKILL_LEVEL_GRAND_MASTER])}

    def __init__(self, agent):
        self.agent = agent
        self.prop = Property(agent)
        self.role = None
        self.alignment = None
        self.race = None
        self.gender = None
        self.self_glyph = None
        self.skill_levels = np.zeros(max(self.name_to_skill_type.values()) + 1, dtype=int)
        self.upgradable_skills = dict()

        self.is_lycanthrope = False
        # 'rat' / 'jackal' / 'wolf' (were.c were_beastie families), None if unknown (LYCAN_FIXES)
        self.lycanthrope_kind = None
        # HP before the current polymorph: while polymorphed, blstats show the form's HP, and that form
        # dying only rehumanizes us (polyself.c rehumanize) back to this HP (LYCAN_FIXES)
        self.hp_before_poly = None
        self._was_polymorphed = False
        # intrinsics from corpses (eat.c cpostfx), attrcurse takes them away: read by jf_config.SHOP_GUARD
        self.teleportitis = False
        self.teleport_control = False
        # intrinsic telepathy from a floating eye corpse (eat.c cpostfx), lost to sit.c attrcurse: read by
        # jf_config.FEYE_TELE (an extrinsic source -- helm of telepathy, amulet of ESP -- prints nothing and is not seen)
        self.telepathic = False

    def _track_telepathy(self, msg):
        # eat.c cpostfx TELEPAT: 'You feel a strange mental acuity.' ('in touch with the cosmos.' hallucinating);
        # sit.c attrcurse takes it away with 'Your senses fail!'
        if 'You feel a strange mental acuity' in msg or 'You feel in touch with the cosmos' in msg:
            if not self.telepathic:
                self.agent.log('INTRINSIC telepathy')
            self.telepathic = True
        if 'Your senses fail' in msg:
            self.telepathic = False

    def _track_teleport(self, msg):
        # eat.c cpostfx: 'You feel very jumpy.' (hallucinating: 'diffuse.') gives teleportitis, 'You feel in control
        # of yourself.' ('centered in your personal space.') teleport control; sit.c attrcurse: 'You feel less jumpy.'
        if 'You feel very jumpy' in msg or 'You feel diffuse' in msg:
            if not self.teleportitis:
                self.agent.log('INTRINSIC teleportitis')
            self.teleportitis = True
        if 'You feel less jumpy' in msg:
            self.teleportitis = False
        if 'You feel in control of yourself' in msg or 'centered in your personal space' in msg:
            self.teleport_control = True

    _WERE_KIND = re.compile(r'\bwere(rat|jackal|wolf)\b')
    _TURN_INTO_WERE = re.compile(r'You turn into an? were(rat|jackal|wolf)!')

    def update(self):
        from . import jf_config
        if not jf_config.LYCAN_FIXES:
            if 'You feel feverish.' in self.agent.message:
                self.is_lycanthrope = True
            if 'You feel purified.' in self.agent.message:
                self.is_lycanthrope = False
            self._track_teleport(self.agent.message)
            self._track_telepathy(self.agent.message)
            return
        # every message since the last update: infections and changes happen inside atomic operations
        # (fights, searches), and the old check of the last message alone missed some of them
        history = self.agent._message_history
        start = getattr(self, '_history_seen', 0)
        if start > len(history):   # a fresh agent after a driver restart
            start = 0
        self._history_seen = len(history)
        msg = ' '.join(history[start:] + [self.agent.message])
        self._track_teleport(msg)
        self._track_telepathy(msg)
        # infected while fainted or asleep the message is 'You dream that you feel feverish.' (75 of 591
        # infections in the dev runs): the old exact match never saw it, so no cure prayer came and the
        # bot went on eating jackal corpses (jf25 s10: 'You cannibal!', Luck -2..-5, next prayer failed)
        pos = msg.rfind('feel feverish')
        if pos >= 0:
            self.is_lycanthrope = True
            # the bite may come a turn earlier than the fever ('The wererat bites!' T9061, 'You feel
            # feverish.' T9062 in public s4): look a few messages back for the biter
            context = ' '.join(history[max(0, start - 8):start]) + ' ' + msg[:pos]
            kinds = self._WERE_KIND.findall(context)
            self.lycanthrope_kind = kinds[-1] if kinds else None
            self.agent.log(f'LYCAN infected (kind {self.lycanthrope_kind})')
        m = None
        for m in self._TURN_INTO_WERE.finditer(msg):
            pass
        if m is not None:
            if not self.is_lycanthrope or self.lycanthrope_kind != m.group(1):
                self.agent.log(f'LYCAN changed into a were{m.group(1)}')
            self.is_lycanthrope = True
            self.lycanthrope_kind = m.group(1)
        if 'You feel purified' in msg[max(pos, 0):]:
            self.is_lycanthrope = False
            self.lycanthrope_kind = None
            self.agent.log('LYCAN cured')
        polymorphed = self.prop.polymorph
        if polymorphed and not self._was_polymorphed:
            bl = self.agent.blstats
            # the last HP seen before the change (this step already shows the new form's HP)
            self.hp_before_poly = getattr(self, '_last_normal_hp', None)
            self.agent.log(f'POLY changed form; HP before: {self.hp_before_poly} now {bl.hitpoints}/{bl.max_hitpoints}')
        if not polymorphed:
            bl = self.agent.blstats
            self._last_normal_hp = (bl.hitpoints, bl.max_hitpoints)
            self.hp_before_poly = None
        self._was_polymorphed = polymorphed

    def were_family(self):
        """Corpse species that are cannibalism for us now (eat.c maybe_cannibal: were_beastie(pm) ==
        u.ulycn): the whole family of our lycanthropy, or of all three when we don't know which."""
        families = {'rat': ('sewer rat', 'giant rat', 'rabid rat', 'wererat'),
                    'jackal': ('jackal', 'fox', 'coyote', 'werejackal'),
                    'wolf': ('wolf', 'warg', 'winter wolf', 'werewolf')}
        if not self.is_lycanthrope:
            return ()
        if self.lycanthrope_kind in families:
            return families[self.lycanthrope_kind]
        return tuple(n for names in families.values() for n in names)

    def poly_hp_is_buffer(self):
        """Polymorphed (a were form) with healthy HP to return to: the form's low HP is no emergency
        -- when it hits 0 we just rehumanize (polyself.c) with the HP we had before the change."""
        if not self.prop.polymorph or self.hp_before_poly is None:
            return False
        hp, maxhp = self.hp_before_poly
        return hp >= max(12, maxhp / 3)

    @property
    def carrying_capacity(self):
        # TODO: levitation, etc
        cap = min(1000, (self.agent.blstats.strength_percentage + self.agent.blstats.constitution) * 25 + 50)
        # UNSQUEEZE: boxed in by diagonal squeezes (hack.c cant_squeeze_thru: > 600 carried), keep under 600
        if self.agent.blstats.time < getattr(self.agent, '_squeeze_cap_until', -1):
            from . import jf_config
            # SQUEEZE_KEEP: Item.unit_weight takes the heaviest candidate object, so the estimate never undershoots
            # hack.c's 600; 550 left ~10 for the wands once the weapon, armour and tool were in (cmp-main jf41 s13)
            cap = min(cap, jf_config.SQUEEZE_KEEP_CAP if jf_config.SQUEEZE_KEEP else 550)
        return cap

    def parse(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.ATTRIBUTES)
            text = ' '.join(self.agent.popup)
            self._parse(text)
            self.self_glyph = self.agent.glyphs[self.agent.blstats.y, self.agent.blstats.x]

    def _parse(self, text):
        matches = re.findall('You are a ([a-z]+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+).', text)
        if len(matches) == 1:
            alignment, _, gender, race, role = matches[0]
        else:
            matches = re.findall(
                # rank titles can contain hyphens (a Valkyrie at XL 10-13 is a "Woman-at-arms")
                'You are an? ([a-zA-Z -]+), a level (\d+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+). *You are ([a-z]+)',
                text)
            assert len(matches) == 1, repr(text)
            _, _, _, gender, race, role, alignment = matches[0]

        if not gender:
            if role == 'Priestess':
                gender = 'female'
            elif role == 'Priest':
                gender = 'male'
            elif role == 'Caveman':
                gender = 'male'
            elif role == 'Cavewoman':
                gender = 'female'
            elif role == 'Valkyrie':
                gender = 'female'
            else:
                assert 0, repr(text)

        self.role = self.name_to_role[role]
        self.alignment = self.name_to_alignment[alignment]
        self.race = self.name_to_race[race]
        self.gender = self.name_to_gender[gender]

    def parse_spellcast_view(self):
        self.known_spells = dict()
        self.spell_fail_chance = dict()

        # TODO: parse for other spellcaster classes
        if self.role not in (self.HEALER,):
            return

        with self.agent.atom_operation():
            self.agent.step(A.Command.CAST)
            if not self.agent.popup:
                self.known_spells[self.agent.message] = None
                return
            if self.agent.popup[0] not in ('Choose which spell to cast') or \
                    not self.agent.popup[1].startswith('Name'):
                raise ValueError(f'Invalid cast popup text format: {self.agent.popup}')
            for line in self.agent.popup[2:]:
                matches = re.findall(r'^([a-zA-Z]) - *' +
                                     r'(' + '|'.join(ALL_SPELL_NAMES) + ') *' +
                                     r'([0-9]*) *' +
                                     r'(' + '|'.join(ALL_SPELL_CATEGORIES) + ') *' +
                                     r'([0-9]*)\% *' +
                                     r'([0-9]*\%|\(gone\))', line)
                assert len(matches) == 1, (matches, line)
                letter, spell_name, level, category, fail, retention = matches[0]
                assert len(letter) == 1, letter
                self.known_spells[spell_name] = letter
                self.spell_fail_chance[spell_name] = int(fail) / 100
        self.agent.step(A.Command.ESC)

    def parse_enhance_view(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.ENHANCE)
            self._parse_enhance_view()
            while self.upgradable_skills:
                to_upgrade = self.select_skill_to_upgrade()
                old_skill_level = self.skill_levels.copy()
                letter = self.upgradable_skills[to_upgrade]

                def type_letter():
                    while f'{letter} - ' not in '\n'.join(self.agent.single_popup):
                        yield A.TextCharacters.SPACE
                    yield letter

                self.agent.step(A.Command.ENHANCE, type_letter())

                self.agent.step(A.Command.ENHANCE)
                self._parse_enhance_view()
                assert (old_skill_level != self.skill_levels).any(), (old_skill_level, self.skill_levels)

    def select_skill_to_upgrade(self):
        assert self.upgradable_skills
        # TODO: logic
        return next(iter(self.upgradable_skills.keys()))

    def _parse_enhance_view(self):
        if self.agent.popup[0] not in ('Current skills:', 'Pick a skill to advance:'):
            raise ValueError('Invalid ehance popup text format.' + str(self.agent.popup))
        self.upgradable_skills = dict()
        for line in self.agent.popup[1:]:
            if line.strip() in self.possible_skill_types or \
                    line.strip() == '(Skill flagged by "#" cannot be enhanced any further.)' or \
                    line.strip() == '(Skills flagged by "#" cannot be enhanced any further.)' or \
                    line.strip() == '(Skill flagged by "*" may be enhanced when you\'re more experienced.)' or \
                    line.strip() == '(Skills flagged by "*" may be enhanced when you\'re more experienced.)':
                continue
            matches = re.findall(r'^([a-zA-Z] -)?#?\*? *' +
                                 r'(' + '|'.join(self.name_to_skill_type.keys()) + ') *' +
                                 r'\[(' + '|'.join(self.possible_skill_levels) + ')\]', line)
            assert len(matches) == 1, (matches, line)
            letter, skill_type, skill_level = matches[0]
            if letter:
                letter = letter[0]
                assert letter not in self.upgradable_skills.values()
                self.upgradable_skills[self.name_to_skill_type[skill_type]] = letter
            self.skill_levels[self.name_to_skill_type[skill_type]] = self.name_to_skill_level[skill_level]

    def _get_str_dex_to_hit_bonus(self):
        bonus = 0
        # /* attack bonus for strength & dexterity */
        # int
        # abon()
        # {
        #     int sbon;
        #     int str = ACURR(A_STR), dex = ACURR(A_DEX);
        strength = self.agent.blstats.strength
        dexterity = self.agent.blstats.dexterity

        #     if (Upolyd)
        #         return (adj_lev(&mons[u.umonnum]) - 3);

        #     if (str < 6)
        #         sbon = -2;
        #     else if (str < 8)
        #         sbon = -1;
        #     else if (str < 17)
        #         sbon = 0;
        #     else if (str <= STR18(50))
        #         sbon = 1; /* up to 18/50 */
        #     else if (str < STR18(100))
        #         sbon = 2;
        #     else
        #         sbon = 3;
        if strength < 6:
            bonus -= 2
        elif strength < 8:
            bonus -= 1
        elif strength < 17:
            pass
        elif strength <= 18 + 50:
            bonus += 1
        elif strength < 18 + 100:
            bonus += 2
        else:
            bonus += 3

        #     /* Game tuning kludge: make it a bit easier for a low level character to
        #      * hit */
        #     sbon += (u.ulevel < 3) ? 1 : 0;

        #     if (dex < 4)
        #         return (sbon - 3);
        #     else if (dex < 6)
        #         return (sbon - 2);
        #     else if (dex < 8)
        #         return (sbon - 1);
        #     else if (dex < 14)
        #         return sbon;
        #     else
        #         return (sbon + dex - 14);
        #

        if dexterity < 4:
            bonus -= 3
        elif dexterity < 6:
            bonus -= 2
        elif dexterity < 8:
            bonus -= 1
        elif dexterity < 14:
            pass
        else:
            return bonus + dexterity - 14

        return bonus

    def _get_weapon_skill_bonus(self, item):
        """ Retuns a pair (to hit bonus, damage bonus) """

        if item is None:
            if self.role in (self.MONK, self.SAMURAI):
                return self.martial_bonus[self.skill_levels[O.P_BARE_HANDED_COMBAT]]
            else:
                return self.unarmed_bonus[self.skill_levels[O.P_BARE_HANDED_COMBAT]]
        else:
            # TODO: two weapon, riding

            # TODO: proper sub
            assert all(item.objs[0].sub == obj.sub for obj in item.objs), item.objs
            sub = abs(item.objs[0].sub)

            if not (0 <= sub < len(self.skill_levels)):
                raise ValueError('Invalid item sub: ' + str(item) + ' sub: ' + str(sub))
            # TODO:
            return self.weapon_bonus[self.skill_levels[sub]]

    def get_ranged_bonus(self, launcher, ammo, monster=None, large_monster=False):
        # TODO: check code/wiki

        if launcher is not None:
            assert launcher.is_launcher()
            assert ammo.is_fired_projectile()
        else:
            assert ammo.is_thrown_projectile()

        roll_offset = 0
        dmg_bonus = 0
        if launcher is not None:
            skill_hit_bonus, dmg_bonus = self._get_weapon_skill_bonus(launcher)
            roll_offset += skill_hit_bonus
            roll_offset += launcher.get_weapon_bonus(large_monster)[0]
            dmg_bonus += launcher.get_weapon_bonus(large_monster)[1]

        roll_offset += ammo.get_weapon_bonus(large_monster)[0]
        dmg_bonus += ammo.get_weapon_bonus(large_monster)[1]

        return roll_offset, max(0, dmg_bonus)

    def get_range(self, launcher, ammo):
        # TODO: implement
        return 7

    def get_melee_bonus(self, item, monster=None, large_monster=False):
        """ Returns a pair (to_hit, damaga)
        https://github.com/facebookresearch/nle/blob/master/src/uhitm.c : find_roll_to_hit
         """
        if monster is not None:
            raise NotImplementedError()

        # TODO:
        # tmp = 1 + Luck + abon() + find_mac(mtmp) + u.uhitinc
        # + maybe_polyd(youmonst.data->mlevel, u.ulevel);
        roll_offset = 1 + self._get_str_dex_to_hit_bonus() + self.agent.blstats.experience_level

        # TODO:
        # /* some actions should occur only once during multiple attacks */
        # if (!(*attk_count)++) {
        #     /* knight's chivalry or samurai's giri */
        #     check_caitiff(mtmp);
        # }

        # TODO:
        # /* adjust vs. (and possibly modify) monster state */
        # if (mtmp->mstun)
        #     tmp += 2;
        # if (mtmp->mflee)
        #     tmp += 2;

        # TODO:
        # if (mtmp->msleeping) {
        #     mtmp->msleeping = 0;
        #     tmp += 2;
        # }

        # TODO:
        # if (!mtmp->mcanmove) {
        #     tmp += 4;
        #     if (!rn2(10)) {
        #         mtmp->mcanmove = 1;
        #         mtmp->mfrozen = 0;
        #     }
        # }

        # /* role/race adjustments */

        # TODO: polymorphed
        # if (Role_if(PM_MONK) && !Upolyd) {
        if self.role == self.MONK:
            # if (uarm)
            if self.agent.inventory.items.suit is not None:
                # tmp -= (*role_roll_penalty = urole.spelarmr);
                roll_offset -= 20
            elif item is None and self.agent.inventory.items.off_hand is None:
                #  else if (!uwep && !uarms)
                # tmp += (u.ulevel / 3) + 2;
                roll_offset += (self.agent.blstats.experience_level // 3) + 2

        # TODO:
        # if (is_orc(mtmp->data)
        #     && maybe_polyd(is_elf(youmonst.data), Race_if(PM_ELF)))
        #     tmp++;

        # TODO:
        # /* encumbrance: with a lot of luggage, your agility diminishes */
        # if ((tmp2 = near_capacity()) != 0)
        #     tmp -= (tmp2 * 2) - 1;
        # if (u.utrap)
        #     tmp -= 3;

        # /*
        #  * hitval applies if making a weapon attack while wielding a weapon;
        #  * weapon_hit_bonus applies if doing a weapon attack even bare-handed
        #  * or if kicking as martial artist
        #  */
        # if (aatyp == AT_WEAP || aatyp == AT_CLAW) {
        #     if (weapon)
        #         tmp += hitval(weapon, mtmp);
        #     tmp += weapon_hit_bonus(weapon);
        # } else if (aatyp == AT_KICK && martial_bonus()) {
        #     tmp += weapon_hit_bonus((struct obj *) 0);
        # }
        skill_hit_bonus, dmg_bonus = self._get_weapon_skill_bonus(item)
        roll_offset += skill_hit_bonus

        if item is not None:
            if item.is_launcher() or item.is_fired_projectile() or item.objs[0].name in ['dart', 'shuriken']:
                # TODO: rocks, boomerang
                dmg_bonus = 1.5  # 1d2
            else:
                dmg_bonus += item.get_weapon_bonus(large_monster)[1]
            roll_offset += item.get_weapon_bonus(large_monster)[0]
        else:
            # TODO: proper unarmed base damage
            dmg_bonus += 1.5
        return roll_offset, max(0, dmg_bonus)

    def get_skill_str_list(self):
        inv_skill_type = {v: k for k, v in self.name_to_skill_type.items()}
        inv_skill_level = {v: k for k, v in self.name_to_skill_level.items()}
        return list(inv_skill_type[skill_type] + '-' + inv_skill_level[level]
                    for skill_type, level in enumerate(self.skill_levels)
                    if level in inv_skill_level and skill_type in inv_skill_type and level != 0)

    def __str__(self):
        if self.role is None:
            return 'unparsed_character'
        skill_str = '| '.join(self.get_skill_str_list())
        if self.upgradable_skills:
            inv_skill_type = {v: k for k, v in self.name_to_skill_type.items()}
            skill_str += '\n Upgradable: ' + '|'.join(letter + '-' + inv_skill_type[skill]
                                                      for skill, letter in self.upgradable_skills.items())
        return '-'.join([f'{list(d.keys())[list(d.values()).index(v)][:3].lower()}'
                         for d, v in [(self.name_to_role, self.role),
                                      (self.name_to_race, self.race),
                                      (self.name_to_gender, self.gender),
                                      (self.name_to_alignment, self.alignment),
                                      ]]) + '\n Skills: ' + skill_str
