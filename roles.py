"""Per-identity configuration: read the identity from the first observation, then override engine settings.

Overrides are module attributes of the nhbot engine, written as "module.NAME" (e.g. "dive_logic.DIVE_XL",
"jf_config.FORCE_BOLT"). Each episode runs in a fresh bot process, so setting them at reset() is safe.
Dev runs may add or replace overrides with JF_ROLE_CFG='{"wiz": {"dive_logic.EARLY_DIVE_XL": 4}}'; keys are
identity prefixes ("wiz", "wiz-elf", "wiz-elf-cha-mal"), applied from the least to the most specific;
"*" applies to every identity.
"""
from __future__ import annotations

import importlib
import json
import os
import re

# identity prefix -> {"module.NAME": value}
# v9p: no profile for ran/rog/tou -- daglar's profiles cost them on the VERIFIED tier (all 10 ranger and 4 rogue
# identities scored lower in v5 than in v4, which had no profiles)
OVERRIDES: dict[str, dict[str, object]] = {'arc': {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True}, 'bar': {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False}, 'cav': {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True}, 'kni': {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False}, 'mon': {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True}, 'pri': {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True}, 'val': {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False}, 'wiz': {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True}}

_ROLES = {"Archeologist": "arc", "Barbarian": "bar", "Caveman": "cav", "Cavewoman": "cav", "Healer": "hea",
          "Knight": "kni", "Monk": "mon", "Priest": "pri", "Priestess": "pri", "Ranger": "ran", "Rogue": "rog",
          "Samurai": "sam", "Tourist": "tou", "Valkyrie": "val", "Wizard": "wiz"}
_RACES = {"human": "hum", "elven": "elf", "dwarven": "dwa", "gnomish": "gno", "orcish": "orc"}
_ALIGNS = {"lawful": "law", "neutral": "neu", "chaotic": "cha"}
_FEMALE_ROLES = {"Cavewoman", "Priestess", "Valkyrie"}
_RE = re.compile(r"You are an? (lawful|neutral|chaotic) (?:(male|female) )?(human|elven|dwarven|gnomish|orcish) "
                 r"(" + "|".join(_ROLES) + r")\b")
_RE_CUT = re.compile(r"You are an? (lawful|neutral|chaotic) (?:(male|female) )?(human|elven|dwarven|gnomish|orcish)")
_RE_TITLE = re.compile(r"Agent the (\w+)")
# Xp 1 rank titles (role.c): the welcome line is cut at 80 columns for long role names
_TITLES = {"Digger": "Archeologist", "Plunderer": "Barbarian", "Plunderess": "Barbarian",
           "Troglodyte": "Caveman", "Rhizotomist": "Healer", "Gallant": "Knight", "Candidate": "Monk",
           "Aspirant": "Priest", "Tenderfoot": "Ranger", "Footpad": "Rogue", "Hatamoto": "Samurai",
           "Rambler": "Tourist", "Stripling": "Valkyrie", "Evoker": "Wizard"}

_RE_ATTRIBUTES = re.compile(r"You are an? [^,]+, a level \d+ (?:(male|female) )?(human|elven|dwarven|gnomish|orcish) "
                            r"(" + "|".join(_ROLES) + r")\.\s+You are (lawful|neutral|chaotic)\b")


def attributes_identity(observation) -> str | None:
    """Identity from the ^X attributes screen (background line + alignment line)."""
    try:
        text = bytes(observation["tty_chars"]).decode("latin-1", "replace")
    except Exception:  # noqa: BLE001
        return None
    m = _RE_ATTRIBUTES.search(text)
    if m is None:
        return None
    gender, race, role, align = m.groups()
    gender = "fem" if gender == "female" or role in _FEMALE_ROLES else "mal"
    return f"{_ROLES[role]}-{_RACES[race]}-{_ALIGNS[align]}-{gender}"


def identity(observation) -> str | None:
    texts = []
    for key in ("message", "tty_chars"):
        try:
            texts.append(bytes(observation[key]).decode("latin-1", "replace"))
        except Exception:  # noqa: BLE001
            pass
    text = " ".join(texts)
    m = _RE.search(text)
    if m is not None:
        align, gender, race, role = m.groups()
    else:
        m = _RE_CUT.search(text)
        t = _RE_TITLE.search(text)
        if m is None or t is None or t.group(1) not in _TITLES:
            t = _RE_TITLE.search(text)
            if t is None or t.group(1) not in _TITLES:
                return None
            return _ROLES[_TITLES[t.group(1)]]
        align, gender, race = m.groups()
        role = _TITLES[t.group(1)]
    gender = "fem" if gender == "female" or role in _FEMALE_ROLES else "mal"
    return f"{_ROLES[role]}-{_RACES[race]}-{_ALIGNS[align]}-{gender}"


def overrides_for(ident: str | None) -> dict[str, object]:
    table = {k: dict(v) for k, v in OVERRIDES.items()}
    raw = os.environ.get("JF_ROLE_CFG")
    if raw:
        for k, v in json.loads(raw).items():
            table.setdefault(k, {}).update(v)
    out: dict[str, object] = dict(table.get("*", {}))   # "*": every identity, before any prefix
    if not ident:
        return out
    for prefix in sorted((k for k in table if k != "*"), key=len):
        if ident == prefix or ident.startswith(prefix + "-"):
            out.update(table[prefix])
    return out


def apply(ident: str | None) -> dict[str, object]:
    applied = overrides_for(ident)
    for name, value in applied.items():
        mod, _, attr = name.rpartition(".")
        module = importlib.import_module(f"nhbot.{mod}")
        if not hasattr(module, attr):
            raise AttributeError(f"roles: nhbot.{mod} has no {attr}")
        if attr == "GRIND_LEVELS" and isinstance(value, dict):
            value = {int(k): int(v) for k, v in value.items()}
        if attr == "ROLE_GRIND_LEVELS" and isinstance(value, dict):
            value = {r: {int(k): int(v) for k, v in t.items()} for r, t in value.items()}
        setattr(module, attr, value)
    return applied

# v10b: dev-confirmed config (held-out dev seeds, paired games on the v9p2 config)
# Rangers on nhbot: UNSEEN_PET_GUARD off +0.022 +- 0.012 over 420 pairs (6 of 7 identities up; v8ab +0.023 on its own)
OVERRIDES["ran"] = {"jf_config.UNSEEN_PET_GUARD": False}
# gnomish Cavemen: the Dlvl-1 grind (an empty level map) instead of the deep grind, +0.073 +- 0.022 over 120 pairs
# (v8ab +0.060); the other roles keep their default entries
OVERRIDES["cav-gno"] = {"jf_config.ROLE_GRIND_LEVELS": {"Rogue": {}, "Knight": {}, "Tourist": {}, "Priest": {}, "Caveman": {}}}

# v11pub (PUBLIC-board variant): full explicit config per nhbot identity, from the source program of its best public score
OVERRIDES["arc-dwa-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["arc-dwa-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["arc-hum-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["arc-hum-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["arc-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["bar-hum-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["bar-hum-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["bar-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["bar-orc-cha-fem"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["bar-orc-cha-mal"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-dwa-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-dwa-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-gno-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}, 'Caveman': {}}}
OVERRIDES["cav-gno-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}, 'Caveman': {}}}
OVERRIDES["cav-hum-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-hum-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["cav-hum-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["mon-hum-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-elf-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-elf-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-hum-law-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-hum-law-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["pri-hum-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-gno-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-gno-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': False, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-hum-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-hum-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-orc-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["ran-orc-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': False, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["rog-orc-cha-fem"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["rog-orc-cha-mal"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["val-hum-law-fem"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["val-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': True, 'jf_config.LONE_WEAK_THREAT': True, 'jf_config.FEYE_GUARD': True, 'jf_config.SHOP_SAFETY': True, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-elf-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-elf-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-gno-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-gno-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-hum-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-hum-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-hum-neu-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-hum-neu-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': True, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': True, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-orc-cha-fem"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}
OVERRIDES["wiz-orc-cha-mal"] = {'jf_config.ALIGN_PRAYER': False, 'jf_config.LONE_WEAK_THREAT': False, 'jf_config.FEYE_GUARD': False, 'jf_config.SHOP_SAFETY': False, 'jf_config.RING_MODULE': False, 'dive_logic.DIVE_XL': 8, 'dive_logic.MEDUSA_HOP': False, 'jf_config.UNSEEN_PET_GUARD': True, 'jf_config.ROLE_GRIND_LEVELS': {'Rogue': {}, 'Knight': {}, 'Tourist': {}, 'Priest': {}}}

# Experimental Wizard spell economy; public/open development hypothesis.
OVERRIDES["wiz"].update({"jf_config.FB_RESERVE": 10, "jf_config.FB_SHOP_KNOWN": True})
