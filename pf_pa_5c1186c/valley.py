"""The Valley of the Dead (Gehennom level 1) as NetHack 3.6.6 builds it: dat/gehennom.des, in bot coordinates.

Pure data, no game logic (dive_logic.DiveLogic.valley_step walks it).

Placement: sp_lev.c spo_map() centres the 76x20 map with xstart = 2 + (x_maze_max - 2 - xsize) / 2 = 2, made
odd = 3, and ystart = 2 + (y_maze_max - 2 - ysize) / 2 = 1. NLE's glyph columns are NetHack x - 1 (the tty
map never uses column 0), so a map square (mx, my) of the level file is bot (row, col) = (my + 1, mx + 2).
(The castle agent's castle map (55,08) = bot (11, 63) follows the same rule with xstart 9, ystart 3.)

The level (FLAGS: noteleport, hardfloor, nommap; NON_DIGGABLE everywhere): a fall through a castle trap door
lands in the east TELEPORT_REGION (58,09)-(72,18); the '>' is in the far north-west corner at (1,1). The only
way there crosses Moloch's temple (a lit room with a peaceful priest of Moloch) and three locked SECRET doors.
Walls can't be dug, but dig.c may_dig() only refuses W_NONDIGGABLE on IS_STWALL squares and a secret door
(SDOOR = 14) is not one: a pick-axe breaks through each door ("You break through a secret door!") in ~3
turns for a dwarf (dig effort doubles every turn), with no search and no Luck involved.
"""

import numpy as np

MAP = (
    '----------------------------------------------------------------------------',
    '|...S.|..|.....|  |.....-|      |................|   |...............| |...|',
    '|---|.|.--.---.|  |......--- ----..........-----.-----....---........---.-.|',
    '|   |.|.|..| |.| --........| |.............|   |.......---| |-...........--|',
    '|   |...S..| |.| |.......-----.......------|   |--------..---......------- |',
    '|----------- |.| |-......| |....|...-- |...-----................----       |',
    '|.....S....---.| |.......| |....|...|  |..............-----------          |',
    '|.....|.|......| |.....--- |......---  |....---.......|                    |',
    '|.....|.|------| |....--   --....-- |-------- ----....---------------      |',
    '|.....|--......---BBB-|     |...--  |.......|    |..................|      |',
    '|..........||........-|    --...|   |.......|    |...||.............|      |',
    '|.....|...-||-........------....|   |.......---- |...||.............--     |',
    '|.....|--......---...........--------..........| |.......---------...--    |',
    '|.....| |------| |--.......--|   |..B......----- -----....| |.|  |....---  |',
    '|.....| |......--| ------..| |----..B......|       |.--------.-- |-.....---|',
    '|------ |........|  |.|....| |.....----BBBB---------...........---.........|',
    '|       |........|  |...|..| |.....|  |-.............--------...........---|',
    '|       --.....-----------.| |....-----.....----------     |.........----  |',
    '|        |..|..B...........| |.|..........|.|              |.|........|    |',
    '----------------------------------------------------------------------------',
)

# "Make the path somewhat unpredictable": three independent 50% IF blocks, each closing one passage and opening
# another ('B' is a CROSSWALL boundary, turned into ROOM floor by remove_boundary_syms). With a door found, each
# variant adds one graveyard to the route (0-3, 1.5 on average); every square of a graveyard holds a sleeping
# undead (mkroom.c fill_zoo), so the walk has to cut through ~5 of them per graveyard.
VARIANT_FLOOR = ((40, 8), (41, 8), (42, 8), (43, 8),                              # IF 1 (closes (50..53, 8))
                 (27, 3), (28, 3), (29, 3),                                        # IF 2 (closes (27, 12), (28, 2))
                 (9, 13), (10, 13), (11, 13), (12, 13), (13, 13), (14, 13))        # IF 3 (closes (16, 10..11))

ROW_OFFSET = 1
COL_OFFSET = 2


def bot(mx, my):
    """Level-file map (x, y) -> bot (row, col)."""
    return my + ROW_OFFSET, mx + COL_OFFSET


DOWN_STAIRS = bot(1, 1)          # (2, 3)
UP_STAIRS = bot(66, 17)          # (18, 68): up to the castle's east edge, with no way back down
ALTAR = bot(3, 10)               # Moloch's altar (don't dig it: altar_wrath + angry priest)
TEMPLE = (bot(1, 6), bot(5, 14))  # ((row, col) top-left, bottom-right), lit, peaceful priest of Moloch

# The secret doors in route order: (door, squares to dig or search it from, on the side we come from).
# The orthogonal square first; the diagonal one when a monster (the priest) stands on it.
DOORS = (
    (bot(6, 6), (bot(5, 6), bot(5, 7))),     # temple's north-east corner -> east pocket
    (bot(8, 4), (bot(9, 4), bot(9, 3))),     # east pocket -> west pocket
    (bot(4, 1), (bot(5, 1), bot(5, 2))),     # west pocket -> the '>' corner ((5, 2) is a spiked pit)
)

# fixed traps of the level file on the only way: spiked pits at (5, 2) and (14, 5), sleeping gas at (3, 1)
SPIKED_PITS = (bot(5, 2), bot(14, 5))
SLEEP_GAS = bot(3, 1)
ARRIVAL = (bot(58, 9), bot(72, 18))


# The three graveyards: REGION (19,01,24,08), (09,14,16,18), (37,09,43,14) 'morgue', filled, irregular. The room is
# sp_lev.c's flood fill (8-connected, mkmap.c flood_fill_rm) over ROOM squares from the region's top-left corner,
# with the 'B' CROSSWALLs still boundaries (remove_boundary_syms runs later), and fill_zoo puts a sleeping
# undead (MM_ASLEEP) on every one of its squares. 48 + 31 + 43 = 122 squares, the same in all 8 wall variants;
# the route to the temple crosses 0 (no IF fired) to 15 (all three) of them.
GRAVEYARD_REGIONS = ((19, 1), (9, 14), (37, 9))


def _graveyard_squares():
    squares = set()
    for sx, sy in GRAVEYARD_REGIONS:
        room = {(sx, sy)}
        todo = [(sx, sy)]
        while todo:
            x, y = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (x + dx, y + dy)
                    if 0 <= n[1] < len(MAP) and 0 <= n[0] < len(MAP[0]) and n not in room and MAP[n[1]][n[0]] == '.':
                        room.add(n)
                        todo.append(n)
        squares |= room
    return frozenset(bot(x, y) for x, y in squares)


GRAVEYARD = _graveyard_squares()


def graveyard_mask(shape):
    mask = np.zeros(shape, bool)
    for p in GRAVEYARD:
        mask[p] = True
    return mask


def floor_mask(shape):
    """Squares that are floor in at least one variant of the level, in bot coordinates (doors excluded)."""
    mask = np.zeros(shape, bool)
    for my, row in enumerate(MAP):
        for mx, ch in enumerate(row):
            if ch in '.B':
                mask[bot(mx, my)] = True
    for mx, my in VARIANT_FLOOR:
        mask[bot(mx, my)] = True
    return mask
