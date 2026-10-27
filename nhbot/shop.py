"""Shop memory grounded in observed rooms, greetings and priced floor stock.

Shopkeeper coordinates are observations, not shop boundaries.  In particular a
keeper in a doorway or following the hero must not turn a corridor into a shop.
"""
from collections import deque
from dataclasses import dataclass, field
import re

import numpy as np

from .glyph import G, SHOP, SS


_ROOM = (SS.S_room, SS.S_darkroom)
_CORRIDOR = (SS.S_corr, SS.S_litcorr)
_WELCOME = re.compile(r"Welcome(?: again)? to .*?(" +
                      '|'.join(re.escape(name) for name in SHOP.name2id if name != 'UNKNOWN') + r")!")


def _neighbors(y, x, shape):
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
            yield yy, xx


def _adjacent(mask):
    result = np.zeros_like(mask)
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


@dataclass
class ShopRecord:
    shop_id: int
    anchor: tuple
    interior: np.ndarray
    entrances: set = field(default_factory=set)
    shop_type: int = SHOP.UNKNOWN
    evidence: set = field(default_factory=set)
    keeper_positions: set = field(default_factory=set)


class ShopModel:
    def __init__(self, shape):
        self.records = {}
        self.shop = np.zeros(shape, dtype=bool)
        self.shop_interior = np.zeros(shape, dtype=bool)
        self.shop_type = np.full(shape, SHOP.UNKNOWN, dtype=np.int32)
        self._keeper_observations = {}  # position -> (shop_id, turn)

    def shop_at(self, y, x, include_entrance=False):
        for record in self.records.values():
            if record.interior[y, x] or (include_entrance and (y, x) in record.entrances):
                return record
        return None

    def _room(self, anchor, walkable):
        if not walkable[anchor]:
            return None
        result = np.zeros_like(walkable)
        result[anchor] = True
        queue = deque([anchor])
        while queue:
            for position in _neighbors(*queue.popleft(), walkable.shape):
                if walkable[position] and not result[position]:
                    result[position] = True
                    queue.append(position)
        return result

    def _register(self, anchor, room, entrances, evidence):
        matches = [record for record in self.records.values() if (record.interior & room).any()]
        if len(matches) > 1:
            # Incomplete terrain must not merge two established owners.
            return None
        if matches:
            record = matches[0]
            record.interior |= room
        else:
            number = len(self.records) + 1
            record = ShopRecord(number, anchor, room.copy())
            self.records[number] = record
        record.entrances.update(zip(*(_adjacent(room) & entrances).nonzero()))
        record.evidence.add(evidence)
        return record

    def observe(self, level, glyphs, peaceful, hero, message, turn, hallucinating=False):
        if hallucinating:
            return
        terrain = level.objects
        walls = np.isin(terrain, tuple(G.WALL))
        opposite = np.zeros_like(walls)
        opposite[1:-1] |= walls[:-2] & walls[2:]
        opposite[:, 1:-1] |= walls[:, :-2] & walls[:, 2:]
        entrances = np.isin(terrain, tuple(G.DOORS) + (SS.S_ndoor,)) | (opposite & level.walkable)
        for record in self.records.values():
            for position in record.entrances:
                entrances[position] = True
        # Unknown terrain beneath an object may be room floor, but known corridors
        # and all doorways are boundaries even when the keeper occupies them.
        walkable = level.walkable & ~entrances & ~np.isin(terrain, _CORRIDOR)
        room_floor = np.isin(terrain, _ROOM)
        welcome = _WELCOME.search(message or '')
        room_cache = {}

        def room_at(position):
            if position not in room_cache:
                candidate = self._room(position, walkable)
                if candidate is None:
                    room_cache[position] = None
                else:
                    # Shop stock may occupy dozens of squares of the same room.
                    # Flood each component once per observation.
                    room_cache.update((p, candidate) for p in zip(*candidate.nonzero()))
            return room_cache[position]

        # Re-observe the anchored room as more terrain becomes visible.  Never
        # use a keeper's new position to enlarge an established room.
        for record in list(self.records.values()):
            room = room_at(record.anchor)
            if room is not None:
                updated = self._register(record.anchor, room, entrances, 'room')
                if updated is record:
                    # Newly revealed doors can disprove an earlier connection.
                    # Keep the owner's id, but retract the unsupported far side.
                    record.interior[:] = room

        # Explicit floor quotes establish ownership even without a visible keeper
        # (for example after falling into a shop).  Free/no-charge items do not.
        from .item import Item
        for y, x in zip(*(level.item_count > 0).nonzero()):
            if any(item.shop_status == Item.FOR_SALE for item in level.items[y, x]):
                anchor = (int(y), int(x))
                room = room_at(anchor)
                if room is not None:
                    self._register(anchor, room, entrances, 'stock')
        if welcome:
            room = room_at(hero)
            if room is None and entrances[hero]:
                # A greeting can arrive while standing in the entrance.  Only
                # infer its inside when exactly one neighboring room is plausible.
                candidates = {}
                for position in _neighbors(*hero, walkable.shape):
                    candidate = room_at(position)
                    if candidate is not None and (candidate & room_floor).any():
                        candidates[candidate.tobytes()] = (position, candidate)
                if len(candidates) == 1:
                    position, room = next(iter(candidates.values()))
                    self._register(position, room, entrances, 'greeting')
            if room is not None:
                if walkable[hero]:
                    self._register(hero, room, entrances, 'greeting')

        keepers = zip(*(np.isin(glyphs, tuple(G.SHOPKEEPER)) & peaceful).nonzero())
        observations = {}
        for y, x in keepers:
            position = (int(y), int(x))
            record = self.shop_at(*position, include_entrance=True)
            if record is None:
                previous = {shop_id for (py, px), (shop_id, seen) in self._keeper_observations.items()
                            if 0 <= turn - seen <= 20 and
                            max(abs(py - y), abs(px - x)) <= 2 * max(1, turn - seen)}
                if len(previous) == 1:
                    # A recently observed keeper has moved outside its shop.
                    record = self.records[next(iter(previous))]
                elif not previous:
                    room = room_at(position)
                    if room is not None and (room & room_floor).any():
                        record = self._register(position, room, entrances, 'keeper')
            if record is not None:
                record.keeper_positions.add(position)
                observations[position] = (record.shop_id, turn)
        self._keeper_observations = {
            position: value for position, value in self._keeper_observations.items()
            if turn - value[1] <= 20}
        self._keeper_observations.update(observations)

        if welcome:
            record = self.shop_at(*hero, include_entrance=True)
            if record is not None:
                record.shop_type = SHOP.name2id[welcome.group(1)]
                record.evidence.add('greeting')

        self.shop.fill(False)
        self.shop_interior.fill(False)
        self.shop_type.fill(SHOP.UNKNOWN)
        for record in self.records.values():
            # Ownership covers the room, while the conservative hazard mask also
            # covers its walls/door and one surrounding square, as legacy users expect.
            interior = record.interior & level.walkable & ~entrances & ~np.isin(terrain, _CORRIDOR)
            record.interior[:] = interior
            self.shop_interior |= interior
            protected = interior.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    sy = slice(max(0, -dy), min(interior.shape[0], interior.shape[0] - dy))
                    sx = slice(max(0, -dx), min(interior.shape[1], interior.shape[1] - dx))
                    ty = slice(max(0, dy), min(interior.shape[0], interior.shape[0] + dy))
                    tx = slice(max(0, dx), min(interior.shape[1], interior.shape[1] + dx))
                    protected[ty, tx] |= interior[sy, sx]
            self.shop |= protected
            self.shop_type[protected] = record.shop_type
