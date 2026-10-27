"""Conservative visible-contact costs when a ray may miss a creature.

This is a bound within the supplied terrain/range model, not an exact NetHack
simulator. It does not infer hidden reflection, resistance, or armor class.
"""
from functools import lru_cache


def miss_contact_penalty_bound(kernel, agent, wand, direction, budget,
                               encounters, hazard_costs, origin):
    """Bound contact cost for all hit probabilities between zero and one.

    A creature hit spends two extra range units; a miss spends none. Charge
    protected-square contact either way and retain the more costly remaining
    path. Terrain branch weights still multiply. Every step spends range, so
    memoization produces a finite acyclic state graph even in enclosed rooms.
    """
    encounters = frozenset(encounters)

    @lru_cache(maxsize=None)
    def value(y, x, dy, dx, left):
        if left < 0:
            return 0.0
        total = 0.0
        for yy, xx, ddy, ddx, probability, penalty in kernel(agent, wand, y, x, dy, dx):
            assert probability >= 0 and penalty >= 0
            if not probability:
                continue
            remaining = left - penalty - 1
            future = value(yy, xx, ddy, ddx, remaining)
            cell = (yy, xx)
            if cell in encounters:
                future = max(future, value(yy, xx, ddy, ddx, remaining - 2))
            total += probability * (hazard_costs.get(cell, 0.0) + future)
        return total

    result = value(*origin, *direction, budget)
    return result, value.cache_info().currsize


def ray_miss_extra_penalty(agent, item, monsters, dy, dx, paths, kernel, glyphs,
                           self_penalty):
    """Only add collateral cost; keep existing target rewards and penalties."""
    if not item.is_ray_wand():
        return 0.0, 0
    budget = 13
    origin = (agent.blstats.y, agent.blstats.x)
    hostile = {(monster[1], monster[2]) for monster in monsters}
    encounters = set(hostile)
    hazards = {}
    # The inherited inclusive budget admits at most fourteen advances.
    radius = budget + 1
    for y in range(max(0, origin[0] - radius), min(agent.glyphs.shape[0], origin[0] + radius + 1)):
        for x in range(max(0, origin[1] - radius), min(agent.glyphs.shape[1], origin[1] + radius + 1)):
            cell = (y, x)
            if cell in hostile:
                continue
            glyph = agent.glyphs[y, x]
            if glyph in glyphs.PETS:
                hazards[cell] = 20.0
            elif glyph in glyphs.MONS and cell != origin:
                hazards[cell] = 200.0
            elif cell == origin:
                hazards[cell] = float(self_penalty)
            else:
                continue
            encounters.add(cell)
    inherited = sum(probability * (20.0 if target == 'pet' else
                                  200.0 if target == 'peaceful' else
                                  self_penalty if target == 'self' else 0.0)
                    for _, _, target, probability in paths)
    bound, states = miss_contact_penalty_bound(kernel, agent, item, (dy, dx), budget,
                                              encounters, hazards, origin)
    return max(0.0, bound - inherited), states
