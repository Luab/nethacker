"""Bounded, observation-driven preparation visits for Gnome Wizards.

The limits below are policy choices, not engine safety guarantees.  This
controller never changes the legacy tour milestone or claims the Mines were
completed.  Combat, debt settlement, eating and equipment actions remain the
ordinary higher-priority strategies.
"""
import hashlib

import numpy as np

from .character import Character
from .glyph import G, Hunger
from .item import Item
from .level import Level
from .strategy import Strategy


START_XL = 4
START_HP = 20
START_HP_FRACTION = 0.8
FOOD_TARGET = 1200
TOTAL_TURNS = 4500
MINES_TURNS = 1200
TOWN_TURNS = 150
TOWN_SURVEY_TURNS = 30
SOKOBAN_TURNS = 2200
RETURN_TURNS = 500
NO_PROGRESS_TURNS = 250
HUNGER_RECOVERY_TURNS = 16
STATUS_PAUSE_TURNS = 30
STATUS_PAUSE_INPUTS = 120
RESUME_TURNS = 1400
RESUME_ENERGY = 10


class WizardRoute:
    def __init__(self, agent):
        self.agent = agent
        self.phase = 'idle'
        self.reason = None
        self.started = self.phase_started = self.last_progress = None
        self.target = self.return_target = None
        self.after_return = None
        self.origin = None
        self.town = None
        self.town_arrived = None
        self.resources = None
        self.initial_resources = None
        self._best = None
        self._soko_layouts = set()
        self._soko_observed = set()
        self._soko_started = None
        self._soko_supplied = set()
        self._mines_entered = None
        self.start_milestone = self._last_milestone = None
        self._later_legacy_sokoban = False
        self._hunger_recovery = None
        self._status_pause = None
        self._meal_retry_after = -1
        self._resumes = 0
        self._previous_attempt = None
        self._finished_sokoban_known = False
        self._attempt_turns = TOTAL_TURNS

    def enabled(self):
        character = self.agent.character
        return getattr(character, 'role', None) == Character.WIZARD and \
            getattr(character, 'race', None) == Character.GNOME

    @property
    def active(self):
        return self.phase not in ('idle', 'done', 'aborted')

    @property
    def preparation_food_target(self):
        """Live supply objective; returning or expired visits cannot buy a detour."""
        if not self.enabled() or self.phase not in ('mines', 'town', 'sokoban') or self._status_pause is not None or \
                self.started is None or \
                self.agent.blstats.time - self.started >= self._attempt_turns - RETURN_TURNS:
            return None
        return FOOD_TARGET

    @property
    def solving_sokoban(self):
        return self.active and self.phase == 'sokoban' and self._status_pause is None and not self._impaired() and \
            self.agent.current_level().dungeon_number == Level.SOKOBAN

    @property
    def owns_sokoban_solver(self):
        return self.started is not None and not self._later_legacy_sokoban

    def _observe_milestone(self):
        from .global_logic import Milestone
        milestone = self.agent.global_logic.milestone
        # Finishing our attempt cannot itself authorize a legacy retry.  A
        # later observed tour transition into SOLVE_SOKOBAN is a new request.
        if self.phase in ('done', 'aborted') and milestone == Milestone.SOLVE_SOKOBAN and \
                self._last_milestone is not None and self._last_milestone != milestone:
            self._later_legacy_sokoban = True
        self._last_milestone = milestone

    def _resources(self):
        bl = self.agent.blstats
        return {'food': int(self.agent.inventory.carried_nutrition()),
                'hp': int(bl.hitpoints), 'max_hp': int(bl.max_hitpoints),
                'energy': int(bl.energy), 'xl': int(bl.experience_level),
                'ac': int(bl.armor_class), 'strength': int(bl.strength),
                'hunger': int(bl.hunger_state),
                'known_tiles': sum(int(level.seen.sum()) for level in self.agent.levels.values()),
                'known_links': sum(sum(value is not None for value in level.stair_destination.values())
                                   for level in self.agent.levels.values())}

    def _impaired(self):
        prop = self.agent.character.prop
        return any(getattr(prop, name, False) for name in
                   ('blind', 'hallu', 'confusion', 'stun', 'polymorph', 'stoned'))

    def _temporary_impaired(self):
        return any(getattr(self.agent.character.prop, name, False)
                   for name in ('blind', 'hallu', 'confusion', 'stun'))

    def _observe_status_pause(self, now):
        # This is independent of phase, including hunger recovery and return.
        # Do not use _change(): recovery must not renew any route deadline.
        if self._temporary_impaired():
            if self._status_pause is None:
                self._status_pause = dict(turn=now, step=int(self.agent.step_count))
                self.agent.log(f'WIZARD_ROUTE status_pause phase={self.phase} turn={now}')
        elif self._status_pause is not None:
            if now - self._status_pause['turn'] >= STATUS_PAUSE_TURNS or \
                    self.agent.step_count - self._status_pause['step'] >= STATUS_PAUSE_INPUTS:
                return  # expiry still applies if recovery followed a long preemption
            self.agent.log(f'WIZARD_ROUTE status_resumed phase={self.phase} '
                           f'elapsed={now - self._status_pause["turn"]}')
            self._status_pause = None

    def _status_pause_expired(self, now):
        pause = self._status_pause
        if pause is None:
            return False
        if now - pause['turn'] >= STATUS_PAUSE_TURNS or \
                self.agent.step_count - pause['step'] >= STATUS_PAUSE_INPUTS:
            # A still-impaired return is unsafe too. Release this attempt;
            # ordinary survival strategies retain control of the character.
            self._finish('temporary_impairment_budget', aborted=True)
            return True
        return False

    def _ready(self):
        bl = self.agent.blstats
        level = self.agent.current_level()
        local = (level.dungeon_number == Level.DUNGEONS_OF_DOOM and level.level_number <= 10) or \
            (level.dungeon_number == Level.GNOMISH_MINES and level.level_number <= 4) or \
            level.dungeon_number == Level.SOKOBAN
        return local and not getattr(self.agent.global_logic.dive, 'diving', False) and \
            bl.experience_level >= START_XL and \
            bl.hitpoints >= max(START_HP, START_HP_FRACTION * bl.max_hitpoints) and \
            bl.hunger_state <= Hunger.NOT_HUNGRY and bl.carrying_capacity <= 1 and \
            not self._impaired() and not self.agent.get_visible_monsters() and \
            not self.agent.inventory.shop_debt_pending()

    def _emergency_rescue_due(self):
        """The existing dive rescue triggers, without starting a dive here.

        The route preempts the tour, removing its should_dive callback while
        we navigate.  Observe these triggers here as well so a new hunger
        emergency releases the route immediately, including its return leg.
        """
        from . import dive_logic
        from .global_logic import Milestone

        agent = self.agent
        first = agent.global_logic.milestone == Milestone.BE_ON_FIRST_LEVEL
        failed = bool(getattr(agent, 'prayer_failed', False))
        recovery = getattr(agent, 'survival', None)
        if first and recovery is not None and recovery.needs_resource_exit():
            return True
        if first and dive_logic.RESCUE_DIVE and failed:
            return True
        if not first and dive_logic.LATE_RESCUE and failed and \
                agent.blstats.hunger_state >= Hunger.WEAK and not agent.edible_carried_food():
            return True
        return bool(first and dive_logic.EARLY_DIVE and agent.blstats.time >= dive_logic.EARLY_DIVE_TURN)

    def _branch_parent(self, branch):
        """Only an actually observed stair connection identifies the parent."""
        candidates = set()
        for key, level in self.agent.levels.items():
            for destination in level.stair_destination.values():
                if destination is None:
                    continue
                other = tuple(destination[0])
                if key[0] == Level.DUNGEONS_OF_DOOM and other[0] == branch:
                    candidates.add(tuple(key))
                elif key[0] == branch and other[0] == Level.DUNGEONS_OF_DOOM:
                    candidates.add(other)
        return min(candidates) if candidates else None

    def _town_seen(self):
        known = getattr(self.agent.global_logic, 'minetown_level', None)
        if known is not None and known[0] == Level.GNOMISH_MINES and known[1] in (3, 4):
            return tuple(known)
        # Fillers do not contain altars or shops.  A town candidate is not a
        # promise of a merchant: Orctown's defiled altar also fits this test.
        for key, level in sorted(self.agent.levels.items()):
            if key[0] == Level.GNOMISH_MINES and key[1] in (3, 4) and \
                    (level.altars or level.shop_interior.any()):
                return tuple(key)
        return None

    def _change(self, phase, reason, target=None):
        self.phase, self.reason, self.target = phase, reason, target
        self.phase_started = self.last_progress = int(self.agent.blstats.time)
        self.agent.log(f'WIZARD_ROUTE phase={phase} reason={reason} target={target} '
                       f'elapsed={self.phase_started - self.started} resources={self.resources}')

    def _finish(self, reason, aborted=False):
        self.after_return = None
        self._hunger_recovery = None
        self._status_pause = None
        self._finished_sokoban_known = self._sokoban_known()
        self._change('aborted' if aborted else 'done', reason)

    def _sokoban_known(self):
        return self.agent.current_level().dungeon_number == Level.SOKOBAN or \
            self._branch_parent(Level.SOKOBAN) is not None

    def _resume_new_sokoban(self, now):
        """One fresh opportunity may reopen a finished supplies visit.

        A repeated old branch, a solver failure or an expired Mines survey
        cannot reset itself.  Only a newly observed Sokoban connection (or an
        actual first arrival there) supplies the new opportunity.  The normal
        solver, food collection, pet care and emergency stack still act.
        """
        pet = getattr(self.agent, '_pet_care', None) or {}
        if self._resumes or self._finished_sokoban_known or \
                self._later_legacy_sokoban or not self._sokoban_known() or \
                not self._ready() or self._emergency_rescue_due() or \
                self.agent.blstats.energy < RESUME_ENERGY or \
                self.agent.inventory.carried_nutrition() >= FOOD_TARGET or \
                pet.get('hunger_warning') is not None:
            return False
        self._previous_attempt = dict(started=self.started, ended=self.phase_started,
                                      phase=self.phase, reason=self.reason,
                                      resources=self.resources)
        self._resumes = 1
        self._attempt_turns = RESUME_TURNS
        self.started = now
        self.origin = tuple(self.agent.current_level().key())
        self.return_target = self.after_return = None
        self._hunger_recovery = self._status_pause = None
        self.initial_resources = self.resources = self._resources()
        self._best = None
        self._soko_started = now
        self._change('sokoban', 'newly_observed_sokoban_supplies', (Level.SOKOBAN, 1))
        return True

    def _return(self, reason, continue_sokoban=False):
        current_branch = self.agent.current_level().dungeon_number
        phase = self._hunger_recovery[1] if self._hunger_recovery is not None else self.phase
        branch = current_branch if current_branch in (Level.GNOMISH_MINES, Level.SOKOBAN) else \
            Level.SOKOBAN if phase == 'sokoban' else Level.GNOMISH_MINES
        self.return_target = self._branch_parent(branch)
        if self.return_target is None:
            # Returning to the observed starting main floor is a fallback,
            # not invented knowledge of an undiscovered branch connection.
            self.return_target = self.origin if self.origin[0] == Level.DUNGEONS_OF_DOOM else None
        self.after_return = 'sokoban' if continue_sokoban else None
        self._hunger_recovery = None
        if self.return_target is None:
            self._finish(reason + ':no_known_parent', aborted=True)
        else:
            self._change('return', reason, self.return_target)

    def _note_progress(self):
        now = int(self.agent.blstats.time)
        current = self._resources()
        quality = (current['known_tiles'], current['known_links'], current['food'],
                   -current['ac'], current['strength'], current['xl'])
        progress = self._best is None or any(a > b for a, b in zip(quality, self._best))
        self._best = quality if self._best is None else tuple(max(a, b) for a, b in zip(quality, self._best))
        level = self.agent.current_level()
        if self.solving_sokoban and level.key() not in self._soko_observed:
            self._soko_observed.add(level.key())
            self._soko_layouts.add(self._soko_layout())
        self.resources = current
        if progress:
            self.last_progress = now
            self.agent.log(f'WIZARD_ROUTE progress phase={self.phase} level={level.key()} resources={current}')

    def _soko_layout(self):
        # Level.objects stores terrain, including the floor under a boulder.
        # Only the public glyph observation contains the visible boulders.
        mask = np.isin(self.agent.glyphs, tuple(G.BOULDER))
        return (self.agent.current_level().key(), hashlib.sha256(np.packbits(mask).tobytes()).digest()[:12])

    def note_sokoban_push(self):
        """Called only after the push handler verifies actual displacement."""
        if not self.solving_sokoban:
            return
        layout = self._soko_layout()
        if layout not in self._soko_layouts and len(self._soko_layouts) < TOTAL_TURNS:
            self._soko_layouts.add(layout)
            self.last_progress = int(self.agent.blstats.time)

    def _supply_meals(self):
        """Accessible owned food, excluding tins, pet tripe and cure corpses."""
        agent = self.agent
        return [item for item in agent.edible_carried_food()
                if item in agent.inventory.items and item.shop_status == Item.NOT_SHOP and
                item.status != Item.CURSED and item.is_unambiguous() and
                not item.is_corpse() and item.object.name not in ('tin', 'tripe ration') and
                item.object.name in agent.inventory.BUY_FOOD_NUTRITION]

    @Strategy.wrap
    def eat_supplies_strategy(self):
        """Spend preparation supplies before Weak interrupts a branch visit."""
        agent = self.agent

        def ready():
            return self.enabled() and self.active and not self._impaired() and \
                agent.blstats.hunger_state >= Hunger.HUNGRY and \
                agent.blstats.carrying_capacity < 4 and \
                agent.blstats.time >= self._meal_retry_after and not agent.get_visible_monsters()

        if not ready() or not self._supply_meals():
            yield False
        yield True
        # A higher-priority strategy may have changed the pack since eligibility.
        if not ready():
            return
        meals = self._supply_meals()
        if not meals:
            return
        self._meal_retry_after = int(agent.blstats.time) + 10
        agent.log(f'WIZARD_ROUTE eating owned supply before travel: {meals[0].text}')
        agent.inventory.eat(meals[0])

    def _recover_hunger(self, now):
        """Pause travel for a real meal/prayer; retain every visit deadline."""
        agent = self.agent
        if self._hunger_recovery is None:
            recovery = getattr(agent, 'survival', None)
            food = bool(recovery is not None and recovery.usable_food())
            prayer = bool(recovery is not None and recovery.pending_prayer is not None)
            if not food and not prayer:
                prayer = agent.is_safe_to_pray(agent.SAFE_HUNGER_PRAYER_GAP)
            if not (food or prayer):
                return False
            self._hunger_recovery = (now, self.phase, self.reason, self.target,
                                     self.phase_started, self.last_progress)
            self._change('recover', 'temporary_hunger')
        return now - self._hunger_recovery[0] < HUNGER_RECOVERY_TURNS

    def _recovery_visit_expired(self, now):
        """Neither hunger nor status recovery can delay a visit's deadline."""
        if self._hunger_recovery is not None:
            _, phase, _, _, phase_started, _ = self._hunger_recovery
        else:
            phase, phase_started = self.phase, self.phase_started
        reason = None
        if phase == 'mines' and now - (
                self._mines_entered if self._mines_entered is not None else phase_started) >= MINES_TURNS:
            reason = 'mines_budget_without_town'
        elif phase == 'town':
            if self.town_arrived is not None and (now - self.town_arrived >= TOWN_TURNS or
                    self._status_pause is not None and now - self.town_arrived >= TOWN_SURVEY_TURNS and
                    self.resources['food'] >= FOOD_TARGET):
                reason = 'town_supply_stop_finished'
            elif self.town_arrived is None and now - (
                    self._mines_entered if self._mines_entered is not None else self.started) >= MINES_TURNS:
                reason = 'town_approach_budget'
        elif phase == 'sokoban' and now - self._soko_started >= SOKOBAN_TURNS:
            reason = 'sokoban_budget'
        if reason is not None:
            self._return(reason, continue_sokoban=self._status_pause is not None and
                         self._hunger_recovery is None and phase == 'town')
        return reason is not None

    def update(self):
        if not self.enabled():
            return
        self._observe_milestone()
        now = int(self.agent.blstats.time)
        level = self.agent.current_level()
        if self.phase in ('done', 'aborted') and not self._resume_new_sokoban(now):
            return
        if self.phase == 'idle':
            if not self._ready() or self._emergency_rescue_due():
                return
            self.started = now
            self.start_milestone = self._last_milestone
            self.origin = tuple(level.key())
            self.initial_resources = self.resources = self._resources()
            if level.dungeon_number == Level.SOKOBAN or \
                    (self.resources['food'] < FOOD_TARGET and self._branch_parent(Level.SOKOBAN) is not None):
                self._soko_started = now
                self._change('sokoban', 'known_food_and_equipment_branch', (Level.SOKOBAN, 1))
            else:
                self._change('mines', 'upper_mines_supplies', (Level.GNOMISH_MINES, 4))
        if self._emergency_rescue_due():
            self._finish('emergency_rescue', aborted=True)
            return
        self._note_progress()
        self._observe_status_pause(now)
        if now - self.started >= self._attempt_turns:
            self._finish('overall_budget', aborted=True)
            return
        if self.phase == 'return':
            if tuple(level.key()) == self.return_target:
                if self.after_return == 'sokoban' and now - self.started < self._attempt_turns - RETURN_TURNS:
                    self.after_return = None
                    self._soko_started = now
                    self._change('sokoban', 'town_visit_returned_to_parent', (Level.SOKOBAN, 1))
                else:
                    self._finish('returned_to_parent:' + self.reason)
            elif now - self.phase_started >= RETURN_TURNS:
                self._finish('return_budget', aborted=True)
            if self.active:
                self._status_pause_expired(now)
            return
        if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES, Level.SOKOBAN):
            self._finish('unexpected_branch', aborted=True)
            return
        bl = self.agent.blstats
        if bl.hitpoints < max(10, 0.55 * bl.max_hitpoints) or any(
                getattr(self.agent.character.prop, name, False) for name in ('polymorph', 'stoned')):
            self._return('unsafe_to_prepare')
            return
        if len(self.agent.get_visible_monsters()) >= 4:
            self._return('hostile_crowd')
            return
        if now - self.started >= self._attempt_turns - RETURN_TURNS:
            self._return('overall_return_reserve')
            return
        if self._status_pause is not None:
            visit_phase = self._hunger_recovery[1] if self._hunger_recovery is not None else self.phase
            if visit_phase == 'mines' and level.dungeon_number == Level.GNOMISH_MINES and \
                    self._mines_entered is None:
                self._mines_entered = now
            if visit_phase == 'town' and tuple(level.key()) == self.town and self.town_arrived is None:
                self.town_arrived = now
                self.agent.log(f'WIZARD_ROUTE town_arrived={self.town} turn={now} during_status_pause')
        if (self.phase == 'recover' or self._status_pause is not None) and self._recovery_visit_expired(now):
            return
        progress_since = self._hunger_recovery[5] if self._hunger_recovery is not None else self.last_progress
        if self._status_pause is not None and now - progress_since >= NO_PROGRESS_TURNS:
            self._return('no_knowledge_or_resource_progress')
            return
        if self._status_pause_expired(now):
            return
        if self.phase == 'recover' and bl.hunger_state < Hunger.WEAK:
            _, phase, reason, target, phase_started, last_progress = self._hunger_recovery
            self._hunger_recovery = None
            self._change(phase, 'observed_hunger_relief:' + str(reason), target)
            self.phase_started, self.last_progress = phase_started, last_progress
        if bl.hunger_state >= Hunger.WEAK:
            if bl.hunger_state == Hunger.WEAK and self._recover_hunger(now):
                return
            self._return('hunger_recovery_unavailable_or_expired')
            return
        if now - self.last_progress >= NO_PROGRESS_TURNS:
            self._return('no_knowledge_or_resource_progress')
            return
        if level.dungeon_number == Level.GNOMISH_MINES and level.level_number > 4:
            self._return('below_upper_mines_cap')
            return
        if self._status_pause is not None:
            return
        if self.phase == 'mines':
            if level.dungeon_number == Level.GNOMISH_MINES and self._mines_entered is None:
                self._mines_entered = now
            self.town = self._town_seen()
            if self.town is not None:
                self._change('town', 'observed_town_candidate', self.town)
            elif now - (self._mines_entered if self._mines_entered is not None else self.phase_started) >= MINES_TURNS:
                self._return('mines_budget_without_town')
        if self.phase == 'town':
            if self.town_arrived is None:
                approach_started = self._mines_entered if self._mines_entered is not None else self.started
                if now - approach_started >= MINES_TURNS:
                    self._return('town_approach_budget')
                    return
                if tuple(level.key()) == self.town:
                    self.town_arrived = now
                    self.agent.log(f'WIZARD_ROUTE town_arrived={self.town} turn={now} '
                                   f'approach_elapsed={now - approach_started} resources={self.resources}')
            if self.town_arrived is not None:
                elapsed = now - self.town_arrived
                if elapsed >= TOWN_TURNS or (elapsed >= TOWN_SURVEY_TURNS and self.resources['food'] >= FOOD_TARGET):
                    self._return('town_supply_stop_finished', continue_sokoban=True)
        elif self.phase == 'sokoban' and now - self._soko_started >= SOKOBAN_TURNS:
            self._return('sokoban_budget')

    def hold_dive(self, rescue=False):
        if rescue:
            if self.active:
                self._finish('emergency_rescue', aborted=True)
            return False
        self.update()
        return self.active

    def solver_failed(self, reason):
        if self.phase == 'sokoban':
            self._return('solver_failed:' + str(reason)[:120])

    @Strategy.wrap
    def sokoban_supplies_strategy(self):
        """Survey and collect after filling the pits, before the next stair."""
        level = self.agent.current_level()
        key = tuple(level.key())
        if not self.solving_sokoban or key in self._soko_supplied or \
                np.isin(level.objects, tuple(G.TRAPS)).any():
            yield False
        yield True

        def changed():
            return not self.solving_sokoban or tuple(self.agent.current_level().key()) != key

        self.agent.global_logic.exploration_strategy(0).until(self.agent, changed).run()
        if not changed():
            self._soko_supplied.add(key)
            self.agent.log(f'WIZARD_ROUTE supplied_sokoban_floor={key} resources={self._resources()}')

    @Strategy.wrap
    def strategy(self):
        self.update()
        if not self.active:
            yield False
        yield True
        while self.active:
            self.update()
            if not self.active:
                break
            if self.phase == 'recover' or self._status_pause is not None:
                # Eating, recovery and combat preempt this ordinary one-turn
                # wait. Never take a branch step while Weak or impaired.
                self.agent.search()
                continue
            token = (self.phase, self.phase_started, self.target)
            step_before = self.agent.step_count

            def changed():
                self.update()
                return self._status_pause is not None or \
                    token != (self.phase, self.phase_started, self.target) or not self.active

            if tuple(self.agent.current_level().key()) != self.target:
                self.agent.exploration.go_to_level_strategy(
                    *self.target, self.agent.exploration.go_to_strategy,
                    self.agent.global_logic.exploration_strategy(None)).until(self.agent, changed).run()
                if not changed() and self.agent.step_count == step_before:
                    self.agent.search()
                continue
            explored = self.agent.global_logic.exploration_strategy(0).until(self.agent, changed).run(
                return_condition=True)
            if changed():
                continue
            if not explored:
                if self.phase == 'mines':
                    self._return('upper_mines_survey_finished_without_town')
                elif self.phase == 'town' and self.town_arrived is not None and \
                        self.agent.blstats.time - self.town_arrived >= TOWN_SURVEY_TURNS:
                    self._return('town_survey_finished', continue_sokoban=True)
                elif self.phase == 'sokoban' and not np.isin(self.agent.current_level().objects, tuple(G.TRAPS)).any():
                    self._return('sokoban_top_survey_finished')
                else:
                    # One ordinary turn lets existing collection/preparation
                    # and the puzzle solver preempt; all budgets keep running.
                    self.agent.search()
            elif self.agent.step_count == step_before:
                # An eligible exploration task can still finish without an
                # action.  Do not spin inside this persistent controller.
                self.agent.search()

    def snapshot(self):
        def plain(value):
            if isinstance(value, np.generic):
                return plain(value.item())
            if isinstance(value, dict):
                return {str(key): plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, int):
                return int(value)  # Includes milestone IntEnum values.
            return value

        # NLE-derived level keys and times can contain NumPy scalars.  Keep
        # the diagnostic contract JSON-ready without mutating route state.
        return plain({'phase': self.phase, 'reason': self.reason, 'active': self.active,
                'started': self.started, 'phase_started': self.phase_started,
                'last_progress': self.last_progress, 'target': self.target,
                'return_target': self.return_target, 'after_return': self.after_return,
                'town_arrived': self.town_arrived,
                'start_milestone': self.start_milestone,
                'later_legacy_sokoban': self._later_legacy_sokoban,
                'mines_entered': self._mines_entered, 'sokoban_supplied': sorted(self._soko_supplied),
                'hunger_recovery': self._hunger_recovery,
                'status_pause': self._status_pause,
                'resumes': self._resumes, 'attempt_turns': self._attempt_turns,
                'finished_sokoban_known': self._finished_sokoban_known,
                'previous_attempt': self._previous_attempt,
                'resources': self.resources, 'initial_resources': self.initial_resources})
