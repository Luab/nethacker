"""Short-lived evidence of attacks whose source cannot be inspected on the map.

Only public HP, turn, form/level context and attack messages are used. An HP
loss alone (including repeated starvation or poison damage) is not an attack.
"""
import re


_UNSEEN_HIT = re.compile(
    r'\b(?:it|something|the invisible [^.!?]+?) '
    r'(?:hits|bites|kicks|butts|stings|touches|claws|mauls|crushes|grabs)'
    r'(?: you)?[.!]', re.IGNORECASE)


class IncomingHarm:
    """One sample per game turn; retain at most four turns of supported damage."""

    def __init__(self):
        self.samples = []  # (turn, latest HP, unseen attack message on this turn)
        self.context = None

    def observe(self, turn, hp, context, message):
        hit = bool(_UNSEEN_HIT.search(message or ''))
        if self.samples:
            last_turn, last_hp, _ = self.samples[-1]
            # Changing HP pools/levels, recovering HP, or skipping a long interval
            # invalidates attribution. Same-turn menus must not count as attacks.
            if (context != self.context or turn < last_turn or
                    turn - last_turn > 4 or hp > last_hp):
                self.samples.clear()
        self.context = context
        if self.samples and self.samples[-1][0] == turn:
            hit = hit or self.samples[-1][2]
            self.samples[-1] = (turn, hp, hit)
        else:
            self.samples.append((turn, hp, hit))
        self.samples = [s for s in self.samples if turn - s[0] <= 4]

    def damage_per_turn(self, turn):
        supported = []
        for before, after in zip(self.samples, self.samples[1:]):
            t0, h0, _ = before
            t1, h1, hit = after
            if hit and h0 > h1 and 0 < t1 - t0 <= 4:
                supported.append((t0, t1, h0 - h1))
        if not supported or not 0 <= turn - supported[-1][1] <= 1:
            return 0.0
        # Include intervening quiet turns, rather than assuming every turn hits.
        return sum(s[2] for s in supported) / max(1, turn - supported[0][0])
