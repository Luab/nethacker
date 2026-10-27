"""Persistent transaction intent; observations, not inventory absence, settle a bill."""
from dataclasses import dataclass
import re


@dataclass
class ShopTransaction:
    level_key: tuple
    shop_id: int
    origin: tuple
    retry_after: int = 0
    credit_budgeted: bool = False


def quantity_price(item, count=1):
    """Floor quotes include the whole stack (NetHack get_cost_of_shop_item)."""
    if count < 1 or item.count < count or item.price <= 0:
        return None
    # Round upward if an unusual quote is not divisible; never under-budget it.
    return (int(item.price) * count + int(item.count) - 1) // int(item.count)


def bill_cleared(message):
    """Only a keeper's explicit zero-debt response confirms complete settlement."""
    return bool(re.search(r'You do not owe .+? anything\.', message or '')) or \
        "You don't owe any money here." in (message or '')


def current_shop_credit(message):
    """Parse a completed `$` report, never another shop's balance.

    NetHack reports the current shop first, then every other shop on this
    level.  A positive balance is attributable only with the following local
    zero-debt sentence.  Credit alongside debt is deliberately left unknown.
    Callers must also establish that the hero is inside the intended shop.
    """
    financial = re.compile(
        r"You have no credit(?: or debt)? in here\."
        r"|You have (\d+) zorkmids? credit at [^.]+\."
        r"|You don't owe any money here\."
        r"|You owe [^.]+\.")
    lines = list(financial.finditer(message or ''))
    if not lines:
        return None
    first = lines[0]
    if first.group() == 'You have no credit or debt in here.':
        return 0
    if first.group(1) is not None and len(lines) > 1 and \
            lines[1].group() == "You don't owe any money here.":
        return int(first.group(1))
    return None
