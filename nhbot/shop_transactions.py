"""Persistent transaction intent; observations, not inventory absence, settle a bill."""
from dataclasses import dataclass
import re


@dataclass
class ShopTransaction:
    level_key: tuple
    shop_id: int
    origin: tuple
    retry_after: int = 0


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
