import re

from .item import Item


def public_buc(item):
    """Return only BUC explicitly present in the observed inventory prefix.

    ItemManager's compatibility fallback calls unknown BUC UNCURSED. Neither
    that fallback nor a known enchantment proves that a consumable is safe.
    Ignore words in names/comments and reject inconsistent parsed evidence.
    """
    match = re.match(r'^(?:a|an|the|\d+) (?:empty )?(blessed|uncursed|cursed)\b',
                     item.text or '')
    if match is None:
        return Item.UNKNOWN
    status = {'blessed': Item.BLESSED, 'uncursed': Item.UNCURSED,
              'cursed': Item.CURSED}[match.group(1)]
    return status if item.status == status else Item.UNKNOWN


def public_noncursed(item):
    return public_buc(item) in (Item.UNCURSED, Item.BLESSED)


def flatten_items(iterable):
    ret = []
    for item in iterable:
        ret.append(item)
        if item.is_container():
            ret.extend(flatten_items(item.content))
    return ret


def find_equivalent_item(item, iterable):
    assert item.text
    for i in iterable:
        assert i.text
        if i.text == item.text:
            return i
    assert 0, (item, iterable)


def check_if_triggered_container_trap(message):
    return ('A cloud of ' in message and ' gas billows from ' in message) or \
           'Suddenly you are frozen in place!' in message or \
           'A tower of flame bursts from ' in message or \
           'You are jolted by a surge of electricity!' in message or \
           'But luckily ' in message
