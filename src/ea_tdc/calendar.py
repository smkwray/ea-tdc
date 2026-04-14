from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from functools import lru_cache


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    current += timedelta(days=7 * (occurrence - 1))
    return current


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    current = date(year, month, monthrange(year, month)[1])
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def us_market_holidays(year: int) -> frozenset[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday_of_month(year, 1, 0, 3),   # MLK Day
        _nth_weekday_of_month(year, 2, 0, 3),   # Presidents Day
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday_of_month(year, 5, 0),      # Memorial Day
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday_of_month(year, 9, 0, 1),   # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),  # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    return frozenset(holidays)


def is_us_market_business_day(value: date) -> bool:
    return value.weekday() < 5 and value not in us_market_holidays(value.year)


def previous_us_market_business_day(value: date) -> date:
    current = value - timedelta(days=1)
    while not is_us_market_business_day(current):
        current -= timedelta(days=1)
    return current


def add_us_market_business_days(value: date, days: int) -> date:
    current = value
    remaining = abs(days)
    direction = 1 if days >= 0 else -1
    while remaining > 0:
        current += timedelta(days=direction)
        if is_us_market_business_day(current):
            remaining -= 1
    return current
