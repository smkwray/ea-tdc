from __future__ import annotations

from datetime import date

from ea_tdc.calendar import add_us_market_business_days, previous_us_market_business_day, us_market_holidays


def test_market_calendar_includes_good_friday_and_juneteenth() -> None:
    holidays_2024 = us_market_holidays(2024)

    assert date(2024, 3, 29) in holidays_2024
    assert date(2024, 6, 19) in holidays_2024


def test_market_business_day_navigation_skips_holidays_and_weekends() -> None:
    assert previous_us_market_business_day(date(2024, 1, 16)) == date(2024, 1, 12)
    assert add_us_market_business_days(date(2024, 3, 28), 1) == date(2024, 4, 1)
