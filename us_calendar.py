from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
us_bd = CustomBusinessDay(calendar=cal)

def next_trading_day(date=None):
    if date is None:
        date = datetime.today()
    next_day = date + us_bd
    return next_day.strftime("%Y-%m-%d")
