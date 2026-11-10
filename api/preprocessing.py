"""Convert API payloads into the DataFrame (dataset column names) the saved preprocessor expects."""
import pandas as pd

COLUMN_MAP = {
    "confirmed": "Confirmed", "deaths": "Deaths", "new_cases": "New cases", "new_deaths": "New deaths",
    "new_recovered": "New recovered", "deaths_per_100_cases": "Deaths / 100 Cases",
    "confirmed_last_week": "Confirmed last week", "who_region": "WHO Region",
}


def to_frame(countries) -> pd.DataFrame:
    rows = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in countries]
    return pd.DataFrame(rows).rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
