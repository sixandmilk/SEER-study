import os
import re

import pandas as pd

FREQ_MAP = {
    "Y": "yearly",
    "A": "yearly",
    "A-DEC": "yearly",
    "A-JAN": "yearly",
    "A-FEB": "yearly",
    "A-MAR": "yearly",
    "A-APR": "yearly",
    "A-MAY": "yearly",
    "A-JUN": "yearly",
    "A-JUL": "yearly",
    "A-AUG": "yearly",
    "A-SEP": "yearly",
    "A-OCT": "yearly",
    "A-NOV": "yearly",
    "AS-DEC": "yearly",
    "AS-JAN": "yearly",
    "AS-FEB": "yearly",
    "AS-MAR": "yearly",
    "AS-APR": "yearly",
    "AS-MAY": "yearly",
    "AS-JUN": "yearly",
    "AS-JUL": "yearly",
    "AS-AUG": "yearly",
    "AS-SEP": "yearly",
    "AS-OCT": "yearly",
    "AS-NOV": "yearly",
    "BA-DEC": "yearly",
    "BA-JAN": "yearly",
    "BA-FEB": "yearly",
    "BA-MAR": "yearly",
    "BA-APR": "yearly",
    "BA-MAY": "yearly",
    "BA-JUN": "yearly",
    "BA-JUL": "yearly",
    "BA-AUG": "yearly",
    "BA-SEP": "yearly",
    "BA-OCT": "yearly",
    "BA-NOV": "yearly",
    "BAS-DEC": "yearly",
    "BAS-JAN": "yearly",
    "BAS-FEB": "yearly",
    "BAS-MAR": "yearly",
    "BAS-APR": "yearly",
    "BAS-MAY": "yearly",
    "BAS-JUN": "yearly",
    "BAS-JUL": "yearly",
    "BAS-AUG": "yearly",
    "BAS-SEP": "yearly",
    "BAS-OCT": "yearly",
    "BAS-NOV": "yearly",
    "Q": "quarterly",
    "Q-DEC": "quarterly",
    "Q-JAN": "quarterly",
    "Q-FEB": "quarterly",
    "Q-MAR": "quarterly",
    "Q-APR": "quarterly",
    "Q-MAY": "quarterly",
    "Q-JUN": "quarterly",
    "Q-JUL": "quarterly",
    "Q-AUG": "quarterly",
    "Q-SEP": "quarterly",
    "Q-OCT": "quarterly",
    "Q-NOV": "quarterly",
    "QS-DEC": "quarterly",
    "QS-JAN": "quarterly",
    "QS-FEB": "quarterly",
    "QS-MAR": "quarterly",
    "QS-APR": "quarterly",
    "QS-MAY": "quarterly",
    "QS-JUN": "quarterly",
    "QS-JUL": "quarterly",
    "QS-AUG": "quarterly",
    "QS-SEP": "quarterly",
    "QS-OCT": "quarterly",
    "QS-NOV": "quarterly",
    "BQ-DEC": "quarterly",
    "BQ-JAN": "quarterly",
    "BQ-FEB": "quarterly",
    "BQ-MAR": "quarterly",
    "BQ-APR": "quarterly",
    "BQ-MAY": "quarterly",
    "BQ-JUN": "quarterly",
    "BQ-JUL": "quarterly",
    "BQ-AUG": "quarterly",
    "BQ-SEP": "quarterly",
    "BQ-OCT": "quarterly",
    "BQ-NOV": "quarterly",
    "BQS-DEC": "quarterly",
    "BQS-JAN": "quarterly",
    "BQS-FEB": "quarterly",
    "BQS-MAR": "quarterly",
    "BQS-APR": "quarterly",
    "BQS-MAY": "quarterly",
    "BQS-JUN": "quarterly",
    "BQS-JUL": "quarterly",
    "BQS-AUG": "quarterly",
    "BQS-SEP": "quarterly",
    "BQS-OCT": "quarterly",
    "BQS-NOV": "quarterly",
    "M": "monthly",
    "BM": "monthly",
    "CBM": "monthly",
    "MS": "monthly",
    "BMS": "monthly",
    "CBMS": "monthly",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "D": "daily",
    "B": "daily",
    "C": "daily",
    "H": "hourly",
    # Minute aliases returned by pandas (including multipliers such as 15T).
    # The lookup helper below also handles arbitrary ``N*T``/``N*min`` values
    # and case variants.
    "T": "minutely",
    "t": "minutely",
    "MIN": "minutely",
    "min": "minutely",
    "15T": "minutely",
    "15t": "minutely",
    "15MIN": "minutely",
    "15min": "minutely",
    "UNKNOWN": "other",
}


_MINUTE_FREQUENCY_PATTERN = re.compile(r"^(?:\d+)?(?:T|MIN)$", re.IGNORECASE)


def get_frequency_category(freq: object) -> str:
    """Map a pandas frequency string to the benchmark frequency category.

    ``pd.infer_freq`` may return aliases such as ``15T`` (pandas 1.x) or
    ``15min`` (newer pandas).  Frequency categories are metadata only; the
    model's actual frequency must be handled separately so that the multiplier
    is not discarded.
    """
    if freq is None:
        return "other"

    key = str(freq).strip()
    category = FREQ_MAP.get(key)
    if category is not None:
        return category

    category = FREQ_MAP.get(key.upper())
    if category is not None:
        return category

    if _MINUTE_FREQUENCY_PATTERN.fullmatch(key):
        return "minutely"

    return "other"


def normalize_model_frequency(freq: object) -> str:
    """Return a pandas-compatible frequency for model time features/padding.

    Preserve minute multipliers (e.g. ``15T`` -> ``15min``), while retaining
    the historical one-letter representation for frequencies whose multiplier
    is not relevant to the existing models (e.g. ``H`` -> ``h``).
    """
    if freq is None:
        raise ValueError("Unable to infer frequency from the time index.")

    key = str(freq).strip()
    minute_match = re.fullmatch(r"(?P<multiple>\d+)?(?:T|MIN)", key, re.IGNORECASE)
    if minute_match:
        multiple = int(minute_match.group("multiple") or 1)
        return "t" if multiple == 1 else f"{multiple}min"

    first = key[0].lower() if key else ""
    if first in ["m", "w", "b", "d", "h", "t", "s"]:
        return first
    return "s"


def get_frequency_granularity(freq: object) -> str:
    """Return the legacy one-letter granularity used by embedding layers."""
    if freq is None:
        return "s"

    key = str(freq).strip()
    if re.fullmatch(r"(?:\d+)?(?:T|MIN)", key, re.IGNORECASE):
        return "t"

    first = key[0].lower() if key else ""
    return first if first in ["m", "w", "b", "d", "h", "t", "s"] else "s"


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read time series data.
    Supports:
    1. Long-format CSV with cols column.
    2. Wide-format CSV with date column, such as WF15.
    """
    data = pd.read_csv(path)

    # ==========================================
    # 1. 普通宽表格式：兼容 WF15
    # ==========================================
    if "cols" not in data.columns:

        if "date" not in data.columns:
            raise ValueError(
                f"Unsupported wide-format CSV: {path}. "
                "Expected a 'date' column."
            )

        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)

        if nrows is not None and isinstance(nrows, int):
            data = data.iloc[:nrows, :]

        return data

    # ==========================================
    # 2. 原有长表格式：保留原始处理逻辑
    # ==========================================
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]
    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]

        col_data = {
            cols_name[j]: data.iloc[
                j * n_points : (j + 1) * n_points, 1
            ].tolist()
            for j in range(n_cols)
        }

        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[
                j * n_points : (j + 1) * n_points, 0
            ].tolist()
            for j in range(n_cols)
        }

        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def load_series_info(file_path: str) -> dict:
    """
    get series info
    :param file_path: series file path
    :return: series info
    :rtype: dict
    """
    data = read_data(file_path)
    file_name = os.path.basename(file_path)
    freq = pd.infer_freq(data.index)
    freq = get_frequency_category(freq)
    if_univariate = data.shape[1] == 1
    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": data.shape[0],
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }
