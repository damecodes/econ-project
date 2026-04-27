from fredapi import Fred
import pandas as pd
from config import FRED_API_KEY

fred = Fred(api_key=FRED_API_KEY)

def get_data():
    series = {
        "GDP": "GDPC1",
        "Unemployment": "UNRATE",
        "CPI": "CPIAUCSL",
        "Industrial_Production": "INDPRO",
        "Retail_Sales": "RSAFS"
    }

    data = pd.DataFrame()

    for name, code in series.items():
        data[name] = fred.get_series(code)

    data = data.sort_index()
    return data