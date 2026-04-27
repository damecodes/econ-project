import pandas as pd

def compute_gdp_growth(df):
    df = df.copy()
    df["GDP_Growth"] = df["GDP"].pct_change() * 400
    return df