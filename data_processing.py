import pandas as pd

def compute_gdp_growth(df):
    df = df.copy()
    df["GDP_Growth"] = df["GDP"].pct_change() * 400
    return df


def compute_feature_growth(df):
    df = df.copy()

    df["CPI"] = df["CPI"].pct_change() * 100
    df["Industrial_Production"] = df["Industrial_Production"].pct_change() * 100
    df["Retail_Sales"] = df["Retail_Sales"].pct_change() * 100

    # unemployment → use difference, not pct change
    df["Unemployment"] = df["Unemployment"].diff()

    return df