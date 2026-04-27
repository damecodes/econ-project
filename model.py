import statsmodels.api as sm
import numpy as np

FEATURES = ["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]

def train_model(df):
    # Use only rows where GDP growth exists
    train_df = df.dropna(subset=["GDP_Growth"]).copy()

    X = train_df[FEATURES]
    y = train_df["GDP_Growth"]

    # Remove inf values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows where features are still missing
    X = X.dropna()

    # Align y with cleaned X
    y = y.loc[X.index]

    # Add constant term
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model


def predict(model, df):
    X = df[FEATURES].copy()

    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values so we can predict future periods
    X = X.interpolate().ffill().bfill()

    # Add constant
    X = sm.add_constant(X)

    # Generate predictions
    df["Predicted_GDP_Growth"] = model.predict(X)

    return df