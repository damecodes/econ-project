import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = ["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]

def train_model(df):
    train_df = df.dropna(subset=["GDP_Growth"]).copy()

    X = train_df[FEATURES]
    y = train_df["GDP_Growth"]

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]

    # ✅ STANDARDIZE HERE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame (important for statsmodels)
    import pandas as pd
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=FEATURES)

    X_scaled = sm.add_constant(X_scaled)

    model = sm.OLS(y, X_scaled).fit()

    # Return BOTH model and scaler
    return model, scaler


def predict(model, scaler, df):
    import pandas as pd

    X = df[FEATURES].copy()

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.interpolate().ffill().bfill()

    # ✅ USE SAME SCALER (DO NOT FIT AGAIN)
    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=FEATURES)

    X_scaled = sm.add_constant(X_scaled)

    df["Predicted_GDP_Growth"] = model.predict(X_scaled)

    return df