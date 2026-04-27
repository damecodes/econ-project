import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURES = ["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]

def train_model(df):
    train_df = df.dropna(subset=["GDP_Growth"]).copy()

    X = train_df[FEATURES]
    y = train_df["GDP_Growth"]

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=FEATURES)

    # Add constant (force it)
    X_scaled = sm.add_constant(X_scaled, has_constant='add')

    model = sm.OLS(y, X_scaled).fit()

    return model, scaler


def predict(model, scaler, df):
    X = df[FEATURES].copy()

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.interpolate().ffill().bfill()

    # Use same scaler
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=FEATURES)

    # Add constant (force it)
    X_scaled = sm.add_constant(X_scaled, has_constant='add')

    # Align columns EXACTLY with training
    X_scaled = X_scaled[model.model.exog_names]

    df["Predicted_GDP_Growth"] = model.predict(X_scaled)

    return df