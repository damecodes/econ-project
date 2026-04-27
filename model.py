import statsmodels.api as sm

FEATURES = ["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]

def train_model(df):
    train_df = df.dropna(subset=["GDP_Growth"])

    X = train_df[FEATURES]
    y = train_df["GDP_Growth"]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def predict(model, df):
    X = df[FEATURES]
    X = sm.add_constant(X)

    df["Predicted_GDP_Growth"] = model.predict(X)
    return df