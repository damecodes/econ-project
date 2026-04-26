import statsmodels.api as sm

def train_model(df):
    X = df[["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]]
    y = df["GDP_Growth"]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def predict(model, df):
    X = df[["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]]
    X = sm.add_constant(X)

    df["Predicted_GDP_Growth"] = model.predict(X)
    return df