import matplotlib.pyplot as plt

def plot_indicators(df):
    cols = ["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]

    df[cols].plot(subplots=True, figsize=(10,8), title="Macroeconomic Indicators")
    plt.tight_layout()
    plt.savefig("charts/indicators.png")
    plt.show()

def plot_data(df):
    df.plot(subplots=True, figsize=(10, 8), title="Macroeconomic Indicators")
    plt.tight_layout()
    plt.savefig("charts/macro_data.png")
    plt.show()

def plot_correlation(df):
    corr = df[["GDP_Growth", "Unemployment", "CPI", 
               "Industrial_Production", "Retail_Sales"]].corr()

    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.savefig("charts/correlation.png")
    plt.show()

def plot_residuals(df):
    df["Residuals"] = df["GDP_Growth"] - df["Predicted_GDP_Growth"]

    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["Residuals"])
    plt.title("Model Residuals")
    plt.savefig("charts/residuals.png")
    plt.show()