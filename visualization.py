import matplotlib.pyplot as plt

def plot_data(df):
    df.plot(subplots=True, figsize=(10, 8), title="Macroeconomic Indicators")
    plt.tight_layout()
    plt.savefig("charts/macro_data.png")
    plt.show()