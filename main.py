from matplotlib import pyplot as plt
from fred_data import get_data
from visualization import plot_data
from model import train_model, predict
import pandas as pd

def main():
    df = get_data()

    df["GDP_Growth"] = df["GDP"].pct_change() * 100
    df = df.dropna()

    plot_data(df)

    model = train_model(df)
    print(model.summary())

    df = predict(model, df)

    df[["GDP_Growth", "Predicted_GDP_Growth"]].plot(figsize=(10,5))
    plt.title("Actual vs Predicted GDP Growth")
    plt.savefig("charts/gdp_prediction.png")
    plt.show()

if __name__ == "__main__":
    main()