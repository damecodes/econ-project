from fred_data import get_data
from data_processing import compute_gdp_growth
from model import train_model, predict
import matplotlib.pyplot as plt
from visualization import plot_indicators, plot_correlation, plot_residuals

def main():
    df = get_data()

    df = compute_gdp_growth(df)

    # Train only on past data
    model = train_model(df)
    print(model.summary())

    # Predict for ALL dates (including future)
    df = predict(model, df)

    # Plot
    plt.figure(figsize=(12,6))

    # Actual GDP growth (only where available)
    plt.plot(df.index, df["GDP_Growth"], label="Actual GDP Growth")

    # Predicted GDP growth (extends forward)
    plt.plot(df.index, df["Predicted_GDP_Growth"], 
             linestyle="dashed", label="Predicted GDP Growth")

    plt.title("GDP Nowcasting Model (Extended into 2026)")
    plt.legend()
    plt.savefig("charts/gdp_nowcast_extended.png")
    plt.show()
    plot_indicators(df)
    plot_correlation(df)
    plot_residuals(df)

if __name__ == "__main__":
    main()