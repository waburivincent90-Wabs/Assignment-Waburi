
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# Styling 
sns.set_theme(style="whitegrid")
BLUE   = "#1F4E79"
ACCENT = "#2E75B6"
LIGHT  = "#D6E4F0"
GREEN  = "#27AE60"
RED    = "#C0392B"
GOLD   = "#F39C12"

def fmt_usd(x, pos=None):
    """Format axis tick as $000,000"""
    return f"${x:,.0f}"



# PHASE 1 — LOAD DATA

df = pd.read_csv('Ames/AmesHousing.csv')   
print(f"[✓] Loading dataset: {"Ames/AmesHousing.csv"}")
print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        


# PHASE 2 — DATA UNDERSTANDING (EDA)


def run_eda(df: pd.DataFrame) -> None:
    # 2.1 Basic overview 
    print(f"\n[2.1] Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"      Numeric cols  : {df.select_dtypes(include=np.number).shape[1]}")
    print(f"      Categorical   : {df.select_dtypes(include='object').shape[1]}")

    #  2.2 Target variable 
    print("\n[2.2] Target variable — SalePrice:")
    print(df["SalePrice"].describe().apply(lambda x: f"${x:,.0f}" if x > 100 else round(x, 2)).to_string())
    _plot_target_distribution(df["SalePrice"])

    # 2.3 Missing values
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"\n[2.3] Columns with missing values: {len(missing)}")
    print(missing.head(15).round(1).to_string())
    if not missing.empty:
        _plot_missing(missing)

    # 2.4 Numeric summary
    print("\n[2.4] Numeric feature summary (key columns):")
    key_cols = ["SalePrice","Overall Qual","Gr Liv Area","Garage Cars",
                "Year Built","Total Bsmt SF","1st Flr SF","Full Bath"]
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].describe().round(1).to_string())

    print("\n[✓] EDA complete. Plots saved to ./outputs/\n")


def _plot_target_distribution(price: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(price.dropna(), bins=50, color=ACCENT, edgecolor="white")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    axes[0].set_title("Sale Price Distribution", fontsize=13, color=BLUE, fontweight="bold")
    axes[0].set_xlabel("Sale Price (USD)")
    axes[0].set_ylabel("Number of Homes")
    mean_val = price.mean()
    axes[0].axvline(mean_val, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Mean: ${mean_val:,.0f}")
    axes[0].legend()

    # Log-scale histogram (shows normality after log transform)
    axes[1].hist(np.log1p(price.dropna()), bins=50, color=LIGHT,
                 edgecolor=ACCENT, linewidth=0.8)
    axes[1].set_title("Sale Price — Log Scale (more normal)",
                      fontsize=13, color=BLUE, fontweight="bold")
    axes[1].set_xlabel("log(Sale Price)")
    axes[1].set_ylabel("Number of Homes")

    plt.tight_layout()
    plt.savefig("outputs/01_saleprice_distribution.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/01_saleprice_distribution.png")


def _plot_missing(missing: pd.Series) -> None:
    top = missing.head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [RED if v > 50 else ACCENT for v in top.values[::-1]]
    ax.barh(top.index[::-1], top.values[::-1], color=colors)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column (Top 20)",
                 fontsize=13, color=BLUE, fontweight="bold")
    for i, v in enumerate(top.values[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/02_missing_values.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/02_missing_values.png")



# PHASE 3 — DATA PREPARATION


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 62)
    print("  PHASE 3 — DATA PREPARATION")
    print("=" * 62)

    df = df.copy()

    # ── 3.1 Drop high-missing columns (>40%) 
    before = df.shape[1]
    df = df.loc[:, df.isnull().mean() <= 0.40]
    dropped = before - df.shape[1]
    print(f"\n[3.1] Dropped {dropped} columns with >40% missing values.")

    # ── 3.2 Drop rows missing target
    df.dropna(subset=["SalePrice"], inplace=True)

    # ── 3.3 Fill numeric NAs with median 
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── 3.4 Fill categorical NAs with mode
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0], inplace=True)

    # ── 3.5 Feature engineering 
    # House age at time of sale
    if "Year Built" in df.columns and "Yr Sold" in df.columns:
        df["House Age"] = df["Yr Sold"] - df["Year Built"]

    # Total bathrooms
    bath_cols = [c for c in ["Full Bath","Half Bath","Bsmt Full Bath","Bsmt Half Bath"]
                 if c in df.columns]
    if bath_cols:
        df["Total Bathrooms"] = (df.get("Full Bath", 0) +
                                  0.5 * df.get("Half Bath", 0) +
                                  df.get("Bsmt Full Bath", 0) +
                                  0.5 * df.get("Bsmt Half Bath", 0))

    # Total SF
    sf_cols = [c for c in ["1st Flr SF","2nd Flr SF","Total Bsmt SF"] if c in df.columns]
    if sf_cols:
        df["Total SF"] = df[sf_cols].sum(axis=1)

    # ── 3.6 Remove outliers from SalePrice (top 1%) 
    upper = df["SalePrice"].quantile(0.99)
    before_rows = len(df)
    df = df[df["SalePrice"] <= upper]
    print(f"[3.2] Removed {before_rows - len(df)} extreme price outliers (top 1%).")

    print(f"[3.3] Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    df.to_csv("outputs/ames_clean.csv", index=False)
    print("    → Saved: outputs/ames_clean.csv\n")
    return df



# PHASE 4A — CORRELATION ANALYSIS


def run_correlation_analysis(df: pd.DataFrame) -> None:
    print("=" * 62)
    print("  PHASE 4A — CORRELATION ANALYSIS")
    print("=" * 62)

    num_df = df.select_dtypes(include=np.number)

    # Correlation with SalePrice
    corr = num_df.corr()["SalePrice"].drop("SalePrice").sort_values()
    print("\n[4A.1] Top 10 features most positively correlated with SalePrice:")
    print(corr.tail(10).round(3).to_string())
    print("\n[4A.2] Top 5 features most negatively correlated:")
    print(corr.head(5).round(3).to_string())

    # ── Ranked correlation bar chart 
    top_corr = pd.concat([corr.tail(12), corr.head(5)]).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, len(top_corr) * 0.42)))
    colors = [ACCENT if v >= 0 else RED for v in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Sale Price",
                 fontsize=13, color=BLUE, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    for bar, val in zip(bars, top_corr.values):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/03_correlation_bar.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/03_correlation_bar.png")

    # ── Heatmap of top 10 features
    top10 = corr.abs().nlargest(9).index.tolist() + ["SalePrice"]
    corr_matrix = num_df[top10].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("Correlation Heatmap — Top 10 Features vs Sale Price",
                 fontsize=13, color=BLUE, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/04_correlation_heatmap.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/04_correlation_heatmap.png")

    # ── Scatter plots: top 4 numeric predictors
    top4 = [c for c in corr.tail(4).index if c != "SalePrice"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(top4):
        sample = df[[col, "SalePrice"]].dropna()
        axes[i].scatter(sample[col], sample["SalePrice"],
                        alpha=0.3, s=12, color=ACCENT)
        axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Sale Price (USD)")
        axes[i].set_title(f"{col} vs Sale Price (r={corr[col]:.2f})",
                          fontsize=11, color=BLUE, fontweight="bold")
    plt.suptitle("Top 4 Numeric Predictors of Sale Price",
                 fontsize=14, color=BLUE, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/05_scatter_top4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved: outputs/05_scatter_top4.png")

    print("\n[✓] Correlation analysis complete.\n")


# PHASE 4B — SEGMENTATION ANALYSIS


def run_segmentation_analysis(df: pd.DataFrame) -> None:
    
    print("  PHASE 4B — SEGMENTATION ANALYSIS")

    # 1. Sale Price by Overall Quality 
    if "Overall Qual" in df.columns:
        qual_stats = (df.groupby("Overall Qual")["SalePrice"]
                      .agg(["mean","median","count"])
                      .rename(columns={"mean":"Avg Price","median":"Median Price","count":"Homes"}))
        print("\n[4B.1] Average Sale Price by Overall Quality (1–10):")
        print(qual_stats.applymap(lambda x: f"${x:,.0f}" if x > 100 else int(x)).to_string())

        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("Blues", len(qual_stats))
        bars = ax.bar(qual_stats.index, qual_stats["Avg Price"],
                      color=palette, edgecolor="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax.set_xlabel("Overall Quality Rating (1 = Poor → 10 = Excellent)")
        ax.set_ylabel("Average Sale Price (USD)")
        ax.set_title("Average Sale Price by Overall Quality",
                     fontsize=13, color=BLUE, fontweight="bold")
        for bar, val in zip(bars, qual_stats["Avg Price"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 2000, f"${val:,.0f}",
                    ha="center", fontsize=7.5, rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/06_price_by_quality.png", dpi=150)
        plt.close()
        print("    → Saved: outputs/06_price_by_quality.png")

    # 2. Sale Price by Neighbourhood 
    neigh_col = next((c for c in ["Neighborhood","Neighbourhood"] if c in df.columns), None)
    if neigh_col:
        neigh_avg = (df.groupby(neigh_col)["SalePrice"]
                     .agg(["mean","count"])
                     .rename(columns={"mean":"Avg Price","count":"Homes"})
                     .sort_values("Avg Price"))
        print(f"\n[4B.2] Average Sale Price by Neighbourhood (top & bottom 5):")
        print(pd.concat([neigh_avg.head(5), neigh_avg.tail(5)])
              .applymap(lambda x: f"${x:,.0f}" if x > 100 else int(x)).to_string())

        fig, ax = plt.subplots(figsize=(10, 9))
        colors_n = [GREEN if i >= len(neigh_avg) - 5 else
                    (RED if i < 5 else LIGHT) for i in range(len(neigh_avg))]
        bars = ax.barh(neigh_avg.index, neigh_avg["Avg Price"], color=colors_n)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax.set_title("Average Sale Price by Neighbourhood",
                     fontsize=13, color=BLUE, fontweight="bold")
        ax.set_xlabel("Average Sale Price (USD)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GREEN, label="Top 5"),
                            Patch(color=RED, label="Bottom 5")], loc="lower right")
        plt.tight_layout()
        plt.savefig("outputs/07_price_by_neighbourhood.png", dpi=150)
        plt.close()
        print("    → Saved: outputs/07_price_by_neighbourhood.png")

    # 3. Sale Price by Building Type 
    bldg_col = next((c for c in ["Bldg Type","BldgType"] if c in df.columns), None)
    if bldg_col:
        bldg_avg = (df.groupby(bldg_col)["SalePrice"]
                    .agg(["mean","count"])
                    .rename(columns={"mean":"Avg Price","count":"Homes"})
                    .sort_values("Avg Price"))
        print(f"\n[4B.3] Average Sale Price by Building Type:")
        print(bldg_avg.applymap(lambda x: f"${x:,.0f}" if x > 100 else int(x)).to_string())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(bldg_avg.index, bldg_avg["Avg Price"],
               color=sns.color_palette("Blues_d", len(bldg_avg)), edgecolor="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax.set_title("Average Sale Price by Building Type",
                     fontsize=13, color=BLUE, fontweight="bold")
        ax.set_ylabel("Average Sale Price (USD)")
        plt.tight_layout()
        plt.savefig("outputs/08_price_by_bldgtype.png", dpi=150)
        plt.close()
        print("    → Saved: outputs/08_price_by_bldgtype.png")

    #  4. Price trend by Year Sold
    if "Yr Sold" in df.columns:
        yr_avg = df.groupby("Yr Sold")["SalePrice"].agg(["mean","count"])
        print(f"\n[4B.4] Average Sale Price by Year Sold:")
        print(yr_avg.applymap(lambda x: f"${x:,.0f}" if x > 100 else int(x)).to_string())

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        ax1.bar(yr_avg.index, yr_avg["count"], color=LIGHT,
                label="Homes Sold", zorder=2, width=0.4)
        ax2.plot(yr_avg.index, yr_avg["mean"], color=ACCENT,
                 marker="o", linewidth=2.5, label="Avg Price", zorder=3)
        ax1.set_ylabel("Number of Homes Sold")
        ax2.set_ylabel("Average Sale Price (USD)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax1.set_title("Sale Volume & Average Price by Year",
                      fontsize=13, color=BLUE, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.tight_layout()
        plt.savefig("outputs/09_price_by_year.png", dpi=150)
        plt.close()
        print("    → Saved: outputs/09_price_by_year.png")

    #  5. Price by number of garage cars 
    if "Garage Cars" in df.columns:
        garage_avg = (df.groupby("Garage Cars")["SalePrice"].mean().sort_index())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(garage_avg.index.astype(str), garage_avg.values,
               color=sns.color_palette("Blues_d", len(garage_avg)), edgecolor="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax.set_title("Average Sale Price by Garage Capacity",
                     fontsize=13, color=BLUE, fontweight="bold")
        ax.set_xlabel("Number of Cars Garage Fits")
        ax.set_ylabel("Average Sale Price (USD)")
        plt.tight_layout()
        plt.savefig("outputs/10_price_by_garage.png", dpi=150)
        plt.close()
        print("    → Saved: outputs/10_price_by_garage.png")

    print("\n[✓] Segmentation analysis complete.\n")



# PHASE 4C — SIMPLE LINEAR REGRESSION


def run_simple_regression(df: pd.DataFrame) -> dict:
    """
    Simple Linear Regression: SalePrice ~ Overall Quality
    The single most correlated numeric feature with SalePrice.
    Kept intentionally simple — satisfies the modelling requirement.
    """
    print("  PHASE 4C — SIMPLE LINEAR REGRESSION")
    print("  Predicting Sale Price from Overall Quality")
    

    predictor = "Overall Qual"
    target    = "SalePrice"

    model_df = df[[predictor, target]].dropna()
    X = model_df[[predictor]]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mae  = np.mean(np.abs(y_test - preds))

    coef  = model.coef_[0]
    intercept = model.intercept_

    print(f"\n  Predictor   : {predictor}")
    print(f"  Records     : {len(model_df):,} homes")
    print(f"  Train/Test  : {len(X_train):,} / {len(X_test):,}")
    print(f"\n  Regression equation:")
    print(f"    SalePrice = {intercept:,.0f} + {coef:,.0f} × (Overall Quality)")
    print(f"\n  Model Performance:")
    print(f"    R² Score : {r2:.4f}  →  model explains {r2*100:.1f}% of price variance")
    print(f"    RMSE     : ${rmse:,.0f}")
    print(f"    MAE      : ${mae:,.0f}")
    print(f"\n  Interpretation:")
    print(f"    Each 1-point increase in quality is associated with")
    print(f"    an average ${coef:,.0f} increase in sale price.")

    # ── Regression line plot 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_test[predictor], y_test, alpha=0.3, s=15,
               color=ACCENT, label="Actual prices")
    x_line = np.linspace(X[predictor].min(), X[predictor].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color=RED, linewidth=2.5,
            label=f"Regression line (R²={r2:.2f})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    ax.set_xlabel("Overall Quality (1 = Poor → 10 = Excellent)", fontsize=11)
    ax.set_ylabel("Sale Price (USD)", fontsize=11)
    ax.set_title("Simple Linear Regression\nSale Price ~ Overall Quality",
                 fontsize=13, color=BLUE, fontweight="bold")
    ax.legend()

    # Annotate equation
    eq_text = f"SalePrice = {intercept:,.0f} + {coef:,.0f} × Quality\nR² = {r2:.3f}"
    ax.text(0.05, 0.78, eq_text, transform=ax.transAxes,
            fontsize=10, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT, alpha=0.8))
    plt.tight_layout()
    plt.savefig("outputs/11_regression_line.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/11_regression_line.png")

    # ── Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=12, color=ACCENT)
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], color=RED, linestyle="--",
            linewidth=1.5, label="Perfect prediction")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    ax.set_xlabel("Actual Sale Price")
    ax.set_ylabel("Predicted Sale Price")
    ax.set_title(f"Actual vs Predicted Sale Price\n(R² = {r2:.3f}  |  RMSE = ${rmse:,.0f})",
                 fontsize=12, color=BLUE, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/12_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/12_actual_vs_predicted.png")

    # ── Residual distribution 
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=60, color=ACCENT, edgecolor="white")
    ax.axvline(0, color=RED, linewidth=1.5, linestyle="--", label="Zero error")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    ax.set_xlabel("Residual  (Actual Price − Predicted Price)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution",
                 fontsize=12, color=BLUE, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/13_residuals.png", dpi=150)
    plt.close()
    print("    → Saved: outputs/13_residuals.png\n")

    return {"coef": coef, "intercept": intercept, "r2": r2, "rmse": rmse, "mae": mae}



# PHASE 5 — KEY INSIGHTS


def print_insights(df: pd.DataFrame, reg_results: dict) -> None:

    print("  PHASE 5 — KEY BUSINESS INSIGHTS")

    print(f"\n  1. PRICE RANGE")
    print(f"     Homes in Ames range from ${df['SalePrice'].min():,.0f} to "
          f"${df['SalePrice'].max():,.0f}.")
    print(f"     Median price: ${df['SalePrice'].median():,.0f}")

    if "Overall Qual" in df.columns:
        low  = df[df["Overall Qual"] <= 4]["SalePrice"].mean()
        high = df[df["Overall Qual"] >= 8]["SalePrice"].mean()
        print(f"\n  2. QUALITY PREMIUM")
        print(f"     Low-quality homes (≤4) avg ${low:,.0f}.")
        print(f"     High-quality homes (≥8) avg ${high:,.0f}.")
        print(f"     Quality premium: ${high - low:,.0f} ({((high-low)/low)*100:.0f}% more).")

    if "Gr Liv Area" in df.columns:
        corr = df["Gr Liv Area"].corr(df["SalePrice"])
        print(f"\n  3. LIVING AREA")
        print(f"     Above-ground living area has a {corr:.2f} correlation")
        print(f"     with sale price — the #2 driver after quality.")

    neigh_col = next((c for c in ["Neighborhood","Neighbourhood"] if c in df.columns), None)
    if neigh_col:
        best  = df.groupby(neigh_col)["SalePrice"].mean().idxmax()
        worst = df.groupby(neigh_col)["SalePrice"].mean().idxmin()
        best_p  = df.groupby(neigh_col)["SalePrice"].mean().max()
        worst_p = df.groupby(neigh_col)["SalePrice"].mean().min()
        print(f"\n  4. NEIGHBOURHOOD GAP")
        print(f"     Most expensive: {best} (avg ${best_p:,.0f})")
        print(f"     Least expensive: {worst} (avg ${worst_p:,.0f})")
        print(f"     Location gap: ${best_p - worst_p:,.0f}")

    print(f"\n  5. REGRESSION MODEL")
    print(f"     Overall Quality alone explains {reg_results['r2']*100:.1f}% of")
    print(f"     sale price variance (R² = {reg_results['r2']:.3f}).")
    print(f"     Each quality point = +${reg_results['coef']:,.0f} on average.")

    print("\n[✓] All insights generated.\n")



# MAIN


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  AMES HOUSING — SALE PRICE ANALYSIS")
    print("  CRISP-DM Data Analytics Capstone")
    print("=" * 62 + "\n")
    

# Your custom load (replaces load_data)
df_raw = pd.read_csv('Ames/AmesHousing.csv')
print(f"[✓] Loading dataset: {'Ames/AmesHousing.csv'}")
print(f"    Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns\n")

    # ── EDA 
run_eda(df_raw)

    # ── Prepare 
df_clean = prepare_data(df_raw)

    # ── Correlation analysis
run_correlation_analysis(df_clean)

    # ── Segmentation analysis 
run_segmentation_analysis(df_clean)

    # ── Simple regression 
results = run_simple_regression(df_clean)

    # ── Insights 
print_insights(df_clean, results)

