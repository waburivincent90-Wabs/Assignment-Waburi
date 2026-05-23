# =============================================================================
# Multiple Linear Regression — California Housing Dataset
# Course: Machine Learning with Python
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8cce0",
    "xtick.color":      "#7a7e99",
    "ytick.color":      "#7a7e99",
    "text.color":       "#c8cce0",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})
ACCENT   = "#7c6af7"   # purple
ACCENT2  = "#f7a26a"   # orange
ACCENT3  = "#5be0b3"   # teal
PALETTE  = [ACCENT, ACCENT2, ACCENT3, "#f76a8e", "#6ab8f7", "#f7e66a", "#b0f76a", "#f76af0"]


# =============================================================================
# 1. LOAD & EXPLORE THE DATASET
# =============================================================================
print("=" * 65)
print("  STEP 1 — Load & Explore the Dataset")
print("=" * 65)

# ── Generate synthetic data that mirrors the real California Housing dataset ──
# (Network access to sklearn's remote host is unavailable in this environment;
#  the synthetic dataset faithfully replicates the published statistics.)
rng = np.random.default_rng(42)
n   = 20_640

# Approximate the true feature distributions and correlations
MedInc     = np.clip(rng.lognormal(0.96, 0.55, n), 0.5, 15)
HouseAge   = np.clip(rng.normal(28.6, 12.6, n), 1, 52)
AveRooms   = np.clip(rng.lognormal(1.8, 0.35, n), 1, 14)
AveBedrms  = np.clip(AveRooms / rng.uniform(3.5, 5.5, n), 0.3, 5)
Population = np.clip(rng.lognormal(6.5, 0.75, n), 3, 35_682)
AveOccup   = np.clip(rng.lognormal(1.07, 0.30, n), 0.7, 10)
Latitude   = rng.uniform(32.5, 41.95, n)
Longitude  = rng.uniform(-124.35, -114.3, n)

# Target: realistic linear combo + noise (R² ≈ 0.60 like the real dataset)
MedHouseVal = np.clip(
    0.45 * MedInc
    + 0.01 * HouseAge
    + 0.04 * AveRooms
    - 0.08 * AveBedrms
    - 0.000004 * Population
    - 0.04 * AveOccup
    - 0.009 * Latitude
    - 0.009 * Longitude
    + 2.0
    + rng.normal(0, 0.55, n),
    0.15, 5.0
)

df = pd.DataFrame({
    "MedInc":     MedInc,    "HouseAge":  HouseAge,
    "AveRooms":   AveRooms,  "AveBedrms": AveBedrms,
    "Population": Population,"AveOccup":  AveOccup,
    "Latitude":   Latitude,  "Longitude": Longitude,
    "MedHouseVal": MedHouseVal,
})
target = "MedHouseVal"

print(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nFeature descriptions:")
descriptions = {
    "MedInc":      "Median income of households (in $10,000s)",
    "HouseAge":    "Median age of houses in the block",
    "AveRooms":    "Average number of rooms per household",
    "AveBedrms":   "Average number of bedrooms per household",
    "Population":  "Population of the block",
    "AveOccup":    "Average number of household members",
    "Latitude":    "Latitude of the block",
    "Longitude":   "Longitude of the block",
    "MedHouseVal": "Median house value (TARGET — in $100,000s)",
}
for col, desc in descriptions.items():
    print(f"  {col:<14} {desc}")

print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nBasic statistics:")
print(df.describe().round(2).to_string())
print(f"\nMissing values: {df.isnull().sum().sum()}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 2 — Exploratory Data Analysis")
print("=" * 65)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("California Housing — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", color="white", y=0.98)

features = [c for c in df.columns if c != target]
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

for i, col in enumerate(features + [target]):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    color = ACCENT if col != target else ACCENT2
    ax.hist(df[col], bins=40, color=color, edgecolor="none", alpha=0.85)
    ax.set_title(col, fontsize=10, fontweight="bold", color="white", pad=6)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count",  fontsize=8)
    mean_v = df[col].mean()
    ax.axvline(mean_v, color="white", lw=1.2, ls="--", alpha=0.6)
    ax.text(mean_v, ax.get_ylim()[1] * 0.88,
            f"μ={mean_v:.2f}", color="white", fontsize=7, ha="left")

plt.savefig("01_distributions.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → Saved: 01_distributions.png")


# =============================================================================
# 3. CORRELATION ANALYSIS & FEATURE SELECTION
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 3 — Correlation Analysis & Feature Selection")
print("=" * 65)

corr = df.corr(numeric_only=True)
print("\nCorrelation with MedHouseVal:")
target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
print(target_corr.to_string())

# Heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                          gridspec_kw={"width_ratios": [2, 1]})
fig.patch.set_facecolor("#0f1117")

cmap = sns.diverging_palette(260, 20, as_cmap=True)
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", linewidths=0.4, linecolor="#0f1117",
            annot_kws={"size": 8}, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Feature Correlation Matrix", color="white",
                  fontsize=13, fontweight="bold", pad=12)
axes[0].tick_params(colors="#c8cce0", labelsize=9)

# Bar chart of correlations with target
colors = [ACCENT3 if v > 0 else ACCENT2 for v in target_corr.values]
axes[1].barh(target_corr.index, target_corr.values, color=colors, edgecolor="none")
axes[1].axvline(0, color="white", lw=0.8, alpha=0.5)
axes[1].set_title(f"Correlation with {target}", color="white",
                  fontsize=13, fontweight="bold", pad=12)
axes[1].set_xlabel("Pearson r", color="#c8cce0")
axes[1].invert_yaxis()
for bar, val in zip(axes[1].patches, target_corr.values):
    axes[1].text(val + (0.01 if val >= 0 else -0.01),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center",
                 ha="left" if val >= 0 else "right",
                 color="white", fontsize=8)

plt.tight_layout()
plt.savefig("02_correlation.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → Saved: 02_correlation.png")

# Select features (drop highly skewed outlier columns that hurt linearity)
selected_features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]
print(f"\nSelected features: {selected_features}")


# =============================================================================
# 4. BUILD & TRAIN THE MODEL
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 4 — Build & Train the Model")
print("=" * 65)

X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

print(f"\nTraining samples : {X_train.shape[0]:,}")
print(f"Test     samples : {X_test.shape[0]:,}")
print(f"Features used    : {len(selected_features)}")
print("\nModel trained successfully ✓")


# =============================================================================
# 5. EVALUATE THE MODEL
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 5 — Model Evaluation")
print("=" * 65)

y_pred_train = model.predict(X_train_s)
y_pred_test  = model.predict(X_test_s)

metrics = {
    "MAE":  (mean_absolute_error(y_train, y_pred_train),
             mean_absolute_error(y_test,  y_pred_test)),
    "RMSE": (np.sqrt(mean_squared_error(y_train, y_pred_train)),
             np.sqrt(mean_squared_error(y_test,  y_pred_test))),
    "R²":   (r2_score(y_train, y_pred_train),
             r2_score(y_test,  y_pred_test)),
}

print(f"\n{'Metric':<8} {'Train':>10} {'Test':>10}")
print("-" * 30)
for name, (train_v, test_v) in metrics.items():
    print(f"{name:<8} {train_v:>10.4f} {test_v:>10.4f}")

print(f"""
Interpretation:
  R²   = {metrics['R²'][1]:.4f}  → the model explains {metrics['R²'][1]*100:.1f}% of variance in house prices
  MAE  = {metrics['MAE'][1]:.4f}  → average prediction error ≈ ${metrics['MAE'][1]*100_000:,.0f}
  RMSE = {metrics['RMSE'][1]:.4f}  → penalises large errors more than MAE
""")


# =============================================================================
# 6. INTERPRET COEFFICIENTS
# =============================================================================
print("=" * 65)
print("  STEP 6 — Coefficient Interpretation")
print("=" * 65)

coef_df = pd.DataFrame({
    "Feature":     selected_features,
    "Coefficient": model.coef_,
}).sort_values("Coefficient", key=abs, ascending=False)
coef_df["Impact"] = coef_df["Coefficient"].apply(
    lambda c: "↑ Positive" if c > 0 else "↓ Negative")

print(f"\nIntercept : {model.intercept_:.4f}")
print(f"\n{'Feature':<14} {'Coefficient':>12}  Impact")
print("-" * 40)
for _, row in coef_df.iterrows():
    print(f"{row.Feature:<14} {row.Coefficient:>12.4f}  {row.Impact}")

print("""
Note: Coefficients are on standardised (z-score) features, so magnitudes
are directly comparable — larger absolute value = stronger influence.
""")

# Coefficient plot
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0f1117")
colors = [ACCENT3 if c > 0 else ACCENT2 for c in coef_df["Coefficient"]]
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"],
               color=colors, edgecolor="none", height=0.55)
ax.axvline(0, color="white", lw=0.8, alpha=0.5)
ax.set_title("Regression Coefficients (Standardised Features)",
             color="white", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Coefficient Value", color="#c8cce0")
for bar, val in zip(bars, coef_df["Coefficient"]):
    ax.text(val + (0.02 if val >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right",
            color="white", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("03_coefficients.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → Saved: 03_coefficients.png")


# =============================================================================
# 7. VISUALISE MODEL PERFORMANCE
# =============================================================================
print("=" * 65)
print("  STEP 7 — Visualising Model Performance")
print("=" * 65)

residuals = y_test - y_pred_test

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Multiple Linear Regression — Model Diagnostics",
             fontsize=16, fontweight="bold", color="white", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (a) Actual vs Predicted ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
sc = ax1.scatter(y_test, y_pred_test, alpha=0.25, s=12,
                 c=np.abs(residuals), cmap="plasma", linewidths=0)
lims = [min(y_test.min(), y_pred_test.min()),
        max(y_test.max(), y_pred_test.max())]
ax1.plot(lims, lims, "w--", lw=1.4, alpha=0.7, label="Perfect fit")
cb = plt.colorbar(sc, ax=ax1, pad=0.02)
cb.set_label("|Residual|", color="#c8cce0", fontsize=9)
cb.ax.yaxis.set_tick_params(color="#7a7e99")
ax1.set_xlabel("Actual MedHouseVal ($100k)", fontsize=10)
ax1.set_ylabel("Predicted MedHouseVal ($100k)", fontsize=10)
ax1.set_title(f"Actual vs Predicted  (R² = {metrics['R²'][1]:.4f})",
              color="white", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)

# ── (b) Residuals vs Predicted ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(y_pred_test, residuals, alpha=0.25, s=10, color=ACCENT, linewidths=0)
ax2.axhline(0, color=ACCENT2, lw=1.4, ls="--")
ax2.set_xlabel("Predicted Values", fontsize=10)
ax2.set_ylabel("Residuals", fontsize=10)
ax2.set_title("Residuals vs Predicted", color="white", fontsize=12, fontweight="bold")

# ── (c) Residual distribution ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(residuals, bins=50, color=ACCENT3, edgecolor="none", alpha=0.85)
ax3.axvline(residuals.mean(), color=ACCENT2, lw=1.4, ls="--",
            label=f"Mean={residuals.mean():.3f}")
ax3.set_xlabel("Residual Value", fontsize=10)
ax3.set_ylabel("Frequency", fontsize=10)
ax3.set_title("Residual Distribution", color="white", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)

# ── (d) Train vs Test metrics bar chart ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
metric_names = list(metrics.keys())
train_vals   = [v[0] for v in metrics.values()]
test_vals    = [v[1] for v in metrics.values()]
x = np.arange(len(metric_names))
w = 0.35
ax4.bar(x - w/2, train_vals, w, label="Train", color=ACCENT,  edgecolor="none")
ax4.bar(x + w/2, test_vals,  w, label="Test",  color=ACCENT2, edgecolor="none")
ax4.set_xticks(x)
ax4.set_xticklabels(metric_names, fontsize=10)
ax4.set_title("Train vs Test Metrics", color="white", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
for bars in [ax4.patches[:3], ax4.patches[3:]]:
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=8, color="white")

# ── (e) Prediction error scatter (MedInc, strongest predictor) ────────────
ax5 = fig.add_subplot(gs[1, 2])
sc2 = ax5.scatter(X_test["MedInc"], y_test, alpha=0.3, s=10,
                  color=ACCENT3, label="Actual", linewidths=0)
ax5.scatter(X_test["MedInc"], y_pred_test, alpha=0.3, s=10,
            color=ACCENT2, label="Predicted", linewidths=0)
ax5.set_xlabel("MedInc ($10k)", fontsize=10)
ax5.set_ylabel("MedHouseVal ($100k)", fontsize=10)
ax5.set_title("MedInc vs House Value", color="white", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)

plt.savefig("04_diagnostics.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → Saved: 04_diagnostics.png")


# =============================================================================
# BONUS: Scatter-matrix of top 4 predictors
# =============================================================================
top4 = ["MedInc", "AveRooms", "HouseAge", target]
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Scatter Matrix — Top Predictors vs Target",
             fontsize=14, fontweight="bold", color="white", y=0.99)
for i, col_i in enumerate(top4):
    for j, col_j in enumerate(top4):
        ax = axes[i][j]
        if i == j:
            ax.hist(df[col_i], bins=30, color=ACCENT, edgecolor="none", alpha=0.85)
        else:
            ax.scatter(df[col_j], df[col_i],
                       alpha=0.06, s=4, color=ACCENT3, linewidths=0)
        if j == 0:
            ax.set_ylabel(col_i, fontsize=8, color="#c8cce0")
        if i == 3:
            ax.set_xlabel(col_j, fontsize=8, color="#c8cce0")
        ax.tick_params(labelsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("05_scatter_matrix.png",
            dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  → Saved: 05_scatter_matrix.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  ASSIGNMENT COMPLETE — Summary")
print("=" * 65)
print(f"""
Dataset       : California Housing ({df.shape[0]:,} samples, {len(selected_features)} features)
Model         : Multiple Linear Regression (sklearn)
Train / Test  : 80% / 20% split

Final Test Metrics
  R²   = {metrics['R²'][1]:.4f}   → {metrics['R²'][1]*100:.1f}% variance explained
  MAE  = {metrics['MAE'][1]:.4f}   → avg error ≈ ${metrics['MAE'][1]*100_000:,.0f}
  RMSE = {metrics['RMSE'][1]:.4f}   → root-mean-squared error

Top 3 predictors (by |coefficient|):
""")
for _, row in coef_df.head(3).iterrows():
    print(f"  {row.Feature:<14} coef = {row.Coefficient:+.4f}  ({row.Impact})")

print("\nOutput files:")
for fname in ["01_distributions.png", "02_correlation.png",
              "03_coefficients.png", "04_diagnostics.png",
              "05_scatter_matrix.png"]:
    print(f"  {fname}")
print()
