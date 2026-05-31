import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.preprocessing import label_binarize
import time

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg':      '#0F1117',
    'panel':   '#1A1D2E',
    'card':    '#252840',
    'accent1': '#FFB347',   # orange  — Gaussian NB
    'accent2': '#56CFE1',   # cyan    — Bernoulli NB
    'accent3': '#FF8C69',   # salmon  — Complement NB
    'text':    '#E8EAF6',
    'subtext': '#9DA3C8',
    'grid':    '#2A2D3E',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['panel'],
    'axes.edgecolor':   COLORS['grid'],
    'axes.labelcolor':  COLORS['text'],
    'axes.titlecolor':  COLORS['text'],
    'xtick.color':      COLORS['subtext'],
    'ytick.color':      COLORS['subtext'],
    'text.color':       COLORS['text'],
    'grid.color':       COLORS['grid'],
    'grid.linewidth':   0.6,
    'legend.facecolor': COLORS['card'],
    'legend.edgecolor': COLORS['grid'],
    'font.family':      'DejaVu Sans',
})

# ─── 1. Load & Preprocess ─────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('AmesHousing.csv')
print(f"Dataset shape: {df.shape}")

FEATURES = [
    'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Garage Area',
    'Year Built', 'Full Bath', 'TotRms AbvGrd', '1st Flr SF',
    'Lot Area', 'Overall Cond',
]

bins   = [0, 139000, 189000, float('inf')]
labels = ['Low', 'Medium', 'High']
df['PriceClass'] = pd.cut(df['SalePrice'], bins=bins, labels=labels)

df_clean = df[FEATURES + ['PriceClass']].dropna()
print(f"Clean shape: {df_clean.shape}")

le = LabelEncoder()
X  = df_clean[FEATURES].values
y  = le.fit_transform(df_clean['PriceClass'])
class_names = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gaussian NB → StandardScaler  |  Bernoulli & Complement NB → MinMaxScaler
std_scaler = StandardScaler()
mm_scaler  = MinMaxScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)
X_train_mm  = mm_scaler.fit_transform(X_train)
X_test_mm   = mm_scaler.transform(X_test)

# ─── 2. Define Bayesian Classifiers ──────────────────────────────────────────
classifiers = {
    'Gaussian NB':   (GaussianNB(),   X_train_std, X_test_std,  StandardScaler),
    'Bernoulli NB':  (BernoulliNB(),  X_train_mm,  X_test_mm,   MinMaxScaler),
    'Complement NB': (ComplementNB(), X_train_mm,  X_test_mm,   MinMaxScaler),
}
short_names = ['Gaussian NB', 'Bernoulli NB', 'Complement NB']
clf_colors  = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]

# ─── 3. Train & Evaluate ─────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*65)
print(f"{'Classifier':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'CV F1':>10}")
print("="*65)

for sname, col in zip(short_names, clf_colors):
    clf, Xtr, Xte, scaler_cls = classifiers[sname]

    t0 = time.time()
    clf.fit(Xtr, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(
        Pipeline([('sc', scaler_cls()), ('clf', clf.__class__(**clf.get_params()))]),
        X, y, cv=cv, scoring='f1_weighted'
    )
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_score = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='weighted')

    results[sname] = dict(acc=acc, prec=prec, rec=rec, f1=f1,
                          auc=auc_score, cm=cm, y_pred=y_pred, y_prob=y_prob,
                          cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                          train_time=train_time)
    print(f"{sname:<18} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} "
          f"{cv_scores.mean():>6.3f}±{cv_scores.std():.3f}")
print("="*65)

# ─── 4. Dashboard ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20), facecolor=COLORS['bg'])
fig.suptitle('Bayesian Classifiers — Naive Bayes Variants\nAmes Housing Price Classification',
             fontsize=18, fontweight='bold', color=COLORS['text'], y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.93, bottom=0.04)

metrics       = ['acc', 'prec', 'rec', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3'], '#A0E77D']
x_pos = np.arange(len(short_names))
width = 0.2

# Panel A — Grouped bar
ax_bar = fig.add_subplot(gs[0, :])
for i, (m, ml, mc) in enumerate(zip(metrics, metric_labels, metric_colors)):
    vals = [results[s][m] for s in short_names]
    bars = ax_bar.bar(x_pos + (i - 1.5) * width, vals, width,
                      color=mc, alpha=0.85, label=ml,
                      edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(short_names, fontsize=12)
ax_bar.set_ylim(0, 1.08)
ax_bar.set_ylabel('Score', fontsize=11)
ax_bar.set_title('A  |  Evaluation Metrics — All Bayesian Variants', fontsize=12, fontweight='bold')
ax_bar.legend(fontsize=9, loc='lower right')
ax_bar.axhline(0.7, color=COLORS['grid'], linestyle='--', linewidth=1, alpha=0.7)
ax_bar.grid(axis='y', alpha=0.4)

# Panel B — CV F1
ax_cv = fig.add_subplot(gs[1, 0])
cv_means = [results[s]['cv_mean'] for s in short_names]
cv_stds  = [results[s]['cv_std']  for s in short_names]
bars = ax_cv.barh(short_names, cv_means, xerr=cv_stds, color=clf_colors,
                  edgecolor='white', linewidth=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': 'white', 'capsize': 5})
ax_cv.set_xlim(0, 1.05)
ax_cv.set_xlabel('CV F1-Score (5-fold)', fontsize=10)
ax_cv.set_title('B  |  Cross-Validation F1', fontsize=11, fontweight='bold')
ax_cv.grid(axis='x', alpha=0.4)
for bar, v in zip(bars, cv_means):
    ax_cv.text(v + 0.01, bar.get_y() + bar.get_height()/2,
               f'{v:.3f}', va='center', fontsize=9, color=COLORS['text'])

# Panel C — AUC
ax_auc = fig.add_subplot(gs[1, 1])
aucs = [results[s]['auc'] for s in short_names]
bars = ax_auc.bar(short_names, aucs, color=clf_colors, edgecolor='white', linewidth=0.5)
ax_auc.set_ylim(0, 1.1)
ax_auc.set_ylabel('AUC (OvR Weighted)', fontsize=10)
ax_auc.set_title('C  |  ROC-AUC Scores', fontsize=11, fontweight='bold')
ax_auc.tick_params(axis='x', labelrotation=15, labelsize=9)
ax_auc.grid(axis='y', alpha=0.4)
for bar, v in zip(bars, aucs):
    ax_auc.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', fontsize=9, color=COLORS['text'])

# Panel D — Training time
ax_time = fig.add_subplot(gs[1, 2])
times = [results[s]['train_time'] * 1000 for s in short_names]
bars = ax_time.bar(short_names, times, color=clf_colors, edgecolor='white', linewidth=0.5)
ax_time.set_ylabel('Training Time (ms)', fontsize=10)
ax_time.set_title('D  |  Training Time', fontsize=11, fontweight='bold')
ax_time.tick_params(axis='x', labelrotation=15, labelsize=9)
ax_time.grid(axis='y', alpha=0.4)
for bar, v in zip(bars, times):
    ax_time.text(bar.get_x() + bar.get_width()/2, v + max(times)*0.01,
                 f'{v:.2f}ms', ha='center', va='bottom', fontsize=9, color=COLORS['text'])

# Panels E–G — Confusion matrices
for idx, (sname, col) in enumerate(zip(short_names, clf_colors)):
    ax_cm = fig.add_subplot(gs[2, idx])
    cm = results[sname]['cm']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt='d', ax=ax_cm, cmap='YlOrRd', cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor=COLORS['bg'],
                annot_kws={'size': 10, 'color': 'white', 'weight': 'bold'})
    ax_cm.set_title(f'{chr(69+idx)}  |  {sname}\nAcc={results[sname]["acc"]:.3f}',
                    fontsize=9, fontweight='bold', color=col)
    ax_cm.set_xlabel('Predicted', fontsize=9)
    ax_cm.set_ylabel('Actual', fontsize=9)

plt.savefig('bayesian_classifiers.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("\n✓ Saved: bayesian_classifiers.png")

# ─── 5. ROC Curves ───────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg'])
fig2.suptitle('ROC Curves — Bayesian (Naive Bayes) Classifiers',
              fontsize=15, fontweight='bold', color=COLORS['text'])
y_bin = label_binarize(y_test, classes=[0, 1, 2])
class_colors = [COLORS['accent1'], COLORS['accent2'], COLORS['accent3']]

for ax, (sname, col) in zip(axes2, zip(short_names, clf_colors)):
    y_prob = results[sname]['y_prob']
    for ci, (cname, cc) in enumerate(zip(class_names, class_colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
        ax.plot(fpr, tpr, color=cc, lw=2, label=f'{cname} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1], color=COLORS['subtext'], linestyle='--', lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title(sname, fontsize=11, fontweight='bold', color=col)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_roc.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("✓ Saved: bayesian_roc.png")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Classifier':<18} {'Acc':>6} {'F1':>6} {'CV-F1':>10} {'AUC':>7}")
print("="*60)
for sname in short_names:
    r = results[sname]
    print(f"{sname:<18} {r['acc']:>6.3f} {r['f1']:>6.3f} "
          f"{r['cv_mean']:>6.3f}±{r['cv_std']:.3f} {r['auc']:>7.3f}")
print("="*60)
best = max(short_names, key=lambda s: results[s]['f1'])
print(f"\n Best: {best}  →  F1={results[best]['f1']:.4f}")
