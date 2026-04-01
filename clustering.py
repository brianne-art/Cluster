import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d0f14',
    'axes.facecolor':    '#0d0f14',
    'axes.edgecolor':    '#2a3040',
    'axes.labelcolor':   '#a0aec0',
    'axes.titlecolor':   '#e2e8f0',
    'xtick.color':       '#a0aec0',
    'ytick.color':       '#a0aec0',
    'grid.color':        '#1a1f2e',
    'grid.linewidth':    0.8,
    'text.color':        '#e2e8f0',
    'font.family':       'monospace',
    'legend.facecolor':  '#161b27',
    'legend.edgecolor':  '#2a3040',
    'legend.labelcolor': '#a0aec0',
    'savefig.facecolor': '#0d0f14',
})

SPECIES_COLORS  = ['#4a9eff', '#ff6b6b', '#ffd93d']
CLUSTER_COLORS  = ['#00ff88', '#c77dff', '#ff9f43']
MARKER_SIZE     = 40
DPI             = 150

def save(name):
    plt.savefig(name, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved {name}')

# ── Load data ─────────────────────────────────────────────────────────────────
iris   = load_iris()
X      = iris.data          # (150, 4)
y      = iris.target        # 0=setosa, 1=versicolor, 2=virginica
names  = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# ── PCA (fit once, reuse) ─────────────────────────────────────────────────────
pca      = PCA(n_components=2)
X_pca    = pca.fit_transform(X)
var      = pca.explained_variance_ratio_ * 100
pc1_lbl  = f'PC1 ({var[0]:.1f}%)'
pc2_lbl  = f'PC2 ({var[1]:.1f}%)'

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — PCA scatter, true labels
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 1: PCA scatter — true labels')
fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(names):
    mask = y == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=SPECIES_COLORS[i], s=MARKER_SIZE, alpha=0.85,
               edgecolors='none', label=name)
ax.set_xlabel(pc1_lbl); ax.set_ylabel(pc2_lbl)
ax.set_title('Iris — PCA Projection (true labels)')
ax.legend(); ax.grid(True)
save('plot_pca_true_labels.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — K-Means (k=3) on PCA space
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2: K-Means (k=3)')
km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
km.fit(X)
labels_km     = km.labels_
centers_pca   = pca.transform(km.cluster_centers_)

fig, ax = plt.subplots(figsize=(7, 5))
for i in range(3):
    mask = labels_km == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=CLUSTER_COLORS[i], s=MARKER_SIZE, alpha=0.85,
               edgecolors='none', label=f'Cluster {i+1}')
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='white', s=180, marker='*', zorder=5,
           edgecolors='#00ff88', linewidths=1, label='Centroids')
ax.set_xlabel(pc1_lbl); ax.set_ylabel(pc2_lbl)
ax.set_title('Iris — K-Means (k=3) on PCA Projection')
ax.legend(); ax.grid(True)
save('plot_kmeans_pca.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Petal-only scatter (petal length vs petal width)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3: Petal-only scatter')
petal_len = X[:, 2]
petal_wid = X[:, 3]

fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(names):
    mask = y == i
    ax.scatter(petal_len[mask], petal_wid[mask],
               c=SPECIES_COLORS[i], s=MARKER_SIZE, alpha=0.85,
               edgecolors='none', label=name)
ax.set_xlabel('Petal Length (cm)'); ax.set_ylabel('Petal Width (cm)')
ax.set_title('Iris — Petal Length vs Petal Width (true labels)')
ax.legend(); ax.grid(True)

# Add a rough separation line for setosa
ax.axvline(x=2.5, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4)
ax.text(2.55, 0.1, 'setosa\nseparation', color='#a0aec0', fontsize=7)
save('plot_petal_scatter.png')

# Answer the petal question
print()
print('  Petal measurements: setosa is cleanly separated (petal length < ~2.5 cm).')
print('  Versicolor and virginica overlap somewhat but are largely distinguishable')
print('  by petal size — better separation than with sepal measurements alone.')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Silhouette scores for k=2–10
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 4: Silhouette scores k=2–10')
k_range  = range(2, 11)
sil_scores = []
for k in k_range:
    km_k = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_k = km_k.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels_k))

best_k   = list(k_range)[np.argmax(sil_scores)]
best_sil = max(sil_scores)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(k_range), sil_scores, color='#4a9eff', linewidth=2, marker='o',
        markerfacecolor='#4a9eff', markersize=6)
ax.axvline(x=best_k, color='#00ff88', linewidth=1.2, linestyle='--', alpha=0.7)
ax.annotate(f'best k={best_k}\n({best_sil:.3f})',
            xy=(best_k, best_sil),
            xytext=(best_k + 0.4, best_sil - 0.02),
            color='#00ff88', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1))
ax.set_xlabel('k (number of clusters)'); ax.set_ylabel('Mean silhouette score')
ax.set_title('Silhouette Score vs k — Iris Dataset')
ax.set_xticks(list(k_range)); ax.grid(True)
save('plot_silhouette_scores.png')

print(f'  Best k = {best_k}, silhouette score = {best_sil:.3f}')
print('  Note: best k is likely 2, not 3 — setosa is so distinct that the')
print('  versicolor/virginica overlap actually hurts the score when split into 3.')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Dendrogram (Ward's linkage)
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 5: Dendrogram (Ward\'s linkage)')
Z = linkage(X, method='ward')

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           color_threshold=0.7 * max(Z[:, 2]),
           above_threshold_color='#a0aec0',
           leaf_rotation=90, leaf_font_size=8)
ax.set_xlabel("Sample index or (cluster size)")
ax.set_ylabel("Ward's linkage distance")
ax.set_title("Hierarchical Clustering Dendrogram — Iris (Ward's linkage, last 30 merges)")
save('plot_dendrogram.png')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — GMM cluster assignments vs true labels
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 6: GMM (3 components)')
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels_gmm = gmm.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Iris PCA — GMM assignments vs. true labels', y=1.01)

# Left: GMM
for i in range(3):
    mask = labels_gmm == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=CLUSTER_COLORS[i], s=MARKER_SIZE, alpha=0.85,
                    edgecolors='none', label=f'Component {i+1}')
axes[0].set_xlabel(pc1_lbl); axes[0].set_ylabel(pc2_lbl)
axes[0].set_title('GMM assignments'); axes[0].legend(); axes[0].grid(True)

# Right: true labels
for i, name in enumerate(names):
    mask = y == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=SPECIES_COLORS[i], s=MARKER_SIZE, alpha=0.85,
                    edgecolors='none', label=name)
axes[1].set_xlabel(pc1_lbl); axes[1].set_ylabel(pc2_lbl)
axes[1].set_title('True labels'); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
save('plot_gmm_clusters.png')

# Compare GMM vs K-Means disagreement (after aligning labels by majority vote)
from scipy.stats import mode as scipy_mode

def align_labels(pred, true, n=3):
    """Permute predicted cluster ids to best match true labels."""
    from itertools import permutations
    best_perm, best_acc = None, -1
    for perm in permutations(range(n)):
        mapped = np.array([perm[l] for l in pred])
        acc = np.mean(mapped == true)
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return np.array([best_perm[l] for l in pred]), best_acc

labels_km_aligned,  acc_km  = align_labels(labels_km,  y)
labels_gmm_aligned, acc_gmm = align_labels(labels_gmm, y)
diff = np.sum(labels_km_aligned != labels_gmm_aligned)

print(f'  K-Means accuracy (vs true labels): {acc_km:.1%}')
print(f'  GMM accuracy     (vs true labels): {acc_gmm:.1%}')
print(f'  Points where GMM and K-Means disagree: {diff}')
print()
print('  GMM can model elliptical clusters (full covariance) rather than')
print('  spherical ones, which better captures the versicolor/virginica overlap.')

print('\nDone. All plots saved.')
