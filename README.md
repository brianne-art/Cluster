# Clustering Assignment — Implementation Specs

## Overview

Implement a series of clustering analyses on the Iris dataset using Python. The assignment covers K-Means, silhouette analysis, hierarchical clustering, and Gaussian Mixture Models.

---

## Environment & Dependencies

```
scikit-learn
matplotlib
scipy
numpy
```

Run with Python 3.x. All plots saved as `.png` files.

---

## Data

**Source:** `sklearn.datasets.load_iris()`

- 150 samples, 3 classes (50 each): *setosa*, *versicolor*, *virginica*
- 4 features: sepal length, sepal width, petal length, petal width
- Labels available in `iris.target` and `iris.target_names`

---

## Part 1 — K-Means Clustering

### 1a. PCA Scatter Plot (pre-clustering)

- Run PCA on the full 4-feature dataset, reduce to 2 components
- Scatter plot PC1 vs PC2, colored by **true class label**
- Use a legend with species names
- Expected result: *setosa* clearly separates; *versicolor* and *virginica* overlap

**Save as:** `plot_pca_true_labels.png`

### 1b. K-Means on PCA Space

- Fit `KMeans(n_clusters=3)` on the full 4-feature data
- Plot the same PCA projection, now colored by **cluster assignment**
- Add centroids to the plot (transformed into PCA space via `pca.transform(kmeans.cluster_centers_)`)
- Compare visually to Part 1a

**Save as:** `plot_kmeans_pca.png`

### 1c. Petal-Only Scatter Plot

- Scatter plot of **petal length (x)** vs **petal width (y)**, colored by true class label
- Answer in a comment or print statement: Can you separate one class from the other two using only petal measurements?

**Save as:** `plot_petal_scatter.png`

---

## Part 2 — Silhouette Analysis

### Silhouette Score Overview

For each point *i*, compute:
- **a(i):** mean distance to other points in the same cluster
- **b(i):** mean distance to points in the nearest other cluster
- **s(i) = (b(i) - a(i)) / max(a(i), b(i))**

Score ranges from -1 (wrong cluster) to +1 (well-clustered). Mean score across all points = overall silhouette score.

### Implementation

- Loop `k` from 2 to 10
- For each `k`: fit `KMeans(n_clusters=k)`, compute mean silhouette score using `sklearn.metrics.silhouette_score`
- Plot k (x-axis) vs mean silhouette score (y-axis)
- Mark the best k with a vertical line or annotation

**Save as:** `plot_silhouette_scores.png`

**Print:** The best k and its score. Note in a comment what this implies — the iris data may have 2 "natural" clusters despite having 3 known species, because *versicolor* and *virginica* overlap significantly.

---

## Part 3 — Hierarchical Clustering (Dendrogram)

- Use `scipy.cluster.hierarchy.linkage` with `method='ward'`
- Use `scipy.cluster.hierarchy.dendrogram` to plot
- Truncate the dendrogram to the last 30 merges for readability (`truncate_mode='lastp'`, `p=30`)
- Label axes: "Sample index or (cluster size)" (x), "Ward's linkage distance" (y)
- Add a title

**Save as:** `plot_dendrogram.png`

---

## Part 4 — Gaussian Mixture Model

- Fit `GaussianMixture(n_components=3, covariance_type='full')` on the full 4-feature data
- Predict cluster labels with `.predict()`
- Plot in PCA space (same projection as Part 1), colored by GMM cluster assignment
- Add a second subplot or separate plot showing true labels for direct comparison

**Save as:** `plot_gmm_clusters.png`

**Print or comment:** How do GMM assignments compare to true labels vs K-Means? Note that GMM can model elliptical clusters rather than spherical ones, which may better capture the overlap between *versicolor* and *virginica*.

---

## Output Files Summary

| File | Description |
|---|---|
| `clustering.py` | All code in one script |
| `plot_pca_true_labels.png` | PCA scatter, true labels |
| `plot_kmeans_pca.png` | PCA scatter, K-Means clusters |
| `plot_petal_scatter.png` | Petal-only scatter, true labels |
| `plot_silhouette_scores.png` | Silhouette scores for k=2–10 |
| `plot_dendrogram.png` | Hierarchical clustering dendrogram |
| `plot_gmm_clusters.png` | GMM cluster assignments in PCA space |

---

## Suggested Code Structure

```python
# 1. Imports
# 2. Load data
# 3. PCA setup (fit once, reuse across plots)
# 4. Part 1 — KMeans + scatter plots
# 5. Part 2 — Silhouette loop
# 6. Part 3 — Dendrogram
# 7. Part 4 — GMM
```

Use `plt.savefig('filename.png', dpi=150, bbox_inches='tight')` and `plt.close()` after each plot to avoid figure bleed-through.

---

## Key Questions to Address (in comments or print output)

1. **Part 1c:** Can petal measurements alone separate one iris species from the other two?
2. **Part 2:** What does the silhouette plot suggest as the best k? Why might this differ from the known 3 classes?
3. **Part 4:** Does GMM outperform K-Means on this dataset? What structural property of GMM helps with overlapping clusters?