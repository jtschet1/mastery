# Chapter 7 — Dimensionality Reduction

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. What are the two main approaches to dimensionality reduction, and when does simple projection break down?

Projection relies on the instances lying in or near a lower-dimensional linear subspace, e.g., a 3D dataset that sits close to a plane. That's common in practice, because many features are nearly constant and others are strongly correlated. Projecting every instance perpendicularly onto the subspace gives its new, lower-dimensional coordinates. Manifold learning instead models the curved shape the data lies on: a d-dimensional manifold is a part of an n-dimensional space (d < n) that locally resembles a d-dimensional hyperplane but can bend and twist globally. Projection breaks down when the subspace twists, as in the Swiss roll: flattening it onto a plane squashes different layers on top of one another, whereas manifold learning algorithms such as LLE, Isomap, t-SNE or UMAP aim to unroll it.

## 2. What is the manifold hypothesis, and what extra assumption often comes with it that doesn't always hold?

The manifold hypothesis says that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. MNIST illustrates it: handwritten digits are made of connected strokes, have white borders and are roughly centered, so they occupy a tiny, constrained corner of the space of all possible images. The companion assumption is that the task (classification or regression) will be simpler in the manifold's low-dimensional coordinates. Sometimes it is: a class boundary that's convoluted on the rolled-up Swiss roll can become a straight line once it's unrolled. But sometimes it's the reverse: a boundary like the plane x₁ = 5 is simple in the original 3D space yet turns into several disconnected segments on the unrolled manifold. So reducing dimensionality usually speeds up training, but it doesn't guarantee a better or simpler solution.

## 3. How does PCA choose the axes it projects onto, and why is preserving variance a sensible criterion?

PCA first finds the axis along which the training set has the most variance: the first principal component (PC). The second PC is the axis orthogonal to the first that accounts for the most remaining variance, the third is orthogonal to both, and so on, with as many PCs as the dataset has dimensions. Projecting onto the first d PCs therefore keeps as much variance as any d-dimensional hyperplane can, which should lose the least information. Equivalently, the maximum-variance axis is the one that minimizes the mean squared distance between the original points and their projections onto it, so PCA finds the hyperplane that lies closest to the data.

## 4. Sketch PCA from scratch with NumPy: find the principal components, project the data down to d dimensions, and map it back.

Center the data, compute its singular value decomposition X = UΣVᵀ, and read the principal components from the columns of V, which are the rows of Vᵀ, already ordered by explained variance:

```python
X_centered = X - X.mean(axis=0)       # PCA assumes centered data
U, s, Vt = np.linalg.svd(X_centered)
W2 = Vt[:2].T                         # W_d: the first d = 2 PCs as columns
X2D = X_centered @ W2                 # project down to 2D
X_recovered = X2D @ W2.T + X.mean(axis=0)
```

The projection is X_d-proj = X·W_d, with X centered and W_d holding the first d columns of V, and the inverse transformation is X_recovered = X_d-proj·W_dᵀ (plus the mean you subtracted). The mean squared distance between the original data and X_recovered is the reconstruction error. Scikit-Learn's PCA class centers the data for you, and its components_ attribute holds W_dᵀ, one row per principal component.

## 5. Why must you retrain a downstream model whenever you refit a PCA transformer, even on nearly the same data?

The direction of each principal component's unit vector isn't guaranteed: perturb the training set slightly and a component can come back pointing the opposite way, and two components with nearly equal variance can even rotate or swap. The new projection may then produce features with flipped signs, mixed together or in a different order, and a model trained on the old PCA output would misread them. So treat PCA and the model that consumes its output as one unit and always retrain them together, for example as steps of a single pipeline.

## 6. What are the main ways to choose the number of dimensions d when using PCA in Scikit-Learn?

- Keep enough variance, say 95%: fit PCA() with all components and take the smallest d whose cumulative explained variance reaches the target, d = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1.
- More simply, pass the ratio itself: PCA(n_components=0.95) picks d during fit() and stores it in n_components_.
- Plot the cumulative explained variance against d and look for the elbow, where it stops growing quickly.
- When PCA is preprocessing for a supervised model, treat d as a hyperparameter: put PCA in a pipeline and search over pca__n_components (e.g., with RandomizedSearchCV), keeping whatever makes the final model perform best.
- For visualization, simply use 2 or 3.

The explained_variance_ratio_ attribute gives the proportion of the dataset's variance that lies along each principal component.

## 7. What is randomized PCA, when is it much faster, and how does Scikit-Learn's PCA pick its solver by default?

PCA(svd_solver="randomized") uses a stochastic algorithm that quickly approximates just the first d principal components. It costs O(m × d²) + O(d³) instead of O(m × n²) + O(n³) for a full SVD, so it's dramatically faster when d is much smaller than n. The default, svd_solver="auto", decides as follows:
- if the data has few features (n < 1,000) and at least 10 times more samples (m > 10n), it uses "covariance_eigh", which is very fast in that regime;
- otherwise, if max(m, n) > 500 and n_components is an integer below 80% of min(m, n), it uses "randomized";
- otherwise it computes a full SVD.

Set svd_solver="full" to force a full SVD, trading compute time for a slightly more precise result.

## 8. How do you run PCA with Scikit-Learn on a training set that doesn't fit in memory?

Use IncrementalPCA and feed it one mini-batch at a time with partial_fit() instead of calling fit() on the whole set:

```python
from sklearn.decomposition import IncrementalPCA

inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, 100):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)
```

Alternatively, store the data in a NumPy memmap, a binary file on disk that behaves like an in-memory array and loads only the parts that are needed, and call the usual fit() with a batch_size, e.g., IncrementalPCA(n_components=154, batch_size=batch_size).fit(X_mmap). Only the raw bytes are saved, so when reopening a memmap you must give its dtype and shape (without a shape, np.memmap() returns a 1D array).

## 9. Why does random projection work, and how does the Johnson–Lindenstrauss result pick the target number of dimensions?

A random linear projection is very likely to preserve pairwise distances fairly well, so similar instances stay similar and very different ones stay very different. The Johnson–Lindenstrauss lemma gives the minimum d that ensures, with high probability, that no squared distance changes by more than a tolerance ε: d ≥ 4·log(m) / (ε²/2 − ε³/3). It depends only on the number of instances m and on ε, not on the number of features n; e.g., m = 5,000 and ε = 0.1 give d = 7,300.

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim

d = johnson_lindenstrauss_min_dim(m, eps=0.1)
P = rng.standard_normal((d, n)) / np.sqrt(d)   # Gaussian, mean 0, variance 1/d
X_reduced = X @ P.T
```

"Training" only needs the data's shape, not the data, so it's almost instantaneous, and transforming is a single matrix multiplication. The price is that it loses a bit more signal than PCA.

## 10. Compare Scikit-Learn's GaussianRandomProjection and SparseRandomProjection.

Both work the same way: fit() uses johnson_lindenstrauss_min_dim() to pick the target dimensionality (tune it with eps, default 0.1, or force it with n_components), generates a random matrix stored in components_, and transform() multiplies by it, e.g., GaussianRandomProjection(eps=0.1, random_state=42).fit_transform(X). The difference is the matrix: SparseRandomProjection's is mostly zeros. Its density r, the fraction of nonzero entries, defaults to 1/√n, and each nonzero entry is +v or −v with equal probability, where v = 1/√(d·r). So it uses far less memory, is faster both to generate the matrix and to transform, keeps sparse inputs sparse (unless dense_output=True), and preserves distances nearly as well. It's usually the better choice, especially for large or sparse datasets. Neither offers a cheap inverse: you'd multiply the reduced data by the transpose of components_'s pseudoinverse, which is slow to compute for large matrices.

## 11. Walk through how locally linear embedding (LLE) reduces dimensionality.

1. For each training instance x⁽ⁱ⁾, find its k nearest neighbors.
2. Find the weights wᵢⱼ that best reconstruct x⁽ⁱ⁾ as a linear combination of its neighbors, minimizing the squared distance between x⁽ⁱ⁾ and ∑ⱼ wᵢⱼ·x⁽ʲ⁾, with wᵢⱼ = 0 for non-neighbors and each instance's weights summing to 1. These weights capture the local geometry.
3. Keep those weights fixed and find the low-dimensional points z⁽ⁱ⁾ that minimize the same kind of error, the squared distance between z⁽ⁱ⁾ and ∑ⱼ ŵᵢⱼ·z⁽ʲ⁾, summed over all instances.

Because it only preserves local relationships, LLE is good at unrolling twisted manifolds when there isn't too much noise, but it doesn't preserve distances at larger scales. The last step costs O(d·m²), so it scales poorly to large datasets. In Scikit-Learn: LocallyLinearEmbedding(n_components=2, n_neighbors=10).fit_transform(X).

## 12. Briefly contrast MDS, Isomap, t-SNE and LDA as dimensionality reduction techniques.

- MDS (sklearn.manifold.MDS): tries to preserve the distances between instances; unlike random projection, it also works well on low-dimensional data.
- Isomap (sklearn.manifold.Isomap): connects each instance to its nearest neighbors to build a graph, then tries to preserve the geodesic (shortest-path) distances through that graph. It works best on a smooth, low-dimensional manifold with a single global structure, like the Swiss roll.
- t-SNE (sklearn.manifold.TSNE): keeps similar instances close and dissimilar ones apart. It's mostly used to visualize clusters in 2D or 3D, not as preprocessing for a model.
- LDA (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): a linear classifier that learns the most discriminative axes between the classes; projecting onto them keeps the classes as far apart as possible, a good step before another classifier.

UMAP (in the umap-learn package) is a popular alternative to t-SNE that preserves more of the global structure and scales better.
