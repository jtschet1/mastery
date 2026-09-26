# Chapter 8 — Unsupervised Learning Techniques

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Walk through the k-means algorithm. Why is it guaranteed to converge, and what can still go wrong?

Start by placing k centroids, for example on k instances picked at random. Then alternate two steps until the centroids stop moving:
1. Assign each instance to the cluster whose centroid is closest.
2. Move each centroid to the mean of the instances assigned to it.

Neither step can increase the inertia, the sum of squared distances from each instance to its closest centroid: inertia = ∑ᵢ ‖x⁽ⁱ⁾ − c⁽ⁱ⁾‖². Since it can't go below zero, the algorithm converges in a finite (usually small) number of iterations.

What can go wrong: it may converge to a local optimum that depends on the initial centroids. That's why KMeans runs the whole algorithm n_init times and keeps the solution with the lowest inertia. The fitted model's inertia is in inertia_, and score() returns −inertia to follow Scikit-Learn's greater-is-better convention.

## 2. What's the difference between KMeans.predict() and KMeans.transform(), and what is transform() useful for?

predict() does hard clustering: it returns the index of each instance's closest centroid. After fitting, the training instances' cluster indices are in labels_ (fit_predict() returns that same array) and the centroids are in cluster_centers_.

transform() is a form of soft clustering: it returns an m × k matrix of the distances from each instance to every centroid. It's useful for:
- Nonlinear dimensionality reduction: each instance's n features become k distances.
- Feature engineering: the distances, or similarities derived from them (e.g., with a Gaussian RBF), make good extra features for another model.
- Choosing which instances to label by hand: the instance closest to each centroid is a good representative of its cluster.

Finding those representatives takes one argmin per column of the distance matrix:

```python
X_dist = kmeans.fit_transform(X_train)       # shape [m, k]
representative_idx = X_dist.argmin(axis=0)   # closest instance per cluster
X_representative = X_train[representative_idx]
```

## 3. How does k-means++ initialization work, and how does Scikit-Learn's KMeans initialize centroids by default?

k-means++ spreads the initial centroids out:
1. Pick the first centroid uniformly at random among the instances.
2. Pick the next centroid among the instances, choosing x⁽ⁱ⁾ with probability D(x⁽ⁱ⁾)² / ∑ⱼ D(x⁽ʲ⁾)², where D(x) is the distance from x to the nearest centroid already chosen. Instances far from the existing centroids are much more likely to be picked.
3. Repeat step 2 until there are k centroids, then run regular k-means.

Well-separated starting centroids make it far less likely to miss a cluster or get stuck in a poor local optimum, so many fewer restarts are needed. KMeans uses init="k-means++" by default, in a greedy variant that samples several candidates at each step and keeps the best one; n_init then defaults to 1 (versus 10 with init="random"). If you know roughly where the centroids should be, you can also pass them to init as a NumPy array.

## 4. What are k-means' main limitations, and what should you do to the features before running it?

- You must choose k, and a wrong k merges distinct clusters or chops real ones into pieces.
- You need several runs (or a smart initialization) to avoid poor local optima.
- It assigns instances by their distance to the centroids alone, implicitly assuming roughly round clusters of similar size and density. It does poorly on clusters with very different sizes or densities, or nonspherical shapes such as elongated ellipsoids; there, even a lower inertia doesn't mean a better clustering.
- It favors similar-sized clusters: when segmenting an image by color, a small but vividly colored object may not get its own cluster unless k is large.

Scale the features first, or clusters get stretched along the features with the largest ranges; it doesn't guarantee round clusters, but it generally helps. For ellipsoidal clusters, a Gaussian mixture works better; for arbitrary shapes, try DBSCAN.

## 5. Walk through how DBSCAN forms clusters and flags anomalies. When does it work well, and when does it struggle?

For each instance, DBSCAN counts how many instances lie within a distance ε (the eps hyperparameter), its ε-neighborhood. An instance with at least min_samples instances in its neighborhood, itself included, is a core instance: it sits in a dense region. Every instance in a core instance's neighborhood joins that core instance's cluster, and since neighborhoods can contain other core instances, a chain of neighboring core instances forms a single cluster. Any instance that isn't a core instance and has no core instance in its neighborhood is an anomaly, labeled −1.

It finds any number of clusters of any shape, is robust to outliers, and has just two hyperparameters, though results are sensitive to eps. It struggles when density varies a lot between clusters or when clusters aren't separated by low-density regions, and at roughly O(m²n) it doesn't scale to large datasets. HDBSCAN (sklearn.cluster.HDBSCAN) often copes better with varying densities.

## 6. Scikit-Learn's DBSCAN has fit_predict() but no predict(). What does a fitted DBSCAN expose, and how can you assign new instances to clusters?

After fitting, labels_ holds each instance's cluster index (−1 for anomalies), core_sample_indices_ the indices of the core instances, and components_ the core instances themselves. There's no predict() because the best way to classify new instances depends on the task, so you train a classifier of your choice on the clustering, for example a k-nearest neighbors classifier on the core instances:

```python
core_labels = dbscan.labels_[dbscan.core_sample_indices_]
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, core_labels)
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = core_labels[y_pred_idx]
y_pred[y_dist > 0.2] = -1   # too far from every core instance: anomaly
```

knn.predict() (or predict_proba()) would always pick some cluster, even for an instance far from all of them. Checking the distance to the nearest neighbor with kneighbors() lets you flag such instances as anomalies instead.

## 7. How does the expectation-maximization (EM) algorithm fit a Gaussian mixture model, and how does it compare with k-means?

EM starts from initial (e.g., random) cluster parameters, then alternates two steps until convergence:
- Expectation step: using the current parameters, estimate for each instance the probability that it belongs to each cluster. These probabilities are called the clusters' responsibilities for the instances.
- Maximization step: update each cluster's weight ϕ⁽ʲ⁾, mean μ⁽ʲ⁾ and covariance matrix Σ⁽ʲ⁾ using all the instances, each weighted by the cluster's responsibility for it.

You can see it as a generalization of k-means that uses soft assignments instead of hard ones, and that learns each cluster's size, shape, and orientation (Σ) and relative weight (ϕ), not just its center (μ). Like k-means, it can converge to a poor solution, so run it several times by setting n_init, which defaults to 1 in GaussianMixture. The converged_ and n_iter_ attributes tell you whether it converged and how many iterations it took.

## 8. Besides hard clustering with predict(), what can you do with a fitted GaussianMixture, and how would you use it to detect anomalies?

- predict_proba(X) gives soft clustering: the probability that each instance belongs to each cluster.
- sample(n) generates n new instances plus their cluster indices (sorted by cluster index), since a GMM is a generative model.
- score_samples(X) returns the log of the probability density at each instance. Exponentiating gives the density itself, which can exceed 1: it's a density, not a probability.

For anomaly detection, flag instances in low-density regions. For example, if about 2% of products are known to be defective, use the 2nd percentile of the densities as the threshold:

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]
```

Lower the threshold if too many normal instances get flagged (false positives), raise it if too many anomalies slip through (false negatives). If there are many outliers, they skew the model's idea of normal, so fit, remove the most extreme outliers, and fit again.

## 9. What does GaussianMixture's covariance_type hyperparameter control, and why might you change it from the default?

It constrains the clusters' covariance matrices, and therefore the shapes and orientations the clusters can take:
- "full" (the default): each cluster has its own unconstrained covariance matrix, so any ellipsoidal shape, size, and orientation.
- "tied": all clusters share the same covariance matrix, so the same shape, size, and orientation.
- "diag": any ellipsoidal shape and size, but with axes parallel to the coordinate axes (diagonal covariance matrices).
- "spherical": round clusters, which can still have different diameters (variances).

Constraining the covariances leaves fewer parameters to learn, which helps EM converge to a good solution when there are many dimensions, many clusters, or few instances. It's also much cheaper: training is roughly O(kmn) with "spherical" or "diag", but O(kmn² + kn³) with "tied" or "full", which doesn't scale to large numbers of features.

## 10. What's the difference between probability and likelihood, and how do BIC and AIC use the maximized likelihood?

For a model f(x; θ), probability describes how plausible an outcome x is when the parameters θ are known; likelihood, ℒ(θ|x) = f(x; θ), describes how plausible parameter values θ are once x has been observed. A PDF integrates to 1 over x, but a likelihood needn't integrate to 1 over θ. Maximum likelihood estimation finds the θ that maximizes it, usually via the log-likelihood: same maximum, and a product over independent instances becomes a sum.

The maximized likelihood L̂ measures how well the model fits, and both criteria trade it off against the number of learned parameters p (m is the number of instances):
- BIC = log(m)·p − 2·log(L̂)
- AIC = 2p − 2·log(L̂)

Lower is better. BIC's penalty grows with m, so it tends to pick simpler models than AIC. A fitted GaussianMixture provides bic(X) and aic(X).

## 11. Besides Gaussian mixtures, which anomaly and novelty detection algorithms does Scikit-Learn offer, and how does each one work?

- Fast-MCD (EllipticEnvelope): assumes the inliers come from a single Gaussian distribution and estimates its elliptic envelope while ignoring the instances most likely to be outliers. Handy for cleaning up a dataset.
- Isolation forest (IsolationForest): grows random trees that split on a random feature at a random threshold until every instance is isolated. Anomalies are usually far from the rest, so on average they get isolated in fewer splits. Efficient, even in high dimensions.
- Local outlier factor (LocalOutlierFactor): compares the density around an instance with the density around its k nearest neighbors; an anomaly is more isolated than its neighbors are.
- One-class SVM (OneClassSVM): in the kernel's high-dimensional space, separates the instances from the origin, which amounts to a small region enclosing them. Best for novelty detection; doesn't scale to large datasets.
- PCA or any transformer with inverse_transform(): anomalies have a much larger reconstruction error.
