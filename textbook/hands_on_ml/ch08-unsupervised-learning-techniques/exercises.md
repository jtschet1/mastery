# Chapter 8 — Unsupervised Learning Techniques

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/08_unsupervised_learning.ipynb`, converted to plain text
for the flashcard app.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 10. Cluster the Olivetti faces with k-means (stratified
  train/validation/test split, pick a good number of clusters) and inspect
  the clusters.
- 11. Classify Olivetti faces, then try k-means as a dimensionality
  reduction step (tuning the number of clusters), and try appending the
  cluster features to the originals.
- 12. Fit a Gaussian mixture to Olivetti faces (after PCA to 99% variance),
  generate new faces with sample(), and use score_samples() to spot
  rotated/flipped/darkened images.
- 13. Use PCA reconstruction error to detect those modified faces as
  anomalies.

## 1. How would you define clustering? Can you name a few clustering algorithms?

In Machine Learning, clustering is the unsupervised task of grouping similar instances together. The notion of similarity depends on the task at hand: for example, in some cases two nearby instances will be considered similar, while in others similar instances may be far apart as long as they belong to the same densely packed group. Popular clustering algorithms include K-Means, DBSCAN, agglomerative clustering, BIRCH, Mean-Shift, affinity propagation, and spectral clustering.

## 2. What are some of the main applications of clustering algorithms?

The main applications of clustering algorithms include data analysis, customer segmentation, recommender systems, search engines, image segmentation, semi-supervised learning, dimensionality reduction, anomaly detection, and novelty detection.

## 3. Describe two techniques to select the right number of clusters when using k-means.

The elbow rule is a simple technique to select the number of clusters when using K-Means: just plot the inertia (the mean squared distance from each instance to its nearest centroid) as a function of the number of clusters, and find the point in the curve where the inertia stops dropping fast (the "elbow"). This is generally close to the optimal number of clusters. Another approach is to plot the silhouette score as a function of the number of clusters. There will often be a peak, and the optimal number of clusters is generally nearby. The silhouette score is the mean silhouette coefficient over all instances. This coefficient varies from +1 for instances that are well inside their cluster and far from other clusters, to –1 for instances that are very close to another cluster. You may also plot the silhouette diagrams and perform a more thorough analysis.

## 4. What is label propagation? Why would you implement it, and how?

Labeling a dataset is costly and time-consuming. Therefore, it is common to have plenty of unlabeled instances, but few labeled instances. Label propagation is a technique that consists in copying some (or all) of the labels from the labeled instances to similar unlabeled instances. This can greatly extend the number of labeled instances, and thereby allow a supervised algorithm to reach better performance (this is a form of semi-supervised learning). One approach is to use a clustering algorithm such as K-Means on all the instances, then for each cluster find the most common label or the label of the most representative instance (i.e., the one closest to the centroid) and propagate it to the unlabeled instances in the same cluster.

## 5. Can you name two clustering algorithms that can scale to large datasets? And two that look for regions of high density?

K-Means and BIRCH scale well to large datasets. DBSCAN and Mean-Shift look for regions of high density.

## 6. Can you think of a use case where active learning would be useful? How would you implement it?

Active learning is useful whenever you have plenty of unlabeled instances but labeling is costly. In this case (which is very common), rather than randomly selecting instances to label, it is often preferable to perform active learning, where human experts interact with the learning algorithm, providing labels for specific instances when the algorithm requests them. A common approach is uncertainty sampling (see the Active Learning section in chapter 8).

## 7. What is the difference between anomaly detection and novelty detection?

Many people use the terms anomaly detection and novelty detection interchangeably, but they are not exactly the same. In anomaly detection, the algorithm is trained on a dataset that may contain outliers, and the goal is typically to identify these outliers (within the training set), as well as outliers among new instances. In novelty detection, the algorithm is trained on a dataset that is presumed to be "clean," and the objective is to detect novelties strictly among new instances. Some algorithms work best for anomaly detection (e.g., Isolation Forest), while others are better suited for novelty detection (e.g., one-class SVM).

## 8. What is a Gaussian mixture? What tasks can you use it for?

A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. In other words, the assumption is that the data is grouped into a finite number of clusters, each with an ellipsoidal shape (but the clusters may have different ellipsoidal shapes, sizes, orientations, and densities), and we don't know which cluster each instance belongs to. This model is useful for density estimation, clustering, and anomaly detection.

## 9. Can you name two techniques to find the right number of clusters when using a Gaussian mixture model?

One way to find the right number of clusters when using a Gaussian mixture model is to plot the Bayesian information criterion (BIC) or the Akaike information criterion (AIC) as a function of the number of clusters, then choose the number of clusters that minimizes the BIC or AIC. Another technique is to use a Bayesian Gaussian mixture model, which automatically selects the number of clusters.
