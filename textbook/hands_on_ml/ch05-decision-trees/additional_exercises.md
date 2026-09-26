# Chapter 5 — Decision Trees

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Sketch how to train a DecisionTreeClassifier on two iris features and export the fitted tree for Graphviz.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris(as_frame=True)
features = ["petal length (cm)", "petal width (cm)"]
X_iris, y_iris = iris.data[features].values, iris.target
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=features,
                class_names=iris.target_names, rounded=True, filled=True)
```

In a Jupyter notebook, graphviz.Source.from_file("iris_tree.dot") displays the tree, and Graphviz's dot command-line tool converts .dot files to formats such as PNG or PDF. The fitted structure is also available programmatically through tree_clf.tree_. Both DecisionTreeClassifier and DecisionTreeRegressor handle missing values natively, so no imputer is needed.

## 2. How does a trained decision tree classify a new instance, and what do the samples, value and gini fields shown in each node mean?

Start at the root node and answer its yes/no question about one feature (e.g., is petal length ≤ 2.45 cm?). Go to the left child if the answer is yes and the right child if it's no, and repeat until you reach a leaf, which predicts the most common class among its training instances. In the rendered tree:
- samples: how many training instances reach that node.
- value: how many of those training instances belong to each class.
- gini: the node's Gini impurity, which is 0 when all of them belong to the same class (a pure node).

Scikit-Learn's CART algorithm only builds binary trees, where every split node has exactly two children; other algorithms, such as ID3, can create nodes with more.

## 3. Define the Gini impurity of a node, and compute it for a node whose 54 training instances split 0, 49 and 5 across three classes.

G = 1 − ∑ₖ pₖ², where pₖ is the fraction of the node's training instances that belong to class k. It's 0 for a pure node and grows as more classes are present in more even proportions (the maximum with K classes is 1 − 1/K). One way to read it: the probability that two instances drawn at random (with replacement) from the node belong to different classes.

For this node: G = 1 − (0/54)² − (49/54)² − (5/54)² ≈ 1 − 0.823 − 0.009 ≈ 0.168.

## 4. What distinguishes a white box model from a black box model, and why does the distinction matter?

A white box model's decisions are easy to interpret: a decision tree boils down to simple if/then rules that you can follow, or even apply by hand, to see exactly why an instance got its prediction. Black box models such as random forests and neural networks often predict better, and you can check every calculation they perform, yet it's hard to explain in simple terms why they made a given prediction (which part of a photo made a network recognize someone?). Interpretability matters whenever people must review or justify decisions: a doctor checking a diagnosis, analysts assessing financial risk, a judge making the final call, or HR making sure decisions aren't biased. Interpretable ML is the field that aims to build systems able to explain their decisions to humans.

## 5. How does a decision tree estimate class probabilities, and what's a weakness of those estimates?

It routes the instance down to its leaf and returns the fraction of that leaf's training instances that belong to each class. For a leaf holding 0, 49 and 5 training instances of three classes, predict_proba() returns about [0, 0.907, 0.093], and predict() returns the most probable class. The weakness is that the estimate is constant over the leaf's entire region of feature space: an instance near the far edge of that region, even one that plainly resembles another class, gets exactly the same probabilities as one in the middle. The probabilities are coarse, piecewise-constant values set by the leaf counts rather than a smooth function of the inputs.

## 6. Walk through how the CART algorithm grows a classification tree, including the cost function it minimizes at each split.

1. At the current node, consider each feature k and candidate threshold tₖ (e.g., petal length ≤ 2.45 cm).
2. Pick the pair that minimizes the size-weighted impurity of the two resulting subsets: J(k, tₖ) = (m_left/m)·G_left + (m_right/m)·G_right, where G_left and G_right are the subsets' impurities and m_left and m_right their numbers of instances.
3. Split the node's instances accordingly, then apply the same procedure recursively to each subset.
4. Stop at max_depth, when no split reduces impurity, or when another limit such as min_samples_split, min_samples_leaf or max_leaf_nodes applies.

CART is greedy: each split is the best one available at that moment, with no check on whether another split would lead to purer nodes a few levels down. Finding the optimal tree is NP-complete (O(exp(m)) time), intractable even for small training sets, so we settle for a reasonably good greedy tree.

## 7. What's the computational cost of making a prediction with a trained decision tree, and why doesn't it depend on the number of features?

A prediction just follows one path from the root to a leaf. Trees end up roughly balanced, so that path goes through about log₂(m) nodes, where m is the number of training instances: O(log₂(m)). Each node on the path checks the value of a single feature against a threshold, so the cost doesn't grow with the total number of features n. That's why trees predict very quickly even when trained on large datasets; the expensive part is training, which by default compares all features on all samples at every node.

## 8. How is entropy used as a node's impurity measure, and how do you choose between it and Gini impurity?

A node's entropy is H = −∑ₖ pₖ·log₂(pₖ), summed over the classes present in the node (pₖ ≠ 0), where pₖ is the fraction of the node's instances in class k. Like Gini impurity, it's 0 for a pure node and largest when the classes are evenly mixed. For a node holding 0, 49 and 5 instances of three classes: H = −(49/54)·log₂(49/54) − (5/54)·log₂(5/54) ≈ 0.445. Select it with DecisionTreeClassifier(criterion="entropy"); Gini is the default. Most of the time the choice barely matters, as both lead to similar trees. Gini is slightly faster to compute, which makes it a good default; when they do differ, Gini tends to isolate the most frequent class in its own branch, while entropy tends to produce slightly more balanced trees.

## 9. What makes a decision tree a nonparametric model, and why does that make it prone to overfitting?

Nonparametric doesn't mean it has no parameters (a tree often has many); it means their number isn't fixed before training, so the model's structure is free to adapt to the training data as closely as it likes. A parametric model such as linear regression has a predetermined number of parameters, which limits its degrees of freedom: less risk of overfitting, more risk of underfitting. Decision trees also make very few assumptions about the data, and left unconstrained, CART keeps splitting until it can no longer reduce impurity, usually leaving pure leaves. So an unrestricted tree will most likely overfit unless you limit its freedom during training (regularization).

## 10. Name DecisionTreeClassifier's main regularization hyperparameters, and state the general rule for using them.

- max_depth: maximum depth of the tree (default None, meaning unlimited).
- max_leaf_nodes: maximum number of leaf nodes.
- max_features: maximum number of features evaluated for splitting at each node.
- min_samples_split: minimum number of samples a node must have before it can be split.
- min_samples_leaf: minimum number of samples a leaf node must have.
- min_weight_fraction_leaf: like min_samples_leaf, but as a fraction of the total number of weighted instances.
- min_impurity_decrease: only split a node if the split reduces impurity by at least this amount.
- ccp_alpha: strength of minimal cost-complexity pruning (default 0, no pruning).

To regularize, increase the min-prefixed hyperparameters or ccp_alpha, or decrease the max-prefixed ones. max_depth is a good default, since it also keeps the tree small and readable; min_samples_leaf helps especially on small datasets, and max_features on high-dimensional ones.

## 11. What does it mean to prune a decision tree, and how does pruning differ from restricting the tree's growth?

Hyperparameters like max_depth or min_samples_leaf stop the tree from growing in the first place. Pruning instead trains the tree without restrictions, then deletes unnecessary nodes. In the classic statistical approach, a node whose children are all leaves is checked with a test such as the χ² test, which estimates the probability (the p-value) that its purity improvement is purely due to chance. If that p-value exceeds a threshold, typically 5%, the node is considered unnecessary and its children are deleted; this repeats until no unnecessary nodes remain. Scikit-Learn's pruning option is minimal cost-complexity pruning, set with ccp_alpha: it prunes subtrees whose impurity reduction isn't worth their number of leaves, and a larger ccp_alpha prunes more, giving a smaller tree.

## 12. How does a regression tree make predictions, and what does CART minimize when growing one?

It's traversed exactly like a classification tree, but each leaf predicts a value: the average target of the training instances that reach that leaf. The model is therefore piecewise constant, with one flat prediction per region. To grow it, CART chooses the feature k and threshold tₖ that minimize the size-weighted MSE of the two subsets, J(k, tₖ) = (m_left/m)·MSE_left + (m_right/m)·MSE_right, where each subset's MSE is measured around that subset's own mean target. So each split makes the training instances in each region as close as possible to the region's prediction. Regression trees overfit just as easily as classification trees: with default hyperparameters they chase every noisy point, while simply setting min_samples_leaf=10 gives a much more reasonable model.

## 13. Why are decision trees sensitive to the orientation of the data, and how can you reduce the problem?

Every split compares a single feature with a threshold, so all decision boundaries are perpendicular to an axis. A class boundary that runs diagonally to the axes has to be approximated by a staircase of splits: a linearly separable dataset rotated by 45° gets a needlessly convoluted boundary that probably won't generalize, even if it fits the training set perfectly. One remedy is to scale the data and then rotate it with PCA, which reduces the correlation between features and often (not always) lets the tree separate the classes with fewer splits:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)
```

## 14. Why do decision trees have high variance, and what's the usual remedy?

Small changes to the training data or to the hyperparameters can produce a very different tree: a slightly different split near the root changes everything grown beneath it. Scikit-Learn's training algorithm is also stochastic: at each node it goes through the features in a random order (or evaluates a random subset of them, if max_features is set), which can decide between equally good splits. So even retraining the same tree on exactly the same data can give a very different model unless you set random_state. The remedy is to average the predictions of many trees, which reduces variance significantly; such an ensemble of decision trees is called a random forest.
