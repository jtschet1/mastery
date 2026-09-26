# Chapter 6 — Ensemble Learning and Random Forests

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Why can an ensemble of weak learners be much more accurate than any of its members, and what condition does that depend on?

Picture a coin that lands heads 51% of the time: over 1,000 tosses, the probability of getting a majority of heads is about 75%, and it keeps climbing with more tosses, thanks to the law of large numbers. Likewise, 1,000 classifiers that are each right only 51% of the time could reach up to about 75% accuracy by majority vote, but only if their errors are independent. In practice they're trained on the same data, so they tend to make the same kinds of mistakes, many votes go to the same wrong class, and the gain shrinks. So an ensemble works best when its members are as diverse as possible: trained with very different algorithms, with different hyperparameters, or on different subsets of the data.

## 2. Sketch how to build a VotingClassifier in Scikit-Learn, inspect its members, and switch it to soft voting.

Pass a list of (name, estimator) pairs and use it like any classifier; fit() clones each estimator and trains the clones:

```python
voting_clf = VotingClassifier(estimators=[
    ("lr", LogisticRegression(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42)),
    ("svc", SVC(random_state=42))])
voting_clf.fit(X_train, y_train)
for name, clf in voting_clf.named_estimators_.items():
    print(name, clf.score(X_test, y_test))
```

estimators (a list) and named_estimators (a dict) hold the original, unfitted models; estimators_ and named_estimators_ hold the fitted clones. predict() uses hard voting by default. To switch to soft voting, set voting="soft", make sure every member has a predict_proba() method (e.g., set probability=True on the SVC), and call fit() again.

## 3. Why does bagging mostly reduce variance, and which kinds of base models benefit most from it?

Bagging trains many copies of one algorithm on random samples of the training set drawn with replacement (pasting samples without replacement), then aggregates their predictions: the most frequent class for classification, the average for regression. Each predictor is a bit more biased than one trained on the full set, but their errors are partly independent, so averaging cancels much of the scatter: the average of two independent predictions with equal variance has half that variance. In practice, the ensemble ends up with about the same bias as a single predictor trained on the full set, but a lower variance. So it helps most with low-bias, high-variance models such as deep decision trees, and little with stable, high-bias ones like linear regression. Bagging's resampling adds diversity, making its predictors less correlated than pasting's; that's why it's usually preferred, especially for noisy data or overfitting-prone models.

## 4. Sketch a bagging ensemble of decision trees with Scikit-Learn, and explain its key hyperparameters.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

- n_estimators: the number of predictors.
- max_samples: how many instances are drawn for each predictor, as a count or as a fraction of the training set (by default, as many as the training set has).
- bootstrap: True (the default) samples with replacement, which is bagging; False gives pasting.
- n_jobs: the number of CPU cores used for training and predictions; -1 means all of them.
- oob_score=True: evaluates the ensemble on out-of-bag instances after training; the score lands in oob_score_, and each training instance's OOB class probabilities in oob_decision_function_.

If the base estimator has a predict_proba() method, as decision trees do, the ensemble automatically uses soft voting. BaggingRegressor is the regression counterpart.

## 5. Why does each predictor in a bagging ensemble see only about 63% of the distinct training instances?

By default each predictor draws m instances with replacement from a training set of size m. A given instance is missed by one draw with probability 1 − 1/m, so it's missed by all m draws with probability (1 − 1/m)ᵐ, which approaches e⁻¹ ≈ 0.37 as m grows. So about 37% of the training instances are never sampled for a given predictor, and only about 63% of the distinct instances are actually used (some of them several times). The unused ones are that predictor's out-of-bag (OOB) instances, and they're a different 37% for each predictor. With enough predictors, each training instance is likely to be OOB for several of them, which is what lets the ensemble score itself on data its predictors haven't seen.

## 6. What are the random patches and random subspaces methods, and how do you get each with BaggingClassifier?

Both sample the input features, so each predictor is trained on a random subset of them. BaggingClassifier controls this with max_features and bootstrap_features, which work like max_samples and bootstrap but for features instead of instances.
- Random patches: sample both the training instances and the features.
- Random subspaces: keep all training instances (bootstrap=False and max_samples=1.0) but sample features (bootstrap_features=True and/or max_features below 1.0).

Sampling features adds even more predictor diversity, trading a bit more bias for a lower variance. It's particularly useful for high-dimensional inputs such as images, where it can speed up training considerably.

## 7. How does a random forest differ from simply bagging decision trees, and what BaggingClassifier is equivalent to a RandomForestClassifier?

A random forest is an ensemble of decision trees generally trained with bagging (typically with samples as large as the training set), plus extra randomness when growing each tree: at every node, it searches for the best split among a random subset of the features instead of all n of them, √n features by default for classification. The trees become more diverse, trading a higher bias for a lower variance, which generally yields a better model overall. RandomForestClassifier is more convenient and optimized for trees, and with a few exceptions it accepts both the decision tree hyperparameters and the bagging ones. These two ensembles are equivalent:

```python
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
```

## 8. How does a random forest measure feature importance, and how do you read it in Scikit-Learn?

For each feature, it looks at the tree nodes that split on that feature and measures how much they reduce impurity on average across all the trees, as a weighted average in which each node counts in proportion to the number of training samples that reach it. After training, the scores are scaled to sum to 1 and exposed in the feature_importances_ attribute:

```python
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)
```

On the iris dataset, petal length and petal width dominate while the sepal measurements matter far less. This makes random forests a quick way to find out which features actually matter, for example when you need to perform feature selection.

## 9. Walk through the AdaBoost training loop and how the final ensemble makes predictions.

1. Give every training instance the same weight, w⁽ⁱ⁾ = 1/m.
2. Train a predictor and compute its weighted error rate rⱼ, the total weight of the instances it misclassifies.
3. Give the predictor a weight αⱼ = η·log((1 − rⱼ)/rⱼ), where η is the learning rate (default 1): high if it's accurate, near 0 if it's guessing randomly, negative if it's worse than random.
4. Multiply the weight of each misclassified instance by exp(αⱼ), then normalize the weights to sum to 1.
5. Train the next predictor on the reweighted instances, and repeat until there are enough predictors or one is perfect.

To predict, each predictor votes for its class with weight αⱼ, and the class with the largest total wins. Scikit-Learn's AdaBoostClassifier uses SAMME, a multiclass version of AdaBoost, with decision stumps (trees with max_depth=1) as its default base estimator.

## 10. Walk through gradient boosting for regression by hand with three trees, and explain how the ensemble predicts.

Instead of reweighting instances like AdaBoost, each new tree is trained on the residual errors left by the previous predictor:

```python
tree_reg1 = DecisionTreeRegressor(max_depth=2).fit(X, y)
y2 = y - tree_reg1.predict(X)      # residual errors of the first tree
tree_reg2 = DecisionTreeRegressor(max_depth=2).fit(X, y2)
y3 = y2 - tree_reg2.predict(X)     # what the first two trees still miss
tree_reg3 = DecisionTreeRegressor(max_depth=2).fit(X, y3)
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

The ensemble's prediction is the sum of all the trees' predictions, so each tree adds a correction to what came before. GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0) builds the same ensemble. A learning_rate below 1 scales down each tree's contribution (shrinkage), so you need more trees, but the model usually generalizes better. Where the name comes from: with a squared-error loss, the residuals are proportional to the negative gradient of the loss with respect to the current predictions, so adding each tree amounts to a gradient descent step on the predictions.

## 11. How does GradientBoostingRegressor's built-in early stopping work, and what does its subsample hyperparameter do?

Set n_iter_no_change to an integer, e.g., GradientBoostingRegressor(max_depth=2, learning_rate=0.05, n_estimators=500, n_iter_no_change=10). fit() then splits off a validation set (validation_fraction, 10% by default), evaluates the model each time it adds a tree, and stops adding trees once the last n_iter_no_change trees haven't improved the validation score by more than tol (default 0.0001). The number of trees it actually kept is stored in n_estimators_, often far fewer than n_estimators. Too small an n_iter_no_change may stop training too early and underfit; too large a value lets it overfit. Separately, subsample=0.25 trains each tree on a random 25% of the training instances. This is stochastic gradient boosting: it trades a higher bias for a lower variance and speeds up training considerably.

## 12. What makes histogram-based gradient boosting so much faster than regular gradient boosting, and how does its Scikit-Learn API differ?

HGB bins each input feature into integers, using at most max_bins bins (default 255, which is also the maximum). With far fewer thresholds to evaluate, compact integer data structures, and no need to sort the features for each tree, training drops from O(n × m × log(m)) to O(b × m), where b is the number of bins: often hundreds of times faster on large datasets. The precision lost to binning acts as a regularizer, which may reduce overfitting or cause underfitting. Compared with GradientBoostingRegressor and GradientBoostingClassifier, the HistGradientBoosting classes:
- turn early stopping on automatically above 10,000 instances (set early_stopping to force it on or off);
- don't support subsampling;
- call n_estimators max_iter;
- only expose the max_leaf_nodes, min_samples_leaf, max_depth and max_features tree hyperparameters;
- handle missing values and categorical features natively; categories must be integers below max_bins (e.g., from an OrdinalEncoder), flagged with categorical_features.

## 13. How is the blender in a stacking ensemble trained, and how does Scikit-Learn's StackingClassifier automate it?

The blender must learn from predictions like those the base predictors will make on new data, so its training set is built from out-of-sample predictions: run cross_val_predict() on each base predictor, so every training instance gets predictions from models that didn't see it. Each instance becomes one input feature per base predictor, and its target is copied from the original training set. After training the blender, retrain the base predictors on the full training set, since cross_val_predict() doesn't keep its fitted models. StackingClassifier does all of this:

```python
stacking_clf = StackingClassifier(
    estimators=[("lr", LogisticRegression(random_state=42)),
                ("rf", RandomForestClassifier(random_state=42)),
                ("svc", SVC(probability=True, random_state=42))],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5)  # number of cross-validation folds
stacking_clf.fit(X_train, y_train)
```

It feeds the blender each base estimator's predict_proba() output if available, else decision_function(), else predict(). Without a final_estimator, it uses LogisticRegression (StackingRegressor uses RidgeCV).
