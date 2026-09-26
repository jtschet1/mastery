# Chapter 5 — Decision Trees

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/05_decision_trees.ipynb`, converted to plain text for the
flashcard app.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 7. Train and fine-tune a decision tree on make_moons(n_samples=10000,
  noise=0.4): split, grid-search max_leaf_nodes, evaluate on the test set
  (~85-87%).
- 8. Grow a forest by hand: train 1,000 trees on random 100-instance
  subsets, then take their majority vote (SciPy's mode()) and compare with
  the single tree.

## 1. What is the approximate depth of a decision tree trained (without restrictions) on a training set with one million instances?

The depth of a well-balanced binary tree containing m leaves is equal to log₂(m), rounded up. log₂ is the binary log; log₂(m) = log(m) / log(2). A binary Decision Tree (one that makes only binary decisions, as is the case with all trees in Scikit-Learn) will end up more or less well balanced at the end of training, with one leaf per training instance if it is trained without restrictions. Thus, if the training set contains one million instances, the Decision Tree will have a depth of log₂(10⁶) ≈ 20 (actually a bit more since the tree will generally not be perfectly well balanced).

## 2. Is a node’s Gini impurity generally lower or higher than its parent’s? Is it generally lower/higher, or always lower/higher?

A node's Gini impurity is generally lower than its parent's. This is due to the CART training algorithm's cost function, which splits each node in a way that minimizes the weighted sum of its children's Gini impurities. However, it is possible for a node to have a higher Gini impurity than its parent, as long as this increase is more than compensated for by a decrease in the other child's impurity. For example, consider a node containing four instances of class A and one of class B. Its Gini impurity is 1 – (1/5)² – (4/5)² = 0.32. Now suppose the dataset is one-dimensional and the instances are lined up in the following order: A, B, A, A, A. You can verify that the algorithm will split this node after the second instance, producing one child node with instances A, B, and the other child node with instances A, A, A. The first child node's Gini impurity is 1 – (1/2)² – (1/2)² = 0.5, which is higher than its parent's. This is compensated for by the fact that the other node is pure, so its overall weighted Gini impurity is 2/5 × 0.5 + 3/5 × 0 = 0.2, which is lower than the parent's Gini impurity.

## 3. If a decision tree is overfitting the training set, is it a good idea to try decreasing max_depth?

If a Decision Tree is overfitting the training set, it may be a good idea to decrease max_depth, since this will constrain the model, regularizing it.

## 4. If a decision tree is underfitting the training set, is it a good idea to try scaling the input features?

Decision Trees don't care whether or not the training data is scaled or centered; that's one of the nice things about them. So if a Decision Tree underfits the training set, scaling the input features will just be a waste of time.

## 5. If it takes one hour to train a decision tree on a training set containing one million instances, roughly how much time will it take to train another decision tree on a training set containing ten million instances? Hint: consider the CART algorithm’s computational complexity.

The computational complexity of training a Decision Tree is O(n × m log₂(m)). So if you multiply the training set size by 10, the training time will be multiplied by K = (n × 10 m × log₂(10 m)) / (n × m × log₂(m)) = 10 × log₂(10 m) / log₂(m). If m = 10⁶, then K ≈ 11.7, so you can expect the training time to be roughly 11.7 hours.

## 6. If it takes one hour to train a decision tree on a given training set, roughly how much time will it take if you double the number of features?

If the number of features doubles, then the training time will also roughly double.
