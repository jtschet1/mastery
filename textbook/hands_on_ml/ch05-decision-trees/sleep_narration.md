# Chapter 5 — Decision Trees, to fall asleep to

Chapter five. Decision trees.

---

Decision trees. Decision trees are versatile machine learning algorithms that can perform classification, regression, and even multioutput tasks. They are powerful and can fit complex datasets. In chapter two, a decision tree regressor fit the California housing dataset perfectly, which in fact meant it overfit it. Decision trees are also the fundamental components of random forests, which are among the most powerful machine learning algorithms available.

---

Training and visualizing a decision tree. The book trains Scikit-Learn's decision tree classifier on the iris dataset, using only petal length and petal width as features, with the max depth hyperparameter set to two.

The trained tree can be visualized by exporting it with Scikit-Learn's export graphviz function, which writes a graph definition file in the dot format, then displaying that file in a Jupyter notebook with the Graphviz library. Graphviz is an open source graph visualization package. It also includes a command-line tool, called dot, that converts dot files to formats such as PDF or PNG.

---

Making predictions. A prediction starts at the root node, at depth zero, at the top of the tree. The root asks whether the flower's petal length is smaller than two point four five centimeters. If it is, the flower moves down to the root's left child, at depth one. That node is a leaf node, meaning it has no child nodes, so it asks no question: the tree simply predicts that node's class, Iris setosa.

If the petal length is greater than two point four five centimeters, the flower moves to the root's right child, at depth one. That is a split node, not a leaf, so it asks another question: is the petal width smaller than one point seven five centimeters? If so, the flower is most likely an Iris versicolor, the left leaf at depth two. If not, it is likely an Iris virginica, the right leaf at depth two.

Decision trees require very little data preparation. In particular, they need no feature scaling or centering at all.

Each node displays several attributes. The samples attribute counts how many training instances the node applies to: one hundred training instances have a petal length greater than two point four five centimeters, and of those, fifty-four have a petal width smaller than one point seven five centimeters. The value attribute gives how many training instances of each class the node applies to: the bottom-right node applies to zero Iris setosa, one Iris versicolor, and forty-five Iris virginica. The gini attribute is the node's Gini impurity.

---

Gini impurity. A node is pure, with a Gini impurity of zero, if all the training instances it applies to belong to the same class. The left node at depth one applies only to Iris setosa, so its impurity is zero. The other nodes apply to instances of several classes, so they are impure.

The Gini impurity of a node is one minus the sum, over all classes, of the square of each class's ratio among the node's training instances. The more classes and the more mixed they are, the larger the impurity. The left node at depth two holds zero setosa, forty-nine versicolor, and five virginica, out of fifty-four instances. Its impurity is one, minus forty-nine fifty-fourths squared, minus five fifty-fourths squared, which is about zero point one six eight.

Scikit-Learn uses the CART algorithm, which produces only binary trees: every split node has exactly two children, so every question has a yes or no answer. Other algorithms, such as ID3, can produce trees with nodes that have more than two children.

Decision boundaries. The root node's decision boundary is the line where petal length equals two point four five centimeters. The area on the short-petal side is pure, all Iris setosa, so it cannot be split further. The other area is impure, so the right node at depth one splits it at a petal width of one point seven five centimeters. Since the max depth is two, the tree stops there. With a max depth of three, each node at depth two would add another decision boundary. The full tree structure, including everything shown in each node, is available through the classifier's tree attribute.

---

White box versus black box models. Decision trees are intuitive, and their decisions are easy to interpret. Such models are called white box models. Random forests and neural networks are generally considered black box models. They make great predictions, and the calculations behind each prediction can be checked, but it is usually hard to explain in simple terms why the predictions were made. If a neural network says that a particular person appears in a picture, it is hard to know what contributed: the person's eyes, mouth, nose, shoes, or even the couch they were sitting on. Decision trees provide simple classification rules that can even be applied by hand.

The field of interpretable machine learning aims to create systems that can explain their decisions in a way humans can understand. This matters in many domains: in healthcare, so a doctor can review a diagnosis; in finance, so analysts can understand the risks; in a judicial system, so a human makes the final call; and in human resources, to make sure decisions are not biased.

---

Estimating class probabilities. A decision tree can estimate the probability that an instance belongs to a class k. It traverses the tree to find the instance's leaf node, then returns the proportion of training instances of class k in that leaf.

A flower with petals five centimeters long and one point five centimeters wide lands in the left leaf at depth two. The tree outputs zero percent for Iris setosa, ninety point seven percent for Iris versicolor, which is forty-nine out of fifty-four, and nine point three percent for Iris virginica, which is five out of fifty-four. Asked for a class, it predicts Iris versicolor, class one, because it has the highest probability.

The estimated probabilities are the same anywhere in that leaf's rectangle of the feature space. A flower with petals six centimeters long and one point five centimeters wide gets the same probabilities, even though it would most likely be an Iris virginica.

---

The CART training algorithm. Scikit-Learn trains decision trees, also called growing trees, with the Classification and Regression Tree algorithm, CART. It first splits the training set into two subsets using a single feature k and a threshold t k, such as petal length less than or equal to two point four five centimeters. It chooses k and t k by searching for the pair that produces the purest subsets, weighted by their size.

The CART cost function for classification is the impurity of the left subset times the fraction of instances that go left, plus the impurity of the right subset times the fraction of instances that go right. The algorithm searches for the split that minimizes this cost.

Once the training set is split in two, CART splits each subset with the same logic, then the sub-subsets, and so on, recursively. It stops when it reaches the maximum depth, set by the max depth hyperparameter, or when it cannot find a split that reduces impurity. Other hyperparameters add stopping conditions, including min samples split, min samples leaf, and max leaf nodes.

CART is a greedy algorithm. It searches for an optimal split at the top level, then repeats the process at each level below, without checking whether a split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a reasonably good solution, but not a guaranteed optimal one.

Finding the optimal tree is known to be an NP-complete problem. P is the set of problems that can be solved in polynomial time, and NP is the set of problems whose solutions can be verified in polynomial time. If, as seems likely, P is not equal to NP, no polynomial-time algorithm will ever be found for any NP-complete problem. Finding the optimal tree requires time that grows exponentially with the number of training instances, which makes the problem intractable even for small training sets. So a reasonably good solution has to do.

---

Computational complexity. Making a prediction requires traversing the tree from the root to a leaf. Decision trees are generally roughly balanced, so the traversal goes through about log base two of m nodes, where m is the number of training instances. Each node checks the value of only one feature, so prediction takes time proportional to log base two of m, independent of the number of features. Predictions are very fast, even with large training sets.

By default, training compares all features on all samples at each node, so training takes time proportional to n times m times log base two of m, where n is the number of features.

Setting a maximum depth, or a maximum number of features to consider at each node, with those features chosen randomly, speeds up training considerably. It can also reduce the risk of overfitting, though going too far causes underfitting.

---

Gini impurity or entropy. The decision tree classifier uses Gini impurity by default, but setting the criterion hyperparameter to entropy selects the entropy impurity measure instead.

Entropy originated in thermodynamics as a measure of molecular disorder: it approaches zero when molecules are still and well ordered. It later spread to many other domains, including Shannon's information theory, where it measures the average information content of a message, and is zero when all messages are identical. In machine learning, entropy is often used as an impurity measure: a set's entropy is zero when it contains instances of only one class.

The entropy of a node is minus the sum, over the classes present in the node, of each class's ratio times the log base two of that ratio. The left node at depth two has an entropy of about zero point four four five.

Most of the time, the choice makes no big difference: the two lead to similar trees. Gini impurity is slightly faster to compute, so it is a good default. When they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.

---

Regularization hyperparameters. Decision trees make very few assumptions about the training data, unlike linear models, which assume the data is linear. Left unconstrained, a tree adapts its structure to the training data, fitting it very closely, and most likely overfitting it.

Such a model is called a nonparametric model. That is not because it has no parameters, since it often has many, but because the number of parameters is not determined before training, so the model structure is free to stick closely to the data. A parametric model, such as a linear model, has a predetermined number of parameters, so its degrees of freedom are limited. That reduces the risk of overfitting but increases the risk of underfitting.

To avoid overfitting, the tree's freedom must be restricted during training, which is regularization. The available hyperparameters depend on the algorithm, but the maximum depth can generally be restricted. In Scikit-Learn this is the max depth hyperparameter. Its default value, none, means unlimited. Reducing max depth regularizes the model and reduces the risk of overfitting.

The decision tree classifier has other hyperparameters that restrict the tree's shape. Max features is the maximum number of features evaluated for splitting at each node. Max leaf nodes is the maximum number of leaf nodes. Min samples split is the minimum number of samples a node must have before it can be split. Min samples leaf is the minimum number of samples a leaf node must have to be created. Min weight fraction leaf is the same as min samples leaf, but expressed as a fraction of the total number of weighted instances. Min impurity decrease means a node is split only if the split reduces impurity by at least that amount.

The CCP alpha hyperparameter, where CCP stands for cost-complexity pruning, controls minimal cost-complexity pruning. It prunes subtrees that do not reduce impurity enough compared to their number of leaves. A larger value means more pruning and a smaller tree. The default, zero, means no pruning.

Increasing the min hyperparameters or CCP alpha, or decreasing the max hyperparameters, limits the model's complexity and regularizes it. Tuning max depth is usually a good default: it regularizes effectively and keeps the tree small and easy to interpret. Setting min samples leaf is also a good idea, especially for small datasets. Max features is great for high-dimensional datasets.

Pruning. Other algorithms first train the tree without restrictions, then prune, meaning delete, unnecessary nodes. A node whose children are all leaf nodes is considered unnecessary if the purity improvement it provides is not statistically significant. Standard statistical tests, such as the chi-squared test, estimate the probability that the improvement is purely the result of chance, which is called the null hypothesis. If this probability, called the p-value, is higher than a threshold, typically five percent and set by a hyperparameter, the node is considered unnecessary and its children are deleted. Pruning continues until all unnecessary nodes are gone.

The moons example. The moons dataset is a toy binary classification dataset in which the data points form two interleaving crescent moons. Scikit-Learn's make moons function generates it. The book trains one tree without regularization and another with min samples leaf set to five, both on one hundred fifty noisy points. The unregularized tree is clearly overfitting, with an irregular decision boundary, while the regularized one will probably generalize better. On a test set of one thousand points generated with a different random seed, the unregularized tree reaches about eighty-nine point eight percent accuracy, and the regularized tree reaches ninety-two percent.

---

Regression. Decision trees can also perform regression. Linear regression only works well with linear data, while decision trees can fit all sorts of complex datasets.

The book trains Scikit-Learn's decision tree regressor, with a max depth of two, on a noisy quadratic dataset: two hundred instances with a single input feature, and a target equal to the square of the feature plus a little noise. The resulting tree looks like the classification tree, but each node predicts a value instead of a class.

For a new instance whose feature equals zero point two, the root asks whether the feature is at most zero point three four three. It is, so the instance goes to the left child, which asks whether the feature is at most minus zero point three zero two. It is not, so the instance goes to the right child, a leaf that predicts zero point zero three eight. That prediction is the average target value of the one hundred thirty-three training instances in that leaf, and it gives a mean squared error of zero point zero zero two over those instances.

With a max depth of three, the tree divides the feature range into more regions. The predicted value for each region is always the average target value of the training instances in that region. The algorithm splits each region so that most training instances end up as close as possible to that predicted value.

CART works as before, except that instead of splitting the training set to minimize impurity, it splits it to minimize the mean squared error, MSE. The cost is the MSE of the left subset times the fraction of instances that go left, plus the MSE of the right subset times the fraction that go right. A node's prediction is the mean target of its instances, and its MSE is the average squared difference between that prediction and each instance's target.

Decision trees are prone to overfitting in regression too. With the default hyperparameters, meaning no regularization, the predictions overfit the training set very badly. Just setting min samples leaf to ten gives a much more reasonable model.

---

Sensitivity to axis orientation. Decision trees are relatively easy to understand and interpret, simple to use, versatile, and powerful, but they have some limitations. First, they love orthogonal decision boundaries: every split is perpendicular to an axis. This makes them sensitive to the data's orientation.

A simple linearly separable dataset is easy for a tree to split when the classes are separated along an axis. After the same dataset is rotated by forty-five degrees, the decision boundary looks unnecessarily convoluted. Both trees fit the training set perfectly, but the one trained on the rotated data will very likely not generalize well.

One way to limit this problem is to scale the data, then apply a principal component analysis transformation, or PCA. PCA rotates the data in a way that reduces the correlation between the features, which often, but not always, makes things easier for trees.

The book builds a small pipeline that scales the iris petal features with a standard scaler and rotates them with PCA, then trains a decision tree classifier with a max depth of two on the result. After the rotation, the dataset can be fit pretty well using only one feature, which is a linear function of the original petal length and width.

The decision tree classifier and decision tree regressor both support missing values natively, so no imputer is needed.

---

Decision trees have a high variance. More generally, the main issue with decision trees is their quite high variance: small changes to the hyperparameters or to the data can produce very different models.

The training algorithm Scikit-Learn uses is stochastic: it randomly selects the set of features to evaluate at each node. So even retraining the same decision tree on exactly the same data can produce a very different model, unless the random state hyperparameter is set.

Averaging predictions over many trees reduces variance significantly. Such an ensemble of trees is called a random forest, and it is one of the most powerful types of models available.

---

Summary.

A decision tree classifies an instance by asking a sequence of yes or no questions about single features, from the root node down to a leaf, and predicts the leaf's majority class. Its class probabilities are the class proportions among the training instances in that leaf. Trees need almost no data preparation, and they are white box models whose rules are easy to read.

Scikit-Learn grows trees with the CART algorithm, which greedily chooses at each node the feature and threshold that produce the purest subsets, weighted by size, and recurses until a stopping condition is reached. Impurity is measured with Gini impurity by default, or with entropy; the two usually give similar trees. Prediction is fast, proportional to the log of the number of instances.

Unconstrained trees are nonparametric and overfit easily. They are regularized by limiting max depth, max leaf nodes, or max features, by raising min samples leaf, min samples split, or min impurity decrease, or by cost-complexity pruning.

Regression trees work the same way but predict the average target of each region and split to minimize the MSE.

Trees produce axis-aligned boundaries, so they are sensitive to rotations of the data, which scaling and PCA can help with. Their main weakness is high variance, which averaging many trees in a random forest reduces.
