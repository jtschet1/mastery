# Chapter 6 — Ensemble Learning and Random Forests, to fall asleep to

Chapter six. Ensemble learning and random forests.

---

Ensemble learning. A complex question put to thousands of random people, with their answers aggregated, often gets a better answer than an expert gives. This is called the wisdom of the crowd. Similarly, aggregating the predictions of a group of predictors, such as classifiers or regressors, often gives better predictions than the best individual predictor. A group of predictors is called an ensemble, the technique is called ensemble learning, and an ensemble learning algorithm is called an ensemble method.

For example, a group of decision tree classifiers can each be trained on a different random subset of the training set, and the class that gets the most votes among their predictions is the ensemble's prediction. Such an ensemble of decision trees is called a random forest. Despite its simplicity, it is one of the most powerful machine learning algorithms available.

Ensemble methods are often used near the end of a project to combine a few good predictors into an even better one. Winning solutions in machine learning competitions often involve several ensemble methods, most famously in the Netflix Prize competition. The downsides are that ensembles need much more computing resources than a single model, for both training and inference, can be more complex to deploy and manage, and are harder to interpret. The pros often outweigh the cons.

---

Voting classifiers. Suppose several classifiers have been trained, each reaching about eighty percent accuracy: for example a logistic regression classifier, a support vector machine classifier, or SVM, a random forest classifier, and a k-nearest neighbors classifier. A very simple way to create a better classifier is to aggregate their predictions and predict the class that gets the most votes. This majority-vote classifier is called a hard voting classifier.

The voting classifier often achieves a higher accuracy than the best classifier in the ensemble. Even if each classifier is a weak learner, meaning it does only slightly better than random guessing, the ensemble can still be a strong learner, achieving high accuracy, provided there are enough weak learners and they are sufficiently diverse, meaning they focus on different aspects of the data and make different kinds of errors.

The law of large numbers explains this. A slightly biased coin has a fifty-one percent chance of coming up heads and a forty-nine percent chance of tails. The probability of getting a majority of heads after one thousand tosses is close to seventy-five percent, and after ten thousand tosses it climbs over ninety-seven percent. The more the coin is tossed, the closer the ratio of heads gets to fifty-one percent, so it ends up consistently above fifty percent.

Similarly, an ensemble of one thousand classifiers that are each correct only fifty-one percent of the time can reach up to seventy-five percent accuracy by predicting the majority class. But this only holds if all the classifiers are perfectly independent, making uncorrelated errors. That is clearly not the case when they are trained on the same data. They are likely to make the same types of errors, so there will be many majority votes for the wrong class, which reduces the ensemble's accuracy.

Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them with very different algorithms, which increases the chance that they make very different types of errors. Another is to vary the model hyperparameters, or to train the models on different subsets of the data.

Scikit-Learn's voting classifier class takes a list of name and predictor pairs and is used like a normal classifier. Fitting it clones every estimator and fits the clones, and both the originals and the fitted clones remain available. The book trains one on the moons dataset with three diverse classifiers: logistic regression, a random forest, and a support vector classifier. On the test set, the logistic regression reaches eighty-six point four percent accuracy, and the other two eighty-nine point six percent each. The voting classifier's predict method performs hard voting, and it reaches ninety-one point two percent, outperforming all the individual classifiers.

Soft voting. If all the classifiers can estimate class probabilities, meaning they all have a predict proba method, the ensemble can predict the class with the highest class probability, averaged over all the individual classifiers. This is called soft voting. It often performs better than hard voting, because it gives more weight to highly confident votes.

Soft voting requires setting the voting hyperparameter to soft. The support vector classifier cannot estimate probabilities by default, so its probability hyperparameter must be set to true, which makes it use cross-validation to estimate them and slows down training. With soft voting, the ensemble reaches ninety-two percent accuracy.

Soft voting works best when the estimated probabilities are well calibrated. If they are not, Scikit-Learn's calibrated classifier CV class can calibrate them.

---

Bagging and pasting. Besides using very different training algorithms, another way to get a diverse set of predictors is to use the same training algorithm for every predictor, but to train each one on a different random subset of the training set.

When the sampling is done with replacement, like drawing a card and putting it back before the next draw, the method is called bagging, short for bootstrap aggregating. In statistics, resampling with replacement is called bootstrapping. When the sampling is done without replacement, the method is called pasting.

Both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows an instance to be sampled several times for the same predictor.

Once all predictors are trained, the ensemble makes a prediction for a new instance by aggregating the predictions of all the predictors. For classification, the aggregation function is typically the statistical mode, meaning the most frequent prediction, as with a hard voting classifier. For regression it is usually the average. Each individual predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance.

For intuition, one house price regressor underestimates by forty thousand dollars on average, and another overestimates by fifty thousand. If they are fully independent and normally distributed, their average overestimates by only five thousand dollars, a much lower bias. And if both have a standard deviation of ten thousand dollars, the average has half the variance.

In practice, the ensemble often ends up with a similar bias but a lower variance than a single predictor trained on the original training set. So it works best with high-variance, low-bias models, such as ensembles of decision trees, not ensembles of linear regressors.

Bagging is preferable when the dataset is noisy or the model is prone to overfitting, such as a deep decision tree. Otherwise pasting is preferable, because it avoids redundancy during training, which makes it a bit more computationally efficient.

The predictors can all be trained in parallel, on different central processing unit cores, or CPU cores, or even on different servers. Predictions can be made in parallel too. This is one reason bagging and pasting are so popular: they scale very well.

---

Bagging and pasting in Scikit-Learn. Scikit-Learn's bagging classifier class handles both bagging and pasting, and the bagging regressor handles regression. The book's example trains an ensemble of five hundred decision tree classifiers, each on one hundred training instances sampled randomly with replacement, which is bagging. Setting the bootstrap hyperparameter to false gives pasting instead. The n jobs hyperparameter sets the number of CPU cores to use, and minus one means all available cores.

The bagging classifier automatically performs soft voting instead of hard voting if the base classifier can estimate class probabilities, which is the case for decision trees.

On the moons dataset, the bagging ensemble of five hundred trees will likely generalize much better than a single decision tree. The ensemble has a comparable bias but a smaller variance: it makes roughly the same number of errors on the training set, but its decision boundary is less irregular.

Bagging introduces a bit more diversity in the subsets each predictor is trained on, so it ends up with a slightly higher bias than pasting. But the extra diversity also makes the predictors less correlated, which reduces the ensemble's variance. Overall, bagging often results in better models, which is why it is generally preferred, though cross-validation can compare the two when time and computing power allow.

---

Out-of-bag evaluation. With bagging, some training instances may be sampled several times for a given predictor, while others may not be sampled at all. By default, the bagging classifier samples m training instances with replacement, where m is the size of the training set. It can be shown mathematically that only about sixty-three percent of the training instances are then sampled, on average, for each predictor. As m grows, this ratio approaches one minus the exponential of minus one. The remaining thirty-seven percent, the instances that are not sampled, are called out-of-bag instances, abbreviated OOB. They are not the same thirty-seven percent for every predictor.

A bagging ensemble can be evaluated on its OOB instances, without a separate validation set. With enough estimators, each training instance is likely to be OOB for several of them, and those estimators can make a fair ensemble prediction for it, from which accuracy or any other metric can be computed.

In Scikit-Learn, setting the OOB score hyperparameter to true when creating a bagging classifier requests an automatic OOB evaluation after training, and the resulting score is stored in an attribute. In the book's example, the OOB evaluation estimates about eighty-nine point six percent accuracy, and the accuracy on the test set turns out to be ninety-two percent, so the OOB evaluation was a bit too pessimistic.

The OOB decision function is also available. Since decision trees can estimate probabilities, it gives OOB class probabilities for each training instance.

---

Random patches and random subspaces. The bagging classifier can sample features as well as instances. Two hyperparameters control this, max features and bootstrap features. They work like max samples and bootstrap, but for features instead of instances. Each predictor is then trained on a random subset of the input features.

This is particularly useful with high-dimensional inputs, such as images, because it can considerably speed up training. Sampling both training instances and features is called the random patches method. Keeping all training instances but sampling features is called the random subspaces method.

Sampling features gives even more predictor diversity, trading a bit more bias for a lower variance.

---

Random forests. A random forest is an ensemble of decision trees, generally trained with the bagging method, or sometimes pasting, typically with max samples set to the size of the training set. Instead of building a bagging classifier around a decision tree classifier, Scikit-Learn's random forest classifier class can be used, which is more convenient and optimized for decision trees. There is also a random forest regressor for regression tasks. The bagging classifier remains useful for bagging models other than decision trees.

The book trains one with five hundred trees, each limited to sixteen leaf nodes. With a few exceptions, the random forest classifier has all the hyperparameters of a decision tree classifier, to control how trees are grown, plus all the hyperparameters of a bagging classifier, to control the ensemble itself.

The random forest algorithm introduces extra randomness when growing trees. Instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features. By default it samples the square root of n features, where n is the total number of features. This results in greater tree diversity, which again trades a higher bias for a lower variance, generally yielding a better model overall.

So a bagging classifier of decision trees, with max features set to the square root and max leaf nodes set to sixteen, is equivalent to the book's random forest classifier.

---

Extra-trees. When a tree in a random forest is grown, only a random subset of the features is considered for splitting at each node. Trees can be made even more random by also using random thresholds for each feature, rather than searching for the best possible thresholds as regular decision trees do. For a single tree, this is done by setting the decision tree's splitter hyperparameter to random.

A forest of such extremely random trees is called an extremely randomized trees ensemble, or extra-trees for short. Once again this trades more bias for a lower variance, so extra-trees may perform better than regular random forests when there is overfitting, especially with noisy or high-dimensional datasets.

Extra-trees are also much faster to train than regular random forests, because finding the best possible threshold for each feature at every node is one of the most time-consuming tasks of growing a tree.

Scikit-Learn's extra trees classifier and extra trees regressor have the same application programming interface, or API, as the corresponding random forest classes, except that bootstrap defaults to false.

In recent versions of Scikit-Learn, random forest and extra-trees classes, like decision tree classes, support missing values natively, so no imputer is needed.

---

Feature importance. Random forests make it easy to measure the relative importance of each feature. Scikit-Learn measures a feature's importance by how much the tree nodes that use that feature reduce impurity on average, across all trees in the forest. More precisely, it is a weighted average, where each node's weight equals the number of training samples associated with it.

Scikit-Learn computes this score for each feature automatically after training, then scales the results so that all the importances add up to one. They are available in the feature importances attribute.

A random forest classifier trained on the iris dataset gives petal length an importance of forty-four percent and petal width forty-two percent, while sepal length, at eleven percent, and sepal width, at two percent, are rather unimportant in comparison.

Random forests are very handy for getting a quick understanding of which features actually matter, in particular when feature selection is needed.

---

Boosting. Boosting, originally called hypothesis boosting, refers to any ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor. There are many boosting methods, but by far the most popular are AdaBoost, short for adaptive boosting, and gradient boosting.

---

AdaBoost. One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances the predecessor underfit. New predictors then focus more and more on the hard cases. This is the technique used by AdaBoost.

To train an AdaBoost classifier, the algorithm first trains a base classifier, such as a decision tree, and uses it to make predictions on the training set. It then increases the relative weight of the misclassified training instances. Then it trains a second classifier using the updated weights, again makes predictions on the training set, updates the instance weights, and so on.

The book illustrates this on the moons dataset with five consecutive highly regularized SVM classifiers, though SVMs are generally poor base predictors for AdaBoost, being slow and unstable with it. The first classifier gets many instances wrong, so their weights get boosted, and the second does better on them. With the learning rate halved, the weights are boosted much less at every iteration.

This sequential learning technique has some similarities with gradient descent, except that instead of tweaking a single predictor's parameters to minimize a cost function, AdaBoost adds predictors to the ensemble, gradually making it better.

It has one important drawback: training cannot be parallelized, since each predictor can only be trained after the previous predictor has been trained and evaluated. So it does not scale as well as bagging or pasting.

Once all predictors are trained, the ensemble makes predictions much like bagging or pasting, except that the predictors have different weights depending on their overall accuracy on the weighted training set.

The AdaBoost algorithm in detail. Each instance weight is initially set to one divided by m, so the weights add up to one. A first predictor is trained, and its weighted error rate is computed on the training set: the sum of the weights of the training instances it misclassified.

The predictor's weight, alpha j, is then the learning rate times the log of one minus the error rate, divided by the error rate. The learning rate hyperparameter, eta, defaults to one; the original AdaBoost algorithm does not use a learning rate. The more accurate the predictor, the higher its weight. If it is just guessing randomly, its weight is close to zero. If it is most often wrong, meaning less accurate than random guessing, its weight is negative.

Next, the instance weights are updated. Each misclassified instance's weight is multiplied by the exponential of the predictor's weight, while correctly classified instances keep their weights. This boosts the weights of the misclassified instances and encourages the next predictor to pay more attention to them. Then all instance weights are normalized, by dividing them by their sum, so that they add up to one again.

Then a new predictor is trained using the updated weights, and the whole process repeats. The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found.

To make predictions, AdaBoost computes the predictions of all the predictors and weighs them using the predictor weights. For each class, it adds up the weights of the predictors that voted for that class, and the predicted class is the one with the largest weighted sum, meaning the majority of weighted votes.

Scikit-Learn uses a multiclass version of AdaBoost called SAMME, which stands for Stagewise Additive Modeling using a Multiclass Exponential loss function. With just two classes, SAMME is equivalent to AdaBoost.

The book trains Scikit-Learn's AdaBoost classifier with thirty decision stumps and a learning rate of zero point five. A decision stump is a decision tree with a max depth of one: a single decision node plus two leaf nodes. It is the default base estimator for the AdaBoost classifier. There is also an AdaBoost regressor class.

If an AdaBoost ensemble is overfitting the training set, the number of estimators can be reduced, or the base estimator can be regularized more strongly.

---

Gradient boosting. Gradient boosting is another very popular boosting algorithm. Like AdaBoost, it works by sequentially adding predictors to an ensemble, each one correcting its predecessor. But instead of tweaking the instance weights at every iteration, it tries to fit each new predictor to the residual errors made by the previous predictor.

With decision trees as the base predictors, this is called gradient tree boosting, or gradient boosted regression trees, GBRT. The book's regression example uses one hundred instances of noisy quadratic data, with a target equal to three times the feature squared, plus Gaussian noise.

A first decision tree regressor, with a max depth of two, is fit to the data. A second tree is trained on the residual errors of the first: the targets minus the first tree's predictions. A third tree is trained on the residual errors of the second. The resulting ensemble of three trees makes a prediction for a new instance simply by adding up the predictions of all the trees.

As trees are added, the ensemble's predictions gradually get better.

Scikit-Learn's gradient boosting regressor class trains GBRT ensembles more easily, and there is a gradient boosting classifier for classification. Like the random forest regressor, it has hyperparameters to control tree growth, such as max depth and min samples leaf, and hyperparameters to control ensemble training, such as the number of trees, n estimators. With a max depth of two, three estimators, and a learning rate of one, it builds the same ensemble as the manual example.

The learning rate hyperparameter scales the contribution of each tree. With a low value, such as zero point zero five, the ensemble needs more trees to fit the training set, but the predictions usually generalize better. This regularization technique is called shrinkage. An ensemble with too few trees underfits the training set, and one with too many starts to overfit it. The optimal learning rate can be found with cross-validation, using grid search or randomized search.

The optimal number of trees can also be found with cross-validation, but there is a simpler way. Setting the n iter no change hyperparameter to an integer, such as ten, makes the gradient boosting regressor stop adding trees automatically during training when the last ten trees did not help. This is early stopping with a little patience: it tolerates having no progress for a few iterations before it stops.

If n iter no change is set too low, training may stop too early and the model underfits. If it is set too high, the model overfits instead. In the book's example, with a small learning rate of zero point zero five and up to five hundred estimators, early stopping leaves only fifty-three trees in the trained ensemble.

When n iter no change is set, the fit method automatically splits the training set into a smaller training set and a validation set, to evaluate the model's performance each time it adds a tree. The validation fraction hyperparameter sets the size of the validation set, ten percent by default, and the tol hyperparameter sets the largest improvement that still counts as negligible.

The gradient boosting regressor also supports a subsample hyperparameter, which sets the fraction of training instances used to train each tree. With subsample set to zero point two five, each tree is trained on twenty-five percent of the training instances, selected randomly. This trades a higher bias for a lower variance, and it also speeds up training considerably. It is called stochastic gradient boosting.

---

Histogram-based gradient boosting. Scikit-Learn also provides another GBRT implementation, optimized for large datasets: histogram-based gradient boosting, abbreviated HGB. It works by binning the input features, replacing them with integers. The number of bins is set by the max bins hyperparameter, which defaults to two hundred fifty-five and cannot be set any higher.

Binning greatly reduces the number of possible thresholds the training algorithm needs to evaluate. Working with integers makes it possible to use faster and more memory-efficient data structures. And the way the bins are built removes the need to sort the features when training each tree.

As a result, training takes time proportional to the number of bins times the number of instances, instead of the number of features times the number of instances times the log of the number of instances. In practice, HGB can train hundreds of times faster than regular GBRT on large datasets. However, binning causes a precision loss, which acts as a regularizer: depending on the dataset, it may help reduce overfitting, or it may cause underfitting.

Scikit-Learn provides two classes for HGB, the hist gradient boosting regressor and the hist gradient boosting classifier. They are similar to the regular gradient boosting classes, with a few notable differences. Early stopping is automatically activated if there are more than ten thousand instances, and the early stopping hyperparameter can turn it always on or always off. Subsampling is not supported. The n estimators hyperparameter is renamed max iter. The only decision tree hyperparameters that can be tweaked are max leaf nodes, min samples leaf, max depth, and max features.

The HGB classes support missing values natively, as well as categorical features, which simplifies preprocessing quite a bit. The categorical features must be represented as integers ranging from zero to a number lower than max bins, which an ordinal encoder can produce.

The book's pipeline for the California housing dataset ordinal-encodes the ocean proximity column, passes the other columns through, and tells the HGB regressor which column is categorical. No imputer, scaler, or one-hot encoder is needed. Without any tuning, it reaches a root mean squared error of about forty-seven thousand six hundred.

HGB is a great choice for fairly large datasets, especially with categorical features and missing values: it performs well, needs little preprocessing, and trains fast. It can be a bit less accurate than GBRT because of the binning, so it is worth trying both.

Several other optimized implementations of gradient boosting exist in the Python machine learning ecosystem, in particular XGBoost, CatBoost, and LightGBM. They are specialized for gradient boosting, their APIs are very similar to Scikit-Learn's, and they provide many additional features, including hardware acceleration using graphics processing units. Yggdrasil Decision Forests, abbreviated YDF, provides optimized implementations of a variety of random forest algorithms.

---

Stacking. Stacking, short for stacked generalization, is based on a simple idea: instead of using trivial functions, such as hard voting, to aggregate the predictions of all the predictors in an ensemble, train a model to perform the aggregation. For example, three predictors might predict three point one, two point seven, and two point nine for a regression task. A final predictor, called a blender or a meta learner, takes these predictions as inputs and makes the final prediction, three point zero.

To train the blender, a blending training set must first be built. Scikit-Learn's cross val predict function is run on every predictor in the ensemble, to get out-of-sample predictions for each instance in the original training set. These predictions are used as the input features to train the blender, and the targets are simply copied from the original training set. Regardless of the number of features in the original training set, the blending training set contains one input feature per predictor.

Once the blender is trained, the base predictors must be retrained one last time on the full original training set, since cross val predict does not give access to the trained estimators.

Several different blenders can be trained this way, for example one using linear regression and another using random forest regression, forming a layer of blenders, with another blender on top. This may squeeze out a few more drops of performance, at a cost in training time and system complexity.

Scikit-Learn provides a stacking classifier and a stacking regressor. The book replaces the moons voting classifier with a stacking classifier using the same three base classifiers, a random forest classifier as the final estimator, and five cross-validation folds.

For each predictor, the stacking classifier calls the predict proba method if available; if not, it falls back to the decision function method, and as a last resort, to the predict method. Without a final estimator, the stacking classifier uses logistic regression, and the stacking regressor uses ridge CV.

On the test set, the stacking model reaches ninety-two point eight percent accuracy, a bit better than the soft voting classifier's ninety-two percent. Depending on the use case, this may or may not be worth the extra complexity and computational cost, since there is an extra model to run after all the others.

---

When to use each ensemble method. Ensemble methods are versatile, powerful, and fairly simple to use. They can overfit if used carelessly, but that is true of every powerful model.

Hard voting suits balanced classification datasets with several strong but diverse classifiers, such as spam detection. Soft voting suits probabilistic models where confidence scores matter, such as medical diagnosis.

Bagging suits structured or semi-structured datasets with high-variance, overfitting-prone models, such as financial risk modeling. Pasting suits similar datasets where more independent models are needed, such as customer segmentation.

Random forests suit high-dimensional structured datasets with potentially noisy features, such as fraud detection. Extra-trees suit large structured datasets with many features, where speed is critical and reducing variance matters, such as sensor data analysis.

AdaBoost suits small to medium, low-noise, structured datasets with weak learners such as decision stumps, where interpretability helps, such as credit scoring. Gradient boosting suits medium to large structured datasets where high predictive power is worth extra tuning, such as demand forecasting. Histogram-based gradient boosting suits large structured datasets where training speed and scalability are key, such as click-through rate prediction.

Stacking suits complex, high-dimensional datasets where combining diverse models can maximize accuracy, such as recommendation engines and Kaggle competitions.

Random forests, AdaBoost, GBRT, and HGB are among the first models to test for most machine learning tasks, particularly with heterogeneous tabular data. Because they require very little preprocessing, they are great for getting a prototype up and running quickly.

---

Summary.

An ensemble aggregates the predictions of several predictors and often beats the best of them, especially when the predictors are diverse and make uncorrelated errors.

A voting classifier predicts the majority class, which is hard voting, or the class with the highest average probability, which is soft voting and usually works better.

Bagging and pasting train the same algorithm on random subsets of the training set, with and without replacement. They mainly reduce variance and scale well, and with bagging, out-of-bag instances give a validation estimate for free.

A random forest is a bagging ensemble of decision trees that picks each split from a random subset of features. Extra-trees also use random thresholds, which makes them faster to train. Random forests also measure feature importance.

Boosting trains predictors sequentially, each correcting its predecessor. AdaBoost increases the weights of misclassified instances and weights each predictor by its accuracy. Gradient boosting fits each new tree to the residual errors of the previous one, regularized by a small learning rate, early stopping, and subsampling. Histogram-based gradient boosting bins the features to train much faster on large datasets.

Stacking trains a blender on out-of-sample predictions from the base predictors, instead of aggregating them with a fixed rule.
