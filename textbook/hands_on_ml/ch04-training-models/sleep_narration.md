# Chapter 4 — Training Models, to fall asleep to

Chapter four. Training models.

---

Why training algorithms matter. Many tasks can be solved while treating models and their training algorithms as black boxes. Knowing how they work helps pick the right model, the right training algorithm, and good hyperparameters, and it makes debugging and error analysis more efficient. Most of these topics are also essential for neural networks.

Linear regression can be trained in two very different ways. The first is a closed-form equation that directly computes the parameters that minimize the cost function over the training set. A closed-form equation contains only a finite number of constants, variables, and standard operations, with no infinite sums, limits, or integrals. The second is an iterative optimization approach called gradient descent, abbreviated GD, which gradually tweaks the parameters and converges to the same solution.

---

Linear regression. A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the bias term, also called the intercept term. The prediction equals theta zero, plus theta one times the first feature, plus theta two times the second feature, and so on up to theta n times the nth feature. Here n is the number of features, theta zero is the bias term, and theta one to theta n are the feature weights. The life satisfaction model from chapter one is a linear model with a single feature.

In vectorized form, the prediction is the hypothesis function, h theta, applied to x, and it equals the dot product of the parameter vector theta and the feature vector x. The parameter vector holds the bias term and all the feature weights. The feature vector holds x zero to x n, where x zero is always equal to one, so the bias term is multiplied by one.

Vectors are often represented as column vectors, two-dimensional arrays with a single column, so the prediction is written theta transpose times x. It is the same prediction, as a single-cell matrix.

Training a model means setting its parameters so that it best fits the training set. The most common performance measure for regression is the root mean squared error, RMSE, so training linear regression means finding the theta that minimizes the RMSE. In practice it is simpler to minimize the mean squared error, MSE, and the result is the same, because the value that minimizes a positive function also minimizes its square root.

Learning algorithms often optimize a different loss during training than the metric used to evaluate the final model, because it is easier to optimize or has extra terms needed only during training, such as regularization. A good metric is as close as possible to the business objective. A good training loss is easy to optimize and strongly correlated with the metric. Classifiers, for example, are often trained with the log loss but evaluated with precision and recall.

The MSE of a linear regression model is the average, over all m training instances, of the squared difference between the model's prediction, theta transpose times x, and the target value. It is written MSE of theta.

---

The normal equation. There is a closed-form solution for the theta that minimizes the MSE, called the normal equation. The best theta, written theta hat, equals the inverse of X transpose times X, multiplied by X transpose, multiplied by y. X is the matrix of training instances, one row per instance, and y is the vector of target values.

The book tests it on two hundred random instances whose target is four plus three times the feature, plus Gaussian noise. Scikit-Learn's add dummy feature function adds x zero to each instance, and NumPy computes the inverse and the matrix products. The result is about three point six nine for theta zero and three point three three for theta one, instead of the true four and three. The noise made it impossible to recover the exact parameters, and the smaller and noisier the dataset, the harder that gets.

Scikit-Learn's linear regression class finds the same parameters. It stores the bias term separately from the feature weights, in an intercept attribute and a coefficient attribute. The class is based on SciPy's least squares function, which computes theta hat as the pseudoinverse of X times y. This pseudoinverse is the Moore–Penrose inverse.

The pseudoinverse is computed with a standard matrix factorization technique called singular value decomposition, SVD, which decomposes the training set matrix X into the product of three matrices: U, sigma, and V transpose. The pseudoinverse of X is V, times sigma plus, times U transpose. To compute sigma plus, the algorithm sets to zero every value of sigma smaller than a tiny threshold, replaces every nonzero value with its inverse, and transposes the result.

This approach is more efficient than computing the normal equation, and it handles edge cases well. The normal equation may not work if the matrix X transpose X is not invertible, also called singular, for example when there are fewer instances than features, or when some features are redundant. The pseudoinverse is always defined.

---

Computational complexity. The normal equation inverts X transpose X, a square matrix with n plus one rows and columns, where n is the number of features. Inverting it takes time roughly proportional to somewhere between n to the power two point four and n cubed, depending on the implementation. Doubling the number of features multiplies the computation time by roughly five point three to eight.

The SVD approach used by Scikit-Learn's linear regression class takes time roughly proportional to n squared, so doubling the number of features multiplies the time by roughly four.

Both approaches get very slow when the number of features grows large, for example one hundred thousand. But both are linear in the number of training instances, so they handle large training sets efficiently, as long as the data fits in memory.

Once a linear regression model is trained, predictions are very fast: the cost is linear in both the number of instances to predict and the number of features. Gradient descent is better suited to cases with many features, or with too many training instances to fit in memory.

---

Gradient descent. Gradient descent is a generic optimization algorithm that tweaks the parameters iteratively to minimize a cost function.

Someone lost in the mountains in dense fog, feeling only the slope underfoot, can reach the bottom of the valley quickly by going downhill in the direction of the steepest slope. Gradient descent measures the local gradient of the error function with regard to the parameter vector theta, and moves in the direction of descending gradient. When the gradient is zero, a minimum has been reached.

Theta is first filled with random values, which is called random initialization. Then it is improved one small step at a time, each step attempting to decrease the cost function, until the algorithm converges to a minimum. The step size is proportional to the slope, so the steps get smaller as the cost approaches the minimum.

The size of the steps is set by the learning rate hyperparameter. If the learning rate is too small, the algorithm needs many iterations to converge, which takes a long time. If it is too high, the algorithm may jump across the valley and end up higher than before. This can make it diverge, with larger and larger values, failing to find a good solution.

Not all cost functions are regular bowls. Holes, ridges, and plateaus cause the two main challenges of gradient descent: depending on where it starts, the algorithm may converge to a local minimum, which is not as good as the global minimum, or take a very long time to cross a plateau and stop before reaching the global minimum.

The MSE cost function for linear regression is convex: the line segment joining any two points on the curve is never below the curve. So there are no local minima, just one global minimum. It is also continuous, and its slope never changes abruptly; technically, its derivative is Lipschitz continuous. Together, these guarantee that gradient descent can get arbitrarily close to the global minimum, if it runs long enough and the learning rate is not too high.

The bowl can be elongated if the features have very different scales. When one feature has much smaller values than another, a larger change in its weight is needed to affect the cost, so the bowl is stretched along that weight's axis. Gradient descent then first moves almost orthogonally to the direction of the minimum, and ends with a long march down an almost flat valley. With gradient descent, all features should have a similar scale, for example by using Scikit-Learn's standard scaler class, or convergence takes much longer.
---

Batch gradient descent. Gradient descent needs the gradient of the cost function with regard to each model parameter, theta j: how much the cost changes if theta j changes just a little. This is called a partial derivative. It is like asking for the slope of the mountain facing east, then facing north, and so on for every dimension.

The partial derivative of the MSE with regard to theta j is two divided by m, times the sum, over all training instances, of the prediction error multiplied by the instance's jth feature value. The prediction error is the prediction minus the target.

The gradient vector contains all the partial derivatives, one per model parameter, and they can be computed in one go. It equals two divided by m, times X transpose, times the vector of prediction errors, which is X times theta, minus y.

This formula involves calculations over the full training set at every step. That is why the algorithm is called batch gradient descent: it uses the whole batch of training data at every step. Full gradient descent would arguably be a better name. As a result, it is very slow on very large training sets. But it scales well with the number of features: with hundreds of thousands of features, gradient descent trains linear regression much faster than the normal equation or SVD.

The gradient vector points uphill, so the step goes the opposite way. The gradient descent step subtracts the learning rate, eta, times the gradient vector, from the parameter vector.

Each iteration over the training set is called an epoch. With a learning rate of zero point one and one thousand epochs, the book's implementation finds exactly what the normal equation found. A learning rate that is too low gets there slowly, and one that is too high diverges, jumping further from the solution at every step. A good learning rate can be found with grid search, limiting the number of epochs so that grid search eliminates models that take too long to converge.

Too few epochs stop the algorithm far from the optimal solution, and too many waste time once the parameters stop changing. A simple solution is to set a very large number of epochs and stop when the norm of the gradient vector falls below a tiny number epsilon, called the tolerance, which happens when gradient descent has almost reached the minimum.

Convergence rate. When the cost function is convex and its slope does not change abruptly, as with the MSE, batch gradient descent with a fixed learning rate converges to the optimal solution, but it can take on the order of one divided by epsilon iterations to get within epsilon of the optimum. Dividing the tolerance by ten can make the algorithm run about ten times longer.

---

Stochastic gradient descent. The main problem with batch gradient descent is that it uses the whole training set at every step. Stochastic gradient descent is the opposite extreme: at every step it picks a random instance from the training set and computes the gradients based only on that instance. This makes each step much faster, since there is very little data to manipulate. It also makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration, so it can be implemented as an out-of-core algorithm.

Because it is stochastic, meaning random, the algorithm is much less regular than batch gradient descent. Instead of decreasing gently, the cost function bounces up and down, decreasing only on average. Over time it ends up very close to the minimum, but it keeps bouncing around and never settles. The final parameters are good, but not optimal.

When the cost function is very irregular, this randomness can help the algorithm jump out of local minima, so stochastic gradient descent has a better chance of finding the global minimum than batch gradient descent.

Randomness is good for escaping local optima but bad because the algorithm never settles. One solution is to gradually reduce the learning rate: large steps first, for quick progress and escaping local minima, then smaller and smaller steps, so the algorithm settles at the global minimum. This is akin to simulated annealing, inspired by the slow cooling of molten metal in metallurgy. The function that sets the learning rate at each iteration is called the learning schedule. If the learning rate drops too quickly, the algorithm may get stuck in a local minimum, or freeze halfway to the minimum. If it drops too slowly, the algorithm may jump around the minimum for a long time and end up suboptimal if training stops too early.

The book's implementation runs fifty epochs of m iterations each. Each iteration computes the gradients on one random instance, without dividing by m, and takes a step with a learning rate from a schedule that shrinks as the iteration count grows. Batch gradient descent needed one thousand passes through the training set; this code needs only fifty to reach a pretty good solution.

Since instances are picked randomly, some may be picked several times in one epoch and others not at all. Shuffling the training set each epoch and going through it in order avoids this, but it is more complex and generally does not improve the result.

With stochastic gradient descent, the training instances must be independent and identically distributed, abbreviated IID, so that the parameters get pulled toward the global optimum on average. A simple way to ensure this is to shuffle the instances during training. If the instances are sorted by label, for example, stochastic gradient descent starts by optimizing for one label, then the next, and does not settle close to the global minimum.

Scikit-Learn's SGD regressor class performs linear regression with stochastic gradient descent, optimizing the MSE by default. The book's example runs for at most one thousand epochs, set by max iter, or until the loss improves by less than the tol value over the number of epochs set by n iter no change. Its initial learning rate is set by eta zero, and the penalty hyperparameter is set to none, for no regularization. The solution is again very close to the normal equation's.

Besides the fit method, some estimators have a partial fit method, which runs a single round of training on one or more instances, ignoring max iter and tol; calling it repeatedly trains the model gradually. Others have a warm start hyperparameter: when it is true, calling fit on a trained model continues training instead of resetting it. The fit method resets the learning schedule's iteration counter, while partial fit does not.

---

Mini-batch gradient descent. At each step, mini-batch gradient descent computes the gradients on small random sets of instances called mini-batches, instead of on the full training set or on a single instance. Its main advantage over stochastic gradient descent is the performance boost from hardware acceleration of matrix operations, especially on graphics processing units, or GPUs.

Its progress in parameter space is less erratic than that of stochastic gradient descent, especially with fairly large mini-batches, so it ends up walking around a bit closer to the minimum. But it may find it harder to escape local minima, in problems that have them.

All three end up near the minimum. Batch gradient descent stops at it, while the other two keep walking around, but batch gradient descent takes much longer per step, and the other two would also reach the minimum with a good learning schedule.

Comparison of the algorithms for linear regression, where m is the number of training instances and n is the number of features. The normal equation and SVD are fast with large m, have no out-of-core support, are slow with large n, and have no hyperparameters. Batch gradient descent is slow with large m, has no out-of-core support, is fast with large n, and has two hyperparameters. Stochastic and mini-batch gradient descent are fast with large m, support out-of-core training, are fast with large n, and have two or more hyperparameters.

After training there is almost no difference: all these algorithms end up with very similar models and make predictions in exactly the same way. The normal equation can only perform linear regression, while gradient descent can train many other models.

---

Polynomial regression. A linear model can fit nonlinear data. A simple way is to add powers of each feature as new features, then train a linear model on this extended set of features. This is called polynomial regression.

The book generates two hundred instances from a quadratic equation, one half times x squared, plus x, plus two, plus Gaussian noise. A straight line will never fit this data. Scikit-Learn's polynomial features class, with the degree set to two, adds the square of each feature as a new feature. A linear regression model trained on the extended data estimates coefficients close to the original ones: about one half for the squared term, about one for the linear term, and about two for the bias term.

With multiple features, polynomial regression can find relationships between features, which plain linear regression cannot. This is because the polynomial features class also adds all combinations of features up to the given degree. With features a and b and a degree of three, it adds a squared, a cubed, b squared, and b cubed, and also a times b, a squared times b, and a times b squared.

With n features and degree d, the transformed array has n plus d factorial, divided by d factorial times n factorial, features. Beware of this combinatorial explosion of the number of features.

---

Learning curves. A three-hundred-degree polynomial model on the quadratic data wiggles around to get as close as possible to the training instances. It is severely overfitting, while a linear model is underfitting. The quadratic model generalizes best, which makes sense because the data was generated by a quadratic model. In general the generating function is unknown, so another way is needed to tell whether a model is overfitting or underfitting.

One way is cross-validation. If a model performs well on the training data but generalizes poorly according to the cross-validation metrics, it is overfitting. If it performs poorly on both, it is underfitting.

Another way is learning curves: plots of the model's training error and validation error as a function of the training iteration. The model is evaluated at regular intervals during training on both the training set and the validation set. If the model cannot be trained incrementally, it must be trained several times on gradually larger subsets of the training set.

Scikit-Learn's learning curve function does this using cross-validation, retraining the model on growing subsets of the training set, or training it incrementally if an option is set. It returns the training set sizes, and the training and validation scores for each size and each fold.

For plain linear regression on the quadratic data, the training error starts at zero, because one or two instances can be fit perfectly. As instances are added, the noisy, nonlinear data can no longer be fit perfectly, so the training error rises to a plateau. The validation error starts high, because a model trained on very few instances cannot generalize, then slowly goes down as the model learns. But a straight line cannot model the data well, so it also ends at a plateau, very close to the training curve.

Both curves at a plateau, close together, and fairly high: that is typical of an underfitting model. Adding more training examples will not help an underfitting model. It needs a better model or better features.

For a tenth-degree polynomial model, built as a pipeline of the polynomial features class followed by linear regression, the curves look similar, with two very important differences. The error on the training data is much lower. And there is a gap between the curves: the model performs better on the training data than on the validation data, which is the hallmark of an overfitting model. With a much larger training set, the two curves would continue to get closer. One way to improve an overfitting model is to feed it more training data until the validation error gets close enough to the training error.

---

The bias variance trade-off. A model's generalization error can be expressed as the sum of three very different errors.

Bias is the part due to wrong assumptions, such as assuming the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data. This notion of bias is not the same as the bias term of linear models.

Variance is the part due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom, such as a high-degree polynomial model, is likely to have high variance and to overfit.

Irreducible error is the part due to the noisiness of the data itself. The only way to reduce it is to clean up the data, for example by fixing broken sensors, or by detecting and removing outliers.

Increasing a model's complexity typically increases its variance and reduces its bias. Reducing its complexity, or increasing regularization, increases its bias and reduces its variance. That is why it is called a trade-off.

---

Regularized linear models. A good way to reduce overfitting is to regularize the model, meaning to constrain it: the fewer degrees of freedom it has, the harder it is for it to overfit. A simple way to regularize a polynomial model is to reduce the polynomial degree.

Linear models can be regularized too. Linear regression assumes, among other things, that the input matrix has full rank, meaning the inputs are not colinear and there are at least as many samples as parameters. Inputs are colinear when one input is a linear combination of others, such as temperature in degrees Celsius and in degrees Fahrenheit. Inputs that are close to colinear make linear regression numerically unstable: very small differences in the training set can have a big impact on the trained model. Regularization can stabilize linear models and make them more accurate.

A linear model is usually regularized by constraining its weights. Ridge regression, lasso regression, and elastic net regression are three different ways to do that.

---

Ridge regression. Ridge regression, also called Tikhonov regularization, adds a regularization term to the MSE: alpha divided by m, times the sum of the squares of the feature weights, theta one to theta n. This forces the learning algorithm to fit the data while keeping the weights as small as possible. The model becomes less flexible and cannot stretch itself to fit every data point, which reduces the risk of overfitting.

The regularization term is only used during training; the trained model is evaluated with the unregularized MSE or RMSE. The hyperparameter alpha controls how much the model is regularized. With alpha equal to zero, ridge regression is just linear regression. With a very large alpha, all weights end up very close to zero, and the result is a flat line through the mean of the data.

The bias term, theta zero, is not regularized. If w is the vector of feature weights, the regularization term equals alpha times the square of the L2 norm of w, divided by m. For batch gradient descent, two times alpha times w, divided by m, is added to the part of the MSE gradient vector that corresponds to the feature weights, and nothing is added to the gradient of the bias term.

The data must be scaled, for example with a standard scaler, before ridge regression, because it is sensitive to the scale of the input features. This is true of most regularized models.

With tenth-degree polynomial features, scaled and then fed to ridge, increasing alpha leads to flatter, less extreme, more reasonable predictions, which reduces the model's variance but increases its bias.

Ridge regression can be computed with a closed-form equation or with gradient descent, with the same pros and cons as linear regression. The closed-form solution is the inverse of X transpose X plus alpha times A, multiplied by X transpose y, where A is the identity matrix of size n plus one, except with a zero in the top-left cell, for the bias term. Scikit-Learn's ridge class can solve it with the Cholesky solver, which uses a matrix factorization technique by André-Louis Cholesky.

The SGD regressor with its penalty hyperparameter set to L2 also performs ridge regression. It adds alpha times the square of the L2 norm of the weight vector, without dividing by m, so alpha must be divided by m to match the ridge class.

Scikit-Learn's ridge CV class tunes the hyperparameters automatically with cross-validation, much faster than a general grid search. Lasso CV and elastic net CV are similar.

---

Lasso regression. Least absolute shrinkage and selection operator regression, usually called lasso regression, adds a regularization term that uses the L1 norm of the weight vector instead of the square of the L2 norm. The lasso cost function is the MSE plus two times alpha times the sum of the absolute values of the feature weights. The L1 norm is multiplied by two alpha, while the squared L2 norm in ridge regression is multiplied by alpha divided by m. These factors were chosen so that the optimal alpha does not depend on the training set size.

An important characteristic of lasso regression is that it tends to eliminate the weights of the least important features, setting them to zero. With high-degree polynomial features and alpha equal to zero point zero one, the lasso model looks roughly cubic: all the weights for the high-degree features are zero. So lasso regression automatically performs feature selection and outputs a sparse model, with few nonzero feature weights. If alpha is increased too much, the model becomes very sparse, but its performance plummets.

Why the L1 norm induces sparsity. On the L1 loss alone, the gradient for each weight is always minus one or one, however small the weight, so gradient descent moves every weight toward zero by the same amount. Starting from theta one equal to two and theta two equal to one half, theta two reaches zero first, and the path then rolls down the gutter along that axis. With the full lasso cost, the MSE plus the L1 loss, the path again quickly reaches theta two equal to zero, then ends up bouncing around the global optimum, because the L1 gradients never get close to zero. Increasing alpha moves the optimum along the gutter toward zero.

On the L2 loss alone, gradient descent takes a straight path toward the origin. With the full ridge cost, the gradients get smaller as the parameters approach the optimum, so gradient descent naturally slows down and bounces less, which helps ridge converge faster than lasso. With ridge, the optimal parameters get closer to the origin as alpha increases, but they are never eliminated entirely.

To keep gradient descent from bouncing around the optimum with lasso, the learning rate must be gradually reduced during training.

The lasso cost function is not differentiable where a weight equals zero, but gradient descent still works with a subgradient vector instead. A subgradient at a nondifferentiable point is like an intermediate vector between the gradient vectors around that point. The lasso subgradient is the gradient of the MSE, plus two alpha times the vector of the signs of the weights. The sign of a weight is minus one if it is negative, zero if it is zero, and plus one if it is positive.

Scikit-Learn provides a lasso class, and the SGD regressor with its penalty set to L1 can be used instead.

---

Elastic net regression. Elastic net is a middle ground between ridge and lasso regression. Its regularization term is a weighted sum of the ridge and lasso terms, with a mix ratio r. The cost function is the MSE, plus r times the lasso term, plus one minus r times the ridge term. With r equal to zero, elastic net is equivalent to ridge regression, and with r equal to one, to lasso regression. In Scikit-Learn's elastic net class, the L1 ratio hyperparameter is the mix ratio.

Choosing among them. It is almost always preferable to have at least a little regularization, so plain linear regression should generally be avoided. Ridge is a good default. If only a few features are likely to be useful, lasso or elastic net is better, because they tend to reduce the useless features' weights to zero. Elastic net is generally preferred over lasso, because lasso may behave erratically when there are more features than training instances, or when several features are strongly correlated.

---

Early stopping. Another way to regularize iterative learning algorithms such as gradient descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping.

When a high-degree polynomial model is trained with batch gradient descent, the training and validation errors both go down at first. After a while, the validation error starts to rise: the model has started to overfit. Early stopping stops training when the validation error is at its minimum. It is such a simple and efficient regularization technique that Geoffrey Hinton described it as a free lunch.

Sometimes the validation error comes back down after a while. This is called double descent. It is fairly common with large neural networks and is an area of active research.

With stochastic and mini-batch gradient descent, the curves are not smooth, and it can be hard to know whether the minimum has been reached. One solution is to stop only after the validation error has been above its minimum for some time, then roll the parameters back to the point where the validation error was lowest.

The book's basic implementation trains an unregularized SGD regressor on scaled ninetieth-degree polynomial features, calling partial fit once per epoch. After each epoch it measures the validation RMSE, and whenever it beats the best so far, it saves a copy of the model with Python's deep copy function, which copies both the hyperparameters and the learned parameters. Scikit-Learn's clone function copies only the hyperparameters.

---

Logistic regression. Logistic regression, also called logit regression, is commonly used to estimate the probability that an instance belongs to a particular class, such as the probability that an email is spam. If the estimated probability is greater than a threshold, typically fifty percent, the model predicts the positive class, labeled one. Otherwise it predicts the negative class, labeled zero. This makes it a binary classifier.

Estimating probabilities. Like linear regression, logistic regression computes a weighted sum of the input features plus a bias term. But instead of outputting the result directly, it outputs the logistic of the result. The estimated probability, p hat, equals sigma of theta transpose x.

The logistic function, written sigma, is a sigmoid, meaning S-shaped, function that outputs a number between zero and one. Sigma of t equals one divided by one plus the exponential of minus t.

The prediction is zero if the estimated probability is less than one half, and one if it is one half or more. Sigma of t is less than one half when t is negative, so with the default threshold the model predicts one if theta transpose x is positive, and zero if it is negative.

The score t is often called the logit. The logit function, the log of p divided by one minus p, is the inverse of the logistic function. The logit is also called the log-odds, because it is the log of the ratio between the estimated probability of the positive class and that of the negative class.

---

Training logistic regression. The goal is to set theta so that the model estimates high probabilities for positive instances and low probabilities for negative instances.

The cost for a single training instance is minus the log of p hat if the instance is positive, and minus the log of one minus p hat if it is negative. Minus the log of t grows very large as t approaches zero, and is close to zero when t is close to one. So the cost is large when the model estimates a probability near zero for a positive instance, or near one for a negative instance, and close to zero when the estimate is right.

The cost over the whole training set is the average cost over all training instances, written as a single expression called the log loss: minus the average, over all instances, of the target times the log of p hat, plus one minus the target, times the log of one minus p hat.

Using Bayesian inference, it can be shown that minimizing the log loss gives the model with the maximum likelihood of being optimal, assuming the instances follow a Gaussian distribution around the mean of their class. The more wrong this implicit assumption, the more biased the model. Similarly, the MSE implicitly assumes that the data is purely linear plus Gaussian noise.

There is no known closed-form equation for the theta that minimizes the log loss; there is no equivalent of the normal equation. But the cost function is convex, so gradient descent, or any other optimization algorithm, is guaranteed to find the global minimum, if the learning rate is not too large and it runs long enough.

The partial derivative of the log loss with regard to theta j is the average, over all training instances, of the prediction error multiplied by the jth feature value, where the prediction error is the estimated probability minus the target. It looks very much like the partial derivative for linear regression. The resulting gradient vector works with batch, stochastic, or mini-batch gradient descent.

---

Decision boundaries. The iris dataset is a famous dataset containing the sepal and petal length and width of one hundred fifty iris flowers from three species: Iris setosa, Iris versicolor, and Iris virginica. Scikit-Learn's load iris function loads it. Its instances are not shuffled.

The book trains a logistic regression classifier to detect Iris virginica based only on petal width. Iris virginica petal widths range from one point four to two point five centimeters, and the other irises' from zero point one to one point eight, so there is some overlap. Above about two centimeters, the classifier is highly confident the flower is an Iris virginica. Below one centimeter, it is highly confident it is not. In between, it is unsure. The predict method, as opposed to the predict proba method, returns the most likely class. So there is a decision boundary at about one point six centimeters, where both probabilities equal fifty percent. Above it the classifier predicts Iris virginica, and below it, not Iris virginica, even when it is not very confident.

With two features, petal width and petal length, the decision boundary, where the model estimates a fifty percent probability, is a straight line: the set of points where theta zero, plus theta one times the first feature, plus theta two times the second feature, equals zero. Parallel lines mark other probabilities, from fifteen percent at the bottom left to ninety percent at the top right.

Logistic regression models can be regularized with L1 or L2 penalties, and Scikit-Learn adds an L2 penalty by default. The hyperparameter that controls regularization strength in Scikit-Learn's logistic regression class is not alpha but its inverse, called C. The higher C is, the less the model is regularized.

---

Softmax regression. Logistic regression can be generalized to support multiple classes directly, without training and combining multiple binary classifiers. This is called softmax regression, or multinomial logistic regression.

Given an instance x, the model first computes a score for each class k, then estimates the probability of each class by applying the softmax function, also called the normalized exponential, to the scores. The score for class k is computed just like a linear regression prediction: theta k transpose times x. Each class has its own parameter vector, and all these vectors are typically stored as rows in a parameter matrix, capital theta.

The softmax function computes the exponential of every score, then normalizes them by dividing by the sum of all the exponentials. So the estimated probability of class k is the exponential of class k's score, divided by the sum of the exponentials of all the classes' scores. The scores are generally called logits or log-odds, although they are actually unnormalized log-odds.

The classifier predicts the class with the highest estimated probability, which is simply the class with the highest score. The argmax operator expresses this: it returns the value of k that maximizes the estimated probability.

Softmax regression predicts only one class at a time. It is multiclass, not multioutput, so it should be used only with mutually exclusive classes, such as different species of plants, not to recognize multiple people in one picture.

The training objective is a high estimated probability for the target class, and consequently low probabilities for the other classes. Minimizing the cross entropy cost function achieves this, because it penalizes the model when it estimates a low probability for a target class. The cross entropy cost is minus the average, over all instances, of the sum, over all classes, of the target probability for that class times the log of the estimated probability for that class. The target probability is generally one if the instance belongs to the class, and zero otherwise. With just two classes, the cross entropy is equivalent to the log loss.

Cross entropy originated in Claude Shannon's information theory. Eight weather options can each be sent with three bits. But if it is sunny almost every day, it is more efficient to code sunny on one bit and the other seven options on four bits each. Cross entropy measures the average number of bits actually sent per option. If the assumption about the weather is perfect, cross entropy equals the entropy of the weather itself, its intrinsic unpredictability. If the assumption is wrong, for example if it rains often, cross entropy is greater, by an amount called the Kullback–Leibler divergence, or KL divergence. For discrete distributions p and q, the cross entropy is minus the sum, over all values x, of p of x times the log of q of x.

The gradient of the cross entropy with regard to theta k is the average, over all instances, of the difference between the estimated and target probabilities of class k, multiplied by the instance's feature vector. With the gradient for every class, gradient descent can find the parameter matrix that minimizes the cost.

Scikit-Learn's logistic regression class uses softmax regression automatically when trained on more than two classes, with its default solver, and applies L2 regularization by default, controlled by C. Trained on all three iris classes with petal length and width, and C equal to thirty, it classifies an iris with petals five centimeters long and two centimeters wide as Iris virginica with ninety-six percent probability, and Iris versicolor with four percent.

The decision boundaries between any two classes are linear. The model can predict a class whose estimated probability is below fifty percent: at the point where all decision boundaries meet, all three classes have an equal estimated probability of thirty-three percent.

---

Summary.

Linear regression predicts with a weighted sum of the features plus a bias term, and is trained by minimizing the MSE, either in closed form, with the normal equation or the SVD-based pseudoinverse, or with gradient descent. The closed-form methods are slow when there are many features.

Gradient descent steps against the gradient of the cost function, with a step size set by the learning rate. Batch gradient descent uses the whole training set at each step, stochastic gradient descent one random instance, and mini-batch gradient descent a small random batch. Features should have similar scales, and a learning schedule helps the stochastic variants settle.

Polynomial regression fits nonlinear data by adding powers and combinations of features, then training a linear model on them. Learning curves reveal underfitting, with both curves high and close together, and overfitting, with a gap between a low training error and a higher validation error. Generalization error is the sum of bias, variance, and irreducible error.

Regularization constrains the weights. Ridge regression adds the squared L2 norm of the weights and is a good default. Lasso adds the L1 norm and drives unimportant weights to zero. Elastic net mixes the two. Early stopping halts training when the validation error is lowest.

Logistic regression passes a linear score through the logistic function to estimate the probability of the positive class, and it is trained by minimizing the log loss, which is convex but has no closed-form solution. Softmax regression generalizes it to multiple mutually exclusive classes, with one score per class, the softmax function to turn scores into probabilities, and the cross entropy as its cost function.
