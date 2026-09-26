# Chapter 4 — Training Models

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. State the normal equation for linear regression, and explain why Scikit-Learn's LinearRegression computes a pseudoinverse instead of solving it directly.

θ̂ = (XᵀX)⁻¹Xᵀy gives, in closed form, the parameter vector that minimizes the MSE. X is the m × (n + 1) matrix of training instances with an extra column of 1s for the bias term, and y is the vector of targets. In NumPy, with X_b = add_dummy_feature(X): np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y.

LinearRegression is based on scipy.linalg.lstsq(), which computes θ̂ = X⁺y, where X⁺ is the Moore–Penrose pseudoinverse (np.linalg.pinv() computes it directly). X⁺ comes from the singular value decomposition X = UΣVᵀ: X⁺ = VΣ⁺Uᵀ, where Σ⁺ is Σ with values below a tiny threshold set to zero, the others inverted, and the result transposed. This is more efficient, and X⁺ is always defined, whereas XᵀX can't be inverted when m < n or some features are redundant. After fitting, the bias is in intercept_ and the feature weights are in coef_.

## 2. Give the gradient of the MSE cost for linear regression in vectorized form, and sketch batch gradient descent in NumPy.

∇θMSE(θ) = (2/m)·Xᵀ(Xθ − y): for each parameter θⱼ, the prediction errors multiplied by feature j, averaged over all m instances and doubled (X includes the bias column of 1s). The gradient points uphill, so each step goes the opposite way: θ ← θ − η·∇θMSE(θ), where η is the learning rate.

```python
eta, n_epochs = 0.1, 1000
m = len(X_b)                          # X_b = X plus a bias column of 1s
rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))   # random init: bias + 1 weight
for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

It's called batch GD because every step uses the whole training set. Rather than guessing n_epochs, you can set it very high and stop as soon as the gradient vector's norm drops below a small tolerance ε, which means you're essentially at the minimum.

## 3. How does polynomial regression let a linear model fit nonlinear data, and what exactly does PolynomialFeatures add?

It's plain linear regression trained on an expanded feature set, so the model stays linear in its parameters but becomes nonlinear in the original inputs. PolynomialFeatures(degree=d) adds the powers of each feature and all products of features up to degree d: with features a and b and degree=3, it adds a², ab, b², a³, a²b, ab² and b³. Those cross terms let the model capture interactions between features, which plain linear regression can't. Set include_bias=False when the next estimator fits its own intercept; otherwise it also adds a column of 1s. Beware the growth: n features become (n + d)! / (d!·n!) columns (counting that bias column), so high degrees explode in size and overfit easily.

```python
poly_reg = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                         LinearRegression())
```

## 4. What are the three components of a model's generalization error in the bias/variance trade-off, and how does model complexity affect them?

- Bias: error from wrong assumptions, such as assuming the data is linear when it's actually quadratic. A high-bias model is likely to underfit.
- Variance: error from the model being overly sensitive to small variations in the training data, typical of models with many degrees of freedom such as a high-degree polynomial. A high-variance model is likely to overfit.
- Irreducible error: the noise in the data itself. Only cleaning up the data (fixing broken sensors, removing outliers) reduces it.

Making a model more complex usually lowers its bias but raises its variance; simplifying it or regularizing it more does the reverse. Hence the trade-off: you're looking for the complexity where the total error is lowest. (This bias has nothing to do with a linear model's bias term.)

## 5. Write the ridge and lasso cost functions, and explain why lasso drives some weights exactly to zero while ridge only shrinks them.

- Ridge: J(θ) = MSE(θ) + (α/m)·∑θᵢ²
- Lasso: J(θ) = MSE(θ) + 2α·∑|θᵢ|

Both sums run from i = 1 to n, so the bias term θ₀ isn't penalized, and α sets the strength. The penalty is only used during training; evaluate the model with the unregularized error.

The difference lies in the penalty's gradient. Ridge adds 2αθᵢ/m, which fades as a weight approaches zero, so weights get small but rarely reach it. Lasso pushes every nonzero weight toward zero with the same force, 2α·sign(θᵢ), however small the weight already is, so the weights of features that barely help reduce the MSE get driven all the way to 0. The result is automatic feature selection and a sparse model. Since the lasso cost isn't differentiable at 0, gradient descent uses a subgradient there, and it keeps bouncing around the optimum unless the learning rate is gradually reduced.

## 6. How do you set the regularization strength in Scikit-Learn's Ridge, Lasso, ElasticNet, SGDRegressor and LogisticRegression, and what are the gotchas?

- Ridge(alpha=...) and Lasso(alpha=...): a higher alpha means stronger regularization. Ridge can also use a closed-form solver, e.g., solver="cholesky".
- ElasticNet(alpha=..., l1_ratio=...): l1_ratio is the mix ratio r between the two penalties, 0 for pure ridge and 1 for pure lasso.
- SGDRegressor(penalty="l2", alpha=...): its ℓ2 term isn't divided by m, so matching Ridge(alpha=0.1) takes alpha=0.1 / m. penalty="l1" with the same alpha as Lasso gives lasso, and penalty=None turns regularization off.
- LogisticRegression(C=...): C is the inverse of the regularization strength, so a higher C means less regularization. It applies an ℓ2 penalty by default.

Scale the features first (e.g., with StandardScaler), since these penalties are sensitive to feature scales, and use RidgeCV, LassoCV or ElasticNetCV to tune alpha quickly with built-in cross-validation.

## 7. Sketch a basic early-stopping loop around an SGDRegressor, and explain why it uses partial_fit() and copy.deepcopy().

Train one epoch at a time, measure the validation RMSE, and keep a copy of the best model seen so far (the inputs here are already expanded and scaled):

```python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
best_valid_rmse = float("inf")
for epoch in range(500):
    sgd_reg.partial_fit(X_train_prep, y_train)
    val_error = root_mean_squared_error(y_valid, sgd_reg.predict(X_valid_prep))
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

partial_fit() runs a single round of training and continues where the previous call left off, whereas fit() would restart training and reset the learning schedule's iteration counter. deepcopy() copies the learned parameters as well as the hyperparameters; sklearn.base.clone() copies only the hyperparameters, giving an untrained model. This loop never actually stops early: it trains for all epochs and lets you roll back to best_model afterward.

## 8. Walk through logistic regression: how it estimates a probability, how it turns that into a class, and what cost function trains it.

It computes a linear score t = θᵀx, called the logit, and passes it through the logistic (sigmoid) function σ(t) = 1 / (1 + exp(−t)), so p̂ = σ(θᵀx) lies between 0 and 1. With the default 50% threshold, it predicts the positive class exactly when θᵀx ≥ 0, so the decision boundary is linear. The logit is also called the log-odds, since t = log(p̂ / (1 − p̂)).

Training minimizes the log loss over the m training instances: J(θ) = −(1/m)·∑[y·log(p̂) + (1 − y)·log(1 − p̂)]. A positive instance costs −log(p̂) and a negative one −log(1 − p̂); both blow up as the prediction becomes confidently wrong, while confidently right predictions cost almost nothing. There's no closed-form solution, so it's trained iteratively, e.g., by gradient descent with ∇θJ(θ) = (1/m)·Xᵀ(σ(Xθ) − y): the same shape as the MSE gradient, prediction errors times inputs.

## 9. How does softmax regression compute class probabilities, and what cost function and gradient are used to train it?

Each class k has its own parameter vector θ⁽ᵏ⁾ (stored as the rows of a matrix Θ) and gets a score sₖ(x) = θ⁽ᵏ⁾ᵀx. The softmax function exponentiates the scores and normalizes them, p̂ₖ = exp(sₖ(x)) / ∑ⱼ exp(sⱼ(x)), so the probabilities are positive and sum to 1. The prediction is the argmax, which is simply the class with the highest score.

Training minimizes the cross entropy J(Θ) = −(1/m)·∑ᵢ∑ₖ yₖ⁽ⁱ⁾·log(p̂ₖ⁽ⁱ⁾), where yₖ⁽ⁱ⁾ is 1 if instance i belongs to class k and 0 otherwise, so it penalizes a low probability for the target class. With two classes it reduces to the log loss. The gradient for class k is ∇θ⁽ᵏ⁾J(Θ) = (1/m)·∑ᵢ(p̂ₖ⁽ⁱ⁾ − yₖ⁽ⁱ⁾)·x⁽ⁱ⁾. Scikit-Learn's LogisticRegression uses softmax regression automatically when trained on more than two classes (with its default lbfgs solver).
