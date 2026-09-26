# Chapter 2 — End-to-End Machine Learning Project

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's exercises for this chapter are all
hands-on coding (listed in exercises.md), so all 20 of this chapter's cards
are here.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Walk through the main steps of an end-to-end machine learning project.

Eight steps, looping back as you learn more:

1. Look at the big picture: frame the problem around the business objective and choose a performance measure.
2. Get the data, take a quick look at its structure, and set aside a test set right away.
3. Explore and visualize the training set: correlations, data quirks, promising attribute combinations.
4. Prepare the data with reusable transformations (cleaning, encoding, feature engineering, scaling), ideally in one pipeline.
5. Select and train models: try several quickly, compare them with cross-validation, and shortlist two to five.
6. Fine-tune the shortlist (hyperparameter search, ensembles), analyze the best models' errors and fairness across subgroups, then evaluate the final model once on the test set.
7. Present the solution: what worked, assumptions, limitations, and reproducible code.
8. Launch, monitor, and maintain: deploy, watch live performance and input quality, retrain regularly, and keep backups for rollback.

## 2. Before writing any code for a new ML project, what should you pin down about the problem, and why?

- The business objective: how the model's output will be used and what the company gains. It drives the framing, the choice of algorithms and performance measure, and how much tuning effort is worth it.
- The current solution, if any: it gives a performance baseline and hints for solving the problem (the housing experts' manual estimates were often off by more than 30%).
- The kind of task: supervised or not, classification or regression, batch or online. Housing is a supervised regression task, multiple (several input features) and univariate (one output per district), and plain batch learning suffices: no continuous data flow, no need to adapt rapidly, and the data fits in memory.
- Your assumptions, checked with whoever consumes the output: if the downstream system only used price categories (cheap, medium, expensive), you'd need a classifier, not a regressor: better to learn that now than after months of work.

## 3. Compare RMSE and MAE as regression performance measures: what does each compute, and when would you prefer MAE?

RMSE(X, y, h) = √( (1/m) ∑ᵢ (h(x⁽ⁱ⁾) − y⁽ⁱ⁾)² ) and MAE(X, y, h) = (1/m) ∑ᵢ |h(x⁽ⁱ⁾) − y⁽ⁱ⁾|, where the sums run over the m instances, x⁽ⁱ⁾ is an instance's feature vector, y⁽ⁱ⁾ its label, and h the model's prediction function (the hypothesis).

Both measure the distance between the vector of predictions and the vector of targets: RMSE corresponds to the Euclidean (ℓ₂) norm and MAE to the Manhattan (ℓ₁) norm. The higher the norm index, the more it focuses on large values and neglects small ones, so RMSE weighs large errors heavily and is more sensitive to outliers. It's the usual default and works very well when outliers are exponentially rare, as in a bell-shaped distribution; when the data has many outliers, prefer MAE.

## 4. Why should you set aside the test set before exploring the data, and how can you keep the split stable across runs and dataset updates?

Your brain is a powerful pattern detector, so it overfits too: patterns you notice in the test set can steer your choice of model, so your later test-set estimate of the generalization error is too optimistic. This is data snooping bias.

A plain random split changes on every run, so over time you'd see the whole dataset. Fixing the random seed or saving the test set solves that, but both break when you fetch an updated dataset. Instead, hash each instance's unique, immutable ID and put the instance in the test set if the hash falls in the lowest 20% of the hash range. New instances are split in the same proportion, and no former training instance ever moves into the test set:

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32
```

With no ID column, use the row index (if new rows are only appended, never deleted) or build an ID from stable features.

## 5. What is stratified sampling, and how do you stratify a train/test split on a continuous attribute such as median income?

When the dataset isn't large, especially relative to the number of attributes, a purely random split can produce a skewed test set. Stratified sampling divides the population into homogeneous subgroups (strata) and draws the right number of instances from each, so the test set keeps the full dataset's proportions for an attribute you know matters.

A continuous attribute must first be binned into categories, with few enough strata that each has plenty of instances. Then pass those categories as stratify:

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)
```

For several stratified splits, StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42) has a split(X, strata) method that yields train and test indices, not the data itself. Drop the helper column afterwards.

## 6. What does corr() tell you when exploring a dataset, and what are the blind spots of the standard correlation coefficient?

housing.corr(numeric_only=True) computes Pearson's r between every pair of numerical attributes; sorting the target's column shows which features track it most (median income, at about 0.69, stood out). Values near 1 mean a strong positive linear correlation, near −1 a strong negative one, and near 0 no linear correlation.

Blind spots:
- It only measures linear correlation: r ≈ 0 can hide a strong nonlinear relationship.
- It says nothing about slope: height in inches and height in centimeters have r = 1, as does any variable paired with a positively rescaled copy of itself.

So also plot the most promising pairs, with Pandas' scatter_matrix() or a single scatterplot. The income-versus-value plot also revealed quirks worth cleaning up: a horizontal line at the 500,000 USD price cap and less obvious lines at a few lower values.

## 7. What are your options for handling missing values, and why use Scikit-Learn's SimpleImputer rather than Pandas' fillna()?

You can drop the affected instances (dropna()), drop the whole attribute (drop()), or impute, i.e., replace missing values with something like the median (fillna()). Imputation is the least destructive.

SimpleImputer(strategy="median") learns each column's median in fit() and stores the medians in statistics_, so transform() applies the same values to the validation set, the test set, and new data in production, and it fits into a pipeline. Apply it to every numerical column, not just those with gaps today, since live data could be missing anything. The median needs numbers, so first select them with housing.select_dtypes(include=[np.number]). Other strategies are "mean", "most_frequent", and "constant" with fill_value; the last two also work on non-numerical data. For numerical features, KNNImputer (mean of the k nearest neighbors' values) and IterativeImputer (a regression model per feature, refined over several rounds) are more powerful.

## 8. Describe Scikit-Learn's core object types and the conventions that make estimators easy to inspect.

- Estimators learn from data with fit(X), or fit(X, y) for supervised learning. Anything else that guides the estimation is a hyperparameter, set in the constructor and stored as an instance variable (like SimpleImputer's strategy).
- Transformers are estimators that can also transform(X), usually using what fit() learned; fit_transform() is equivalent to fit() then transform(), and sometimes faster.
- Predictors are estimators with predict(X) for new instances and score() to measure prediction quality (R² for regressors, accuracy for classifiers).

Hyperparameters are public attributes (imputer.strategy), and learned parameters are public attributes ending with an underscore (imputer.statistics_). Datasets are NumPy arrays or SciPy sparse matrices, and hyperparameters are plain strings or numbers, not custom classes. Estimators compose, for example into pipelines, and come with sensible defaults. Gotcha: transformers output NumPy arrays even when fed DataFrames, unless you call sklearn.set_config(transform_output="pandas").

## 9. How should you encode a categorical attribute like ocean_proximity, and why is OneHotEncoder safer than pd.get_dummies() in production?

OrdinalEncoder maps each category to an integer, but models then assume nearby numbers are similar, which only makes sense for ordered categories such as bad, average, good, excellent. For unordered categories, one-hot encoding creates one binary feature per category, exactly one of which is 1.

OneHotEncoder learns the categories in fit() (categories_) and always outputs one column per learned category, in the same order, so production data gets exactly the features the model was trained on. pd.get_dummies() only creates columns for the categories present in whatever data it's given: a batch with fewer categories yields fewer columns, and an unknown category yields a new one. OneHotEncoder raises an error on unknown categories by default, or encodes them as all zeros with handle_unknown="ignore". Its output is a SciPy sparse matrix by default (sparse_output=False gives a dense array), and get_feature_names_out() names the columns.

## 10. Compare min-max scaling and standardization, and explain the rule for fitting scalers.

Most models perform badly when numerical features have very different scales (total rooms: about 6 to 39,320; median income: 0 to 15).

- Min-max scaling (MinMaxScaler, often called normalization): x′ = (x − min) / (max − min), which maps training values to 0–1, or to another range via feature_range, e.g., (-1, 1) for neural networks, which work best with zero-mean inputs.
- Standardization (StandardScaler): x′ = (x − μ) / σ, giving zero mean and unit standard deviation. Values aren't bounded to a range, but outliers affect it much less: a single erroneous income of 100 would squash every other min-max-scaled income into 0–0.15.

Fit a scaler (fit() or fit_transform()) on the training set only, then use its transform() on the validation set, the test set, and new data. New outliers can land outside the min-max range unless you set clip=True. For sparse input, StandardScaler(with_mean=False) preserves sparsity.

## 11. How can you transform heavy-tailed or multimodal features so models can make better use of them?

Scaling a heavy-tailed feature would squash most values into a small range, so shrink the tail first. For a positive feature with a long right tail, take its square root (or a power between 0 and 1); for a really long tail like a power law (population), take the logarithm, which can bring it close to Gaussian. Alternatively, bucketize it into roughly equal-frequency buckets (e.g., replace each value with its percentile), giving a nearly uniform feature that needs no further scaling.

For a multimodal feature (several peaks, like housing_median_age):
- Bucketize it and one-hot encode the bucket IDs as categories, so the model can learn different rules for different ranges.
- Or add a similarity feature per main mode with a Gaussian radial basis function: exp(−γ(x − 35)²) equals 1 at x = 35 and decays as x moves away, faster for larger γ. rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1) computes it.

## 12. What are the two ways to write a custom transformer in Scikit-Learn, and what rules must a custom transformer class follow?

If nothing needs to be learned, wrap a function: FunctionTransformer(np.log, inverse_func=np.exp). kw_args passes extra arguments, feature_names_out ("one-to-one" or a callable) makes get_feature_names_out() work, and inverse_func makes it reversible, e.g., for use in a TransformedTargetRegressor.

If it must learn something in fit(), write a class. Scikit-Learn relies on duck typing, but TransformerMixin gives you fit_transform() for free, and BaseEstimator gives you get_params() and set_params(), which hyperparameter search needs, provided the constructor has no variable-length argument lists. The constructor just stores each hyperparameter under the same name; fit() accepts y=None even if unused, sets learned attributes ending in an underscore (plus n_features_in_), and returns self. This one finds k-means clusters and outputs each instance's RBF similarity to every cluster center:

```python
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        self.n_features_in_ = X.shape[1]
        return self  # always return self
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
```

## 13. How does a Scikit-Learn Pipeline work: what can its steps be, and what happens when you call fit() and predict()?

Pipeline takes a list of (name, estimator) pairs; names must be unique and must not contain double underscores. Every step but the last must be a transformer (have fit_transform()); the last can be any estimator. make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) builds one without explicit names, using the lowercased class names, like "simpleimputer".

fit() calls fit_transform() on each transformer in turn, feeding each output to the next step, then calls fit() on the final estimator. The pipeline exposes the final estimator's methods: with a predictor last, predict() pushes new data through the already-fitted transformers (transform() only) and then calls the model's predict(); with a transformer last, the pipeline is itself a transformer. You can index it (pipe[1], or pipe[:-1] for a sub-pipeline), look steps up by name (pipe["simpleimputer"]), or use steps and named_steps. Bundling preprocessing with the model means they're cross-validated, tuned, and saved together.

## 14. What does ColumnTransformer do, and how do you tell it which columns each transformer should handle?

It applies different transformers to different columns and concatenates their outputs side by side (transformers must never change the number of rows). Its constructor takes (name, transformer, columns) triplets:

```python
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
], remainder="drop")
```

Columns can be names, indices, or a selector such as make_column_selector(dtype_include=np.number). Instead of a transformer you can pass "drop" or "passthrough". Unlisted columns are dropped by default; set remainder to a transformer or "passthrough" to keep them. make_column_transformer() names the transformers for you, like make_pipeline(). When some outputs are sparse, the result is a sparse matrix only if its overall density (ratio of nonzero cells) is below sparse_threshold, 0.3 by default. get_feature_names_out() prefixes each output column with its transformer's name, as in "cat__ocean_proximity_INLAND".

## 15. A decision tree scores an RMSE of 0 on its training set. What does that tell you, and how do you evaluate it properly with k-fold cross-validation?

A perfect training score almost certainly means the model badly overfits, and you shouldn't touch the test set until you're ready to launch. k-fold cross-validation splits the training set into k non-overlapping folds and trains k times, each time evaluating on a different fold after training on the other k − 1:

```python
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
```

Scikit-Learn's cross-validation expects a utility function (greater is better), so the scorer returns negative RMSEs and you flip the sign. You get k scores, hence a mean and a standard deviation that tells you how precise the estimate is, which a single validation set can't. Here the tree's mean validation RMSE was about 66,600: zero training error but high validation error confirms overfitting. The price is k training runs.

## 16. How do you grid-search the hyperparameters of a pipeline that bundles preprocessing and a model, and how do you read the results?

Say full_pipeline has a "preprocessing" step (a ColumnTransformer containing a transformer named "geo") followed by a "random_forest" step. GridSearchCV takes the pipeline and a param_grid (a dict, or a list of dicts explored one after the other); nested hyperparameters are named by joining names with double underscores, so "preprocessing__geo__n_clusters" is the n_clusters of geo inside preprocessing:

```python
param_grid = [
    {"preprocessing__geo__n_clusters": [5, 8, 10],
     "random_forest__max_features": [4, 6, 8]},
    {"preprocessing__geo__n_clusters": [10, 15],
     "random_forest__max_features": [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_labels)
```

That's (3×3 + 2×3) = 15 combinations × 3 folds = 45 training runs. Tuning preprocessing together with the model matters because their hyperparameters often interact. Afterwards, best_params_ holds the winning combination, best_estimator_ the best pipeline retrained on the full training set (refit=True by default), and cv_results_ every score (wrap it in a DataFrame). If the best value sits at the edge of the grid, search beyond it.

## 17. When and why is RandomizedSearchCV preferable to GridSearchCV?

Grid search tries every combination you list. Randomized search evaluates a fixed number of combinations (n_iter), sampling a value for each hyperparameter at every iteration, which pays off in large search spaces:
- Continuous or finely grained hyperparameters get many distinct values tried (1,000 iterations explore 1,000 values each), not just a few listed ones.
- A hyperparameter that turns out not to matter costs nothing extra, whereas adding its 10 values to a grid makes the search 10 times longer.
- You set the budget: with 6 hyperparameters of 10 values each, a grid means a million combinations.

Each hyperparameter gets a list of values or a distribution, e.g., param_distributions={"random_forest__max_features": randint(low=2, high=20)} with scipy.stats.randint. HalvingRandomSearchCV and HalvingGridSearchCV go further: they train many candidates with limited resources (by default, a small part of the training set), keep the best, and give the survivors more resources each round.

## 18. How should you evaluate your final model on the test set, and what must you resist doing afterwards?

Evaluate once, at the very end: split the test set into predictors and labels, call final_model.predict() (the pipeline reuses the preprocessing fitted on the training set), and compute the RMSE. A point estimate may not be convincing, say if it's barely better than the current model, so also compute a 95% confidence interval, for example by bootstrapping the squared errors:

```python
from scipy import stats

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
                              confidence_level=0.95, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
```

After heavy hyperparameter tuning, the test score is usually a bit worse than the cross-validation score, because the system ended up tuned to the validation data. Don't tweak hyperparameters to make the test number look better: those gains are unlikely to generalize to new data.

## 19. How do you save a trained model and serve it in production, and what are the main deployment options?

Save the whole pipeline, preprocessing included, with joblib.dump(final_model, "my_california_housing_model.pkl"), and load it in production with joblib.load(). The file doesn't contain your code, so every custom function and class the pipeline uses (column_ratio, ClusterSimilarity, and so on) must be defined or imported in the production code before loading.

Deployment options:
- Load the model inside your web application, once at server startup rather than on every request, and call predict().
- Wrap it in a dedicated web service that the application queries through a REST API: you can upgrade the model without interrupting the app, scale by load-balancing across several instances, and write the app in any language.
- Use a cloud platform such as Google's Vertex AI: upload the saved model to Google Cloud Storage and create a model version, which gives you a scalable prediction service taking and returning JSON.

## 20. Once a model is deployed, what does monitoring and maintaining it involve?

- Monitor live performance and alert on drops: sudden ones from broken infrastructure, and slow decay from data drift, which can go unnoticed for long. Use downstream metrics where possible (e.g., sales of recommended products); otherwise have human raters (experts, crowd workers, even users) label a sample of predictions, especially uncertain ones.
- Monitor input quality: alert when more inputs are missing a feature, when a feature's mean or standard deviation drifts from the training set's, or when a categorical feature shows new categories. This may catch problems earlier.
- Automate retraining: collect and label fresh data, retrain and fine-tune on a schedule, and deploy the new model only if it does at least as well as the old one on the updated test set, including on subsets such as rich versus poor districts.
- Keep backups of every model and dataset version, for quick rollback and comparison.
