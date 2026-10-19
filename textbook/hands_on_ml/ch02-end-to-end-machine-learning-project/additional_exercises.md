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

- The business objective: how the model's output will be used and what the company gains. It drives the framing, the algorithm and performance measure, and how much tuning is worth it.
- The current solution, if any: it gives a baseline to beat and hints for solving the problem (the housing experts' manual estimates were often off by more than 30%).
- The kind of task: supervised or not, classification or regression, batch or online. Housing is supervised, multiple (several input features), univariate (one output per district) regression, and plain batch learning is enough: no continuous data flow, no need to adapt rapidly, and the data fits in memory.
- Your assumptions, checked with whoever consumes the output: if the downstream system only used price categories (cheap, medium, expensive), you'd need a classifier, not a regressor. Better to learn that now than after months of work.

## 3. Compare RMSE and MAE as regression performance measures: what does each compute, and when would you prefer MAE?

MAE is the mean absolute error: average |prediction − target| over all instances. RMSE is the root mean squared error: square each error, average, then take the square root. Both are in the same units as the target.

The difference is how they treat big errors. RMSE squares each error first, so one large error counts far more than several small ones. Example: errors of 1, 1, 1, 1, 10 give MAE = 2.8 but RMSE ≈ 4.56; the single outlier dominates RMSE. (Technically RMSE is the Euclidean ℓ₂ norm of the error vector and MAE is the Manhattan ℓ₁ norm.)

RMSE is the usual default and works very well when outliers are rare, as in a bell-shaped distribution. Prefer MAE when the data has many outliers, so a few extreme errors don't dominate the score.

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

Stratified sampling divides the population into homogeneous subgroups (strata) and draws the right number of instances from each, so the test set keeps the full dataset's proportions for an attribute you know matters. It helps when the dataset is small relative to its number of attributes, where a purely random split can be skewed.

A continuous attribute must first be binned into a modest number of categories (so each stratum has plenty of instances), then passed as stratify:

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)
```

Drop the helper column (income_cat) afterwards.

## 6. What does corr() tell you when exploring a dataset, and what are the blind spots of the standard correlation coefficient?

housing.corr(numeric_only=True) computes Pearson's r between every pair of numerical attributes. Sort the target's column to see which features track it most. Near 1 means a strong positive linear correlation, near −1 a strong negative one, and near 0 no linear correlation.

Blind spots:
- It only measures linear correlation: r ≈ 0 can hide a strong nonlinear relationship (a U shape, for example).
- It says nothing about slope: any variable paired with a positively rescaled copy of itself has r = 1 (height in inches vs. centimeters).

So also plot the most promising pairs, with Pandas' scatter_matrix() or a single scatterplot. Plots reveal quirks that r hides, such as a price cap showing up as a horizontal line.

## 7. What are your options for handling missing values, and why use Scikit-Learn's SimpleImputer rather than Pandas' fillna()?

Three options: drop the affected instances (dropna()), drop the whole attribute (drop()), or impute, i.e., replace missing values with something like the median (fillna()). Imputation is the least destructive.

SimpleImputer(strategy="median") beats fillna() because it learns each column's median in fit() (stored in statistics_), and transform() then applies those same values to the validation set, the test set, and new data in production. It also fits into a pipeline. Apply it to every numerical column, not just those with gaps today, since live data could be missing anything.

The median needs numbers, so select them first with housing.select_dtypes(include=[np.number]). Other strategies are "mean", "most_frequent" and "constant". KNNImputer and IterativeImputer are more powerful options for numerical features.

## 8. Describe Scikit-Learn's core object types and the conventions that make estimators easy to inspect.

- Estimators learn from data with fit(X), or fit(X, y) for supervised learning. Anything else that guides the learning is a hyperparameter, set in the constructor (like SimpleImputer's strategy).
- Transformers are estimators that also have transform(X), using what fit() learned. fit_transform() is fit() then transform(), sometimes faster.
- Predictors are estimators with predict(X) for new instances and score() to measure quality (R² for regressors, accuracy for classifiers).

Conventions for inspecting them: hyperparameters are public attributes (imputer.strategy), and learned parameters are public attributes ending with an underscore (imputer.statistics_).

Gotcha: transformers output NumPy arrays even when fed DataFrames, unless you call sklearn.set_config(transform_output="pandas").

## 9. How should you encode a categorical attribute like ocean_proximity, and why is OneHotEncoder safer than pd.get_dummies() in production?

For unordered categories, use one-hot encoding: one binary feature per category, exactly one of which is 1. OrdinalEncoder (category to integer) only makes sense for ordered categories such as bad, average, good, excellent, because models assume nearby numbers are similar.

OneHotEncoder is safer in production because it learns the categories in fit() (categories_) and always outputs one column per learned category, in the same order, so production data gets exactly the features the model was trained on. pd.get_dummies() only creates columns for the categories present in the data it's given: a batch with fewer categories yields fewer columns, and an unknown category yields a new one. OneHotEncoder raises an error on an unknown category by default, or encodes it as all zeros with handle_unknown="ignore".

Its output is a sparse matrix by default (sparse_output=False gives a dense array).

## 10. Compare min-max scaling and standardization, and explain the rule for fitting scalers.

Most models perform badly when numerical features have very different scales (total rooms: about 6 to 39,320; median income: 0 to 15). Two fixes:

- Min-max scaling (MinMaxScaler, often called normalization): x′ = (x − min) / (max − min), which maps values to 0–1. feature_range changes that, e.g., (-1, 1) for neural networks.
- Standardization (StandardScaler): x′ = (x − μ) / σ, giving zero mean and unit standard deviation. Values aren't bounded, but outliers hurt much less: a single erroneous income of 100 would squash every other min-max-scaled income into 0–0.15.

The rule: fit the scaler on the training set only, then use its transform() on the validation set, the test set, and new data.

## 11. How can you transform heavy-tailed or multimodal features so models can make better use of them?

Scaling a heavy-tailed feature would squash most values into a small range, so shrink the tail first:
- Long right tail: take the square root (or a power between 0 and 1).
- Really long tail, like a power law (population): take the logarithm, which can bring it close to Gaussian.
- Or bucketize into equal-frequency buckets (replace each value with its percentile), which gives a nearly uniform feature that needs no further scaling.

For a multimodal feature (several peaks, like housing_median_age):
- Bucketize it and one-hot encode the bucket IDs, so the model can learn different rules for different ranges.
- Or add a similarity feature per main mode using a Gaussian radial basis function (RBF): exp(−γ(x − 35)²) equals 1 at x = 35 and decays as x moves away, faster for larger γ. In Scikit-Learn: rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1).

## 12. What are the two ways to write a custom transformer in Scikit-Learn, and what rules must a custom transformer class follow?

1. If nothing needs to be learned, wrap a function: FunctionTransformer(np.log, inverse_func=np.exp). inverse_func makes it reversible (e.g., for a TransformedTargetRegressor), and feature_names_out="one-to-one" makes get_feature_names_out() work.
2. If it must learn something in fit(), write a class.

Class rules: inherit TransformerMixin (gives fit_transform() for free) and BaseEstimator (gives get_params() and set_params(), which hyperparameter search needs; the constructor must not take *args or **kwargs). The constructor just stores each hyperparameter under the same name. fit() takes y=None even if unused, sets learned attributes ending in an underscore, and returns self.

This one finds k-means clusters and outputs each instance's RBF similarity to every cluster center:

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

Pipeline takes a list of (name, estimator) pairs. Every step but the last must be a transformer; the last can be any estimator. Names must be unique and must not contain double underscores. make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) skips the names and uses lowercased class names, like "simpleimputer".

fit() calls fit_transform() on each transformer in turn, feeding each output to the next step, then calls fit() on the final estimator. predict() pushes new data through the already-fitted transformers (transform() only), then calls the model's predict(). If the last step is a transformer, the pipeline is itself a transformer.

You can get at steps by index (pipe[1]), slice (pipe[:-1]), or name (pipe["simpleimputer"]). Bundling preprocessing with the model means they're cross-validated, tuned, and saved together.

## 14. What does ColumnTransformer do, and how do you tell it which columns each transformer should handle?

It applies different transformers to different columns and concatenates their outputs side by side (transformers must keep the same number of rows). Its constructor takes (name, transformer, columns) triplets:

```python
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

Columns can be names, indices, or a selector such as make_column_selector(dtype_include=np.number). Instead of a transformer you can pass "drop" or "passthrough". Unlisted columns are dropped by default; set remainder="passthrough" (or a transformer) to keep them.

make_column_transformer() names the transformers for you. get_feature_names_out() prefixes each output column with its transformer's name, as in "cat__ocean_proximity_INLAND".

## 15. A decision tree scores an RMSE of 0 on its training set. What does that tell you, and how do you evaluate it properly with k-fold cross-validation?

A perfect training score almost certainly means the model badly overfits. Don't touch the test set to check; use k-fold cross-validation on the training set instead. It splits the training set into k folds and trains k times, each time evaluating on a different fold after training on the other k − 1:

```python
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
```

Scikit-Learn's cross-validation expects a utility function (greater is better), so the scorer returns negative RMSEs and you flip the sign. You get k scores, hence a mean and a standard deviation that tells you how precise the estimate is, which a single validation set can't. Here the tree's mean validation RMSE was about 66,600: zero training error but high validation error confirms overfitting. The price is k training runs.

## 16. How do you grid-search the hyperparameters of a pipeline that bundles preprocessing and a model, and how do you read the results?

Say full_pipeline has a "preprocessing" step (a ColumnTransformer containing a transformer named "geo") followed by a "random_forest" step. Pass GridSearchCV the pipeline and a param_grid (a dict, or a list of dicts explored one after the other). Nested hyperparameters are named by joining names with double underscores, so "preprocessing__geo__n_clusters" is the n_clusters of geo inside preprocessing:

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

That's (3×3 + 2×3) = 15 combinations × 3 folds = 45 training runs. Tuning preprocessing together with the model matters because their hyperparameters interact.

To read the results: best_params_ is the winning combination, best_estimator_ the best pipeline retrained on the full training set (refit=True by default), and cv_results_ every score (wrap it in a DataFrame). If the best value sits at the edge of the grid, search beyond it.

## 17. When and why is RandomizedSearchCV preferable to GridSearchCV?

Grid search tries every combination you list. Randomized search evaluates a fixed number of combinations (n_iter), sampling a value for each hyperparameter at every iteration. That pays off in large search spaces:
- Same budget, more distinct values: a 3×3 grid is 9 runs but only 3 values of each hyperparameter. 9 random iterations try up to 9 values of each, because every iteration samples each hyperparameter independently. (A grid can list more values, but its cost multiplies across hyperparameters.)
- Unimportant hyperparameters don't inflate the cost: adding one with 10 values to a grid makes the search 10 times longer, mostly repeating the same effective model. In randomized search the cost stays n_iter no matter how many hyperparameters you add, and the important ones still get n_iter distinct values.
- You set the budget: with 6 hyperparameters of 10 values each, a grid means a million combinations.

Give each hyperparameter a list of values or a distribution, e.g., param_distributions={"random_forest__max_features": randint(low=2, high=20)}, with randint from scipy.stats.

HalvingRandomSearchCV and HalvingGridSearchCV go further: they train many candidates on a small part of the training set, keep the best, and give the survivors more resources each round.

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

After heavy hyperparameter tuning, the test score is usually a bit worse than the cross-validation score, because the system ended up tuned to the validation data. Resist tweaking hyperparameters to make the test number look better: those gains are unlikely to generalize to new data.

## 19. How do you save a trained model and serve it in production, and what are the main deployment options?

Save the whole pipeline, preprocessing included, with joblib.dump(final_model, "my_california_housing_model.pkl"), and load it in production with joblib.load(). The file doesn't contain your code, so every custom function and class the pipeline uses (column_ratio, ClusterSimilarity, and so on) must be defined or imported in the production code before loading.

Deployment options:
- Load the model inside your web application, once at server startup rather than on every request, and call predict().
- Wrap it in a web service that the application queries through a REST API: you can upgrade the model without interrupting the app, load-balance across several instances, and write the app in any language.
- Use a cloud platform such as Google's Vertex AI: upload the saved model to Google Cloud Storage and create a model version, which gives you a scalable prediction service taking and returning JSON.

## 20. Once a model is deployed, what does monitoring and maintaining it involve?

- Monitor live performance and alert on drops: sudden ones from broken infrastructure, and slow decay from data drift, which can go unnoticed for long. Use downstream metrics where possible (e.g., sales of recommended products); otherwise have human raters (experts, crowd workers, even users) label a sample of predictions, especially uncertain ones.
- Monitor input quality: alert when more inputs are missing a feature, when a feature's mean or standard deviation drifts from the training set's, or when a categorical feature shows new categories. This may catch problems earlier.
- Automate retraining: collect and label fresh data, retrain and fine-tune on a schedule, and deploy the new model only if it does at least as well as the old one on the updated test set, including on subsets such as rich versus poor districts.
- Keep backups of every model and dataset version, for quick rollback and comparison.
