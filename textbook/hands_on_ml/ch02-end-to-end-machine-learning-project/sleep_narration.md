# Chapter 2 — End-to-End Machine Learning Project, to fall asleep to

Chapter two. End-to-end machine learning project.

---

The chapter follows one example project from start to finish, from the point of view of a newly hired data scientist at a fictitious real estate company. The goal is to show the main steps of a typical machine learning project.

There are eight main steps. Look at the big picture. Get the data. Explore and visualize the data to gain insights. Prepare the data for machine learning algorithms. Select a model and train it. Fine-tune the model. Present the solution. Launch, monitor, and maintain the system.

It is best to learn with real-world data from open repositories, such as OpenML, Kaggle, or Hugging Face Datasets, rather than artificial datasets.

The project uses the California Housing Prices dataset, based on the 1990 California census. It is not recent, but it has many qualities for learning. For teaching purposes, a categorical attribute was added and a few features were removed.

---

Framing the problem. The task is to build a model of housing prices in California from census data. The data includes metrics such as population, median income, and median housing price for each block group, the smallest geographical unit for which the census publishes sample data, typically six hundred to three thousand people. Block groups are called districts. The model must predict the median housing price in any district, given all the other metrics.

The first question is what exactly the business objective is. Building a model is probably not the end goal. How the company expects to use the model determines how the problem is framed, which algorithms are selected, which performance measure is used, and how much effort goes into tweaking.

Here, the model's prediction will be fed, along with other signals, to another machine learning system that decides whether it is worth investing in an area.

The second question is what the current solution looks like, which gives a reference for performance. Currently, experts estimate district prices manually, using complex rules when they cannot get the actual median price. This is costly and slow, and their estimates are often off by more than thirty percent.

---

Data pipelines. A sequence of data processing components is called a data pipeline. Components typically run asynchronously: each one pulls in a large amount of data, processes it, and writes the result to another data store, where the next component picks it up later. The interface between components is simply the data store, so each component is self-contained and different teams can own different components.

If a component breaks, downstream components can often keep running for a while on its last output, which makes the architecture robust. But without proper monitoring, a broken component can go unnoticed, the data gets stale, and the overall system's performance drops.

---

The type of task. This is a supervised learning task, because each instance comes with its expected output, the district's median housing price. It is a regression task, because the model predicts a value. It is a multiple regression problem, because it uses multiple features, and a univariate regression problem, because it predicts a single value per district. Predicting several values per district would be multivariate regression.

There is no continuous flow of incoming data, no need to adapt quickly, and the data fits in memory, so plain batch learning is enough. With huge data, batch learning could be split across servers with the MapReduce technique, or online learning could be used.

---

Selecting a performance measure. A typical performance measure for regression is the root mean squared error, or RMSE. It is the square root of the average, over all instances, of the squared difference between the prediction and the label. It shows how much error the system typically makes, giving a higher weight to large errors.

In the book's notation, the letter m is the number of instances. x sub i is the feature vector of instance i, and y sub i is its label. Capital X is the matrix of all feature vectors, one row per instance, each row being the transpose of an instance's feature vector. The prediction function h is also called the hypothesis, and its prediction for instance i is written y-hat sub i.

When there are many outliers, the mean absolute error, or MAE, may be preferable, since the RMSE is quite sensitive to them. The MAE is the average of the absolute differences between predictions and labels.

Both measure the distance between the vector of predictions and the vector of targets. The RMSE corresponds to the Euclidean norm, also called the L2 norm. The MAE corresponds to the L1 norm, also called the Manhattan norm, because it measures distance in a city where travel is only possible along orthogonal blocks. More generally, the L-k norm is the k-th root of the sum of the absolute values raised to the power k. The L-zero norm counts the nonzero elements, and the L-infinity norm gives the maximum absolute value. The higher the norm index, the more it focuses on large values and neglects small ones. That is why the RMSE is more sensitive to outliers than the MAE. When outliers are exponentially rare, as in a bell-shaped curve, the RMSE performs very well and is generally preferred.

---

Checking the assumptions. Listing and verifying assumptions catches serious issues early. If the downstream system actually converted prices into categories, such as cheap, medium, and expensive, the task should be framed as classification, not regression, and that is better discovered before months of work. In this case the downstream team confirms that it needs actual prices.

---

Getting the data. In a typical environment, data is spread across databases and files, and requires credentials, access authorizations, and knowledge of the data schema. Here, the data is a single compressed archive containing one comma-separated values file, or CSV file. Rather than downloading it by hand, a function downloads and extracts the archive if it is missing, then loads the CSV file into a Pandas data frame. Automating this helps when the data changes regularly, and when it must be installed on several machines.

---

A quick look at the data structure. Each row is one district, with ten attributes: longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, median house value, and ocean proximity.

The data frame's info method shows the number of rows, each attribute's type, and the number of non-null values. There are twenty thousand six hundred forty instances, which is small by machine learning standards. Total bedrooms has only twenty thousand four hundred thirty-three non-null values, so two hundred seven districts are missing it.

All attributes are numerical except ocean proximity, a text attribute with repeated values, so probably categorical. The value counts method shows its five categories: less than one hour from the ocean, inland, near ocean, near bay, and island, which has only five districts.

The describe method summarizes the numerical attributes: count, mean, minimum, maximum, standard deviation, and percentiles. The standard deviation measures how dispersed the values are; it is the square root of the variance, the average squared deviation from the mean. A percentile is the value below which a given percentage of observations fall. Twenty-five percent of districts have a housing median age below eighteen, fifty percent below twenty-nine, and seventy-five percent below thirty-seven. These are the first quartile, the median, and the third quartile.

The hist method plots a histogram of each numerical attribute, showing the number of instances in each value range. The histograms reveal four things.

First, median income is not in dollars. It was scaled and capped at fifteen for high incomes and at zero point five for low ones, and it represents roughly tens of thousands of dollars, so three means about thirty thousand dollars. Preprocessed attributes are common and not necessarily a problem, but it is important to understand how they were computed.

Second, housing median age and median house value were also capped. The cap on median house value, the target, may be a serious problem, because algorithms may learn that prices never exceed it. If the client team needs precise predictions beyond five hundred thousand dollars, there are two options: collect proper labels for the capped districts, or remove those districts from both the training set and the test set.

Third, the attributes have very different scales. Fourth, many histograms are skewed right, extending much farther right of the median than left, which can make patterns harder for some algorithms to detect.

---

Creating a test set. A test set should be created and set aside before looking at the data any further. The brain is a powerful pattern detector, and therefore prone to overfitting. Noticing a pattern in the test data could lead to choosing a particular model, and the generalization error estimated on the test set would then be too optimistic. This is called data snooping bias.

In theory, creating a test set is simple: pick instances randomly, typically twenty percent of the dataset, or less for very large datasets. For the housing data, that gives sixteen thousand five hundred twelve training instances and four thousand one hundred twenty-eight test instances.

But each run of a random split produces a different test set, so eventually the whole dataset gets seen. Saving the test set on the first run, or setting the random number generator's seed, for example to forty-two, fixes this until the dataset is updated. A stable solution is to decide based on each instance's identifier, assuming identifiers are unique and immutable: compute a hash of the identifier, and put the instance in the test set if the hash is at most twenty percent of the maximum hash value. The test set then stays consistent across refreshes: it gets twenty percent of the new instances, and never an instance that was previously in the training set.

The housing data has no identifier column. The row index can serve as one, but only if new data is always appended at the end and no row is ever deleted. Alternatively, stable features such as latitude and longitude can be combined into an identifier, but the coarse location data makes many districts share an identifier, which introduces some sampling bias.

Scikit-Learn's train test split function does a random split. Its random state parameter sets the seed, and it can split several datasets with the same number of rows on the same indices, such as separate features and labels.

---

Stratified sampling. Purely random sampling is fine for large datasets, especially relative to the number of attributes. Otherwise it risks significant sampling bias.

A survey company polling one thousand people tries to keep the sample representative. In the United States, fifty-one point six percent of voting-age citizens are female, so a good survey would include five hundred sixteen women and four hundred eighty-four men. This is stratified sampling: the population is divided into homogeneous subgroups called strata, and the right number of instances is sampled from each. With purely random sampling, there would be over a ten percent chance of a sample with less than forty-nine or more than fifty-four percent women.

Experts say median income is very important for predicting housing prices, so the test set should represent the income categories of the whole dataset. Since median income is continuous, an income category attribute is created with the Pandas cut function: five categories, with boundaries at one point five, three, four point five, and six. Most incomes fall between one point five and six. There should not be too many strata, and each must be large enough, or the estimate of a stratum's importance may be biased.

Scikit-Learn's splitter classes implement various splitting strategies. Each has a split method that yields the training and test indices of different splits. The stratified shuffle split class can generate, for example, ten stratified splits. For a single split, the train test split function with its stratify argument set to the income category is shorter.

The stratified test set has income category proportions almost identical to the full dataset, while a purely random test set is skewed. The income category column is then dropped. Test set generation is an often neglected but critical part of a project.

---

Visualizing geographical data. Exploration uses only the training set, and works on a copy so the original can be restored. A very large training set can be sampled into a smaller exploration set.

A scatterplot of latitude against longitude looks like California. Setting the alpha option, the point transparency, to zero point two reveals the high-density areas: the Bay Area, Los Angeles, San Diego, and a long line through the Central Valley, around Sacramento and Fresno.

A second plot sizes each circle by population and colors it by price, with the jet color map, from blue for low prices to red for high ones. Prices are strongly related to location, such as closeness to the ocean, and to population density. A clustering algorithm could detect the main clusters and add features measuring proximity to their centers. Ocean proximity may also help, although coastal districts in Northern California are not very expensive, so it is not a simple rule.

---

Looking for correlations. The corr method computes the standard correlation coefficient, also called Pearson's r, between every pair of numerical attributes. Median income has the strongest correlation with median house value, at about zero point six nine. Total rooms and housing median age are weakly positive, and latitude is weakly negative, at about minus zero point one four, meaning prices go down slightly going north.

The coefficient ranges from minus one to one. Close to one means a strong positive correlation, close to minus one a strong negative correlation, and close to zero no linear correlation.

The Pandas scatter matrix function plots every numerical attribute against every other. Nine attributes would give eighty-one plots, so it is restricted to the most promising ones: median house value, median income, total rooms, and housing median age. The diagonal shows a histogram of each attribute instead of straight lines.

The plot of median house value against median income shows a strong upward trend, with noise. The price cap shows as a horizontal line at five hundred thousand dollars. Less obvious horizontal lines appear around four hundred fifty thousand, three hundred fifty thousand, and perhaps two hundred eighty thousand. Removing those districts may stop algorithms from learning to reproduce these quirks.

The correlation coefficient only measures linear correlation. It can completely miss nonlinear relationships, and a coefficient of zero does not mean the variables are independent. It also says nothing about the slope: height in inches has a correlation of one with height in feet or in nanometers.

---

Experimenting with attribute combinations. Total rooms is not very useful without the number of households, and total bedrooms is more useful compared with total rooms. So three attributes are created: rooms per house, the bedrooms ratio, and people per house.

The bedrooms ratio has a correlation of about minus zero point two six with median house value, much stronger than total rooms or total bedrooms: houses with a lower ratio of bedrooms to rooms tend to be more expensive. Rooms per house is also more informative than total rooms, since larger houses are more expensive.

New features should not be too linearly correlated with existing features, because collinearity causes problems for some models, such as linear regression. Simple weighted sums of existing features should be avoided.

This exploration does not need to be thorough. The goal is quick insights for a first prototype, and the process is iterative.

---

Preparing the data. Data preparation should be written as functions, so the transformations can be reproduced on any dataset, reused in future projects, applied to new data in the live system, and easily combined to see which work best. The training set is restored to a clean copy, and the predictors and labels are separated, since they do not necessarily get the same transformations.

---

Cleaning the data. Most algorithms cannot handle missing features. There are three options for the missing total bedrooms values: drop those districts, drop the whole attribute, or set the missing values to some value, such as zero, the mean, or the median. The third option is called imputation, and it is the least destructive.

Scikit-Learn's simple imputer, with its strategy set to median, replaces each attribute's missing values with that attribute's median. Unlike filling values with Pandas, it stores the median of each feature, so it can impute the validation set, the test set, and new data in production the same way. The median only applies to numerical attributes, so the imputer is fitted on the numerical columns only. Fitting computes the medians and stores them in the imputer's statistics attribute, and the transform method fills in the missing values. Although only total bedrooms has missing values, the imputer is applied to all numerical attributes, since new data could have gaps anywhere.

Other strategies are the mean, the most frequent value, and a constant; the last two also work for non-numerical data. More powerful options exist. The k-nearest neighbors imputer replaces each missing value with the mean of that feature over the k nearest neighbors. The iterative imputer trains a regression model per feature to predict its missing values from the other features, and repeats the process several times, improving the models and the values at each iteration.

---

Scikit-Learn's design. Scikit-Learn's application programming interface, or API, follows a few principles.

Consistency. An estimator is any object that estimates parameters from a dataset, through its fit method, which takes the data and, for supervised learning, the labels. Any other parameter guiding the estimation is a hyperparameter, set as an instance variable, usually through the constructor. A transformer is an estimator that can also transform a dataset, with its transform method, generally using the learned parameters. Its fit transform method fits and transforms in one step, sometimes faster. A predictor is an estimator that can make predictions, with its predict method, and measure their quality on a test set, with its score method.

Inspection. Hyperparameters are public instance variables, and learned parameters are public instance variables whose names end with an underscore.

Nonproliferation of classes. Datasets are NumPy arrays or SciPy sparse matrices, and hyperparameters are plain Python strings or numbers.

Composition. Existing building blocks are reused; for example, a pipeline is built from a sequence of transformers followed by a final estimator.

Sensible defaults. Most parameters have reasonable defaults, so a working baseline is quick to build.

Transformers output NumPy arrays even when given Pandas data frames, so column names and the index are lost unless the output is wrapped back into a data frame.

---

Handling text and categorical attributes. Ocean proximity is not arbitrary text: it has a limited set of values, each representing a category. Most algorithms prefer numbers, so it must be encoded.

Scikit-Learn's ordinal encoder maps each category to an integer. The problem is that algorithms assume nearby values are more similar than distant ones. That suits ordered categories, such as bad, average, good, and excellent, but not ocean proximity, where categories zero and four, less than one hour from the ocean and near ocean, are more similar than categories zero and one.

The common solution is one binary attribute per category, equal to one when the instance belongs to that category and zero otherwise. This is called one-hot encoding, because only one attribute is one, or hot, while the others are zero, or cold. The new attributes are sometimes called dummy attributes. Scikit-Learn's one-hot encoder does this.

By default, the one-hot encoder outputs a SciPy sparse matrix, which stores only the nonzero values and their positions. With thousands of categories, one-hot encoding produces a huge matrix of zeros with a single one per row, so a sparse matrix saves a lot of memory and computation. It can be converted to a dense array, or the encoder's sparse output option can be turned off.

The Pandas get dummies function also one-hot encodes, but the one-hot encoder remembers the categories it was trained on. A model in production must receive exactly the same features as during training. Given data with only two categories, get dummies outputs two columns, while the trained encoder outputs one column per learned category, in the right order. Given an unknown category, get dummies creates a new column, while the encoder raises an exception, or, with its handle unknown hyperparameter set to ignore, encodes it as all zeros.

When a categorical attribute has many categories, such as country code, profession, or species, one-hot encoding creates many features, which can slow training and degrade performance. Alternatives are to replace the category with related numerical features, such as distance to the ocean instead of ocean proximity, or a country's population and gross domestic product per capita instead of its code; to use the category encoders package; or, with neural networks, to replace each category with a learnable low-dimensional vector called an embedding, an example of representation learning.

An estimator fitted on a data frame stores its column names, as its feature names in attribute, and checks that later inputs have the same columns. Transformers also have a get feature names out method, useful for wrapping their output in a data frame and for debugging.

---

Feature scaling. With few exceptions, algorithms perform poorly when numerical inputs have very different scales. Total rooms ranges from about six to thirty-nine thousand, while median income ranges from zero to fifteen. Without scaling, most models would focus on the number of rooms and largely ignore median income.

Scalers, like all estimators, must be fitted to the training set only. The trained scaler then transforms the validation set, the test set, and new data. Outliers in new data may be scaled outside the target range, unless the clip hyperparameter is set.

Min-max scaling, often called normalization, subtracts the minimum and divides by the difference between the maximum and the minimum, so values range from zero to one. Scikit-Learn's min max scaler does this, and its feature range hyperparameter changes the range. Neural networks work best with zero-mean inputs, so a range from minus one to one suits them better.

Standardization subtracts the mean and divides by the standard deviation, giving values with a mean of zero and a standard deviation of one. It does not bound values to a range, but it is much less affected by outliers. A median income of one hundred, entered by mistake, would make min-max scaling crush all the other values into the range from zero to zero point one five, while standardization would barely change. Scikit-Learn's standard scaler does this. For sparse matrices, its with mean hyperparameter can be set to false, so it only divides by the standard deviation, since subtracting the mean would destroy sparsity.

---

Heavy tails and multimodal distributions. When a feature has a heavy tail, meaning values far from the mean are not exponentially rare, both scaling methods squash most values into a small range, which models dislike. So the feature should first be transformed to shrink the tail and make the distribution roughly symmetrical. For a positive feature with a heavy right tail, replacing it with its square root, or a power between zero and one, often works. For a really long tail, such as a power law, the logarithm may help. Population roughly follows a power law: districts with ten thousand inhabitants are only ten times less frequent than districts with one thousand. Its logarithm is very close to a Gaussian, bell-shaped distribution.

Another option for heavy tails is bucketizing: chopping the distribution into roughly equal-sized buckets and replacing each value with its bucket index, for example its percentile. The result is almost uniform, so it needs no further scaling.

A multimodal distribution has two or more clear peaks, called modes, as housing median age does. It can also be bucketized, but with the bucket IDs treated as categories and one-hot encoded, so there should not be too many buckets. This allows the model to learn different rules for different ranges, for example if houses built around thirty-five years ago have an unfashionable style that makes them cheaper.

Another option is to add a feature for each main mode, measuring similarity to that mode with a radial basis function, or RBF: a function that depends only on the distance between the input and a fixed point. The Gaussian RBF decays exponentially with distance. The similarity between housing age x and thirty-five is the exponential of minus gamma times the square of x minus thirty-five. The hyperparameter gamma controls how fast the similarity decays as x moves away from thirty-five. Scikit-Learn's RBF kernel function computes it. The resulting feature peaks at thirty-five, near a spike in the age distribution, so it will help if that age group correlates with lower prices.

---

Transforming the target. A heavy-tailed target can be replaced with its logarithm, but then the model predicts the log of the median house value, and its predictions must be exponentiated. Most transformers have an inverse transform method for this. The labels can be scaled with a standard scaler, a model trained on the scaled labels, and its predictions converted back with the scaler's inverse transform method. The transformed target regressor does all of this automatically and with less risk of error. It takes a regression model and a label transformer, is fitted on the original labels, and returns predictions on the original scale.

---

Custom transformers. Some tasks need custom transformations, cleanup operations, or attribute combinations. A transformation that needs no training can be a plain function wrapped in Scikit-Learn's function transformer. A log transformer applies the logarithm, with the exponential as an optional inverse, needed inside a transformed target regressor. The function can take hyperparameters as keyword arguments: wrapping the RBF kernel function, with thirty-five as the fixed point, produces the age similarity feature. The RBF has no inverse, because for any nonzero distance there are two values at that distance from the fixed point. Given several features, the RBF kernel uses the Euclidean distance between them, so applied to latitude and longitude with San Francisco's coordinates, it measures each district's geographic similarity to San Francisco. A function transformer can also compute the ratio of two columns.

A transformer that learns parameters in fit and uses them in transform needs a custom class. Scikit-Learn relies on duck typing: what matters is that an object has the right methods, not its type. A custom transformer needs a fit method that returns self, a transform method, and a fit transform method. Inheriting from the transformer mixin class provides fit transform for free. Inheriting from the base estimator class provides get params and set params, which automatic hyperparameter tuning needs, as long as the constructor has no variable arguments.

The book's example re-implements the standard scaler. Its fit method validates the input, stores each column's mean and standard deviation, records the number of input features, and returns self. Its transform method checks that the scaler is fitted and the input has the right number of features, then subtracts the mean, unless that option is off, and divides by the standard deviation.

The rules this illustrates: the fit method must accept both X and y, with y defaulting to none, because pipelines pass both; every estimator records its number of input features in fit and checks it later; and fit must return self. A complete transformer also records feature names, provides a get feature names out method, and provides an inverse transform when possible.

---

The cluster similarity transformer. A custom transformer can use other estimators. The cluster similarity transformer runs k-means in its fit method to find clusters in the training data, then uses the RBF kernel in its transform method to measure each sample's similarity to each cluster center. Its hyperparameters are the number of clusters, gamma, and the random state.

K-means is a clustering algorithm. Its fit method accepts sample weights; weighting districts by median income would bias the clusters toward wealthy districts. It is stochastic, relying on randomness, so the random state must be set for reproducible results.

Fitted on latitude and longitude with ten clusters and a gamma of one, the transformer outputs a matrix with one row per district and one column per cluster. Most clusters land in highly populated areas.

---

Transformation pipelines. Scikit-Learn's pipeline class runs transformation steps in the right order. A small numerical pipeline imputes missing values with the median, then standardizes.

The pipeline is built from a list of name and estimator pairs. Names must be unique and must not contain double underscores. Every step except the last must be a transformer; the last can be any estimator. The make pipeline function names the steps automatically, after their class names in lowercase.

Fitting the pipeline calls fit transform on each transformer in turn, passing each output to the next step, and calls fit on the final estimator. The pipeline exposes the same methods as its final estimator: with a scaler at the end it acts as a transformer, and with a predictor at the end it has a predict method that applies all the transformations, then predicts.

---

The column transformer. A column transformer applies different transformers to different columns. Here, it applies the numerical pipeline to the numerical columns and a categorical pipeline, which imputes the most frequent category and then one-hot encodes, to ocean proximity.

It is built from a list of triplets: a name, a transformer, and the columns it applies to. A transformer can be replaced by drop or passthrough. Columns that are not listed are dropped by default, unless the remainder hyperparameter specifies a transformer for them. Scikit-Learn's make column selector selects columns by type, such as all numerical or all categorical columns, and the make column transformer function names the transformers automatically.

The column transformer concatenates the transformed columns horizontally, so transformers must never change the number of rows. When outputs mix sparse and dense matrices, it returns a sparse matrix only if the share of nonzero cells is below a threshold, by default thirty percent.

---

The full preprocessing pipeline. The final preprocessing pipeline does six things. It imputes missing numerical values with the median and missing categorical values with the most frequent category. It one-hot encodes the categorical feature. It adds the three ratio features: the bedrooms ratio, rooms per house, and people per house. It adds cluster similarity features, likely more useful than raw latitude and longitude. It replaces long-tailed features, namely total bedrooms, total rooms, population, households, and median income, with their logarithm. And it standardizes all numerical features. The one remaining column, housing median age, is imputed and scaled by the remainder. The output has twenty-four features.

---

Training and evaluating on the training set. The first model chains the preprocessing pipeline and linear regression. On the first five districts, one prediction is off by over two hundred thousand dollars, two by about twenty-five percent, and two by less than ten percent. The RMSE on the training set is about sixty-nine thousand dollars, while most districts' values lie between one hundred twenty thousand and two hundred sixty-five thousand dollars.

The score method of a regressor returns the R squared coefficient of determination, not the RMSE. It is the share of the variance in the data that the model explains: one at best, zero for a model that always predicts the mean, and negative for a model that does even worse.

Linear regression underfits: the features do not carry enough information, or the model is not powerful enough. The fixes are a more powerful model, better features, or fewer constraints, and since this model is not regularized, the next step is a more powerful model.

A decision tree regressor can find complex nonlinear relationships. Its training RMSE is zero, which almost certainly means it badly overfits. The test set must stay untouched, so part of the training set is needed for validation.

---

Cross-validation. One option is to split the training set into a smaller training set and a validation set. A great alternative is k-fold cross-validation: the training set is split into k nonoverlapping subsets, called folds, and the model is trained and evaluated k times, each time evaluated on a different fold and trained on the other k minus one folds. This produces k scores. Scikit-Learn's cross val score function does this.

Scikit-Learn's cross-validation expects a utility function, where greater is better, rather than a cost function, so the scoring is the negative RMSE, and its sign must be flipped.

With ten folds, the decision tree's mean RMSE is about sixty-six thousand six hundred, with a standard deviation of about one thousand one hundred. Linear regression scores about seventy thousand, with a standard deviation of about four thousand two hundred. So the decision tree is barely better than linear regression, because it overfits severely: its training error is zero, while its validation error is high. Cross-validation measures not only performance, but also how precise the estimate is, at the cost of training the model several times.

A random forest regressor trains many decision trees on random subsets of the features and averages their predictions. A model made of many models is an ensemble. If the models are diverse, their errors are not very correlated, so averaging smooths out the errors, reduces overfitting, and improves performance. The random forest's mean validation RMSE is about forty-seven thousand, much better. But its training RMSE is about seventeen thousand five hundred, so it still overfits a lot. The remedies are to simplify or regularize the model, or to get much more training data.

Before tuning one model, it is better to try many models from different categories, such as support vector machines with various kernels and possibly a neural network, without spending much time tweaking hyperparameters, and to shortlist two to five promising ones.

---

Grid search. Tuning hyperparameters by hand is tedious. Scikit-Learn's grid search class takes the hyperparameters to explore and the values to try, and evaluates every combination with cross-validation.

The search runs on a full pipeline of preprocessing followed by a random forest. A hyperparameter of a nested estimator is named by its path, with steps joined by double underscores. For example, preprocessing, geo, and number of clusters, joined that way, refers to the step named preprocessing, which is the column transformer, then its transformer named geo, the cluster similarity transformer, then that transformer's number of clusters hyperparameter. This allows preprocessing hyperparameters to be tuned together with model hyperparameters, which matters because they often interact.

The grid has two parts. The first combines three cluster counts with three values of the random forest's max features hyperparameter, nine combinations. The second combines two cluster counts with three max features values, six combinations. That is fifteen combinations, each trained three times with three-fold cross-validation, for forty-five rounds of training.

The best combination is fifteen clusters and max features of six. Since fifteen was the largest cluster count tried, a search with higher values might do even better. By default, the grid search then retrains the best estimator on the whole training set, which usually improves it. The best mean RMSE is about forty-three thousand six hundred, better than the forty-seven thousand obtained with default hyperparameters.

---

Randomized search. Randomized search is often preferable, especially for large search spaces. It evaluates a fixed number of combinations, choosing a random value for each hyperparameter at each iteration.

It has three advantages. A continuous hyperparameter, or one with many possible values, gets explored at as many values as there are iterations, instead of just the few listed in a grid. A hyperparameter that turns out not to matter multiplies grid search time by its number of values, but costs randomized search nothing. And six hyperparameters with ten values each would require a million grid search trainings, while randomized search runs for any chosen number of iterations.

Each hyperparameter is given either a list of values or a probability distribution, for example a random integer distribution for the number of clusters.

Scikit-Learn also has halving grid search and halving randomized search. In the first round, many candidate combinations are trained with limited resources, by default on a small part of the training set, and evaluated with cross-validation. Only the best candidates move on to the next round, with more resources. After several rounds, the final candidates are evaluated with full resources.

---

Ensemble methods. Combining the best models into an ensemble often beats the best individual model, just as a random forest beats its individual trees, especially when the models make very different types of errors. For example, an ensemble could predict the mean of a random forest's and a k-nearest neighbors model's predictions.

---

Analyzing the best models and their errors. A random forest reports the relative importance of each feature. The log of median income is by far the most important, at about zero point one nine, followed by the inland category, the bedrooms ratio, rooms per house, people per house, and some cluster similarities. Only the inland ocean proximity category matters much, so the others could be dropped. Scikit-Learn's select from model transformer drops the least useful features automatically, based on a model's feature importances.

The system's specific errors should also be examined, to understand them and find fixes, such as adding features, removing uninformative ones, or cleaning up outliers.

This is also the time to check fairness. The model should work well not only on average, but on every category of districts, such as rural or urban, rich or poor, northern or southern, and minority or not. This bias analysis creates a subset of the validation set for each category and measures performance on it. If the model performs poorly on a whole category, it should probably not be deployed until the issue is resolved, or at least not used for that category, since it may do more harm than good.

---

Evaluating on the test set. The final model makes predictions on the test set, and the test RMSE is about forty-one thousand four hundred. A point estimate may not be enough to decide on launching, for example if it is only slightly better than the current production model. SciPy's bootstrap function computes a ninety-five percent confidence interval, here from about thirty-nine thousand five hundred to about forty-three thousand seven hundred.

After a lot of hyperparameter tuning, test performance is usually slightly worse than the cross-validation estimate, because the system is fine-tuned to the validation data. That is not the case here. When it does happen, the hyperparameters must not be tweaked to improve the test score, because the gains would not generalize to new data.

---

Presenting the solution. The prelaunch phase presents the solution: concise reports or slides, visualizations of the key insights, and a message tailored to the audience, technical for peers and high-level for stakeholders. Memorable statements work well, such as "median income is the number one predictor of housing prices". The presentation covers what was learned, what worked and what did not, the assumptions, and the system's limitations.

Results should be reproducible: code shared with the team, a README with installation steps, clear notebooks, a list of required libraries with exact versions or a Docker image, and fixed random seeds.

The final system is not much better than the experts, who were often off by thirty percent, but launching it may still be worthwhile if it frees the experts for more interesting and productive work.

---

Launching the system. The code is polished, documented, and tested for production. The simplest deployment is to save the trained model with the joblib library, transfer the file to production, and load it there. Any custom classes and functions the model uses must be importable in production too. Saving every model tried, with its cross-validation scores and validation predictions, makes later comparisons easy.

A website could load the model once at server startup and call its predict method for each user query. Alternatively, the model can run as a dedicated web service, queried through a REST API using HTTP requests and JSON. This makes upgrading the model easier without interrupting the main application, simplifies scaling with load balancing across several service instances, and allows the application to be written in any language. The model can also be deployed to the cloud, for example on Google's Vertex AI, which serves it as a web service that handles load balancing and scaling.

---

Monitoring and maintaining the system. Monitoring code must check live performance regularly and trigger alerts when it drops. Performance can drop suddenly, for example when an infrastructure component breaks, or decay slowly, which can go unnoticed for a long time. Slow decay is common because of data drift: a model trained on last year's data may not fit today's data.

Sometimes performance can be inferred from downstream metrics. For a recommender system, if sales of recommended products drop relative to other products, the prime suspect is the model: the data pipeline may be broken, or the model may need retraining on fresh data.

Sometimes human analysis is needed. For a model detecting product defects on a production line, a sample of the pictures it classified, especially those it was unsure about, can be sent to human raters. They may need to be experts, or they can be workers on a crowdsourcing platform, such as Amazon Mechanical Turk, or the users themselves, through surveys or repurposed captchas.

Monitoring, and the processes for handling failures, are often much more work than building and training the model.

If the data keeps evolving, the model must be retrained regularly, and as much as possible should be automated: collecting and labeling fresh data; a script that trains and fine-tunes the model, for example daily or weekly; and a script that evaluates the new and previous models on the updated test set, including subsets such as rich or poor and rural or urban districts, and deploys the new model only if performance has not decreased.

Input data quality should also be monitored, because performance can degrade slightly before any alert fires, for example when a sensor malfunctions or another team's output goes stale. Alerts can fire when more inputs are missing a feature, when the mean or standard deviation drifts far from the training set, or when a categorical feature contains new categories.

Every model and every version of the datasets should be backed up, with tools to roll back quickly if a new model fails or new data is corrupted. Backups also make it easy to compare models and to evaluate any model on any previous dataset.

All this infrastructure is called machine learning operations, or MLOps. A first project takes a lot of time to build and deploy, but once the infrastructure exists, going from idea to production is much faster.

---

The overall process. Much of the work in a project is not the algorithms: it is data preparation, monitoring tools, human evaluation pipelines, and automated retraining. Knowing the overall process well, and three or four algorithms well, is worth more than spending all the time on advanced algorithms. A good way to practice is to take a dataset, for example from a Kaggle competition, through the whole process.

---

Summary.

A machine learning project has eight main steps: frame the problem, get the data, explore it, prepare it, select and train a model, fine-tune it, present the solution, and launch, monitor, and maintain the system.

Predicting district housing prices is a supervised, multiple, univariate regression task suited to batch learning. It is measured with the RMSE, or with the MAE when there are many outliers.

A test set is set aside before exploring the data, to avoid data snooping bias. Stratified sampling on an important attribute, here income category, keeps it representative.

Exploration uses plots, correlation coefficients, and attribute combinations. Median income is the strongest predictor, and the bedrooms ratio and rooms per house are more informative than raw totals.

Preparation imputes missing values, one-hot encodes categories, transforms heavy-tailed features, adds similarity features, and scales everything, fitting every transformer on the training set only. Pipelines and column transformers chain these steps into one reproducible preprocessing step.

Models are compared with k-fold cross-validation. Linear regression underfits, a decision tree overfits, and a random forest does best. Grid search and randomized search tune hyperparameters, including preprocessing hyperparameters, and the final model is evaluated once on the test set, with a confidence interval.

After launch, the system is monitored for sudden failures and slow data drift, retrained regularly with as much automation as possible, and backed up with its datasets so it can be rolled back.
