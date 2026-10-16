# ## Exercises

# Step 1: Imports
import pandas as pd
import numpy as np

from scipy.stats import randint, loguniform, expon

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Step 2: Load and Split Data
def load_housing_data():
    return pd.read_csv("textbook/hands_on_ml/ch02-end-to-end-machine-learning-project/datasets/housing/housing.csv")

housing = load_housing_data()

train, test = train_test_split(housing, test_size=0.2, random_state=67)

# Step 3: Separate features and labels
train_labels = train["median_house_value"].copy()
train_features = train.drop("median_house_value", axis=1)



# Step 4: Preprocessing Transformer

default_numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
def ratio_func(x):
    return x[:, [0]] / x[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

ratio_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(ratio_func, feature_names_out=ratio_name),
    StandardScaler()
)

geo_pipeline = make_pipeline(
    KMeans(n_clusters=10, random_state=42),
    StandardScaler()
)

class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = np.asarray(X) # from pd dataframe to np array
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn_.fit(X[:, :2], X[:, 2]) # features are the first two columns labels are the 3rd column
        return self

    def transform(self, X):
        X = np.asarray(X)
        predictions = self.knn_.predict(X[:, :2]) # every row, first two columns (0 and 1)
        return predictions.reshape(-1, 1) # single column of predictions, not sure whata data type


    def get_feature_names_out(self, names=None):
        return ["neighborhood_income"]

preprocessing = ColumnTransformer([  
    ("categorical", categorical_pipeline, ["ocean_proximity"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "households", "population", "median_income"]),
    ("bedrooms", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline, ["total_rooms", "households"]),
    ("population_per_house", ratio_pipeline, ["population", "households"]),
    ("geo", geo_pipeline, ["longitude", "latitude"]),
    ],
    remainder=default_numeric_pipeline
)

# Step 5: SVR Search
svr_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("svr", SVR())
])

param_distributions = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0)
}

rnd_search = RandomizedSearchCV(
    svr_pipeline, 
    param_distributions, 
    n_iter=10, 
    cv=3, 
    scoring="neg_root_mean_squared_error", 
    random_state=42
)

# first 5k rows
rnd_search.fit(train_features[:5000], train_labels[:5000])

print("Best parameters found: ", rnd_search.best_params_)
print("Best score found: ", -rnd_search.best_score_)

# Step 6: Feature Selection with Best Parameters

svr_c = rnd_search.best_params_["svr__C"]
svr_gamma = rnd_search.best_params_["svr__gamma"]
svr_kernel = rnd_search.best_params_["svr__kernel"]

selector = SelectFromModel(
    RandomForestRegressor(n_estimators=100, random_state=42), 
    threshold=0.005
)

svr_pipeline_with_selection = Pipeline([
    ("preprocessing", preprocessing),
    ("selector", selector),
    ("svr", SVR(kernel=svr_kernel, C=svr_c, gamma=svr_gamma))
])


# Step 7: Cross Validation


cv_scores = cross_val_score(
    svr_pipeline_with_selection, 
    train_features[:5000], 
    train_labels[:5000], 
    cv=3, 
    scoring="neg_root_mean_squared_error"
)

print("Cross-validation scores: ", -cv_scores)
