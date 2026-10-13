# ## Exercises

# Step 1: Imports
import pandas as pd
import numpy as np

from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

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

