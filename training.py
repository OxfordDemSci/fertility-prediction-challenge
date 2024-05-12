"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""
import random
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

def data_prepartion(X_var, outcome=None):
    vars_model = ["nomem_encr", "gender_bg", "age_bg", "partnership_status", "domestic_situation", 'hh_net_income', "fertility_intentions", "high_edu_level", 'n_children_in_hh']
    var_cate = ["gender_bg", "partnership_status", "domestic_situation", "fertility_intentions", "high_edu_level"]

    X = X_var[vars_model]
    X=X.dropna(how='any')

    X[var_cate] = X[var_cate].astype('int').astype("category")


    if isinstance(outcome, pd.DataFrame):
        y = outcome[outcome["nomem_encr"].isin(X["nomem_encr"])][["nomem_encr", "new_child"]]
        y["new_child"] = y["new_child"].astype('int')
        y = y.drop(columns="nomem_encr")

        return X, y
    else:

        return X




def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    random.seed(1080126904)  # not useful here because logistic regression deterministic

    # Filter cases for whom the outcome is not available
    training_var=cleaned_df[cleaned_df["outcome_available"]==1]
    training_outcome=outcome_df[outcome_df["nomem_encr"].isin(training_var["nomem_encr"])]


    X_train, y_train = data_prepartion(training_var, training_outcome)

    X_train = X_train.drop(columns="nomem_encr")

    # Define the model
    model = GradientBoostingClassifier(learning_rate=0.001, n_estimators=1800,max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=1080126904)
    model.fit(X_train, y_train)

    # Fit the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "model.joblib")
