"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
from training import data_prepartion
import pandas as pd
import numpy as np
import joblib

def get_last_valid(row):
    try:
        last_valid_idx = row.last_valid_index()
        return row[last_valid_idx]
    except KeyError:
        return np.nan


def most_recent(df, cols):
    sorted_cols = sorted(cols, reverse=False)
    data = df[sorted_cols]
    return data.apply(get_last_valid, axis=1)

def clean_df(raw_df, df_b):
    # base dataframe
    df = raw_df[["nomem_encr",
                 "outcome_available",
                 "gender_bg",
                 "age_bg"]].copy()

    df["partnership_status"] = most_recent(raw_df, ["partner_2020",
                                                    "partner_2019",
                                                    "partner_2018"])
    df["domestic_situation"] = most_recent(raw_df, ["woonvorm_2020",
                                                    "woonvorm_2019",
                                                    "woonvorm_2018"])
    df["lenght_partnership"] = 2024 - most_recent(raw_df, ["cf20m028",
                                                           "cf19l028",
                                                           "cf18k028"])
    df["age_of_partner"] = 2024 - most_recent(raw_df, ["cf20m026",
                                                       "cf19l026",
                                                       "cf18k026"])
    df["satisf_partnership"] = most_recent(raw_df, ["cf20m180",
                                                    "cf19l180",
                                                    "cf18k180"])
    df["gender_of_partner"] = most_recent(raw_df, ["cf20m032",
                                                   "cf19l032",
                                                   "cf18k032"])

    conditions_f = [
        (df['gender_bg'] == 1),
        (df['gender_bg'] == 1) & (df['gender_of_partner'] == 1),
        (df['gender_bg'] == 1) & (df['gender_of_partner'] == 2),
        (df['gender_bg'] == 1) & (df['gender_of_partner'] == 4)
    ]

    choices_f = [
        df['age_bg'],
        df['age_of_partner'],
        df['age_of_partner'],
        np.nan
    ]

    conditions_m = [
        (df['gender_bg'] == 1),
        (df['gender_bg'] == 2) & (df['gender_of_partner'] == 1),
        (df['gender_bg'] == 2) & (df['gender_of_partner'] == 3),
        (df['gender_bg'] == 2) & (df['gender_of_partner'] == 5)
    ]

    choices_m = [
        df['age_bg'],
        df['age_of_partner'],
        df['age_of_partner'],
        np.nan
    ]

    df["age_of_female"] = np.select(conditions_f, choices_f, default=np.nan)
    df["age_of_male"] = np.select(conditions_m, choices_m, default=np.nan)

    df["hh_net_income"] = most_recent(raw_df, ["nettohh_f_2020",
                                               "nettohh_f_2019",
                                               "nettohh_f_2018"])
    df["fertility_intentions"] = most_recent(raw_df, ["cf20m128",
                                                      "cf19l128",
                                                      "cf18k128"])
    df["parity"] = most_recent(raw_df, ["cf20m455",
                                        "cf19l455",
                                        "cf18k455"])
    df["high_edu_level"] = most_recent(raw_df, ["oplcat_2020",
                                                "oplcat_2019",
                                                "oplcat_2018"])
    raw_df["child_soon_2020"] = np.nan
    raw_df.loc[(raw_df["cf20m130"] <= 5), "child_soon_2020"] = 1
    raw_df.loc[(raw_df["cf20m130"] > 5), "child_soon_2020"] = 0
    raw_df["child_soon_2019"] = np.nan
    raw_df.loc[(raw_df["cf19l130"] <= 6), "child_soon_2019"] = 1
    raw_df.loc[(raw_df["cf19l130"] > 6), "child_soon_2019"] = 0
    raw_df["child_soon_2018"] = np.nan
    raw_df.loc[(raw_df["cf18k130"] <= 7), "child_soon_2018"] = 1
    raw_df.loc[(raw_df["cf18k130"] > 7), "child_soon_2018"] = 0
    df["child_soon"] = most_recent(raw_df, ["child_soon_2020",
                                            "child_soon_2019",
                                            "child_soon_2018"])
    df_b[["nomem_encr", "wave", "aantalki"]]
    wide_b = df_b.pivot(index='nomem_encr', columns='wave', values='aantalki').loc[:, 201801:]
    wide_b["n_children_in_hh"] = wide_b.apply(get_last_valid, axis=1)
    wide_b.reset_index(inplace=True)
    wide_b = wide_b[["nomem_encr", "n_children_in_hh"]]
    merged_df = pd.merge(df, wide_b, on='nomem_encr')

    return merged_df


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


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    X = data_prepartion(df)

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(X.drop(columns="nomem_encr"))

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": X["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict

