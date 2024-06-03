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
from training import clean_df
import pandas as pd
import numpy as np
import joblib


def data_prepartion(X_var, outcome=None):


    vars_model = ["nomem_encr", 'gender_bg', 'age_bg',
                  'partnership_status', 'domestic_situation', 'lenght_partnership',
                  'satisf_partnership',
                  'age_of_female', 'age_of_male', 'hh_net_income', 'fertility_intentions',
                  'parity', 'high_edu_level', 'child_soon', 'n_children_in_hh', 'fert_int_index_5y', 'hh_income_sd_5y',
                  'stability_hh_5y', 'personal_income_2020', 'dutch', 'non_wstrn_1gen',
                  'non_wstrn_2gen', 'wstrn_1gen', 'wstrn_2gen', 'sted', 'cost-free',
                  'rental', 'self-owned', 'weight', 'height', 'bmi', 'len_partnership',
                  'satis_partnership', 'irregular_work', 'ends_meet', 'religion_they_1.0',
                  'religion_they_2.0', 'religion_they_3.0', 'religion_they_4.0',
                  'religion_they_5.0', 'religion_they_6.0', 'religion_they_7.0',
                  'religion_they_8.0', 'religion_they_9.0', 'religion_they_10.0',
                  'religion_they_11.0', 'religion_they_12.0', 'religion_they_13.0',
                  'religion_they_14.0', 'religious', 'religion_you_1.0',
                  'religion_you_2.0', 'religion_you_3.0', 'religion_you_4.0',
                  'religion_you_5.0', 'religion_you_6.0', 'religion_you_7.0',
                  'religion_you_8.0', 'religion_you_9.0', 'religion_you_10.0',
                  'religion_you_11.0', 'religion_you_12.0', 'religion_you_13.0',
                  'religion_you_14.0', 'freq_see_father', 'freq_see_mother', 'life_satis',
                  'satis_relationship', 'satis_family_life', 'satis_house',
                  'satis_financial', 'satis_contacts', 'perc_health', 'long_disease',
                  'hinder', 'nettocat_clean']
    var_cate = ["gender_bg", "partnership_status", "domestic_situation", "satisf_partnership"
        , "fertility_intentions", "high_edu_level", "n_children_in_hh"]

    # Check which columns are missing
    missing_columns = [col for col in vars_model if col not in X_var.columns]

    # Add missing columns to the DataFrame and fill them with zeros
    for col in missing_columns:
        X_var[col] = 0

    imputed_media_median = X_var[['gender_bg', 'age_bg',
                                  'partnership_status', 'domestic_situation', 'lenght_partnership',
                                  'satisf_partnership',
                                  'age_of_female', 'age_of_male', 'hh_net_income', 'fertility_intentions',
                                  'parity', 'high_edu_level', 'child_soon', 'n_children_in_hh', 'fert_int_index_5y', 'hh_income_sd_5y',
                                  'stability_hh_5y', 'personal_income_2020', 'dutch', 'non_wstrn_1gen',
                                  'non_wstrn_2gen', 'wstrn_1gen', 'wstrn_2gen', 'sted', 'cost-free',
                                  'rental', 'self-owned', 'weight', 'height', 'bmi', 'len_partnership',
                                  'satis_partnership', 'irregular_work', 'ends_meet', 'religion_they_1.0',
                                  'religion_they_2.0', 'religion_they_3.0', 'religion_they_4.0',
                                  'religion_they_5.0', 'religion_they_6.0', 'religion_they_7.0',
                                  'religion_they_8.0', 'religion_they_9.0', 'religion_they_10.0',
                                  'religion_they_11.0', 'religion_they_12.0', 'religion_they_13.0',
                                  'religion_they_14.0', 'religious', 'religion_you_1.0',
                                  'religion_you_2.0', 'religion_you_3.0', 'religion_you_4.0',
                                  'religion_you_5.0', 'religion_you_6.0', 'religion_you_7.0',
                                  'religion_you_8.0', 'religion_you_9.0', 'religion_you_10.0',
                                  'religion_you_11.0', 'religion_you_12.0', 'religion_you_13.0',
                                  'religion_you_14.0', 'freq_see_father', 'freq_see_mother', 'life_satis',
                                  'satis_relationship', 'satis_family_life', 'satis_house',
                                  'satis_financial', 'satis_contacts', 'perc_health', 'long_disease',
                                  'hinder', 'nettocat_clean']].median()

    X = X_var[vars_model]
    X[vars_model] = X[vars_model].fillna(imputed_media_median)
    # X=X.dropna(how='any')
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



