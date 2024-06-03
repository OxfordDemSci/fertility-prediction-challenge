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
import numpy as np


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


def count_changes(row):
    changes = 0
    for i in range(1, len(row)):
        if row[i] != row[i - 1]:
            changes += 1
    return changes


def clean_df(raw_df, df_b):
    # raw_df = pd.read_csv("./rawdata/training_data/PreFer_train_data.csv")
    # df_b = pd.read_csv("./rawdata/other_data/PreFer_train_background_data.csv")

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
    raw_df["cf20m128_recoded"] = raw_df["cf20m128"].replace({1: 2, 3: 1, 2: 0})
    raw_df["cf19l128_recoded"] = raw_df["cf19l128"].replace({1: 2, 3: 1, 2: 0})
    raw_df["cf18k128_recoded"] = raw_df["cf18k128"].replace({1: 2, 3: 1, 2: 0})
    raw_df["cf17j128_recoded"] = raw_df["cf17j128"].replace({1: 2, 3: 1, 2: 0})
    raw_df["cf16i128_recoded"] = raw_df["cf16i128"].replace({1: 2, 3: 1, 2: 0})

    df['fert_int_index_5y'] = raw_df[[
        "cf20m128_recoded",
        "cf19l128_recoded",
        "cf18k128_recoded",
        "cf17j128_recoded",
        "cf16i128_recoded"
    ]].mean(axis=1)

    df["hh_income_sd_5y"] = raw_df[[
        "nettohh_f_2020",
        "nettohh_f_2019",
        "nettohh_f_2018",
        "nettohh_f_2017",
        "nettohh_f_2016"]].std(axis=1)

    df["stability_hh_5y"] = raw_df[[
        "woonvorm_2020",
        "woonvorm_2019",
        "woonvorm_2018",
        "woonvorm_2017",
        "woonvorm_2016"
    ]].apply(count_changes, axis=1)

    df["personal_income_2020"] = raw_df["netinc_2020"]

    raw_df['migration_background_bg_recoded'] = raw_df['migration_background_bg'].replace({0: 'dutch',
                                                                                           101: 'wstrn_1gen',
                                                                                           102: 'non_wstrn_1gen',
                                                                                           201: 'wstrn_2gen',
                                                                                           202: 'non_wstrn_2gen' })

    df = pd.concat([df, pd.get_dummies(raw_df['migration_background_bg_recoded'])], axis=1)

    df["sted"] = most_recent(raw_df, ["sted_2007",
                                      "sted_2008",
                                      "sted_2009",
                                      "sted_2010",
                                      "sted_2011",
                                      "sted_2012",
                                      "sted_2013",
                                      "sted_2014",
                                      "sted_2015",
                                      "sted_2016",
                                      "sted_2017",
                                      "sted_2018",
                                      "sted_2019",
                                      "sted_2020"
                                      ])

    raw_df["woning"] = most_recent(raw_df, ["woning_2007",
                                            "woning_2008",
                                            "woning_2009",
                                            "woning_2010",
                                            "woning_2011",
                                            "woning_2012",
                                            "woning_2013",
                                            "woning_2014",
                                            "woning_2015",
                                            "woning_2016",
                                            "woning_2017",
                                            "woning_2018",
                                            "woning_2019",
                                            "woning_2020"
                                            ])

    raw_df["woning"] = raw_df["woning"].replace({9: np.nan,
                                                 1: "self-owned",
                                                 2: "rental",
                                                 3: "sub-rented",
                                                 4: "cost-free"})

    df = pd.concat([df, pd.get_dummies(raw_df['woning'])], axis=1)

    df["weight"] = most_recent(raw_df, ["ch07a017",
                                        "ch08b017",
                                        "ch09c017",
                                        "ch10d017",
                                        "ch11e017",
                                        "ch12f017",
                                        "ch13g017",
                                        "ch15h017",
                                        "ch16i017",
                                        "ch17j017",
                                        "ch18k017",
                                        "ch19l017",
                                        "ch20m017"
                                        ])

    df["height"] = most_recent(raw_df, ["ch07a016",
                                        "ch08b016",
                                        "ch09c016",
                                        "ch10d016",
                                        "ch11e016",
                                        "ch12f016",
                                        "ch13g016",
                                        "ch15h016",
                                        "ch16i016",
                                        "ch17j016",
                                        "ch18k016",
                                        "ch19l016",
                                        "ch20m016"
                                        ])
    df["bmi"] = (df["weight"] / (df["height"] ** 2)) * 10000

    df['len_partnership'] = most_recent(raw_df, [
        "cf20m028", "cf19l028", "cf18k028"
    ])

    df['satis_partnership'] = most_recent(raw_df, [
        "cf20m180", "cf19l180", "cf18k180"
    ])

    df["irregular_work"] = most_recent(raw_df, [
        "cw08a425",
        "cw09b425",
        "cw10c425",
        "cw11d425",
        "cw12e425",
        "cw13f425",
        "cw14g425",
        "cw15h425",
        "cw16i425",
        "cw17j425",
        "cw18k425",
        "cw19l425",
        "cw20m425"
    ])

    df["ends_meet"] = most_recent(raw_df, [
        "ci08a244",
        "ci09b244",
        "ci10c244",
        "ci11d244",
        "ci12e244",
        "ci13f244",
        "ci14g244",
        "ci15h244",
        "ci16i244",
        "ci17j244",
        "ci18k244"
    ])

    raw_df["religion_they"] = most_recent(raw_df, [
        "cr08a003",
        "cr09b003",
        "cr10c003",
        "cr11d003",
        "cr12e003",
        "cr13f003",
        "cr14g003",
        "cr15h003",
        "cr16i003",
        "cr17j003",
        "cr18k003"
    ])

    df = pd.concat([df, pd.get_dummies(raw_df["religion_they"], prefix="religion_they")], axis=1)

    df["religious"] = most_recent(raw_df, [
        "cr08a012",
        "cr09b012",
        "cr10c012",
        "cr11d012",
        "cr12e012",
        "cr13f012",
        "cr14g012",
        "cr15h012",
        "cr16i012",
        "cr17j012",
        "cr18k012"
    ])

    df["religious"] = df["religious"].replace({99: np.nan})

    raw_df["religion_you"] = most_recent(raw_df, [
        "cr08a013",
        "cr09b013",
        "cr10c013",
        "cr11d013",
        "cr12e013",
        "cr13f013",
        "cr14g013",
        "cr15h013",
        "cr16i013",
        "cr17j013",
        "cr18k013"
    ])

    df = pd.concat([df, pd.get_dummies(raw_df["religion_you"], prefix="religion_you")], axis=1)

    df["freq_see_father"] = most_recent(raw_df, [
        "cf08a020",
        "cf09b020",
        "cf10c020",
        "cf11d020",
        "cf12e020",
        "cf13f020",
        "cf14g020",
        "cf15h020",
        "cf16i020",
        "cf17j020",
        "cf18k020",
        "cf19l020",
        "cf20m020"
    ])

    df["freq_see_mother"] = most_recent(raw_df, [
        "cf08a022",
        "cf09b022",
        "cf10c022",
        "cf11d022",
        "cf12e022",
        "cf13f022",
        "cf14g022",
        "cf15h022",
        "cf16i022",
        "cf17j022",
        "cf18k022",
        "cf19l022",
        "cf20m022"
    ])

    df["life_satis"] = most_recent(raw_df, [
        "cp08a011",
        "cp09b011",
        "cp10c011",
        "cp11d011",
        "cp12e011",
        "cp13f011",
        "cp14g011",
        "cp15h011",
        "cp17i011",
        "cp18j011",
        "cp19k011",
        "cp20l011"
    ])

    df["life_satis"] = df["life_satis"].replace({999: np.nan})

    df["satis_relationship"] = most_recent(raw_df, [
        "cf08a180",
        "cf09b180",
        "cf10c180",
        "cf11d180",
        "cf12e180",
        "cf13f180",
        "cf14g180",
        "cf15h180",
        "cf16i180",
        "cf17j180",
        "cf18k180",
        "cf19l180",
        "cf20m180"
    ])

    df["satis_relationship"] = df["satis_relationship"].replace({999: np.nan})

    df["satis_family_life"] = most_recent(raw_df, [
        "cf08a181",
        "cf09b181",
        "cf10c181",
        "cf11d181",
        "cf12e181",
        "cf13f181",
        "cf14g181",
        "cf15h181",
        "cf16i181",
        "cf17j181",
        "cf18k181",
        "cf19l181",
        "cf20m181"
    ])

    df["satis_family_life"] = df["satis_family_life"].replace({999: np.nan})

    df["satis_house"] = most_recent(raw_df, [
        "cd08a001",
        "cd09b001",
        "cd10c001",
        "cd11d001",
        "cd12e001",
        "cd13f001",
        "cd14g001",
        "cd15h001",
        "cd16i001",
        "cd17j001",
        "cd18k001"
    ])

    df["satis_house"] = df["satis_house"].replace({999: np.nan})

    df["satis_financial"] = most_recent(raw_df, [
        "ci08a006",
        "ci09b006",
        "ci10c006",
        "ci11d006",
        "ci12e006",
        "ci13f006",
        "ci14g006",
        "ci15h006",
        "ci16i006",
        "ci17j006",
        "ci18k006",
        "ci19l006",
        "ci20m006"
    ])

    df["satis_financial"] = df["satis_financial"].replace({999: np.nan})

    df["satis_contacts"] = most_recent(raw_df, [
        "cs08a283",
        "cs09b283",
        "cs10c283",
        "cs11d283",
        "cs12e283",
        "cs13f283",
        "cs14g283",
        "cs15h283",
        "cs16i283",
        "cs17j283",
        "cs18k283",
        "cs19l283",
        "cs20m283"
    ])

    df["satis_contacts"] = df["satis_contacts"].replace({999: np.nan})

    df["perc_health"] = most_recent(raw_df, [
        "ch07a004",
        "ch08b004",
        "ch09c004",
        "ch10d004",
        "ch11e004",
        "ch12f004",
        "ch13g004",
        "ch15h004",
        "ch16i004",
        "ch17j004",
        "ch18k004",
        "ch19l004",
        "ch20m004"
    ])

    df["long_disease"] = most_recent(raw_df, [
        "ch07a018",
        "ch08b018",
        "ch09c018",
        "ch10d018",
        "ch11e018",
        "ch12f018",
        "ch13g018",
        "ch15h018",
        "ch16i018",
        "ch17j018",
        "ch18k018",
        "ch19l018",
        "ch20m018"
    ])

    df["hinder"] = most_recent(raw_df, [
        "ch07a020",
        "ch08b020",
        "ch09c020",
        "ch10d020",
        "ch11e020",
        "ch12f020",
        "ch13g020",
        "ch15h020",
        "ch16i020",
        "ch17j020",
        "ch18k020",
        "ch19l020",
        "ch20m020"
    ])

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

    wide_b = df_b.pivot(index='nomem_encr', columns='wave', values="aantalki").loc[:, 201801:]
    wide_b["n_children_in_hh"] = wide_b.apply(get_last_valid, axis=1)
    wide_b.reset_index(inplace=True)
    wide_b = wide_b[["nomem_encr", "n_children_in_hh"]]

    merged_df = pd.merge(df, wide_b, on='nomem_encr')

    wide_c = df_b.pivot(index='nomem_encr', columns='wave', values="nettocat")
    wide_c["nettocat_clean"] = wide_c.apply(get_last_valid, axis=1)
    wide_c.reset_index(inplace=True)
    wide_c = wide_c[["nomem_encr", "nettocat_clean"]]

    merged_df = pd.merge(merged_df, wide_c, on='nomem_encr')

    return merged_df


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
    model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=10)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "model.joblib")


