from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    object_features = (
        pd.DataFrame(working_train_df.select_dtypes(include="object").nunique())
        .reset_index()
        .rename(columns={"index": "feature", 0: "count"})
    )
    bin_features = object_features[object_features["count"] == 2]["feature"].tolist()
    cat_features = object_features[object_features["count"] > 2]["feature"].tolist()

    ordinalencoder = OrdinalEncoder()
    onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    transformed_bin_train = ordinalencoder.fit_transform(working_train_df[bin_features])
    transformed_cat_train = onehotencoder.fit_transform(working_train_df[cat_features])
    cat_cols = onehotencoder.get_feature_names_out(cat_features)
    encoded_bin_train = pd.DataFrame(
        transformed_bin_train, index=working_train_df.index, columns=bin_features
    )
    encoded_cat_train = pd.DataFrame(
        transformed_cat_train, index=working_train_df.index, columns=cat_cols
    )

    transformed_bin_val = ordinalencoder.transform(working_val_df[bin_features])
    transformed_cat_val = onehotencoder.transform(working_val_df[cat_features])
    encoded_bin_val = pd.DataFrame(
        transformed_bin_val, index=working_val_df.index, columns=bin_features
    )
    encoded_cat_val = pd.DataFrame(
        transformed_cat_val, index=working_val_df.index, columns=cat_cols
    )

    transformed_bin_test = ordinalencoder.transform(working_test_df[bin_features])
    transformed_cat_test = onehotencoder.transform(working_test_df[cat_features])
    encoded_bin_test = pd.DataFrame(
        transformed_bin_test, index=working_test_df.index, columns=bin_features
    )
    encoded_cat_test = pd.DataFrame(
        transformed_cat_test, index=working_test_df.index, columns=cat_cols
    )

    working_train_df[bin_features] = encoded_bin_train
    working_train_df.drop(columns=cat_features, inplace=True)
    working_train_df = pd.concat([working_train_df, encoded_cat_train], axis=1)

    working_val_df[bin_features] = encoded_bin_val
    working_val_df.drop(columns=cat_features, inplace=True)
    working_val_df = pd.concat([working_val_df, encoded_cat_val], axis=1)

    working_test_df[bin_features] = encoded_bin_test
    working_test_df.drop(columns=cat_features, inplace=True)
    working_test_df = pd.concat([working_test_df, encoded_cat_test], axis=1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    mean_imputer = SimpleImputer(strategy="median")
    transformed_missing_train = mean_imputer.fit_transform(working_train_df)
    transformed_missing_val = mean_imputer.transform(working_val_df)
    transformed_missing_test = mean_imputer.transform(working_test_df)

    working_train_df = pd.DataFrame(
        transformed_missing_train, index=working_train_df.index
    )
    working_val_df = pd.DataFrame(transformed_missing_val, index=working_val_df.index)
    working_test_df = pd.DataFrame(
        transformed_missing_test, index=working_test_df.index
    )

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    mm_scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
    train = mm_scaler.fit_transform(working_train_df)
    val = mm_scaler.transform(working_val_df)
    test = mm_scaler.transform(working_test_df)

    return train, val, test
