"""Helper module for EDA notebook to perform 
data cleaning and preprocessing"""


from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, roc_curve)
from sklearn.model_selection import KFold
from unidecode import unidecode
import textblob
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
pd.plotting.register_matplotlib_converters()
import os


"""Statistics"""
alpha = 0.05  # Significance level
confidence_level = 0.95


def csv_download(relative_path: str) -> pd.DataFrame:
    """Download data."""
    absolute_path = os.path.abspath(relative_path)
    df = pd.read_csv(absolute_path, index_col=False, header=0)

    return df


def first_look(df: pd.DataFrame) -> None:
    """Performs initial data set analysis."""
    df_size = df.shape

    df_type = df.dtypes.to_frame().T.rename(index={df.index[0]: 'dtypes'})
    df_null = df.apply(lambda x: x.isna().sum()).to_frame().T.rename(index={df.index[0]: 'Null values, Count'})

    # Copy of df_null for Null %
    df_null_proc = round(df_null / df_size[0] * 100, 1)
    df_null_proc = df_null_proc.rename(index={df_null.index[0]: 'Null values, %'})

    info_df = pd.concat([df_type, df_null, df_null_proc])

    print(f'Dataset has {df.shape[0]} observations and {df_size[1]} features')
    print(f'Columns with all empty values {df.columns[df.isna().all(axis=0)].tolist()}')
    print(f'Dataset has {df.duplicated().sum()} duplicates')

    return info_df


    
def remove_empty_rows(df: pd.DataFrame, exception_col: str):
    """ Removing empty rows, where all rows must be empty, apart from exception columns, i.e. 'id' """

    columns_to_check = [col for col in df.columns if col != exception_col]
    non_empty_rows = df[columns_to_check].notna().any(axis=1)
    df_filtered = df[non_empty_rows]

    return df_filtered

def distribution_check(df: pd.DataFrame) -> None:
    """Box plot graph for identifying numeric column outliers, normality of distribution."""
    df = df.reset_index(drop=True)

    for feature in df.columns:

        if df[feature].dtype.name in ['object', 'bool']:
            pass

        else:

            fig, axes = plt.subplots(1, 3, figsize=(12, 3))

            print(f'{feature}')

            # Outlier check (Box plot)
            df.boxplot(column=feature, ax=axes[0])
            axes[0].set_title(
                f'{feature} ranges from {df[feature].min()} to {df[feature].max()}')

            # Distribution check (Histogram).
            sns.histplot(data=df, x=feature, kde=True, bins=20, ax=axes[1])
            axes[1].set_title(f'Distribution of {feature}')

            # Normality check (QQ plot).
            sm.qqplot(df[feature].dropna(), line='s', ax=axes[2])
            axes[2].set_title(f'Q-Q plot of {feature}')

            plt.tight_layout()
            plt.show()


def heatmap(df: pd.DataFrame, name: str, method: str) -> None:
    """ Plotting the heatmap of correlation matrix """
    plt.figure(figsize=(8, 5))
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                vmin=-1, vmax=1, mask=mask)
    plt.title(f'Correlation {name.capitalize()} Attributes')
    plt.show()


def dummy_columns(df, feature_list):
    """ Created a dummy and replaces the old feature with the new dummy """
    df_dummies = pd.get_dummies(df[feature_list])
    df_dummies = df_dummies.astype(int)

    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=feature_list, inplace=True)

    # Drop '_No' features and leave '_Yes'
    # Replace the original column with new dummy
    df = df.drop(columns=[col for col in df.columns if col.endswith('_No')])
    df.columns = [col.replace('_Yes', '') for col in df.columns]
    return df

def phi_corr_matrix(df, feature_list):
    """ Phi correlation for binary features"""
    phi_corr_matrix = pd.DataFrame(index=feature_list, columns=feature_list)

    for feature1 in feature_list:
        for feature2 in feature_list:
            phi_corr_matrix.loc[feature1, feature2] = matthews_corrcoef(
                df[feature1], df[feature2])

    thresholded_matrix = phi_corr_matrix[(
        phi_corr_matrix <= (-1) * alpha) | (phi_corr_matrix >= alpha)]

    sns.heatmap(thresholded_matrix.astype(float),
                annot=True, annot_kws={"size": 8}, cmap='rocket', fmt=".2f")
    plt.title(
        f'Phi correlation coefficient of Binary Attributes (Thresholds +/- {alpha})')
    plt.show()