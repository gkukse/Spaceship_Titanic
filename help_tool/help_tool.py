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
from scipy.stats import chi2_contingency


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

    return info_df.T


def cabin_split(df):
    # Form deck/num/side, where side can be either P for Port or S for Starboard.
    df[['Deck', 'Deck_num', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # Available Deck values are A-G, T
    df['Deck'] = df['Deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})

    # Side has only 2 values and will be updated to bool feature
    df['Side'] = (df['Side'] == 'S').astype(int) 

    for feature in ['Deck_num']:
        try:
            df[feature] = df[feature].astype(int)
        except: 
            df[feature] = df[feature].astype(float)

    return df


def cleaning(df, binary_features):
    # Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group.
    df[['Group', 'Group_Id']] = df['PassengerId'].str.split('_', expand=True)

    # Separating last name for family identifications
    df[['First_Name', 'Last_Name']] = df['Name'].str.split(' ', expand=True)

    cabin_split(df)

    
    for feature in binary_features + ['Group', 'Group_Id']:
        try:
            df[feature] = df[feature].astype(int)
        except: 
            df[feature] = df[feature].astype(float)


    df.drop(columns=['PassengerId', 'Name'], inplace=True)
    return df
   


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
    """Compute and visualize Phi correlation matrix for binary features"""
    corr_matrix = pd.DataFrame(index=feature_list, columns=feature_list)

    # Calculate correlation coefficients
    for i in range(len(feature_list)):
        for j in range(i, len(feature_list)):
            feature1 = feature_list[i]
            feature2 = feature_list[j]
            corr_coef = matthews_corrcoef(df[feature1], df[feature2])
            corr_matrix.loc[feature1, feature2] = corr_coef

    # Filter to upper or lower triangular part based on the parameter
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    filtered_matrix = corr_matrix.where(mask)

    # Plot the correlation matrix
    sns.heatmap(filtered_matrix.astype(float), annot=True, annot_kws={"size": 8},
                cmap='rocket', fmt=".2f", vmin=-1, vmax=1)  # Adjust vmin and vmax as needed
    plt.title(f'Phi Correlation Matrix of Binary Attributes')
    plt.show()

def countplot_per_feature(df, feature_list):
    for i, feature_to_exclude in enumerate(feature_list):
        features_subset = [
            feature for feature in feature_list if feature != feature_to_exclude]

        """ Countplot for 5 features """
        fig, axes = plt.subplots(
            1, len(feature_list)-1, figsize=(20, 3))  # Changed the number of columns to 5

        palette = 'rocket'

        for i, feature in enumerate(features_subset):
            sns.countplot(data=df, x=feature, hue=feature_to_exclude,
                          ax=axes[i], palette=palette)
            axes[i].get_legend().remove()
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.suptitle("Binary feature analysis", size=16, y=1.02)
        plt.legend(title=feature_to_exclude,
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

def chi_squared_test(df, feature_tuple):
    """ Chi-squared test for binary features """

    contingency_table = pd.crosstab(df[feature_tuple[0]], df[feature_tuple[1]])

    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    a = f"Chi-squared statistic: {chi2}."
    b = f"P-value: {p_value}."

    if p_value < alpha:
        print(f"{a} {b} Reject the null hypothesis.")
    else:
        print(f"{a} {b} Do not reject the null hypothesis.")


def biserial_heatmap(df, continues_features, binary_features):
    """ Biserial correlation for binary and continues features. """
    correlation_matrix = pd.DataFrame(
        index=binary_features, columns=continues_features)


    for binary_feature in binary_features:

        for continuous_feature in continues_features:

            biserial_corr, _ = stats.pointbiserialr(

                df[binary_feature], df[continuous_feature])

            correlation_matrix.loc[binary_feature,

                                continuous_feature] = biserial_corr


    correlation_matrix = correlation_matrix.apply(pd.to_numeric)


    sns.heatmap(pd.DataFrame(correlation_matrix),
                annot=True, cmap="rocket", fmt=".2f")

    plt.title("Biserial Correlation Heatmap")

    plt.show()


def confidence_intervals(data, type) -> None:
    """Calculate Confidence Intervals for a given dataset."""

    sample_mean = np.mean(data)

    if type == 'Continuous':
        # Continuous feature
        # ddof=1 for sample standard deviation
        sample_std = np.std(data, ddof=1)
        critical_value = stats.norm.ppf((1 + confidence_level) / 2)

    elif type == 'Discrete':
        # Discrete feature
        # Sample standard deviation for discrete data
        sample_std = np.sqrt(np.sum((data - sample_mean)**2) / (len(data) - 1))
        # t-distribution for discrete data
        critical_value = stats.t.ppf(
            (1 + confidence_level) / 2, df=len(data) - 1)

    standard_error = sample_std / np.sqrt(len(data))
    margin_of_error = critical_value * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    print(f"Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")

def significance_t_test(df: pd.DataFrame, feature: str, change_feature: str,
                        min_change_value: float, max_change_value: float) -> None:
    """Perform a t-test (sample size is small or when 
    the population standard deviation is unknown) and follows a normal distribution."""
    t_stat, p_value = stats.ttest_ind(df[df[change_feature] == min_change_value][feature],
                                      df[df[change_feature] == max_change_value][feature], equal_var=False)

    if p_value < alpha:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Reject null hypothesis')
    else:
        print(
            f'p-value = {p_value:.4f} between {feature} and {change_feature}. Fail to reject null hypothesis')


def feature_transpose(df, feature_list):
    """ Transpose a few features into a new dataframe"""
    thresholds = df[feature_list].T
    thresholds.reset_index(inplace=True)
    thresholds.columns = thresholds.iloc[0]
    thresholds.drop(thresholds.index[0], inplace=True)

    return thresholds


def cross_val_thresholds(fold, X, y, thresholds_df, classifiers):
    """ Cross validation with threshold adjustments """
    kf = KFold(n_splits=fold)
    # Initialize lists to store metric scores and confusion matrices
    metric_scores = {metric: {clf_name: [] for clf_name in classifiers.keys(
    )} for metric in ['accuracy', 'precision', 'recall', 'f1']}
    confusion_matrices = {clf_name: np.zeros(
        (2, 2)) for clf_name in classifiers.keys()}

    for train_index, val_index in kf.split(X):
        X_train_i, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_i, y_val = y.iloc[train_index], y.iloc[val_index]

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_i, y_train_i)

            # Threshold update
            # Assuming binary classification
            scores = clf.predict_proba(X_val)[:, 1]
            optimal_threshold = thresholds_df[clf_name].iloc[0]
            y_pred = (scores > optimal_threshold).astype(int)

            # Calculate metrics
            metric_scores['accuracy'][clf_name].append(
                accuracy_score(y_val, y_pred))
            metric_scores['precision'][clf_name].append(
                precision_score(y_val, y_pred))
            metric_scores['recall'][clf_name].append(
                recall_score(y_val, y_pred))
            metric_scores['f1'][clf_name].append(f1_score(y_val, y_pred))

            # Compute confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            confusion_matrices[clf_name] += cm

    # Calculate average scores
    avg_metric_scores = {metric: {clf_name: np.mean(scores) for clf_name, scores in clf_scores.items(
    )} for metric, clf_scores in metric_scores.items()}

    # Average confusion matrices
    avg_confusion_matrices = {
        clf_name: matrix / fold for clf_name, matrix in confusion_matrices.items()}

    cv_results = []
    for clf_name, scores in avg_metric_scores['accuracy'].items():
        cv_results.append({
            'Classifier': classifiers[clf_name].__class__.__name__,
            'CV Mean Accuracy': np.mean(scores),
            'CV Mean Precision': np.mean(avg_metric_scores['precision'][clf_name]),
            'CV Mean Recall': np.mean(avg_metric_scores['recall'][clf_name]),
            'CV Mean F1': np.mean(avg_metric_scores['f1'][clf_name]),
            'Confusion Matrix': avg_confusion_matrices[clf_name]
        })

    model_info = pd.DataFrame(cv_results)
    return model_info

def cross_validation_param(model_info):
    """ Parameter heatmap """
    heatmap_data = model_info

    heatmap_data.set_index('Classifier', inplace=True)

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5)
    plt.title('Model Performance Metrics')
    plt.show()


def cross_validation_confusion_matrix(model_info):
    """ Cross Validation Matrix """
    f, ax = plt.subplots(2, 4, figsize=(15, 6))
    ax = ax.flatten()

    for i, row in model_info.iterrows():
        cm = row['Confusion Matrix']
        sns.heatmap(cm, ax=ax[i], annot=True, fmt='2.0f')
        ax[i].set_title(f"Matrix for {row['Classifier']}")
        ax[i].set_xlabel('Predicted Label')
        ax[i].set_ylabel('True Label')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()