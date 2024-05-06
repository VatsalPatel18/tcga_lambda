import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

class SurvivalDataProcessor:
    def __init__(self, path_expression, path_survival):
        self.path_expression = path_expression
        self.path_survival = path_survival
        self.df_expression = None
        self.df_survival = None
        self.train_inputs = None
        self.train_labels = None
        self.test_inputs = None
        self.test_labels = None

    def load_data(self):
        # Load expression data
        self.df_expression = pd.read_csv(self.path_expression)
        self.df_expression = self.df_expression.rename(columns={'Unnamed: 0': 'PatientID'})
        self.df_expression.set_index('PatientID', inplace=True)

        # Load survival data
        self.df_survival = pd.read_csv(self.path_survival)
        self.df_survival.set_index('PatientID', inplace=True)

        # Ensure data consistency
        common_ids = self.df_expression.index.intersection(self.df_survival.index)
        self.df_expression = self.df_expression.loc[common_ids]
        self.df_survival = self.df_survival.loc[common_ids]

    def preprocess_data(self):
        # Log transform and robust scale the expression data
        df_log_scaled = np.log1p(self.df_expression)
        scaler = RobustScaler()
        self.df_expression = pd.DataFrame(scaler.fit_transform(df_log_scaled), index=self.df_expression.index, columns=self.df_expression.columns)

    def split_data(self, long_criteria=36, short_criteria=12, test_size=0.3, random_state=42):
        # Filtering based on survival criteria
        long_df = self.df_survival[(self.df_survival['Overall Survival (Months)'] > long_criteria) & (self.df_survival['Overall Survival Status'] == 0)]
        short_df = self.df_survival[(self.df_survival['Overall Survival (Months)'] < short_criteria) & (self.df_survival['Overall Survival Status'] == 1)]

        # Join expression data
        long_df = long_df.join(self.df_expression, how='inner')
        short_df = short_df.join(self.df_expression, how='inner')

        # Prepare features and labels
        features_long = long_df.drop(['Overall Survival Status', 'Overall Survival (Months)'], axis=1)
        labels_long = pd.Series([0] * len(features_long), index=features_long.index)
        features_short = short_df.drop(['Overall Survival Status', 'Overall Survival (Months)'], axis=1)
        labels_short = pd.Series([1] * len(features_short), index=features_short.index)

        # Train-test split
        train_features_long, test_features_long, train_labels_long, test_labels_long = train_test_split(
            features_long, labels_long, test_size=test_size, random_state=random_state)
        train_features_short, test_features_short, train_labels_short, test_labels_short = train_test_split(
            features_short, labels_short, test_size=test_size, random_state=random_state)

        # Combine training and test sets
        self.train_inputs = pd.concat([train_features_long, train_features_short])
        self.train_labels = pd.concat([train_labels_long, train_labels_short])
        self.test_inputs = pd.concat([test_features_long, test_features_short])
        self.test_labels = pd.concat([test_labels_long, test_labels_short])

    def get_dataset(self):
        return {
            'train_inputs': self.train_inputs,
            'train_labels': self.train_labels,
            'test_inputs': self.test_inputs,
            'test_labels': self.test_labels
        }
