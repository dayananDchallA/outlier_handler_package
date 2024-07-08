# handlers/lof_handler.py

from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np

class LOFHandler:
    def __init__(self, df):
        self.df = df
    
    def handle_outliers(self, column_name, n_neighbors=20, contamination=0.1, impute_logic='median', remove_or_fill_with_quartile=None):
        """
        Detect outliers using LocalOutlierFactor and handle them based on specified logic.

        Parameters:
        - column_name (str): Name of the column in the DataFrame to detect outliers and impute.
        - n_neighbors (int): Number of neighbors to consider for LocalOutlierFactor.
        - contamination (float): Proportion of outliers expected in the data.
        - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').
        - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.

        Returns:
        - df (DataFrame): DataFrame with outliers handled based on specified logic.
        """
        # Initialize LocalOutlierFactor model
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        
        # Fit model and predict outliers
        outlier_scores = lof.fit_predict(self.df[[column_name]])
        outlier_mask = outlier_scores == -1  # -1 indicates outlier according to LOF
        
        # Calculate quartiles for imputation logic
        if remove_or_fill_with_quartile == "fill":
            if impute_logic == 'median':
                imputed_value = self.df[column_name].median()
            elif impute_logic == 'mean':
                imputed_value = self.df[column_name].mean()
            else:
                raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
            
            self.df[column_name] = np.where(outlier_mask, imputed_value, self.df[column_name])
        
        elif remove_or_fill_with_quartile == "drop":
            self.df = self.df.loc[~outlier_mask]
        
        return self.df
