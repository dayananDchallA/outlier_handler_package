# handlers/dbscan_handler.py

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

class DBSCANHandler:
    def __init__(self, df):
        self.df = df
    
    def handle_outliers(self, col, eps=0.5, min_samples=5, remove_or_fill_with_quartile=None, impute_logic='median'):
        """
        Detect outliers using DBSCAN and handle them based on specified logic.

        Parameters:
        - col (str): Name of the column in the DataFrame to detect outliers and handle.
        - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.
        - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

        Returns:
        - df (DataFrame): DataFrame with outliers handled based on specified logic.
        """
        # Reshape the data for DBSCAN fitting
        X = self.df[[col]].values

        # Fit DBSCAN model
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)

        # Identify outliers
        outlier_mask = dbscan.labels_ == -1  # -1 indicates outliers

        # Handle outliers based on specified logic
        if remove_or_fill_with_quartile == "drop":
            self.df = self.df.loc[~outlier_mask]
        
        elif remove_or_fill_with_quartile == "fill":
            if impute_logic == 'median':
                imputed_value = self.df[col].median()
            elif impute_logic == 'mean':
                imputed_value = self.df[col].mean()
            else:
                raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
            
            self.df[col] = np.where(outlier_mask, imputed_value, self.df[col])
        
        return self.df
