# handlers/zscore_handler.py

import pandas as pd
import numpy as np
from scipy import stats as st

class ZScoreHandler:
    def __init__(self, df):
        self.df = df
    
    def handle_outliers(self, column_name, threshold=3, remove_or_fill_with_quartile=None, impute_logic='median'):
        """
        Detect outliers using z-score and handle them based on specified logic.

        Parameters:
        - column_name (str): Name of the column in the DataFrame to detect outliers and handle.
        - threshold (float): Z-score threshold to identify outliers.
        - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.
        - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

        Returns:
        - df (DataFrame): DataFrame with outliers handled based on specified logic.
        """
        # Calculate z-score for the column
        self.df['z_score'] = st.zscore(self.df[column_name])
        
        # Identify outliers based on z-score
        outlier_mask = np.abs(self.df['z_score']) > threshold
        
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
        
        # Drop z_score column
        self.df = self.df.drop(columns=['z_score'])
        
        return self.df
