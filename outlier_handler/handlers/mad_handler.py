# handlers/mad_handler.py

import pandas as pd
import numpy as np

class MADHandler:
    def __init__(self, df):
        self.df = df
    
    def mad_based_outlier(self, col, threshold=3.5, remove_or_fill_with_quartile=None, impute_logic='median'):
        """
        Detect outliers using Mean Absolute Deviation (MAD) and handle them based on specified logic.

        Parameters:
        - col (str): Name of the column in the DataFrame to detect outliers and handle.
        - threshold (float): Threshold value to determine outliers based on MAD (default is 3.5).
        - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.
        - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

        Returns:
        - df (DataFrame): DataFrame with outliers handled based on specified logic.
        """
        # Calculate MAD
        mad = np.abs(self.df[col] - self.df[col].median()).median()
        
        # Calculate lower and upper fences
        lower_fence = self.df[col].median() - threshold * mad
        upper_fence = self.df[col].median() + threshold * mad
        
        # Print information about outliers
        print('Lower Fence:', lower_fence)
        print('Upper Fence:', upper_fence)
        
        # Identify outliers
        outlier_mask = (self.df[col] < lower_fence) | (self.df[col] > upper_fence)

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
