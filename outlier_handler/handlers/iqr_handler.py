# handlers/iqr_handler.py

import pandas as pd
import numpy as np

class IQRHandler:
    def __init__(self, df):
        self.df = df
    
    def handle_outliers(self, col, remove_or_fill_with_quartile=None, impute_logic='median'):
        """
        Detect outliers using IQR (Interquartile Range) and handle them based on specified logic.

        Parameters:
        - col (str): Name of the column in the DataFrame to detect outliers and handle.
        - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). 
          If 'fill', impute with quartile-based values.
        - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

        Returns:
        - df (DataFrame): DataFrame with outliers handled based on specified logic.
        """
        # Calculate quartiles and IQR
        q1 = self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Calculate fences for outlier detection
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Print information about outliers
        print('Lower Fence:', lower_fence)
        print('Upper Fence:', upper_fence)
        print('Total number of outliers left:', self.df[(self.df[col] < lower_fence) | (self.df[col] > upper_fence)].shape[0])
        
        # Handle outliers based on specified logic
        if remove_or_fill_with_quartile == "drop":
            self.df = self.df.loc[(self.df[col] >= lower_fence) & (self.df[col] <= upper_fence)]
        
        elif remove_or_fill_with_quartile == "fill":
            if impute_logic == 'median':
                imputed_value = self.df[col].median()
            elif impute_logic == 'mean':
                imputed_value = self.df[col].mean()
            else:
                raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
            
            self.df[col] = np.where(self.df[col] < lower_fence, imputed_value, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_fence, imputed_value, self.df[col])
        
        return self.df
