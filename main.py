from outlier_handler.handlers import IQRHandler, LOFHandler, ZScoreHandler, DBSCANHandler, MADHandler
import pandas as pd
import numpy as np

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'Value': np.random.randn(100),  # Example data
})

# Instantiate handlers
iqr_handler = IQRHandler(df.copy())
lof_handler = LOFHandler(df.copy())
zscore_handler = ZScoreHandler(df.copy())
dbscan_handler = DBSCANHandler(df.copy())
mad_handler = MADHandler(df.copy())

# Handle outliers using each handler
df_iqr_handled = iqr_handler.handle_outliers('Value', remove_or_fill_with_quartile='fill', impute_logic='median')
df_lof_handled = lof_handler.handle_outliers('Value', remove_or_fill_with_quartile='fill', impute_logic='median')
df_zscore_handled = zscore_handler.handle_outliers('Value', remove_or_fill_with_quartile='fill', impute_logic='median')
df_dbscan_handled = dbscan_handler.handle_outliers('Value', remove_or_fill_with_quartile='fill', impute_logic='median')
df_mad_handled = mad_handler.mad_based_outlier('Value', remove_or_fill_with_quartile='fill', impute_logic='median')

# Display modified DataFrames
print("IQR Handler Result:")
print(df_iqr_handled)

print("\nLOF Handler Result:")
print(df_lof_handled)

print("\nZ-Score Handler Result:")
print(df_zscore_handled)

print("\nDBSCAN Handler Result:")
print(df_dbscan_handled)

print("\nMAD Handler Result:")
print(df_mad_handled)
