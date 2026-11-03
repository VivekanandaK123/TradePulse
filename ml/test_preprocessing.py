
import pandas as pd

from ml.preprocessing import preprocess_bars

# Load raw data
df = pd.read_csv("data/raw/META_5min.csv")

# Preserve original data for comparison
original = df.copy(deep=True)

# Preprocess
processed = preprocess_bars(df)

# Results
print("\nProcessed shape:", processed.shape)
print("\nProcessed dtypes:")
print(processed.dtypes)

print("\nFirst 5 rows:")
print(processed.head())

# Verify original dataset was not modified
pd.testing.assert_frame_equal(df, original)

print("\nRaw dataset remains unchanged.")

# Verify chronological ordering
assert processed["timestamp"].is_monotonic_increasing

print("Chronological ordering verified.")
