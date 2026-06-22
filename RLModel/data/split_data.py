print("Start Date:", df.index.min())
print("End Date:", df.index.max())
import pandas as pd
test_start = df.index.max() - pd.DateOffset(months=6)
print("Test starts from:", test_start)
train_df = df[df.index < test_start]
test_df = df[df.index >= test_start]

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))

print("\nTrain Start:", train_df.index.min())
print("Train End:", train_df.index.max())

print("\nTest Start:", test_df.index.min())
print("Test End:", test_df.index.max())

print("\nTrain Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

train_df.to_csv("BTC_USDT_train.csv")
test_df.to_csv("BTC_USDT_test.csv")

print("Train and Test files saved successfully!")
