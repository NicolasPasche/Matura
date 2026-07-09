import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/merton_dataset2.csv')

# Just to reassure
print(df.shape)

# x = input dataframe, y = target dataframe
x = df.drop(columns=['price','standard_error','lower_CI','upper_CI'])
y = df['price']

# First split: train + validation + temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    x, 
    y, 
    test_size=0.3, 
    random_state=42
)

# Second split: validation + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.5, 
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Standardize the features
x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

print("\nAfter scaling:")
print("Train mean:", X_train.mean())
print("Train std:", X_train.std())

print("Validation mean:", X_val.mean())
print("Validation std:", X_val.std())

# Standardize the target
y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val = y_scaler.transform(y_val.values.reshape(-1, 1))
y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

print("\nAfter scaling target:")
print("Train mean:", y_train.mean())
print("Train std:", y_train.std())


# Save processed data
np.save("../data/processed/X_train2.npy", X_train)
np.save("../data/processed/X_val2.npy", X_val)
np.save("../data/processed/X_test2.npy", X_test)

np.save("../data/processed/y_train2.npy", y_train)
np.save("../data/processed/y_val2.npy", y_val)
np.save("../data/processed/y_test2.npy", y_test)


joblib.dump(
    x_scaler,
    "../data/processed/x_scaler2.pkl"
)

joblib.dump(
    y_scaler,
    "../data/processed/y_scaler2.pkl"
)

print("Dataset preparation finished.")