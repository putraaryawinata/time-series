import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_windowed_dataset(series, window_size, stride, num_features=1):
    """
    Create a windowed time series dataset with a specified window size and stride.

    Parameters:
    - series: numpy array or pandas Series, representing the time series data.
    - window_size: integer, size of the input window (number of time steps).
    - stride: integer, number of steps to move the window forward.
    - num_features: integer, number of features in each time step.

    Returns:
    - X: numpy array of shape (num_samples, window_size, num_features), input data.
    - y: numpy array of shape (num_samples, num_features), target data.
    """

    # Calculate the total number of samples
    if np.any(series == 0.0):
        raise("cannot use this series") 
    num_samples = (len(series) - window_size) // stride

    # Initialize arrays to store input and output data
    X = np.zeros((num_samples, window_size, num_features))
    y = np.zeros((num_samples, num_features))

    # Fill in input and output data arrays
    for i in range(num_samples):
        start_idx = i * stride
        end_idx = start_idx + window_size
        X[i] = series[start_idx:end_idx].reshape(window_size, num_features)

        y[i] = series[end_idx]

    return X, y

if __name__ == "__main__":
    dataset =  pd.read_csv("GlobalElectricityStatistics.csv")
    features = ['net generation', 'net consumption', 'imports ', 'exports ', 'net imports ', 'installed capacity ', 'distribution losses ']
    
    # Filter rows where 'City' is 'New York'
    consum_dataset = dataset[dataset['Features'] == 'net consumption']
    countries = list(consum_dataset["Country"])

    window_size = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    kernels = ['linear', 'poly', 'sigmoid'] # 'rbf', 
    stride = 1
    for kernel in kernels:
        for ws in window_size:
            for i, country in enumerate(countries):
                # print(country)
                consum = consum_dataset[consum_dataset['Country'] == country].values[:, 3:]
                consum = consum.reshape((-1))
                
                # num_obs = 42 # number of observation years
                try:
                    X_i, y_i = create_windowed_dataset(consum, ws, stride)
                    X_i, y_i = np.squeeze(X_i, axis=-1), np.squeeze(y_i, axis=-1)
                    if i == 0:
                        X, y = X_i, y_i
                    else:
                        X, y = np.append(X, X_i, axis=0), np.append(y, y_i, axis=0)
                except:
                    pass
            
            nan_indices = np.isnan(X)
            remove_index = list(np.unique(np.where(nan_indices)[0]))
            # Create a boolean mask where True indicates elements to keep
            mask = np.ones(len(X), dtype=bool)
            mask[remove_index] = False

            # Use boolean indexing to select elements to keep
            X = X[mask]
            y = y[mask]
            # print(f"shape X: {X.shape}")
            # print(f"shape y: {y.shape}")
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create SVR model
            svr = SVR(kernel=kernel, C=1)  # You can specify other kernel functions as well

            # Train the SVR model
            svr.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = svr.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            print(f"Mean Squared Error on window_size={ws} with {kernel}: {mse}")