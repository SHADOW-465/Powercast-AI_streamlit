import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

class LoadPreprocessor:
    def __init__(self, window_length=11, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.scaler = MinMaxScaler()
        
    def smooth_data(self, data):
        """
        Applies Savitzky-Golay filtering to smooth the load signal.
        """
        # Ensure window_length is odd and less than data length
        w = self.window_length
        if len(data) < w:
            w = len(data) if len(data) % 2 != 0 else len(data) - 1
        
        if w < 3: # Savgol needs at least window 3
            return data
            
        return savgol_filter(data, w, self.polyorder)

    def normalize_data(self, data):
        """
        Normalizes data to [0, 1] range.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        """
        Converts normalized data back to original scale.
        """
        return self.scaler.inverse_transform(data)

    def prepare_sliding_window(self, data, look_back):
        """
        Prepares X, y pairs for LSTM training.
        """
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)
