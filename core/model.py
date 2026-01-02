import numpy as np
from sklearn.neural_network import MLPRegressor
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

class ForecastModel:
    def __init__(self, look_back=24, horizon=1):
        self.look_back = look_back
        self.horizon = horizon
        self.model = None

    def build_model(self):
        """
        Builds a Multi-Layer Perceptron (MLP) as a robust, 
        computationally lightweight alternative to LSTM.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        return self.model

    def train(self, X, y):
        """
        Trains the model. 
        X shape: (samples, look_back)
        y shape: (samples,)
        """
        if self.model is None:
            self.build_model()
        
        # Squeeze X if it's 3D (samples, look_back, 1) to (samples, look_back)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1])
            
        self.model.fit(X, y)

    def forecast(self, last_window):
        """
        Infers the next value.
        """
        if len(last_window.shape) == 1:
            last_window = last_window.reshape(1, -1)
        elif len(last_window.shape) == 3:
            last_window = last_window.reshape(1, -1)
            
        prediction = self.model.predict(last_window)
        return prediction.reshape(1, 1)

    def multi_step_forecast(self, current_window, steps):
        """
        Recursive forecasting for a user-defined horizon.
        """
        predictions = []
        # Ensure window is 1D for manipulation
        if len(current_window.shape) > 1:
            temp_window = current_window.flatten().tolist()
        else:
            temp_window = current_window.tolist()
        
        for _ in range(steps):
            win_arr = np.array(temp_window[-self.look_back:]).reshape(1, -1)
            pred = self.model.predict(win_arr)[0]
            predictions.append(pred)
            temp_window.append(pred)
            
        return np.array(predictions)
