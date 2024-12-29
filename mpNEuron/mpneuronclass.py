import numpy as np
from sklearn.metrics import accuracy_score

class MPNeuron:
    
    def __init__(self):
        self.b = None  # Initialize threshold to None
    
    def model(self, x):
        return np.sum(x) >= self.b  # Model logic: returns True or False
    
    def predict(self, X):
        y = []
        for sample in X:  # Iterate over each sample in X
            result = self.model(sample)
            y.append(result)
        return np.array(y)  # Return predictions as a NumPy array
    
    def fit(self, X, y):
        accuracy = {}  # Dictionary to store accuracy for each threshold value
        for b in range(X.shape[1] + 1):  # Iterate over possible threshold values
            self.b = b
            y_pred = self.predict(X)
            accuracy[b] = accuracy_score(y, y_pred)  # Compute and store accuracy
        
        # Find the threshold `b` that gives the highest accuracy
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b  # Set the best threshold as the model parameter
        print('Optimal value of b is', best_b)
        print('Highest accuracy is', accuracy[best_b])
