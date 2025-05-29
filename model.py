import numpy as np
from typing import List
from loguru import logger
import json

class  UnivariteLinearRegression():

    def __init__(self, bias : float = 0, theta1: float = 0):
        
        self.bias = bias
        self.theta1 = theta1
        self.loss_history = []
    
    def dump_params(self, path : str):
        """Serialize and save model"""

        try:
            with open(path, "w") as f:
                params = {"bias" : self.bias,
                        "theta1" : self.theta1}
                json.dump(params, f)
                logger.info(f"Params {params} saved to {path}")
        except Exception as e:
            logger.error(f"Failed to dump params: {e}")
    
    def load_params(self, path : str):
        """Load model weights from file"""

        try:
            with open(path, "r") as f:  
                params = json.load(f)

                self.bias = params["bias"]
                self.theta1 = params["theta1"]
                logger.info(f"Params {params} loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load params: {e}")

    
    def get_mse_loss(self, y_predict : np.array, y_true : np.array) -> float:
        """Computes Mean Squared Error Loss"""

        errors = (y_predict - y_true)
        mse = float(np.mean(errors ** 2))
        return mse
    
    def get_gradient(self, X: np.array, y_true: np.array, y_pred: np.array) -> List[float]:
        """Computes gradient with respect to parameter vector"""

        errors = (y_pred - y_true)

        #compute partial derivs
        bias_partial_derivative = float(np.mean(errors))
        theta1_partial_derivative = float(np.mean(errors * X))

        gradient = [bias_partial_derivative, theta1_partial_derivative]
        return gradient



    def predict(self, X: np.array) -> np.array:

        Y_predicts = self.bias + self.theta1 * X

        return Y_predicts
    
    def fit(self, X: List[float], Y: List[float], n_iterations : int = 100, lr : float = 0.001):
        
        logger.info(f"Start Training with {n_iterations} iterations and Learning rate = {lr}")

        X = np.array(X)
        Y = np.array(Y)


        #Scale X and Y by STD to avoid Gradient Explosion
        X_std = (X - np.mean(X)) / np.std(X)
        Y_std = (Y - np.mean(Y)) / np.std(Y)

        self.loss_history = []

        for i in range(n_iterations):
            
            #predict
            Y_predicts = self.predict(X_std)

            #compute loss
            loss = self.get_mse_loss(Y_predicts, Y_std)
            self.loss_history.append(loss)
            #compute gradient & update params
            bias_partial_deriv, theta1_partial_deriv = self.get_gradient(X_std, Y_std, Y_predicts)

            print(bias_partial_deriv, theta1_partial_deriv)

            self.bias -= lr * bias_partial_deriv
            self.theta1 -= lr * theta1_partial_deriv

            logger.debug(f"Iteration #{i + 1}: (STD SCALED) bias={self.bias} ; theta1={self.theta1} | MSE LOSS (STD SCALED)= {loss}")

        #Scale learned parameters back to original units
        self.theta1 = float(self.theta1 * np.std(Y) / np.std(X))
        self.bias = float(np.mean(Y) - self.theta1 * np.mean(X))
        
        logger.info(f'Learned parameters in Original Scale: {(self.bias,  self.theta1)}')