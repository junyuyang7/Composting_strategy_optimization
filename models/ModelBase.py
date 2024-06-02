from sklearn.ensemble import RandomForestRegressor
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn import metrics

class ModelBase:
    def __init__(self, X_train=None, 
                 y_train=None, 
                 X_test=None, 
                 y_test=None, 
                 kf=None, 
                 model_save_file=None, 
                 target=None, 
                 method=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kf = kf
        self.model_save_file = model_save_file
        self.target = target
        self.method = method

    def train(self):
        pass

    def test(self):
        pass

    def save_result(self):
        pass

    def get_important_analyse(self):
        pass