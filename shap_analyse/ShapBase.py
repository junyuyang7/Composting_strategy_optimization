import shap
import pandas as pd

class ShapAnalyse:
    def __init__(self, df: pd.DataFrame, features: list, target: str, model_name: str, model_path: str):
        self.df = df
        self.features = features
        self.target = target
        self.model_name = model_name
        self.model_path = model_path
    
    def get_featrue(self):
        pass

    def get_force_plot(self):
        pass