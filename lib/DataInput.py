import pandas as pd
from sklearn.impute import KNNImputer

class AdvancedImputer:
    def __init__(self, dataframe: pd.DataFrame, n_neighbors=5):
        self.df = dataframe
        self.n_neighbors = n_neighbors

    def knn_impute(self):
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        return self.df