import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f

class Dataset():

    def __init__(self, path) -> None:
        self.column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        self.X, self.Y = self.read_data(path)
        self.lm = LinearRegression()
        pass

    def read_data(self, path):
        data = []
        with open(path, "r") as file:
            Lines = file.readlines()
            for i, line in enumerate(Lines):
                data.append(list(map(float, line.split())))
                
            data = pd.DataFrame(data, columns=self.column_names)
            self.data = data
        file.close()
        return data.drop(columns=["MEDV"]), data["MEDV"]

    def pearson_correlation_coefficient(self):
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.show()
        return corr_matrix
    
    def f_test(self, feature_name):
        X = np.array(self.X[feature_name])[: , np.newaxis]
        Y = np.array(self.Y)[:, np.newaxis]
        Y_mean = np.mean(Y)
        self.lm.fit(X, Y)
        Y_hats = self.lm.predict(np.array(self.X[feature_name][:, np.newaxis]))
        SSM = sum((Y_hats - Y_mean)**2)
        SSE = sum((Y - Y_hats)**2)
        DFM = 1 
        DFE = len(X) - 1 
        F = (SSM/DFM)/(SSE/DFE)
        SL = 0.95
        critical_value = f.ppf(SL, DFM, DFE)
        print("Confidence Interval: [0, %d]" % critical_value)
        print("F : %d" % F)
        return True if 0 < F < critical_value else False
    
    def f_test_all(self):
        column_names = self.column_names[:-1]
        for feature in column_names:
            res = self.t_test(feature)
            print(res)