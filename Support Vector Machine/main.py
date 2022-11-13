import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

def getData(path:str, output_attr:str):
    df = pd.read_csv(path)


    output_values = df[output_attr].unique()
    map_dict = dict()
    values = [-1,1]
    for i in range(len(output_values)):
        print(output_values[i] ,' --> ', values[i])
        map_dict[output_values[i]] = values[i]
    df[output_attr] = df[output_attr].map(map_dict)

    output = df[output_attr]
    df.drop(output_attr, axis=1, inplace=True)

    n = len(df)
    train_n = int(np.round(n*0.90))

    train_data, train_labels = df.loc[train_n:], output.loc[train_n:]
    test_data, test_labels = df.loc[:train_n], output.loc[:train_n]
    
    return train_data, train_labels, test_data, test_labels

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.03, n_iters=2000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    
    def train(self, X, y):
        X = np.asarray(X)
        n_samples, n_features = X.shape
            
        y_ = np.where(y <= 0, -1, 1)
            
        self.w = np.zeros(n_features)
        self.b = 0
    
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    
    def predict(self, X):
        approximate = np.dot(X, self.w) - self.b
        return np.sign(approximate)
    
    def predictTest(self, X, Y):
        length = len(X)
        X = np.asarray(X)
        a = 0
        for x,y in zip(X,Y):
            approx = np.dot(x, self.w) - self.b
            pred = np.sign(approx)
            if pred == y:
                a += 1
        acc = a/length
        print(f'Accuracy : {acc:.2f}')



train_x, train_y, test_x, test_y = getData('./dataset/gender_classification_v7.csv', 'gender')

svm = SVM()

svm.train(train_x, train_y)

svm.predictTest(test_x, test_y)

svm.learning_curve()

