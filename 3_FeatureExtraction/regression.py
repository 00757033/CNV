from sklearn.model_selection import train_test_split ,KFold
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.linear_model import LogisticRegression ,LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, LassoLars, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor, HuberRegressor, TheilSenRegressor
import tools.tools as tools
import matplotlib.pyplot as plt
import seaborn as sns
regression_model = {
    'LinearRegression': LinearRegression(),
    # 'Lasso': Lasso(),
    # 'Ridge': Ridge(),
    # 'ElasticNet': ElasticNet(),
    # 'BayesianRidge': BayesianRidge(),
    # 'LassoLars': LassoLars(),
    # 'ARDRegression': ARDRegression(),
    # 'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
    # 'RANSACRegressor': RANSACRegressor(),
    
}

class regression():
    def normalization (self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data
    def standardization (self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data
    
    def KFold(self, X, y):
        kf = KFold(n_splits=5, shuffle=True)
        print(len(X), len(y))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model(X_train, y_train, X_test, y_test)
    
    def model(self, X_train, y_train, X_test, y_test):
        for model in regression_model:
            clf = regression_model[model]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(model)
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            print('R2 Score:', metrics.r2_score(y_test, y_pred))
            print('len(X_test)' , len(X_test))
            print('len(y_test)' , len(y_test))
            # plot

            data = np.column_stack((X_test, y_test))
            data = data[data[:,0].argsort()]
            plt.scatter(data[:,0], data[:,1], color='gray')
            plt.show()

            
            
    
    
    def plot(self, X, y):
        plt.scatter(X, y,  color='gray')
        plt.show()
        
    def feature_selection(self, X, y):
        clf = SVC(kernel='linear')
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        return X_new

    def feature_selection_plot(self, X, y):
        clf = SVC(kernel='linear')
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        plt.scatter(X_new, y)
        plt.show()
    
    def get_data(self, data):
        print( len(data))
        X =  data['vision']
        y = data['regression']
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        return X, y    
    
        
    
if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240411'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    PATH_FEATURE = PATH_BASE + 'feature/'
    
    data = './record/' + disease + '_' + date +'/'+'regression.csv'
    label = pd.read_csv(data)
    # label = label.dropna()
    print(len(label))

    Regression = regression()
    X, y =  Regression.get_data(label)
    # X Y kind =“reg”
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    plt.scatter(X, y,  color='gray')
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.show()


    
    