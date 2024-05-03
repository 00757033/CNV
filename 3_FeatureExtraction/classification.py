from sklearn.model_selection import train_test_split ,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier ,BaggingClassifier ,GradientBoostingClassifier , RandomForestClassifier , AdaBoostClassifier , StackingClassifier , VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix ,recall_score, f1_score, precision_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier , NearestNeighbors
from sklearn.naive_bayes import ComplementNB , BernoulliNB , CategoricalNB , GaussianNB , MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb 
from catboost import CatBoostClassifier 
import os
import csv
import pymrmr 
from sklearn.feature_selection import SelectFromModel , SelectKBest, chi2, f_classif, mutual_info_classif ,RFECV
import sys
import shap
shap.initjs()
from genetic_selection import GeneticSelectionCV
xgb_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.3,
    'seed': 42,
    'nthread': 8,
}

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 6,
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
}

param_grid = {
    'SVM': {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['sigmoid']},
    'AdaBoost': {'n_estimators': [50, 100, 150, 200, 250], 'learning_rate': [0.001, 0.01, 0.1, 1, 10]},
    'Random Forest': { 'n_estimators': [50, 100, 150, 200, 250], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    # 'Decision Tree': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    # 'KNN': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    'XGBoost': {'n_estimators': [ 100, 200, 300, 400, 500], 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 3, 5, 7]},
    # 'LightGBM': {'n_estimators': [ 100 , 200, 300, 400, 500], 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'num_leaves': [20, 30, 40, 50, 60, 70, 80, 90, 100], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    'CatBoost' : {'iterations': [200,500], 'depth': [2,3, 5, 8, 9, 10,16], 'learning_rate': [0.0001, 0.001, 0.1], 'loss_function': ['Logloss'] , 'border_count': [32, 64, 128, 254]}
}



classifiers = {
    # 'SVM': svm.SVC( kernel='sigmoid',C=1000, gamma=0.001),
    # 'AdaBoost': AdaBoostClassifier( n_estimators=100, random_state=42),    #
    'Random Forest': RandomForestClassifier( n_estimators=100),
    # # 'KNN': KNeighborsClassifier( n_neighbors=5),
    'XGBoost': xgb.XGBClassifier(**xgb_params),
    # # 'LightGBM': lgb.LGBMClassifier( **lgb_params),
    # 'CatBoost': CatBoostClassifier( iterations=500, depth=10, learning_rate=0.001 , loss_function='Logloss', verbose=False,border_count=254)
}

def normalization(X_train, Y_train, X_test, Y_test):
    print("normalization")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    Y_train = Y_train
    Y_test = Y_test
    return X_train, Y_train, X_test, Y_test


def evaluate_classifier(classifier_name, classifier,kf, X, y, split_method):
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    specifity_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, split_method, 'original')
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)
        specifity_scores.append(1 - recall)
        
    Avg_accuracy = np.mean(accuracy_scores)
    Avg_recall = np.mean(recall_scores)
    Avg_precision = np.mean(precision_scores)
    Avg_f1 = np.mean(f1_scores)
    Avg_specifity = np.mean(specifity_scores)
    std_accuracy = np.std(accuracy_scores, ddof=1)
    std_recall = np.std(recall_scores, ddof=1)
    std_precision = np.std(precision_scores, ddof=1)
    std_f1 = np.std(f1_scores, ddof=1)
    std_specifity = np.std(specifity_scores, ddof=1)
    # print('Classifier:', classifier_name)
    # print('Accuracy:', Avg_accuracy, std_accuracy)
    # print('Recall:', Avg_recall, std_recall)
    # print('Precision:', Avg_precision, std_precision)
    # print('F1:', Avg_f1, std_f1)
    # print('Specifity:', Avg_specifity, std_specifity)
    
    if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
        with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
            
    with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original', classifier_name,  split_method, Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
        
    return Avg_accuracy, Avg_recall, Avg_precision, Avg_f1, Avg_specifity, std_accuracy, std_recall, std_precision, std_f1, std_specifity

def feature_importance(classifier_name, classifier, X, y, split_method):
    model = classifier.fit(X, y)
    importance = model.feature_importances_
    print('importance:', importance)
    if not os.path.isfile(os.path.join('record', disease + '_' + date ,'feature_importance.csv')):
        with open( os.path.join('record', disease + '_' + date , 'feature_importance.csv') ,'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['classifier', 'split_method', 'importance'])
    with open( os.path.join('record', disease + '_' + date ,'feature_importance.csv') ,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([classifier_name,  split_method, importance])
    return importance

def feature_importance_SHAP(classifier_name, classifier, X, y, method):
    if os.path.exists(os.path.join('record', disease + '_' + date ,'classification','shap')) == False:
        os.makedirs(os.path.join('record', disease + '_' + date ,'classification','shap'))
    model = classifier.fit(X, y)
    if classifier_name == 'SVM' :
        explainer = shap.KernelExplainer(classifier.decision_function, X)
        shap_values = explainer.shap_values(X)
    elif classifier_name == 'Random Forest' or classifier_name == 'Decision Tree' or classifier_name == 'XGBoost' or classifier_name == 'LightGBM' or classifier_name == 'CatBoost' or classifier_name == 'AdaBoost':
        # explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X,check_additivity=False)
    else:
        shap_values = None    
    if shap_values is not None:    
        plt.title('SHAP summary plot')  
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join('record', disease + '_' + date ,'classification','shap', method + '_' + classifier_name + '.png'))
        plt.clf()
    return shap_values
        

class Classified:
    def __init__(self, file_path,disease):
        self.file_path = file_path
        self.disease = disease
    
    def read_csv(self):
        return pd.read_csv(self.file_path,index_col = 0)
    
    def get_data(self, data):
        X =  data.loc[:, data.columns != 'classification']
        y = data.loc[:, data.columns == 'classification'].values.ravel()
        X = X.astype(float)
        return X, y

    def split_data(self, X, y, data, test_size = 0.3):


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=22)
        print('y_test:', y_test)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_test, y_pred, classifier = 'random_forest', split_method = 'K-Fold', method = 'original'):
        print('evaluate_model')
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        specifity = 1 - recall
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # write_csv
        # print('Accuracy:', accuracy)
        # print('Recall:', recall)
        # print('Precision:', precision)
        # print('F1:', f1)
        print('Confusion Matrix:', cm)
        
        # save the result to csv
        if not os.path.isfile(os.path.join('record', self.disease , 'result.csv')):
            with open( os.path.join('record', self.disease , 'result.csv') ,'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'classifier', 'split_method', 'accuracy','precision', 'recall', 'specifity',  'f1'])
        with open( os.path.join('record', self.disease ,'result.csv') ,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([method, classifier,  split_method, accuracy, precision, recall, specifity, f1])
            
        return accuracy, recall, precision, f1
    

    





    # def feature_importance(self, X, y, classifier = 'random_forest'):
    #     if classifier == 'random_forest':
    #         model = RandomForestClassifier()
    #     elif classifier == 'decision_tree':
    #         model = DecisionTreeClassifier()
    #     elif classifier == 'svm':
    #         model = svm.SVC()
    #     elif classifier == 'knn':
    #         model = KNeighborsClassifier()
    #     elif classifier == 'xgboost':
    #         model = xgb.XGBClassifier()
    #     elif classifier == 'lightgbm':
    #         model = lgb.LGBMClassifier()
    #     elif classifier == 'catboost':
    #         model = CatBoostClassifier()
            
        model.fit(X, y)
        importance = model.feature_importances_
        return importance


        


        
        
        
        
        
        return 
  


    
if __name__ == '__main__':
    disease = 'PCV'
    date = '20240320'
    PATH_DATA = '../../Data/'
    
    data = './record/' + disease + '_' + date +'/'+'classification.csv'
    classified = Classified(data,disease + '_' + date)
    classified_data = classified.read_csv()
    classified_data = classified_data.dropna()
    X, y = classified.get_data(classified_data)
    # 使用 K-Fold 交叉驗證'
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # for classifier_name, classifier in classifiers.items():
    #     # feature_importance_SHAP(classifier_name, classifier, X, y, 'original')
    #     print('Classifier:', classifier_name)
    #     accuracy_scores = []
    #     recall_scores = []
    #     precision_scores = []
    #     f1_scores = []
    #     specifity_scores = []
    #     for train_index, test_index in kf.split(X):
    #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         x_train, y_train, x_test, y_test = normalization(X_train, y_train, X_test, y_test)
    #         model = classifier.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'original')
    #         accuracy_scores.append(accuracy)
    #         recall_scores.append(recall)
    #         precision_scores.append(precision)
    #         f1_scores.append(f1)
    #         specifity_scores.append(1 - recall)    
    #     Avg_accuracy = np.mean(accuracy_scores)
    #     Avg_recall = np.mean(recall_scores)
    #     Avg_precision = np.mean(precision_scores)
    #     Avg_f1 = np.mean(f1_scores)
    #     Avg_specifity = np.mean(specifity_scores)
    #     std_accuracy = np.std(accuracy_scores, ddof=1)
    #     std_recall = np.std(recall_scores, ddof=1)
    #     std_precision = np.std(precision_scores, ddof=1)
    #     std_f1 = np.std(f1_scores, ddof=1)
    #     std_specifity = np.std(specifity_scores, ddof=1)
        
    #     # print('Accuracy:', Avg_accuracy, std_accuracy)
    #     # print('Recall:', Avg_recall, std_recall)
    #     # print('Precision:', Avg_precision, std_precision)
    #     # print('F1:', Avg_f1, std_f1)
    #     # print('Specifity:', Avg_specifity, std_specifity)
        
    #     # save the result to csv
    #     if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
    #         with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
                
    #     with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['original', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
            
        
    # # 使用 GridSearchCV 調參
    # print('GridSearchCV')
    # for classifier_name, classifier in classifiers.items():
    #     print('Classifier:', classifier_name)
    #     model = GridSearchCV(classifier, param_grid[classifier_name], cv=kf, scoring='accuracy')
    #     model.fit(X, y)

        
    #     print('Best parameters:', model.best_params_)
    #     print('Best score:', model.best_score_)
        
    #     # feature_importance_SHAP(classifier_name, model.best_estimator_, X, y, 'GridSearchCV')
        
    #     for  param_name in model.best_params_:
    #         print(param_name, ":", model.best_params_[param_name])
            
    #     accuracy_scores = []
    #     recall_scores = []
    #     precision_scores = []
    #     f1_scores = []
    #     specifity_scores = []
        
    #     for train_index, test_index in kf.split(X):
    #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
    #         model = classifier.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'GridSearchCV')
    #         accuracy_scores.append(accuracy)
    #         recall_scores.append(recall)
    #         precision_scores.append(precision)
    #         f1_scores.append(f1)
    #         specifity_scores.append(1 - recall)
            
    #     Avg_accuracy = np.mean(accuracy_scores)
    #     Avg_recall = np.mean(recall_scores)
    #     Avg_precision = np.mean(precision_scores)
    #     Avg_f1 = np.mean(f1_scores)
    #     Avg_specifity = np.mean(specifity_scores)
    #     std_accuracy = np.std(accuracy_scores, ddof=1)
    #     std_recall = np.std(recall_scores, ddof=1)
    #     std_precision = np.std(precision_scores, ddof=1)
    #     std_f1 = np.std(f1_scores, ddof=1)
    #     std_specifity = np.std(specifity_scores, ddof=1)
    #     # print('Classifier:', classifier_name)
    #     # print('Accuracy:', Avg_accuracy, std_accuracy)
    #     # print('Recall:', Avg_recall, std_recall)
    #     # print('Precision:', Avg_precision, std_precision)
    #     # print('F1:', Avg_f1, std_f1)
    #     # print('Specifity:', Avg_specifity, std_specifity)
        
    #     if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
    #         with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
            
    #     with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['GridSearchCV', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
            
            
        
    # # 使用 RFECV 進行特徵選擇
    # print('RFECV')
    
    # for classifier_name, classifier in classifiers.items():
    #     if classifier_name in ['SVM']:
    #         svc = svm.SVC(kernel="linear")
    #         rfecv = RFECV(estimator=svc, step=1 ,scoring='accuracy')
    #         rfecv.fit(X, y)
    #     if classifier_name in ['AdaBoost', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
    #         rfecv = RFECV(estimator=classifier, step=1, cv=kf, scoring='accuracy')
    #         rfecv.fit(X, y)
    #         # feature_importance_SHAP(classifier_name, rfecv.estimator_, X, y, 'RFECV')
    #         # print('Classifier:', classifier_name)
    #         # print('Optimal number of features:', rfecv.n_features_)
    #         # print('Ranking of features:', rfecv.ranking_)
    #         # print('Support of features:', rfecv.support_)
    #         # print('Grid scores:', rfecv.grid_scores_)
    #         # print('Feature importances:', rfecv.estimator_.feature_importances_)
    #         accuracy_scores = []
    #         recall_scores = []
    #         precision_scores = []
    #         f1_scores = []
    #         specifity_scores = []
    #         # explainer = shap.Explainer(rfecv.estimator_)
    #         # shap_values = explainer(X)
    #         # shap.summary_plot(shap_values, X)
    #         # plt.show()
    #         # plt.savefig(os.path.join('record', disease + '_' + date , 'shap_'+ classifier_name + '.png'))
            
    #         for train_index, test_index in kf.split(X):
    #             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #             y_train, y_test = y[train_index], y[test_index]
    #             X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
    #             model = classifier.fit(X_train, y_train)
    #             y_pred = model.predict(X_test)
    #             accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'RFECV')
    #             accuracy_scores.append(accuracy)
    #             recall_scores.append(recall)
    #             precision_scores.append(precision)
    #             f1_scores.append(f1)
    #             specifity_scores.append(1 - recall)
                
    #         Avg_accuracy = np.mean(accuracy_scores)
    #         Avg_recall = np.mean(recall_scores)
    #         Avg_precision = np.mean(precision_scores)
    #         Avg_f1 = np.mean(f1_scores)
    #         Avg_specifity = np.mean(specifity_scores)
    #         std_accuracy = np.std(accuracy_scores, ddof=1)
    #         std_recall = np.std(recall_scores, ddof=1)
    #         std_precision = np.std(precision_scores, ddof=1)
    #         std_f1 = np.std(f1_scores, ddof=1)
    #         std_specifity = np.std(specifity_scores, ddof=1)
    #         # print('Classifier:', classifier_name)
    #         # print('Accuracy:', Avg_accuracy, std_accuracy)
    #         # print('Recall:', Avg_recall, std_recall)
    #         # print('Precision:', Avg_precision, std_precision)
    #         # print('F1:', Avg_f1, std_f1)
    #         # print('Specifity:', Avg_specifity, std_specifity)
            
    #         if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
    #             with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
                    
    #         with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['RFECV', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
                
    # # 使用 GeneticSelectionCV 進行特徵選擇
    # print('GeneticSelectionCV')
     
    # for classifier_name, classifier in classifiers.items():
    #     print('Classifier:', classifier_name)
    #     selector = GeneticSelectionCV(classifier,
    #                                   cv=kf,
                                      
    #                                     )
    #     selector = selector.fit(X, y)
    #     accuracy_scores = []
    #     recall_scores = []
    #     precision_scores = []
    #     f1_scores = []
    #     specifity_scores = []
        
    #     for train_index, test_index in kf.split(X):
    #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
    #         model = classifier.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'GeneticSelectionCV')
    #         accuracy_scores.append(accuracy)
    #         recall_scores.append(recall)
    #         precision_scores.append(precision)
    #         f1_scores.append(f1)
    #         specifity_scores.append(1 - recall)
            
    #     Avg_accuracy = np.mean(accuracy_scores)
    #     Avg_recall = np.mean(recall_scores)
    #     Avg_precision = np.mean(precision_scores)
    #     Avg_f1 = np.mean(f1_scores)
    #     Avg_specifity = np.mean(specifity_scores)
    #     std_accuracy = np.std(accuracy_scores, ddof=1)
    #     std_recall = np.std(recall_scores, ddof=1)
    #     std_precision = np.std(precision_scores, ddof=1)
    #     std_f1 = np.std(f1_scores, ddof=1)
    #     std_specifity = np.std(specifity_scores, ddof=1)
    #     # print('Classifier:', classifier_name)
    #     # print('Accuracy:', Avg_accuracy, std_accuracy)
    #     # print('Recall:', Avg_recall, std_recall)
    #     # print('Precision:', Avg_precision, std_precision)
    #     # print('F1:', Avg_f1, std_f1)
    #     # print('Specifity:', Avg_specifity, std_specifity)
        
    #     if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
    #         with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
                
    #     with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['GeneticSelectionCV', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
            

    # # # # 使用 PCA 進行特徵選擇
    # print('PCA')
    # from sklearn.decomposition import PCA

    # for classifier_name, classifier in classifiers.items():
    #     print('Classifier:', classifier_name)
    #     if classifier_name in ['CatBoost']:
    #         pca = PCA(n_components=0.95)
    #     else:
    #         pca = PCA(n_components=0.95)
    #     X_pca = pca.fit_transform(X)
    #     # feature_importance_SHAP(classifier_name, classifier, X_pca, y, 'PCA')
    #     accuracy_scores = []
    #     recall_scores = []
    #     precision_scores = []
    #     f1_scores = []
    #     specifity_scores = []
        
    #     for train_index, test_index in kf.split(X_pca):
    #         X_train, X_test = X_pca[train_index], X_pca[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
    #         model = classifier.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'PCA')
    #         accuracy_scores.append(accuracy)
    #         recall_scores.append(recall)
    #         precision_scores.append(precision)
    #         f1_scores.append(f1)
    #         specifity_scores.append(1 - recall)
            
    #     Avg_accuracy = np.mean(accuracy_scores)
    #     Avg_recall = np.mean(recall_scores)
    #     Avg_precision = np.mean(precision_scores)
    #     Avg_f1 = np.mean(f1_scores)
    #     Avg_specifity = np.mean(specifity_scores)
    #     std_accuracy = np.std(accuracy_scores, ddof=1)
    #     std_recall = np.std(recall_scores, ddof=1)
    #     std_precision = np.std(precision_scores, ddof=1)
        
    #     std_f1 = np.std(f1_scores, ddof=1)
    #     std_specifity = np.std(specifity_scores, ddof=1)
    #     print('Classifier:', classifier_name)
    #     # print('Accuracy:', Avg_accuracy, std_accuracy)
    #     # print('Recall:', Avg_recall, std_recall)
    #     # print('Precision:', Avg_precision, std_precision)
    #     # print('F1:', Avg_f1, std_f1)
    #     # print('Specifity:', Avg_specifity, std_specifity)
        
    #     if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
    #         with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
                
    #     with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['PCA', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
            
        

    # 使用 PCA & GridSearchCV 進行特徵選擇
    print('PCA & GridSearchCV')
    for classifier_name, classifier in classifiers.items():
        print('Classifier:', classifier_name)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X)
        model = GridSearchCV(classifier, param_grid[classifier_name], cv=kf, scoring='f1')
        model.fit(X_pca, y)
        print('Best parameters:', model.best_params_)
        print('Best score:', model.best_score_)
        # feature_importance_SHAP(classifier_name, model.best_estimator_, X_pca, y, 'PCA_GridSearchCV')
        for  param_name in model.best_params_:
            print(param_name, ":", model.best_params_[param_name])
        accuracy_scores = []
        recall_scores = []
        precision_scores = []
        f1_scores = []
        specifity_scores = []
        
        for train_index, test_index in kf.split(X_pca):
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train , y_train, X_test, y_test = normalization(X_train, y_train, X_test, y_test)
            model = classifier.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy, recall, precision, f1 = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), 'PCA_GridSearchCV')
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)
            specifity_scores.append(1 - recall)
            
        Avg_accuracy = np.mean(accuracy_scores)
        Avg_recall = np.mean(recall_scores)
        Avg_precision = np.mean(precision_scores)
        Avg_f1 = np.mean(f1_scores)
        Avg_specifity = np.mean(specifity_scores)
        std_accuracy = np.std(accuracy_scores, ddof=1)
        std_recall = np.std(recall_scores, ddof=1)
        std_precision = np.std(precision_scores, ddof=1)
        std_f1 = np.std(f1_scores, ddof=1)
        std_specifity = np.std(specifity_scores, ddof=1)
        # print('Classifier:', classifier_name)
        # print('Accuracy:', Avg_accuracy, std_accuracy)
        # print('Recall:', Avg_recall, std_recall)
        # print('Precision:', Avg_precision, std_precision)
        
        # print('F1:', Avg_f1, std_f1)
        
        # print('Specifity:', Avg_specifity, std_specifity)
        
        if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
            with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1'])
                
        
        with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['PCA_GridSearchCV', classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1])
    