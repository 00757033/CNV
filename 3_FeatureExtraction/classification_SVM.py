from sklearn.model_selection import train_test_split ,KFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier ,BaggingClassifier ,GradientBoostingClassifier , RandomForestClassifier , AdaBoostClassifier , StackingClassifier , VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix ,recall_score, f1_score, precision_score ,roc_auc_score
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
from sklearn.feature_selection import SelectFromModel , SelectKBest, chi2, f_classif, mutual_info_classif ,RFECV ,RFE
import sys
import shap
shap.initjs()
from genetic_selection import GeneticSelectionCV
from sklearn.feature_selection import VarianceThreshold
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
    'Random Forest': { 'n_estimators': [50, 100, 150, 200, 250], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 20,50, 100]},
    'GaussianNB' : { 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    # 'Decision Tree': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    'KNN': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    'XGBoost': {'n_estimators': [ 100, 200, 300, 400, 500], 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'min_child_weight': [1, 3, 5, 7]},
    'LightGBM': {'n_estimators': [ 100 , 200, 300, 400, 500], 'learning_rate': [0.001, 0.01, 0.1, 1, 10], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'num_leaves': [20, 30, 40, 50, 60, 70, 80, 90, 100], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    'CatBoost' : {'iterations': [300,500], 'depth': [5,10,16], 'learning_rate': [0.01, 0.1], 'loss_function': ['Logloss', 'CrossEntropy']},
    'BernoulliNB': {'alpha': [0.1, 0.5, 1.0], 'binarize': [0.0, 0.5, 1.0], 'fit_prior': [True, False]}    
}



classifiers = {
    # 'BernoulliNB': BernoulliNB( alpha=0.1, binarize=0.5, fit_prior=True, class_prior=None),
    'SVM': svm.SVC( kernel='sigmoid',C=1000, gamma=0.001),
    'AdaBoost': AdaBoostClassifier( n_estimators=100, random_state=42),    #
    # 'Random Forest': RandomForestClassifier( n_estimators=100),
    'KNN': KNeighborsClassifier( n_neighbors=10),
    # 'XGBoost': xgb.XGBClassifier(**xgb_params),
    # # 'LightGBM': lgb.LGBMClassifier( **lgb_params),
    # 'CatBoost': CatBoostClassifier( iterations=300, depth=5, learning_rate=0.01 , loss_function='Logloss', verbose=0),
}

def normalization2(data):
    scaler = preprocessing.StandardScaler()
    data_normalized = scaler.fit_transform(data)
    data_df = pd.DataFrame(data_normalized, columns=data.columns)
    return data_df

def standardization (self, data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data
def normalization(X_train, Y_train, X_test, Y_test):
    print("normalization")
    min_max_scaler = preprocessing.StandardScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
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
        X =  data.loc[:, (data.columns != 'classification') & (data.columns != 'patient')]
        y = data.loc[:, data.columns == 'classification'].values.ravel()
        X = X.astype(np.float64)
        y = y.astype(np.float64)
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
        roc_auc = roc_auc_score(y_test, y_pred) # ROC曲線下的面積
        
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
                writer.writerow(['method', 'classifier', 'split_method', 'accuracy','precision', 'recall', 'specifity',  'f1', 'roc_auc'])
        with open( os.path.join('record', self.disease ,'result.csv') ,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([method, classifier,  split_method, accuracy, precision, recall, specifity, f1, roc_auc])
            
        return accuracy, recall, precision, f1 , roc_auc
    

    





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
            
    
if __name__ == '__main__':
    disease = 'PCV'
    date = '20240401'
    PATH_DATA = '../../Data/'
    ROI = True
    file_name = 'classification' +  '_ROI.csv' if ROI else 'classification.csv'
    data = './record/' + disease + '_' + date +'/'+file_name
    classified = Classified(data,disease + '_' + date)
    classified_data = classified.read_csv()
    classified_data = classified_data.dropna()
    # classified_data = classified_data.drop('patient', axis=1)
    classified_data = classified_data.drop('FD_CC', axis=1)
    
    print('original:', len(classified_data.columns[(classified_data.columns != 'vision')&(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    # 移除常數特徵
    del_columns = []
    for column in classified_data.columns:
        if len(classified_data[column].unique()) == 1:
            del_columns.append(column)
    classified_data = classified_data.loc[:, classified_data.apply(pd.Series.nunique) != 1]
    print('Removel Basic Filter Feature :', len(classified_data.columns[(classified_data.columns != 'vision')&(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    # 設定半常數特徵的的門檻  whithout classification
    threshold = 0.97
    # 移除半常數特徵 whithout classification
    constant_filter = VarianceThreshold(threshold=threshold)
    constant_filter.fit(classified_data)
    constant_columns = [column for column in classified_data.columns if column not in classified_data.columns[constant_filter.get_support()]]
    for feature in constant_columns:
        if feature != 'classification' and feature != 'patient' and feature != 'vision':
            if feature in classified_data.columns:
                classified_data = classified_data.drop(feature, axis=1)
                
    print('constant_columns:', constant_columns)
    
    print('Removel Constant Filter Feature :', len(classified_data.columns[(classified_data.columns != 'vision')&(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    
    # 移除Duplicated Feature
    classified_data = classified_data.T.drop_duplicates().T
    print('Removel Duplicate Filter Feature :', len(classified_data.columns[(classified_data.columns != 'vision')&(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    
    
    # 移除相關性高的特徵 kendall tau 相關係數 
    feature_del = ['GLCM_SumAverage_CC', 'GLDM_DependenceNonUniformity_CC', 'LBP skewness_CC']
    
    for feature in feature_del:
        if feature in classified_data.columns:
            classified_data = classified_data.drop(feature, axis=1)
            
    print(len(classified_data.columns)-1)
    
    
    
    X, y = classified.get_data(classified_data)
    X = normalization2(X)
    # classification == 1
    print('classification == 1:', len(classified_data[classified_data['classification'] == 1]))
    # classification == 0
    print('classification == 0:', len(classified_data[classified_data['classification'] == 0]))
    print(len(X), len(y))
    # 使用 K-Fold 交叉驗證'
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cases = ['RFECV']# 'original','PCA', 'Genetic', 'mRMR', 'chi2', 'f_classif', 'mutual_info_classif', 'RFE','RFECV', 'VarianceThreshold', 'SelectFromModel'
    for case in cases:

        for classifier_name, classifier in classifiers.items():
            X, y = classified.get_data(classified_data)
            X = normalization2(X)
            if case == 'PCA':
                pca = PCA(n_components=0.9)
                X = pca.fit_transform(X)
                # X = pd.DataFrame(X)
            elif case == 'Genetic':
                selector = GeneticSelectionCV(classifier, cv=5)
                selector = selector.fit(X, y)
                X = selector.transform(X)    
                
            elif case == 'mRMR':
                mr = pymrmr.mRMR(X, 'MIQ', 10)
                X = X.iloc[:, mr]
            elif case == 'chi2':
                X = SelectKBest(chi2, k=10).fit_transform(X, y)
            elif case == 'f_classif':
                X = SelectKBest(f_classif, k=10).fit_transform(X, y)
            elif case == 'mutual_info_classif':
                X = SelectKBest(mutual_info_classif, k=10).fit_transform(X, y)
            elif case == 'RFECV':
                # 先 SVM 做特徵選擇
                
                classifier_rfe = svm.SVC(kernel='linear')
                selector = RFECV(classifier_rfe, step=1, cv=10, scoring='accuracy')
                selector = selector.fit(X, y)
                X = selector.transform(X)
                
                
                
                
                if not classifier_name == 'KNN':
                    if classifier_name == 'SVM':
                        classifier = svm.SVC(kernel='linear')
                    selector = RFECV(classifier, step=1, cv=10, scoring='accuracy')
                    selector = selector.fit(X, y)
                    X = selector.transform(X)
                else:
                    classifier = svm.SVC(kernel='linear')
                    selector = RFECV(classifier, step=1, cv=10, scoring='accuracy')
                    selector = selector.fit(X, y)
                    X = selector.transform(X)
                    classifier = KNeighborsClassifier()
                
            elif case == 'RFE':
                if classifier_name == 'SVM':
                    classifier = svm.SVC(kernel='linear')
                elif classifier_name == 'KNN':
                    continue
                selector = RFE(classifier, n_features_to_select=10)
                selector = selector.fit(X, y)
                print('selector:', X.columns[selector.support_])
                X = selector.transform(X)
                
            elif case == 'VarianceThreshold':
                selector = VarianceThreshold(threshold=0.1)
                X = selector.fit_transform(X)
            elif case == 'SelectFromModel':
                selector = SelectFromModel(classifier)
                selector = selector.fit(X, y)
                X = selector.transform(X)
            
            # feature_importance_SHAP(classifier_name, classifier, X, y, 'original')
            print('Classifier:', classifier_name)
            accuracy_scores = []
            recall_scores = []
            precision_scores = []
            f1_scores = []
            specifity_scores = []
            roc_auc_scores = []
            for train_index, test_index in kf.split(X):
                if case == 'PCA' or case == 'Genetic' or case =='RFECV' or case =='RFE' or case == 'SelectFromModel':
                    X_train, X_test = X[train_index], X[test_index]
                else:
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # x_train, y_train, x_test, y_test = normalization(X_train, y_train, X_test, y_test)
                model = classifier.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, recall, precision, f1 , roc_auc = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), case)
                accuracy_scores.append(accuracy)
                recall_scores.append(recall)
                precision_scores.append(precision)
                f1_scores.append(f1)
                specifity_scores.append(1 - recall)    
                roc_auc_scores.append(roc_auc)
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
            Avg_roc_auc = np.mean(roc_auc_scores)
            std_roc_auc = np.std(roc_auc_scores, ddof=1)
            
            # print('Accuracy:', Avg_accuracy, std_accuracy)
            # print('Recall:', Avg_recall, std_recall)
            # print('Precision:', Avg_precision, std_precision)
            # print('F1:', Avg_f1, std_f1)
            # print('Specifity:', Avg_specifity, std_specifity)
            
            # save the result to csv
            if not os.path.isfile(os.path.join('record', disease + '_' + date ,'avg_result.csv')):
                with open( os.path.join('record', disease + '_' + date , 'avg_result.csv') ,'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specifity', 'Std specifity',  'Avg f1', 'Std f1', 'Avg roc_auc', 'Std roc_auc'])
                    
            with open( os.path.join('record', disease + '_' + date ,'avg_result.csv') ,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([case, classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specifity, std_specifity, Avg_f1, std_f1, Avg_roc_auc, std_roc_auc])
                
            
