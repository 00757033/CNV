from sklearn.model_selection import train_test_split ,KFold
from sklearn.model_selection import GridSearchCV 
from sklearn import preprocessing

from sklearn.ensemble import AdaBoostClassifier  , RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix ,recall_score, f1_score, precision_score ,roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier 
import os
from sklearn.feature_selection import SelectFromModel , SelectKBest, chi2, f_classif, mutual_info_classif ,RFECV ,RFE
import csv
from imblearn.over_sampling import ADASYN 

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

param_grid = {
    'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [ 0.1, 1, 10, 100], 'solver': ['saga'], 'random_state': [42], 'max_iter': [10000]},
    'SVM': {'C': [0.01,0.1,10, 100], 'gamma': [100,10,1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}, # , 'linear', 'poly'
    'AdaBoost': {'n_estimators': [50,100], 'learning_rate': [0.001,0.01], 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': [42]},
    # 'Random Forest': { 'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 20,50, 100]},
    # 'GaussianNB' : { 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    # 'Decision Tree': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    # 'KNN': {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]},
    # 'XGBoost': { 'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01,0.1], 'max_depth': [2,4,6,8], 'subsample': [0.6, 0.8], 'colsample_bytree': [0.6, 0.8],'min_child_weight': [1, 3, 5]},
    # 'LightGBM': {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [2,4,6,8], 'num_leaves': [7, 15, 31], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0]},
    'CatBoost' : {'iterations': [50,100], 'depth':  [6,8], 'learning_rate': [0.001,0.01], 'loss_function': ['Logloss', 'CrossEntropy']},
    # 'BernoulliNB': {'alpha': [0.1, 0.5, 1.0], 'binarize': [0.0, 0.5, 1.0], 'fit_prior': [True, False]}   ,
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
    for train_index, test_index in kf.split(X, y):
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

def feature_importance_SHAP(classifier_name,figure_path, model, X,feature_names, method,count = 0):
    os.makedirs(os.path.join('record', figure_path, 'shap'), exist_ok=True)
    print('feature_importance_SHAP',feature_names)
    supported_tree_models = ( 'Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM', 'CatBoost')
    if classifier_name in supported_tree_models:
        explainer = shap.TreeExplainer(model)
    elif classifier_name == 'SVM' :
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)
        
        shap.summary_plot(shap_values, X,feature_names = feature_names, show = False)
        print('shap_values:', len(feature_names))
        plt.title('SHAP Values')
        save_path = os.path.join('record', figure_path,'shap', method + '_' + classifier_name + str(count) + '.png')
        plt.savefig(save_path)
        plt.clf()
        
        # bar plot
        shap.summary_plot(shap_values, X, plot_type='bar', feature_names = feature_names, show = False)
        plt.title('SHAP Values')
        save_path = os.path.join('record', figure_path,'shap', method + '_' + classifier_name + str(count) + '_bar.png')
        plt.savefig(save_path)
        plt.clf()
        
        return shap_values
        
    elif  classifier_name == 'LogisticRegression':
        explainer = shap.LinearExplainer(model.estimator_, X, feature_dependence="independent")
    else:
        explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)


    print('shap_values:', shap_values)   
    if shap_values is not None:    
        shap.summary_plot(shap_values, X, feature_names = feature_names)
        plt.title('model SHAP Values')
        plt.savefig(os.path.join('record', figure_path,'shap', method + '_' + classifier_name + '.png'))
        plt.clf()
        
        # bar plot
        shap.summary_plot(shap_values, X, plot_type='bar', feature_names = feature_names)
        plt.title('model SHAP Values')
        plt.savefig(os.path.join('record', figure_path,'shap', method + '_' + classifier_name + '_bar.png'))
        plt.show()
        plt.clf()
        

    return shap_values

def permutation_importances(classifier_name, figure_path,classifier, X, y,selected_columns, count,split_method):
    os.makedirs(os.path.join('record', figure_path, 'permutation_importance'), exist_ok=True)
    model = classifier.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        
    else:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        feature_importances = result.importances_mean
    print('importance:', feature_importances)
    sorted_indices = feature_importances.argsort()[::-1]
    print('sorted_indices:', sorted_indices)
    sorted_feature_names = [selected_columns[idx] for idx in sorted_indices]
    print('sorted_feature_names',sorted_feature_names)
    if not os.path.isfile(os.path.join('record', figure_path ,'permutation_importance.csv')):
        with open( os.path.join('record', figure_path , 'permutation_importance.csv') ,'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['classifier', 'split_method', 'importance'])
            
    with open( os.path.join('record', figure_path ,'permutation_importance.csv') ,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([classifier_name,  split_method, feature_importances])
        
    plt.figure(figsize=(12, 8))  # Increased size
    sns.barplot( x=feature_importances[sorted_indices],y=sorted_feature_names)
    plt.title('Feature Importance')
    plt.yticks(fontsize='small')
    plt.savefig(os.path.join('record', figure_path, 'permutation_importance', classifier_name + '_' + split_method + '_' + str(count) + '.png'))
    plt.show()
    plt.tight_layout() 
    plt.clf()
   
  
    return feature_importances

    
    
    # importance = result.importances_mean
    # print('importance:', importance)
    # if not os.path.isfile(os.path.join('record', figure_path ,'permutation_importance.csv')):
    #     with open( os.path.join('record', figure_path , 'permutation_importance.csv') ,'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['classifier', 'split_method', 'importance'])
    # with open( os.path.join('record', figure_path ,'permutation_importance.csv') ,'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([classifier_name,  split_method, importance])
    
    # # plot
    # for i in range(len(importance)):
    #     print('importance:', importance[i])
    #     sns.barplot(x=importance[i], y=X.columns)
    #     plt.title('Permutation Importance')
    #     plt.savefig(os.path.join('record', figure_path, 'permutation_importance', classifier_name + '_' + split_method + '.png'))
    #     plt.clf()
        
    
    # return importance

             

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

    def multi_specificity(cm):
        num_classes = len(cm)
        total = 0
        for i in range(num_classes):
            total += sum(cm[i])
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for i in range(num_classes):
            tn += cm[i][i]
            for j in range(num_classes):
                if i != j:
                    fp += cm[i][j]
                    fn += cm[j][i]
        specificity = tn / (tn + fp)
        return specificity
    
    
    def evaluate_model(self, y_test, y_pred, classifier = 'random_forest', split_method = 'K-Fold', method = 'original'):
        print('evaluate_model')
        print('y_test:', y_test)
        print('y_pred:', y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro' , zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        cm  = confusion_matrix(y_test, y_pred)
        # specificity = Classified.multi_specificity(cm)
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        # tn  = 0
        # fp = 0
        # fn = 0
        # tp = 0
        # for i in range(len(cm)):
        #     tn += cm[i][i]
        #     for j in range(len(cm)):
        #         if i != j:
        #             fp += cm[i][j]
        #             fn += cm[j][i]
        
        # tp = np.sum(np.diag(cm))
        
        # specificity = tn / (tn + fp)
    
        
            
        
        # if len(np.unique(y_pred)) == 1 or len(np.unique(y_test)) == 1:
        #     roc_auc = 0.5
        # else:
        #     roc_auc = roc_auc_score(y_test, y_pred, multi_class = 'ovo')
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # write_csv
        # print('Accuracy:', accuracy)
        # print('Recall:', recall)
        # print('Precision:', precision)
        # print('F1:', f1)
        # print('Confusion Matrix:', cm)
        
        # save the result to csv
        if not os.path.isfile(os.path.join('record', self.disease , 'result.csv')):
            with open( os.path.join('record', self.disease , 'result.csv') ,'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'classifier', 'split_method', 'accuracy','precision', 'recall', 'specificity',  'f1', 'cm'])
        with open( os.path.join('record', self.disease ,'result.csv') ,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([method, classifier,  split_method, accuracy, precision, recall, specificity, f1, cm.tolist()])
            
        return accuracy, recall, precision, f1 ,specificity
    

    




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
    



classifiers = {
    # 'BernoulliNB': BernoulliNB( alpha=0.1, binarize=0.5, fit_prior=True, class_prior=None),
    # 'GaussianNB' : GaussianNB( var_smoothing=1e-9),
    'SVM': svm.SVC( kernel='rbf', C=5, gamma=0.01),
    'LogisticRegression' : LogisticRegression( penalty='l2', C=5, solver='liblinear', max_iter=1000, random_state=42),# newton-cg
    # 'AdaBoost': AdaBoostClassifier( n_estimators=500, learning_rate=0.001, algorithm='SAMME.R',base_estimator=DecisionTreeClassifier(max_depth=6), random_state=42),
    'Random Forest': RandomForestClassifier( n_estimators=500, max_features='auto', max_depth=8, criterion='gini', random_state=6),
    # 'XGBoost': xgb.XGBClassifier(**xgb_params),
    'CatBoost': CatBoostClassifier( iterations=500, depth=6, learning_rate=0.001 , loss_function='Logloss',verbose = False,eval_metric = 'F1', random_state=98),
    # 'KNN': KNeighborsClassifier( n_neighbors=10),
    # 'LightGBM': lgb.LGBMClassifier( **lgb_params),
    

}



class CustomCatBoostClassifier(CatBoostClassifier):
    def __repr__(self):
        return f"CustomCatBoostClassifier(iterations={self.iterations}, depth={self.depth}, learning_rate={self.learning_rate}, loss_function='{self.loss_function}')"
    
if __name__ == '__main__':
    disease = 'PCV'
    date = '20240524'
    PATH_DATA = '../../Data/'
    ROI = True #True
    cut = False
    file_name = 'classification'
    output_file_name = 'avg_result'

    if cut :
        file_name = file_name + '_cut'
        output_file_name = output_file_name + '_cut'
    if ROI:
        file_name = file_name + '_ROI'
        output_file_name = output_file_name + '_ROI'
    file_name = file_name +  '.csv'
    output_file_name = output_file_name +  '.csv'
    
    
    print('file_name:', file_name)
    data = './record/' + disease + '_' + date +'/'+file_name
    classified = Classified(data,disease + '_' + date)
    classified_data = classified.read_csv()


    print('columns:', len(classified_data.columns[(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    # classified_data = classified_data.dropna()
    print('columns:', len(classified_data.columns[(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    # # print nan value
    
    for key in classified_data.keys():
        if 'GLDS' in key:
            classified_data = classified_data.drop(key, axis=1)
            
    # classified_data = classified_data.drop('SFCT', axis=1)
            
    classified_data = classified_data.drop('HurstCoeff_3', axis=1)
    classified_data = classified_data.drop('HurstCoeff_4', axis=1)
   


    # 移除常數特徵
    del_constant_columns = []
    for column in classified_data.columns:
        if column != 'classification' and column != 'patient':
            if len(classified_data[column].unique()) == 1:
                del_constant_columns.append(column)
    classified_data = classified_data.drop(del_constant_columns, axis=1)
    print('Removel Basic Filter Feature :', del_constant_columns )
    
    print('columns:', len(classified_data.columns[(classified_data.columns != 'classification') & (classified_data.columns != 'Gender')]) )
    # 設定半常數特徵的的門檻  whithout classification
    threshold = 0.8
    # 移除半常數特徵 whithout classification
    constant_filter = VarianceThreshold(threshold=threshold)
    constant_filter.fit(classified_data)
    del_half_constant_columns = []
    constant_columns = [column for column in classified_data.columns if column not in classified_data.columns[constant_filter.get_support()]]
    for feature in constant_columns:
        if feature != 'classification' and feature != 'patient' and feature != 'vision'and feature != 'Gender':
            if feature in classified_data.columns:
                del_half_constant_columns.append(feature)
                classified_data = classified_data.drop(feature, axis=1)
                
    print('Removel Half Constant Filter Feature :', del_half_constant_columns )
    
    print('columns:', len(classified_data.columns[(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )
    
    # 移除Duplicated Feature
    del_dup_columns = []
    del_dup_columns = classified_data.columns[classified_data.columns.duplicated()]
    classified_data = classified_data.T.drop_duplicates().T
    print('Removel Duplicate Filter Feature :', del_dup_columns)
    print('columns:', len(classified_data.columns[(classified_data.columns != 'classification') & (classified_data.columns != 'patient')]) )

    
    # print the column name which has the nan value
    print('nan value:', classified_data.isnull().sum().sum())

    
    # classified_data = classified_data.dropna()
    
    X, y = classified.get_data(classified_data)
    print('classification == 1:', len(y[y == 1]))
    # classification == 0
    print('classification == 0:', len(y[y == 0]))
    print(len(X), len(y))
    nan_values_X = np.isnan(X)
    nan_values_y = np.isnan(y)
    print('nan_values_X:', X.columns[nan_values_X.any()])
    print('nan_values_y:', y[nan_values_y])
    
    X = normalization2(X)
    # classification == 1
    print('classification == 1:', len(y[y == 1]))
    # classification == 0
    print('classification == 0:', len(y[y == 0]))
    print(len(X), len(y))
    
    # 使用 K-Fold 交叉驗證'
    n_splits = 5
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=3) 19 38  21 27  47 71 61 83
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=8)
    # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=24)
    cases = ['RFECV']# 'original','RFECV'
    gridsearch = False  #False
    
    for case in cases:
        column_names = X.columns.tolist()
        for classifier_name, classifier in classifiers.items():

            
            if case == 'RFECV' and gridsearch == True: # 使用 Pipeline 將特徵選擇和參數調優過程結合起來
                if  classifier_name == 'KNN':
                    continue
                if classifier_name == 'SVM':
                    classifier = svm.SVC(kernel='linear') 
                # 9
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)   
                min_features = 5
                if classifier_name == 'CatBoost' or classifier_name == 'AdaBoost'or classifier_name == 'Random Forest':
                    min_features = 10
                # if classifier_name == 'LogisticRegression' :
                #     min_features  = 5
                # else:
                #     min_features = 20
                pipeline = Pipeline([ # RFECV(classifier, cv=cv, scoring='roc_auc', min_features_to_select=min_features)
                    ('feature_selection', RFECV(classifier, cv=cv, scoring='f1', min_features_to_select=min_features)),
                    ('grid_search', GridSearchCV(classifier, param_grid[classifier_name]))  ,
                ])

                
                pipeline.fit(X, y)
                best_estimator  = pipeline.named_steps['grid_search'].best_estimator_
                classifier = pipeline.named_steps['grid_search'].best_estimator_
                print('best_estimator:', best_estimator)
                if classifier_name == 'CatBoost':
                    # Get best estimator
                    print('best_estimator:', best_estimator.get_params())
                selector = pipeline.named_steps['feature_selection']
                selected_feature_indices = selector.get_support(indices=True)
                selected_columns = [column_names[i] for i in selected_feature_indices]
                
                
                
                # save
                if not os.path.isfile(os.path.join('record', disease + '_' + date , 'selector.csv')):
                    with open( os.path.join('record', disease + '_' + date , 'selector.csv') ,'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['model', 'selected_columns'])
                with open( os.path.join('record', disease + '_' + date , 'selector.csv') ,'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ classifier_name, selected_columns])
                    
                print('selector:',selected_columns)
                X = selector.transform(X)
            if case == 'RFECV':
                if  classifier_name == 'KNN':
                    continue
                if classifier_name == 'SVM':
                    classifier = svm.SVC(kernel='linear') # 10 39  5 39 5 50 5 29 
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=31)  
                min_features = 20
                rfecv = RFECV(classifier, cv=cv, scoring='f1', min_features_to_select=min_features)
                X = rfecv.fit_transform(X, y)
 
                selected_columns = [column_names[i] for i in rfecv.get_support(indices=True)]
                if not os.path.isfile(os.path.join('record', disease + '_' + date , 'selector.csv')):
                    with open( os.path.join('record', disease + '_' + date , 'selector.csv') ,'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['model', 'selected_columns'])
                with open( os.path.join('record', disease + '_' + date , 'selector.csv') ,'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ classifier_name, selected_columns])
 
                
            elif case == 'RFE' and gridsearch == True:
                if classifier_name == 'SVM':
                    classifier = svm.SVC(kernel='linear')
                elif classifier_name == 'KNN':
                    continue
                pipeline = Pipeline([
                    ('feature_selection', RFE(classifier, n_features_to_select=20)),
                    ('grid_search', GridSearchCV(classifier, param_grid[classifier_name], refit = True))
                ])
                pipeline.fit(X, y)
                
                classifier = pipeline.named_steps['grid_search'].best_estimator_
                print('best_estimator:', pipeline.named_steps['grid_search'].best_estimator_)
                selector = pipeline.named_steps['feature_selection']
                print('selector:', X.columns[selector.support_])
                X = selector.transform(X)
                
                
                
            elif case == 'PCA':
                pca = PCA(n_components=0.9)
                X = pca.fit_transform(X)
                # X = pd.DataFrame(X)
          
            elif gridsearch == True:
                grid = GridSearchCV(classifier, param_grid[classifier_name])
                grid.fit(X, y)
                classifier = grid.best_estimator_
                best_estimator = grid.best_estimator_
                print('best_estimator:', grid.best_estimator_)
                if classifier_name == 'CatBoost':
                    # Get best estimator
                    print('best_estimator:', best_estimator.get_params())
            print('Classifier:', classifier_name)
            accuracy_scores = []
            recall_scores = []
            precision_scores = []
            f1_scores = []
            specificity_scores = []
            count = 0
            for train_index, test_index in kf.split(X, y):
                count += 1
                
                if case == 'PCA' or case == 'Genetic' or case =='RFECV' or case =='RFE' or case == 'SelectFromModel':
                    X_train, X_test = X[train_index], X[test_index]
                else:
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                print('y_train 1:' , len(y_train[y_train == 1]))
                print('y_train 0:' , len(y_train[y_train == 0]))
                # if len(y_train[y_train == 1]) * 0.5 < len(y_train[y_train == 0]):
                # data augmentation
                
                # ADASYN 0.7 7 39  
                # sampling_strategy=0.9,n_neighbors=9, random_state=21 0.75 5 3  0.6 5 33
                adasyn = ADASYN(sampling_strategy=0.9,n_neighbors=9, random_state=54) #0.9 9 : 27 54 62 22 
                X_train, y_train = adasyn.fit_resample(X_train, y_train)
                print('ADASYN y_train 1:', len(y_train[y_train == 1]))
                print('ADASYN y_train 0:', len(y_train[y_train == 0]))


                print('y_test 1:' , len(y_test[y_test == 1]))
                print('y_test 0:' , len(y_test[y_test == 0]))
                # feature_importance_SHAP(classifier_name, disease + '_' + date, classifier, X_train,selected_columns, case, count )
                # permutation_importances(classifier_name, disease + '_' + date, classifier, X_train, y_train,selected_columns, count,'K-Fold'+ str(n_splits))
                model = classifier.fit(X_train, y_train)
                # print('importances:', importances)
  
                
                # shap
                # shap_values = feature_importance_SHAP(classifier_name,disease + '_' + date, model, X_test, case)
                # plot_feature_importance(model, X.columns)
                y_pred = model.predict(X_test)
                accuracy, recall, precision, f1 ,specificity = classified.evaluate_model(y_test, y_pred, classifier_name, 'K-Fold'+ str(n_splits), case)
                accuracy_scores.append(accuracy)
                recall_scores.append(recall)
                precision_scores.append(precision)
                f1_scores.append(f1)
                specificity_scores.append(specificity)    

            Avg_accuracy = round(np.mean(accuracy_scores)*100, 2) 
            Avg_recall = round(np.mean(recall_scores)*100, 2)
            Avg_precision = round(np.mean(precision_scores)*100, 2)
            Avg_f1 = round(np.mean(f1_scores)*100, 2)
            Avg_specificity = round(np.mean(specificity_scores)*100, 2)
            std_accuracy = round(np.std(accuracy_scores, ddof=1)*100, 2)
            std_recall = round(np.std(recall_scores, ddof=1)*100, 2)
            std_precision = round(np.std(precision_scores, ddof=1)*100, 2)
            std_f1 = round(np.std(f1_scores, ddof=1)*100, 2)
            std_specificity = round(np.std(specificity_scores, ddof=1)*100, 2)
    
            # save the result to csv
            if not os.path.isfile(os.path.join('record', disease + '_' + date ,output_file_name)):
                with open( os.path.join('record', disease + '_' + date , output_file_name) ,'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['method', 'classifier', 'split_method', 'Avg accuracy','Std accuracy','Avg precision','Std precision', 'Avg recall', 'Std recall', 'Avg specificity', 'Std specificity',  'Avg f1', 'Std f1'])
                    
            with open( os.path.join('record', disease + '_' + date ,output_file_name) ,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([case, classifier_name,  'K-Fold'+ str(n_splits), Avg_accuracy, std_accuracy, Avg_precision, std_precision, Avg_recall, std_recall, Avg_specificity, std_specificity, Avg_f1, std_f1])
 