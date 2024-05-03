
from re import S
from sklearn import svm, ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from csv import reader


# 評估
def evaluation(X_train, Y_train, X_test, Y_test, clf):
    clf.fit(X_train, Y_train) #訓練
    Y_pred = clf.predict(X_test) #預測
    print(Y_pred, Y_test)

    # 評估
    tn, fp, fn, tp = confusion_matrix(Y_test,Y_pred).ravel()
    sensitivity = recall_score(Y_test,Y_pred)
    specificity = tn / (fp+tn)
    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred)
    f1 = f1_score(Y_test,Y_pred)
    print('tn:', tn, 'fp:', fp, 'fn:', fn, 'tp:', tp)
    print('Accuracy:', accuracy)
    print('Specificity:', specificity)
    print('Sensitivity:',sensitivity)
    print('Precision:', precision)
    print('F1-score:', f1)
    return accuracy, sensitivity, specificity, precision, f1

# 分類
def SVM(X_train,Y_train,X_test,Y_test):      
    clf = svm.SVC(kernel='rbf',gamma=0.001,C=1000)
    accuracy, sensitivity, specificity, precision, f1 = evaluation(X_train, Y_train, X_test, Y_test, clf)
    return accuracy, sensitivity, specificity, precision, f1
   
def adaboost(X_train,Y_train,X_test,Y_test):
    clf = ensemble.AdaBoostClassifier(n_estimators = 100)
    accuracy, sensitivity, specificity, precision, f1 = evaluation(X_train, Y_train, X_test, Y_test, clf)
    return accuracy, sensitivity, specificity, precision, f1

def randomforest(X_train,Y_train,X_test,Y_test):
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    accuracy, sensitivity, specificity, precision, f1 = evaluation(X_train, Y_train, X_test, Y_test, forest)
    return accuracy, sensitivity, specificity, precision, f1

# 正規化
def normalization(X_train, Y_train, X_test, Y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    Y_train = Y_train
    Y_test = Y_test
    return X_train, Y_train, X_test, Y_test 

# 挑最大值
def select_max(a_list):    
    maxx = a_list[0]
    index = 0
    for i in range(1, len(a_list)):
        if maxx < a_list[i]:
            maxx = a_list[i]
            index = i
    return index

########################### 降維 ###########################
class Dimension_reduction:
    def __init__(self,X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def principal_component_analysis(self, pc_begin, pc_end):
        accuracy_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []
        f1_list = []
        for i in range(pc_begin, pc_end):
            ncomp = i
            pca = PCA(ncomp)
            pca.fit(X_train)
            X_train_proj = pca.transform(X_train)
            X_test_proj = pca.transform(X_test)
            print('降維數:',ncomp)
            accuracy, sensitivity, specificity, precision, f1 = randomforest(X_train_proj,Y_train,X_test_proj,Y_test) # or adaboost
            
            accuracy_list.append(accuracy)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            precision_list.append(precision)
            f1_list.append(f1)
        
        # 挑選陣列中最大值
        index = select_max(accuracy_list)
        # 回傳最大的accuracy值
        return accuracy_list[index], sensitivity_list[index], specificity_list[index], precision_list[index], f1_list[index]


def train_test_splitting(path, x_train, y_train, x_test, y_test, test_list, no, day, num_normal, num_aki):
    df = pd.read_excel(path)
    df = np.array(df) 
    if no in test_list: # 如果是testing set的老鼠
        for img_feat in df: #將df內的所有影像的特徵放入x_test中
            x_test.append(img_feat)
            if day == 1: #第一天為正常影像,label為0
                y_test.append(0)
                num_normal += 1
            else:
                y_test.append(1)
                num_aki += 1
    else:
        for img_feat in df: # 其餘都是training set的老鼠
            x_train.append(img_feat)
            if day == 1:
                y_train.append(0)
            else:
                y_train.append(1)

    return x_train, y_train, x_test, y_test, num_normal, num_aki

def data_shuffle(X_train, Y_train):           
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    return X_train, Y_train




def label_func_diff(features, diff_value):
    label = []
    for feature in features:
        if int(feature[0]) - int(feature[1]) < diff_value:
            label.append(0)
        else:
            label.append(1)
    return label

def label_func_diff_ratio(features, diff_value):
    label = []
    for feature in features:
        if (int(feature[0]) - int(feature[1]))/int(feature[0]) < diff_value:
            label.append(0)
        else:
            label.append(1)
    return label




if __name__ == '__main__':

    selected_list = []
    sets_feat_importance = [] #紀錄每個集合中，所有特徵的重要
    
    path_feature = 'feature_final.csv'
    with open(path_feature, 'r') as csv_file:
        csv_reader = reader(csv_file)
        features = list(csv_reader)
    data_list = []
    label_list =[]

    for feature in features:
        data_list.append(feature[:-2])   
        label_list.append(feature[-2:])
    label_list = label_func_diff_ratio(label_list, 0.05)

    # 評估指標
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    f1_list = []
    # k-fold迴圈數
    for i in range(5):
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            # testing data中normal&aki個別的數量
            num_normal = 0
            num_aki = 0

            x_train, x_test, y_train, y_test = train_test_split(data_list, label_list, train_size=0.7)
            print("LABEL", label_list)
            ###### training data, testing data包成array #####
            X_train = np.array(x_train)
            Y_train = np.array(y_train)
            X_test = np.array(x_test)
            Y_test = np.array(y_test)

            ############# PCA+SVM or adaboost分類 ##############
            pc_begin = 1
            pc_end = 10
            X_train, Y_train, X_test, Y_test = normalization(X_train, Y_train, X_test, Y_test) # 正規化
            X_train, Y_train = data_shuffle(X_train, Y_train)
            pca = Dimension_reduction(X_train, Y_train, X_test, Y_test)
            accuracy, sensitivity, specificity, precision, f1 = pca.principal_component_analysis(pc_begin, pc_end)
            randomforest(X_train,Y_train,X_test,Y_test)
            ############### 所有c10取2的list ###################
            accuracy_list.append(accuracy)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            precision_list.append(precision)
            f1_list.append(f1)
    print("//////////////////////////////////////////////////////")    
    print('Accuracy:', round(np.mean(accuracy_list),4), '±', round(np.std(accuracy_list),4))
    print('Sensitivity:', round(np.mean(sensitivity_list),4), '±', round(np.std(sensitivity_list),4))
    print('Specificity:', round(np.mean(specificity_list),4), '±', round(np.std(specificity_list),4))
    print('Precision:', round(np.mean(precision_list),4), '±', round(np.std(precision_list),4))
    print('F1-score:', round(np.mean(f1_list),4), '±', round(np.std(f1_list),4))
   