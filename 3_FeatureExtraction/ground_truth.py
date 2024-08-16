import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.preprocessing import StandardScaler






if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240502'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    PATH_FEATURE = PATH_BASE + 'feature/'
    
    data = './record/' + disease + '_' + date +'/'+'classification_ROI.csv'
    feature = pd.read_csv(data)
    feature = feature.dropna()
    # scaler = MinMaxScaler()
    features = feature[['vision','regression']]
    # normalization
    # features = StandardScaler().fit_transform(features)
    features = pd.DataFrame(features, columns = ['vision','regression'])
    # 假設 data 是包含所有數據的 DataFrame，'vision' 是特徵列，'regression' 是目標列
    # vision regression 的相關係數
    label2 = features[['vision', 'regression']]
    # rename
    label2.columns = ['Relative Vision Change', 'Relative Thickness Change ']
    print(label2.corr (method='kendall'))
    # plot
    sns.heatmap(label2.corr(method='kendall'), annot=True, cmap='coolwarm', annot_kws={"size": 16})
    plt.title('Correlation Heatmap')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    
    # 計算線性回歸線 
    slope, intercept = np.polyfit(features['vision'], features['regression'], 1)
    print('slope:',slope)
    print('intercept:',intercept)
    # 'vision', 'regression' 分布 含斜率
    sns.scatterplot(x='vision', y='regression', data=features)
    plt.plot(features['vision'], slope * features['vision'] + intercept, color='red')
    # y = slope * x + intercept
    plt.text(10, 10, f'y = {slope:.2f}x + {intercept:.2f}', color='red', fontsize=12)
    plt.title("Scatter Plot")
    plt.xlabel("Relative Vision Change (%) ")
    plt.ylabel("Relative Thickness Change (%)")
    plt.title('Relationship between Relative Vision Change and Relative Thickness Change')
    plt.show()
    