import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.stats import chi2_contingency
from sklearn.feature_selection import VarianceThreshold
def single_feature_histogram(df:pd.DataFrame):
    """單特徵分析(分布、盒狀圖)

    Args:
        df (pd.DataFrame)
    """
    feature_names = [col for col in df.columns if col not in ['classification',  'patient']]

    # Perform univariate analysis for each feature
    for f in feature_names:
        plt.figure(figsize=(8, 4))
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=f, kde=True)
        plt.xlabel(f)
        plt.ylabel("Frequency")
        plt.title("Histogram")
        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, y=f)
        plt.ylabel(f)
        plt.title("Box Plot")
        plt.tight_layout()
        plt.show()


def pearson_correlation_heatmap(df:pd.DataFrame,feature):
    """特徵相關性分析(heatmap)

    Args:
        df (pd.DataFrame)
    """
    # feature_name= df[[key + '_' + f + '_CC' for f in feature[key]]]
    feature = df
    # Calculate correlation matrix
    print('df.columns',df.columns)
    corr = feature.corr( method='pearson')
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 16})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Correlation Heatmap")
    plt.show()
    


def delete_feature(df:pd.DataFrame,feature):
    feature = df
    feature_name = feature.columns
    # kendall spearman pearson
    corr = feature.corr( method='kendall')
    dup = {}
    for f in feature.columns:
        if len(corr[f][corr[f] == 1]) > 0 or len(corr[f][corr[f] ==-1]) > 0:
            for item in corr[f][((corr[f] ==1)  | (corr[f] ==-1)) & (corr[f].index != f)].index:
                print('-'*50)
                print(f,item,corr[f][item])
                if item in dup:
                    dup[item].append(f)
                else:
                    dup[item] = [f]
    print(len(dup),dup)
    # # 高度相關的特徵每組僅保留一個
    # new_del_feature = set()
    # for f in feature_name:
    #     for item in dup[f]:
    #         if f in feature_name:
    #             new_del_feature.add(f)
    #             feature_name.drop(f)
    # print('new_del_feature',len(new_del_feature),new_del_feature)
    # print('feature_name',feature_name)

                
    
    

    
    feature_name = [f for f in feature.columns if f  in dup ]
    # feature_name 
    # feature = feature[feature_name]
    # corr = feature.corr( method='kendall')
    # # Plot heatmap
    # # plt.figure(figsize=(10, 8))
    # # sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 16})
    # # plt.xticks(fontsize=12)
    # # plt.yticks(fontsize=12)
    # # plt.title("Correlation Heatmap")
    # # plt.show()
    # 倆倆配對僅保留一個
    for f in feature_name:
        for item in dup[f]:
            if f in feature_name:
                feature_name.remove(f)
        
    
    return feature_name
    
    
        
        


    # # Plot heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Correlation Heatmap")
    # plt.show()

if __name__ == '__main__':
    disease   = 'PCV'
    date    = '20240524'
    PATH_BASE    = "../../Data/" + disease + '_' + date + '/'
    PATH_FEATURE = PATH_BASE + 'feature/'
    
    data = './record/' + disease + '_' + date +'/'+'classification_ROI.csv'
    feature = pd.read_csv(data)
    feature = feature.dropna()
    # 移除常數特徵
    feature = feature.loc[:, feature.apply(pd.Series.nunique) != 1]
    print(len(feature.columns))
    feature_names = {
        'Morphology' :[ 'VD','VLD','VDI','FD'],
        'GLCM' : ['Autocorrelation',
                  'ClusterProminence',
                    'ClusterShade',
                    'ClusterTendency',
                    'Contrast',
                    'Correlation',
                    'DifferenceEntropy',
                    'DifferenceVariance',
                    'Dissimilarity',
                    'JointEnergy',
                    'JointEntropy',
                    'IMC1',
                    'IMC2',
                    'Homogeneity',
                    'IDMN',
                    'ID',
                    'IDN',
                    'InverseVariance',
                    'MaximumProbability',
                    'SumAverage',
                    'SumEntropy',
                    'SumSquares',
                    ],
        'GLRLM' : ['ShortRunEmphasis',
                    'LongRunEmphasis',
                    'GrayLevelNonUniformity',
                    'RunLengthNonUniformity',
                    'RunPercentage',
                    'LowGrayLevelRunEmphasis',
                    'HighGrayLevelRunEmphasis',
                    'ShortRunLowGrayLevelEmphasis',
                    'ShortRunHighGrayLevelEmphasis',
                    'LongRunLowGrayLevelEmphasis',
                    'LongRunHighGrayLevelEmphasis'],
        'GLSZM' : ['SmallAreaEmphasis',
                    'LargeAreaEmphasis',
                    'GrayLevelNonUniformity',
                    'SizeZoneNonUniformity',
                    'GrayLevelVariance',
                    'SizeZoneVariance',
                    'ZonePercentage',
                    'GrayLevelEntropy',
                    'ZoneEntropy',
                    'LowGrayLevelZoneEmphasis',
                    'HighGrayLevelZoneEmphasis',
                    'SmallAreaLowGrayLevelEmphasis',
                    'SmallAreaHighGrayLevelEmphasis',
                    'LargeAreaLowGrayLevelEmphasis',
                    'LargeAreaHighGrayLevelEmphasis'],
        'NGTDM' : ['Coarseness',
                    'Contrast',
                    'Busyness',
                    'Complexity',
                    'Strength'],
        'GLDM' : ['LowGrayLevelEmphasis',
                    'HighGrayLevelEmphasis',
                    'GrayLevelNonUniformity',
                    'GrayLevelNonUniformityNormalized',
                    'GrayLevelVariance',
                    'GrayLevelVarianceNormalized']
        }
        
  
    
    # not regression vision and classification
    feature = feature.drop(columns=['classification','patient'])

    feature_columns = feature.columns
    #normalization
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    feature = pd.DataFrame(feature, columns = feature_columns)
    feature_del = delete_feature(feature,feature_names)
    #save  delete feature
    output_file = './record/' + disease + '_' + date +'/'+'classification_ROI_del.csv'
    
    feature_del_df = pd.DataFrame(feature_del,columns = ['feature'])
    feature_del_df.to_csv(output_file, index=False)
    feature = feature.drop(columns=feature_del)
    # print(feature.columns,len(feature.columns))
    #save 
    # pearson_correlation_heatmap(feature,feature_names)
    
    


