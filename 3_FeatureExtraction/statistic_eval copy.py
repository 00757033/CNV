import pandas as pd
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats.morestats import stats
from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
import json 
import scipy.stats as stats
import tools.tools as tools
import csv
class Statistic:
    def __init__( self, PATH_BASE,disease,compare_path, layers = ['CC']):
        self.PATH_BASE = PATH_BASE
        self.compare_path = compare_path
        self.layers = layers
        self.disease = disease
        self.Morphologyfeature = [ 'VAD','VSD','VDI','VPI','FD_2','FD_4','FD_8']
        self.GLCMfeature = ['Contrast','Autocorrelation','JointEnergy','Homogeneity','DifferenceEntropy','Dissimilarity']
        self.GLCMfeature = ['GLCM' + '_' + feature for feature in self.GLCMfeature]
        self.GLRLMfeature = ['ShortRunEmphasis','LongRunEmphasis','GrayLevelNonUniformity','RunLengthNonUniformity','RunPercentage','LowGrayLevelRunEmphasis','HighGrayLevelRunEmphasis','ShortRunLowGrayLevelEmphasis','ShortRunHighGrayLevelEmphasis','LongRunLowGrayLevelEmphasis','LongRunHighGrayLevelEmphasis']
        self.GLSZMfeature = ['SmallAreaEmphasis','LargeAreaEmphasis','GrayLevelNonUniformity','SizeZoneNonUniformity','GrayLevelVariance','SizeZoneVariance','ZonePercentage','GrayLevelEntropy']
        self.NGTDMfeature = ['Coarseness','Contrast','Busyness','Complexity','Strength']
        self.GLDMfeature = ['LowGrayLevelEmphasis','HighGrayLevelEmphasis','GrayLevelNonUniformity','GrayLevelNonUniformityNormalized','GrayLevelVariance','GrayLevelVarianceNormalized']
        self.SFMfeature = ['Periodicity','Roughness']
        self.HOGfeature = ['HOG_Kurtosis','HOG_Skewness','HOG_Mean','HOG_Std']
        self.LBPfeature = ['LBP']
        self.DWTfeature = ['avg_LH','avg_HL','std_LH','std_HL']
        
        
        
    def stat(self, file_name,feature_names = "Morphology"):
        # read csv file
        with open(file_name) as f:
            data = pd.read_csv(f)
        
        if feature_names == "Morphology":
            feature = self.Morphologyfeature
            
        elif feature_names == "GLCM":
            feature = self.GLCMfeature
            
        elif feature_names == "GLRLM":
            feature = self.GLRLMfeature
            
        elif feature_names == "GLSZM":
            feature = self.GLSZMfeature
            
        elif feature_names == "NGTDM":
            feature = self.NGTDMfeature
            
        elif feature_names == "GLDM":
            feature = self.GLDMfeature
            
        elif feature_names == "SFM":
            feature = self.SFMfeature
            
        elif feature_names == "Statistic":
            feature = self.Statisticfeature
            
        elif feature_names == "LBP":
            feature = self.LBPfeature
        
        elif feature_names == "HOG":
            feature = self.HOGfeature
            
        elif feature_names == "DWT":
            feature = self.DWTfeature
            
        poor = pd.DataFrame(data[data['classification'] == 0]).reset_index(drop=True)
        good = pd.DataFrame(data[data['classification'] == 1]).reset_index(drop=True)
            
        # statistic
        for layer in self.layers:
            for feature_name in feature:
                MannWhitney2(poor[feature_name + '_' + layer], good[feature_name + '_' + layer], feature_name,layer, ["Poor Treatment", "Good Treatment"],self.disease,feature_names)

            
            
# def ArrayMannWhitney( data_0, data_1, data_2, feature_name,layer, injection,disease,feature_names): 
#     # for data_0[0] is array
#     print(feature_name,layer)
#     statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1)
#     statistic_1 , pvalue_1 = stats.mannwhitneyu(data_1, data_2)
#     statistic_2 , pvalue_2 = stats.mannwhitneyu(data_0, data_2)
#     print(statistic_0,pvalue_0)
#     print(statistic_1,pvalue_1)
#     print(statistic_2,pvalue_2)
#     order = ["Pre-treatment", "1-injection", "2-injection"]
#     group_pairs = [("Pre-treatment", "1-injection"), ("1-injection", "2-injection"), ("Pre-treatment", "2-injection")]
#     fig, ax = plt.subplots()
#     print(len(statistic_0))
#     for i in range(len(statistic_0)):
        
#         print("plot boxplot")
#         feature_data_0 = [t for t in zip(*data_0)][i]
#         feature_data_1 = [t for t in zip(*data_1)][i]
#         feature_data_2 = [t for t in zip(*data_2)][i]
#         df2 = pd.DataFrame (list(zip(feature_data_0,feature_data_1,feature_data_2)),columns=['Pre-treatment', '1-injection', '2-injection'])
#         ax = sns.boxplot(data=df2, palette="Set3")
#         ax.set_title(feature_names + ' : ' + feature_name  + ' ' + str(i) + ' in  ' + layer)
#         ax.set_ylabel('Value')
#         ax.set_xlabel('Injection')
#         ax.set_xticklabels(injection)
#         annotator = Annotator( ax,pairs = group_pairs , data=df2, order=order)
#         annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=3)
#         annotator.apply_and_annotate()
#         # show title p < 0.05 p < 0.01 p < 0.001
        
        
        
#         tools.makefolder('./record/'+ disease+ '/'+ feature_name + '/' + layer)
#         plt.savefig('./record/'+ disease+ '/'+ feature_name + '/' + layer + '/' + str(i) + '.png')
#         # plt.show()
#         plt.clf()
        
        
        
#     # 0 : Pre-treatment
#     # 1 : 1-injection
#     # 2 : 2-injection

#     print(data_0[0][0])
#     for i in range(len(data_0[0])):
#         print(i)
        
        
               
                             
    
def MannWhitney2( data_0, data_1, feature_name,layer, injection,disease,feature_names):
    if not os.path.exists('./record/'+ disease+ '/'+ feature_names ):
        os.makedirs('./record/'+ disease+ '/'+ feature_names )
    print(feature_name,layer)
    statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1)
    
    # len(data_0) != len(data_1)
    # static
    print(statistic_0,pvalue_0)
    fig, ax = plt.subplots()
    feature_data_0 = data_0
    feature_data_1 = data_1
    df2 = pd.DataFrame({'Poor Treatment': data_0, 'Good Treatment': data_1})
    plt.figure(figsize=(6, 6))  # 设置图形大小
    ax = sns.boxplot(data=df2, palette="Set3", width=0.8)
    ax.set_title(feature_names + ' : ' + feature_name  + ' in  ' + layer)
    ax.set_ylabel('Value')
    ax.set_xlabel('Classification')
    ax.set_xticklabels(injection)
    annotator = Annotator( ax, pairs = [("Poor Treatment", "Good Treatment")], data=df2, order=["Poor Treatment", "Good Treatment"])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=3)
    annotator.apply_and_annotate()
    plt.savefig('./record/'+ disease+ '/'+ feature_names + '/' + feature_name +'_' + layer + '.png')
    plt.clf()
    df2.to_csv('./record/'+ disease+ '/'+ feature_names + '/' + feature_name +'_' + layer + '.csv')
    
    avg_0 = np.mean(data_0)
    avg_1 = np.mean(data_1)
    std_0 = np.std(data_0, ddof=1)
    std_1 = np.std(data_1, ddof=1)
    
    
    with open('./record/'+ disease+ '/'+ feature_names + '/' + feature_name +'_' + layer + '.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['avg', avg_0, avg_1])
        writer.writerow(['std', std_0, std_1])
        f.close()
    
                    
                
            
        
        
        
        
                
        
        
    
    
    
if __name__ == '__main__':
    PATH_BASE = '../../Data/'
    data_class = 'PCV'
    data_date = '20240524'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    disease = data_class + '_' + data_date
    
    statistic = Statistic(PATH_BASE,disease,path)
    feature_list = ['Morphology','GLCM','GLRLM','GLSZM','NGTDM','GLDM','HOG','DWT']
    
    statistic.stat('./record/' + disease + '/classification_ROI.csv','GLCM')
    