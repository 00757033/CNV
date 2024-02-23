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

class Statistic:
    def __init__( self, PATH_BASE,disease,compare_path, layers = ['CC', 'OR']):
        self.PATH_BASE = PATH_BASE
        self.compare_path = compare_path
        self.layers = layers
        self.disease = disease
        
        self.Morphologyfeature = [ 'VD','VLD','VAPR','VLA','VDI']
        self.GLCMfeature = ['Contrast','Autocorrelation','Energy','Homogeneity','DifferenceEntropy','Dissimilarity']
        self.GLRLMfeature = ['ShortRunEmphasis','LongRunEmphasis','GrayLevelNonUniformity','RunLengthNonUniformity','RunPercentage','LowGrayLevelRunEmphasis','HighGrayLevelRunEmphasis','ShortRunLowGrayLevelEmphasis','ShortRunHighGrayLevelEmphasis','LongRunLowGrayLevelEmphasis','LongRunHighGrayLevelEmphasis']
        self.GLSZMfeature = ['SmallAreaEmphasis','LargeAreaEmphasis','GrayLevelNonUniformity','SizeZoneNonUniformity','GrayLevelVariance','SizeZoneVariance','ZonePercentage','GrayLevelEntropy']
        self.NGTDMfeature = ['Coarseness','Contrast','Busyness','Complexity','Strength']
        self.GLDMfeature = ['LowGrayLevelEmphasis','HighGrayLevelEmphasis','GrayLevelNonUniformity','GrayLevelNonUniformityNormalized','GrayLevelVariance','GrayLevelVarianceNormalized']
        self.SFMfeature = ['Periodicity','Roughness']
        self.HOGfeature = ['HOG']
        self.LBPfeature = ['LBP']
        self.DWTfeature = ['avg_LH','avg_HL','std_LH','std_HL']
        
        
        
    def stat(self, file_name,feature_names = "Morphologyfeature"):
        # read json file
        with open('./record/' + self.disease+ '/'+ file_name) as f:
            data = json.load(f)
        
        

        injection = ["0","1","2"]
        
        feature_avg = {}
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
            
            
            
        
        
        
        for feature_name in feature:
            feature_avg[feature_name] = {}
            OR_0 = []
            OR_1 = []
            OR_2 = []
            CC_0 = []
            CC_1 = []
            CC_2 = []
            for patient in data.keys():
                if str(0)  in data[patient]:
                    if feature_name + '_OR' in data[patient][str(0)]:
                        OR_0.append(data[patient][str(0)][feature_name + '_OR'])
                    if feature_name + '_CC' in data[patient][str(0)]:
                        CC_0.append(data[patient][str(0)][feature_name + '_CC'])
                if str(1)  in data[patient]:
                    if feature_name + '_OR' in data[patient][str(1)]:
                        OR_1.append(data[patient][str(1)][feature_name + '_OR'])
                    if feature_name + '_CC' in data[patient][str(1)]:
                        CC_1.append(data[patient][str(1)][feature_name + '_CC'])
                if str(2)  in data[patient]:
                    if feature_name + '_OR' in data[patient][str(2)]:
                        OR_2.append(data[patient][str(2)][feature_name + '_OR'])
                    if feature_name + '_CC' in data[patient][str(2)]:
                        CC_2.append(data[patient][str(2)][feature_name + '_CC'])
                        
            print(feature_name)
            print("OR_0",len(OR_0))
            print("OR_1",len(OR_1))
            print("OR_2",len(OR_2))
            print("CC_0",len(CC_0))
            print("CC_1",len(CC_1))
            print("CC_2",len(CC_2))

            # Mann-Whitney U test
            print("Mann-Whitney U test")
            if len(OR_0) != 0 and len(OR_1) != 0 and len(OR_2) != 0:
                if feature_names == "LBP":
                    ArrayMannWhitney(OR_0, OR_1, OR_2, feature_name ,'OR', injection, self.disease,feature_names)     
                else:
                    OR_0_avg,OR_1_avg,OR_2_avg = MannWhitney(OR_0, OR_1, OR_2, feature_name ,'OR', injection, self.disease,feature_names)
                    feature_avg[feature_name]['OR'] = {}
                    feature_avg[feature_name]['OR']['Pre-treatment'] = OR_0_avg
                    feature_avg[feature_name]['OR']['1-injection'] = OR_1_avg
                    feature_avg[feature_name]['OR']['2-injection'] = OR_2_avg  
 
                    
            if len(CC_0) != 0 and len(CC_1) != 0 and len(CC_2) != 0:
                if feature_names == "LBP":
                    ArrayMannWhitney(CC_0, CC_1, CC_2, feature_name ,'CC', injection, self.disease,feature_names)
                else:
                    CC_0_avg,CC_1_avg,CC_2_avg = MannWhitney(CC_0, CC_1, CC_2, feature_name ,'CC', injection, self.disease,feature_names)
            
                    feature_avg[feature_name]['CC'] = {}
                    feature_avg[feature_name]['CC']['Pre-treatment'] = CC_0_avg
                    feature_avg[feature_name]['CC']['1-injection'] = CC_1_avg
                    feature_avg[feature_name]['CC']['2-injection'] = CC_2_avg
            

        # save feature_avg
        with open('./record/' + self.disease+ '/'+ 'feature_avg.json', 'w') as fp:
            json.dump(feature_avg, fp, indent=4)
            
    
        return feature_avg    
            
            
def ArrayMannWhitney( data_0, data_1, data_2, feature_name,layer, injection,disease,feature_names): 
    # for data_0[0] is array
    print(feature_name,layer)
    statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1)
    statistic_1 , pvalue_1 = stats.mannwhitneyu(data_1, data_2)
    statistic_2 , pvalue_2 = stats.mannwhitneyu(data_0, data_2)
    print(statistic_0,pvalue_0)
    print(statistic_1,pvalue_1)
    print(statistic_2,pvalue_2)
    order = ["Pre-treatment", "1-injection", "2-injection"]
    group_pairs = [("Pre-treatment", "1-injection"), ("1-injection", "2-injection"), ("Pre-treatment", "2-injection")]
    fig, ax = plt.subplots()
    print(len(statistic_0))
    for i in range(len(statistic_0)):
        
        print("plot boxplot")
        feature_data_0 = [t for t in zip(*data_0)][i]
        feature_data_1 = [t for t in zip(*data_1)][i]
        feature_data_2 = [t for t in zip(*data_2)][i]
        df2 = pd.DataFrame (list(zip(feature_data_0,feature_data_1,feature_data_2)),columns=['Pre-treatment', '1-injection', '2-injection'])
        ax = sns.boxplot(data=df2, palette="Set3")
        ax.set_title(feature_names + ' : ' + feature_name  + ' ' + str(i) + ' in  ' + layer)
        ax.set_ylabel('Value')
        ax.set_xlabel('Injection')
        ax.set_xticklabels(injection)
        annotator = Annotator( ax,pairs = group_pairs , data=df2, order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=3)
        annotator.apply_and_annotate()
        # show title p < 0.05 p < 0.01 p < 0.001
        
        
        
        tools.makefolder('./record/'+ disease+ '/'+ feature_name + '/' + layer)
        plt.savefig('./record/'+ disease+ '/'+ feature_name + '/' + layer + '/' + str(i) + '.png')
        # plt.show()
        plt.clf()
        
        
        
    # 0 : Pre-treatment
    # 1 : 1-injection
    # 2 : 2-injection

    print(data_0[0][0])
    for i in range(len(data_0[0])):
        print(i)
        
        
               
                             
def MannWhitney( data_0, data_1, data_2, feature_name,layer, injection,disease,feature_names):
    print(feature_name,layer)
    statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1)
    statistic_1 , pvalue_1 = stats.mannwhitneyu(data_1, data_2)
    statistic_2 , pvalue_2 = stats.mannwhitneyu(data_0, data_2)
    print("0 vs 1",stats.mannwhitneyu(data_0, data_1))
    print("1 vs 2",stats.mannwhitneyu(data_1, data_2))
    print("0 vs 2",stats.mannwhitneyu(data_0, data_2))   
    print("plot boxplot")
    # 0 : Pre-treatment
    # 1 : 1-injection
    # 2 : 2-injection
    order = ["Pre-treatment", "1st Post-treatment", "2nd Post-treatment"]
    df2 = pd.DataFrame( list(zip( data_0, data_1, data_2)), columns=['Pre-treatment', '1st Post-treatment', '2nd Post-treatment'])

    
    print(df2)
    group_pairs = [("Pre-treatment", "1st Post-treatment"), ("1st Post-treatment", "2nd Post-treatment"), ("Pre-treatment", "2nd Post-treatment")]
    # fig, ax = plt.subplots(figsize=(3, 10))
    ax = sns.boxplot(data=df2, palette="Set3")
    # ax.set_title(feature_names + ' : ' + feature_name + ' in  ' + layer)
    # ax.set_ylabel('Value')
    # ax.set_xlabel('Injection')
    feature_vis = feature_name
    if feature_name == 'avg_LH' or feature_name == 'avg_HL' or feature_name == 'std_LH' or feature_name == 'std_HL':
        if 'avg' in feature_name:
            feature_vis = 'Average of ' + feature_name[4:]
        else:
            feature_vis = 'Standard deviation of ' + feature_name[4:]
                    
    if layer == 'OR':
        ax.set_title('Outer Retina : ' + feature_vis)
        
    elif layer == 'CC':
        ax.set_title('Choriocapillaris : ' + feature_vis)
    
    annotator = Annotator( ax,pairs = group_pairs , data=df2, order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=3)
    annotator.apply_and_annotate()
    # legend 在外面 內容為 order 並標註顏色
    # ax.legend(loc='upper right', labels=order)
    # add average and std in new row
    df2.loc['mean'] = df2.mean().round(2) 
    df2.loc['std'] = df2.std().round(2)
    # relative change
    relative1 = (df2['1st Post-treatment']['mean'] - df2['Pre-treatment']['mean'] ) / df2['Pre-treatment']['mean']
    relative2 = (df2['2nd Post-treatment']['mean'] - df2['Pre-treatment']['mean'] ) / df2['Pre-treatment']['mean']
    df2.loc['relative change'] = [0,relative1 * 100,relative2 * 100]
    
    
    # show title p < 0.05 p < 0.01 p < 0.001
    tools.makefolder('./record/'+ disease+ '/'+ feature_names )
    plt.savefig('./record/'+ disease+ '/'+ feature_names + '/'+ feature_name + '_' + layer + '.png')
    plt.show()
    plt.clf()
    
    # save df2
    df2.to_csv('./record/'+ disease+ '/'+ feature_names + '/'+ feature_name + '_' + layer + '.csv')
    
    data_0_avg = round(sum(data_0)/len(data_0),2)
    data_1_avg = round(sum(data_1)/len(data_1),2)
    data_2_avg = round(sum(data_2)/len(data_2),2)
    
    data_0_std = round(np.std(data_0),2)
    data_1_std = round(np.std(data_1),2)
    data_2_std = round(np.std(data_2),2)
    
    # return avg and std
    
    data0 = [data_0_avg,data_0_std]
    data1 = [data_1_avg,data_1_std]
    data2 = [data_2_avg,data_2_std]
    
    return data0,data1,data2
    
    
                  
                    
                
            
        
        
        
        
                
        
        
    
    
    
if __name__ == '__main__':
    PATH_BASE = '../../Data/'
    data_class = 'PCV'
    data_date = '1120'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    disease = data_class + '_' + data_date
    
    statistic = Statistic(PATH_BASE,disease,path)
    feature_list = ['Morphology','GLCM','GLRLM','GLSZM','NGTDM','GLDM','SFM','DWT']
    
    statistic.stat('VesselFeature_Morphology.json','DWT')
    