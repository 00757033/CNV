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
        
        self.Morphologyfeature = [ 'VAD','VSD','VDI','VPI']
        
        
        
    def stat(self, file_name):
        # read json file
        with open('./record/' + self.disease+ '/'+ file_name) as f:
            data = json.load(f)   
        
        feature_avg = {}
        # if feature_names == "Morphology":
        #     feature = self.Morphologyfeature
              
        #     for feature_name in feature:
        #         feature_avg[feature_name] = {}
        #         CC_0 = []
        #         CC_1 = []
        #         for patient in data.keys():
        #             if "Pre-treatment"  in data[patient]:
        #                 CC_0.append(data[patient]["Pre-treatment"][feature_name + '_CC'])
        #             if "Post-treatment"  in data[patient]:
        #                 CC_1.append(data[patient]["Post-treatment"][feature_name + '_CC'])
                        
        #         print('feature_name',feature_name)        
        #         # Mann-Whitney U test
        #         print("Mann-Whitney U test")
        #         if len(CC_0) != 0 and len(CC_1) != 0 :
        #             CC_0_avg,CC_1_avg = MannWhitney2(CC_0, CC_1,  feature_name ,'CC', self.disease,feature_names)
        #             feature_avg[feature_name]['CC'] = {}
        #             feature_avg[feature_name]['CC']['Pre-treatment'] = CC_0_avg
        #             feature_avg[feature_name]['CC']["Post-treatment"] = CC_1_avg
                    
                        
        # else:
        print('keys',data[list(data.keys())[0]]["Pre-treatment"].keys())
        feature= data[list(data.keys())[0]]["Pre-treatment"].keys()
        for feature_name in feature:
            if feature_name == 'Date':
                continue
            if  feature_name == 'HOG Hist8_CC' or feature_name == 'HOG Median _CC' :
                continue
            if 'freq_0.05' in  feature_name :
                continue
            feature_avg[feature_name] = {}
            CC_0 = []
            CC_1 = []
            for patient in data.keys():
                if "Pre-treatment"  in data[patient]:
                    CC_0.append(data[patient]["Pre-treatment"][feature_name])
                if "Post-treatment"  in data[patient]:
                    CC_1.append(data[patient]["Post-treatment"][feature_name])
            print('feature_name',feature_name)        
            # Mann-Whitney U test
            print("Mann-Whitney U test")
            if len(CC_0) != 0 and len(CC_1) != 0 :
                CC_0_avg,CC_1_avg = MannWhitney2(CC_0, CC_1,  feature_name ,'CC', self.disease)
                feature_avg[feature_name]= {}
                feature_avg[feature_name]['Pre-treatment'] = CC_0_avg
                feature_avg[feature_name]["Post-treatment"] = CC_1_avg

    
def MannWhitney2( data_0, data_1, feature_name,layer,disease):
    print(feature_name,layer)
    feature_name = feature_name.split('_CC')[0]
    statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1,alternative='two-sided')
    print(statistic_0,pvalue_0)
    #plot boxplot
    # 繪製直方圖
    # sns.histplot(data_0, kde=True)
    # plt.title('Histogram')
    # plt.show()
    # sns.histplot(data_1, kde=True)
    # plt.title('Histogram')
    # plt.show()
    Morphology = ['VAD','VSD','VDI','VPI','VCI']
    clinical = ['CMT','SFCT']

    if pvalue_0 < 0.05 or (feature_name in Morphology) or (feature_name in clinical):
        
        orders = ["Pre-Treatment", "Post-Treatment"]
        df2 = pd.DataFrame( list(zip( data_0, data_1)), columns=['Pre-Treatment', 'Post-Treatment'])
        # plt.figure(figsize=(8,8), dpi= 80)
        ax = sns.boxplot(data=df2, palette="Set3", width=0.8, linewidth=1.5,order=orders)
        ax.set_title( feature_name, fontsize=15)
        ax.set_ylabel('Value',fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'], fontsize=15)
        annotator = Annotator( ax, pairs = [("Pre-Treatment", "Post-Treatment")], data=df2, order=orders)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',fontsize=15)
        p_value_annotation = annotator.apply_and_annotate()
        
        # del ns in plot
        for annotation in ax.texts:
            if 'ns' in annotation.get_text():
                annotation.set_visible(False)
                
        # print(p_value_annotation)
        

            
        # Extract and print the p-value
        p_value = p_value_annotation[1][0]
        print("Mann-Whitney U test p-value:", p_value)
        tools.makefolder('./record/'+ disease+ '/' + 'features')
        plt.savefig('./record/'+ disease+ '/'+ 'features' + '/' + feature_name + '.png')
        plt.clf()
    
        df2.to_csv('./record/'+ disease+ '/'+ 'features' + '/' + feature_name + '.csv')
    
        avg_0 = np.mean(data_0)
        avg_1 = np.mean(data_1)
        std_0 = np.std(data_0, ddof=1)
        std_1 = np.std(data_1, ddof=1)
        
        relative = (avg_1 - avg_0) / avg_0 * 100
        
        # save statistic
        with open('./record/'+ disease+ '/'+ 'features' + '/' + feature_name + '.csv', 'a') as f:
            f.write('Pre-treatment,Post-treatment\n')
            f.write(str(avg_0) + ',' + str(avg_1) + '\n')
            f.write(str(std_0) + ',' + str(std_1) + '\n')
            f.write(str(relative) + '\n')
            f.write(str(p_value) + '\n')
            
        feature_static_path =  os.path.join('./record/'+ disease+ '/'+'features'+ '/' +'feature_static' + '.csv')
        if not os.path.exists(feature_static_path):
            with open(feature_static_path, 'a') as f:
                f.write('Feature,Pre-treatment,std-pre,Post-treatment,std-post,relative\n')
        with open(feature_static_path, 'a') as f:
            f.write(feature_name + ',' + str(avg_0) + ',' + str(std_0) + ',' + str(avg_1) + ',' + str(std_1) + ',' + str(relative) +'\n')
            
            
             
    avg_0 = np.mean(data_0)
    avg_1 = np.mean(data_1)                
        
    return avg_0,avg_1

                    
                
            
        
        
        
        
                
        
        
    
    
    
if __name__ == '__main__':
    PATH_BASE = '../../Data/'
    data_class = 'PCV'
    data_date = '20240524'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    disease = data_class + '_' + data_date
    
    statistic = Statistic(PATH_BASE,disease,path)
    # feature_list = ['Morphology','GLCM','GLRLM','GLSZM','NGTDM','GLDM','SFM','DWT']
    
    statistic.stat('VesselFeature_ROI.json')
    