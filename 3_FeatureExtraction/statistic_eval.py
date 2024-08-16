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

        
        
        
    def stat(self, file_name,feature_names = "Morphologyfeature"):
        # read json file
        with open(file_name) as f:
            data = pd.read_csv(f)
        poor = pd.DataFrame(data[data['classification'] == 0]).reset_index(drop=True)
        good = pd.DataFrame(data[data['classification'] == 1]).reset_index(drop=True)   
        
        features = ['VAD','VSD','VDI','VPI','VCI']
        for feature in features:
            if feature == 'classification' or feature == 'patient':
                continue
            # if feature_names == "Morphology":
            #     continue
            
            data_0 = poor[feature]
            data_1 = good[feature]
            
            MannWhitney2(data_0, data_1, feature, 'CC', ['Poor Treatment', 'Good Treatment'], self.disease, feature_names)
            
             
def MannWhitney2( data_0, data_1, feature_name,layer, injection,disease,feature_names):
    print(feature_name)
    if not os.path.exists('./record/'+ disease+ '/'+ feature_names ):
        os.makedirs('./record/'+ disease+ '/'+ feature_names )
        
    statistic_0 , pvalue_0 = stats.mannwhitneyu(data_0, data_1,alternative='two-sided')
    print(statistic_0,pvalue_0)
    #plot boxplot
    if pvalue_0 < 1:
        orders = ['Poor Treatment', 'Good Treatment']
        data = pd.DataFrame({'Poor Treatment':data_0,'Good Treatment':data_1})
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=data, palette="Set3", width=0.6)
        ax.set_title( feature_name)
        ax.set_ylabel('Value')
        ax.set_xlabel('Classification')
        
        annotator = Annotator(ax, pairs=[("Poor Treatment", "Good Treatment")], data=data, order=orders)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=3)
        annotator.apply_and_annotate()
        plt.savefig('./record/'+ disease+ '/'+ feature_names + '/' + feature_name+ '.png')
        plt.clf()
        data.to_csv('./record/'+ disease+ '/'+ feature_names + '/' + feature_name + '.csv')
        
        avg_0 = np.mean(data_0)
        avg_1 = np.mean(data_1)
        std_0 = np.std(data_0, ddof=1)
        std_1 = np.std(data_1, ddof=1)
        relative = (avg_1 - avg_0) / avg_0 * 100
        with open('./record/'+ disease+ '/'+ feature_names + '/' + feature_name  + '.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['avg', avg_0, avg_1])
            writer.writerow(['std', std_0, std_1])
            f.close()
                    
    
        feature_static_path =  os.path.join('./record/'+ disease+ '/'+ feature_names + '/' +'feature_static' + '.csv')
        if not os.path.exists(feature_static_path):
            with open(feature_static_path, 'a') as f:
                f.write('Feature,Pre-treatment,std-pre,Post-treatment,std-post,relative\n')
        with open(feature_static_path, 'a') as f:
            f.write(feature_name + ',' + str(avg_0) + ',' + str(std_0) + ',' + str(avg_1) + ',' + str(std_1) + ',' + str(relative) +  '\n')
            
                
    
if __name__ == '__main__':
    PATH_BASE = '../../Data/'
    data_class = 'PCV'
    data_date = '20240524'
    PATH_BASE  =  PATH_BASE + data_class + '_' + data_date + '/'
    path = PATH_BASE +  '/compare/'
    disease = data_class + '_' + data_date
    
    statistic = Statistic(PATH_BASE,disease,path)
    feature_list = ['Morphology','GLCM','GLRLM','GLSZM','NGTDM','GLDM','HOG','DWT']
    
    statistic.stat('./record/' + disease + '/classification_ROI.csv','feature_importance2')
    