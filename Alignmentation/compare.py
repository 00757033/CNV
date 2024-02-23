import csv
import os
import json

def compare(date,filename,crop =False):
    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mathod','mse','mse_std','psnr','psnr_std','ssim','ssim_std','match_mse','match_mse_std','match_psnr','match_psnr_std','match_ssim','match_ssim_std' , 'difference_mse,difference _psnr','difference_ssim'])
    for folder in os.listdir('./record/'+date):
        if folder.endswith('.json'):
            if 'eval' in folder or 'Eval' in folder:
                # get the avg 
                if crop :
                    if folder.startswith('crop'):
                        
                        with open('./record/'+date+ '/'+folder,'r') as f:
                            data = json.load(f)
                            original = data['original']
                            avg =  data['avg']
                    else:
                        continue
                else:
                    if not folder.startswith('crop'):
                        with open('./record/'+date+ '/'+folder,'r') as f:
                            data = json.load(f)
                            original = data['original']
                            avg =  data['avg']
                    else:
                        continue             

                # write the avg to csv
                with open(filename,'a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ folder,
                                        original['mse'][0],original['mse'][1],
                                        original['psnr'][0],original['psnr'][1],
                                        original['ssim'][0],original['ssim'][1],
                                        avg['mse'][0],avg['mse'][1],
                                        avg['psnr'][0],avg['psnr'][1],
                                        avg['ssim'][0],avg['ssim'][1],
                                        round(avg['mse'][0] - original['mse'][0],2),
                                        round(avg['psnr'][0] - original['psnr'][0],2),
                                        round(avg['ssim'][0] - original['ssim'][0],2)
                                        ])
                    
                    
if __name__ == '__main__':

    date = '1121'
    disease = 'PCV'
    
    PATH_BASE = disease + '_' + date
    file_name = 'crop_'+ PATH_BASE
    compare(PATH_BASE,'./record/'+ file_name+'_eval.csv',crop = True)
    