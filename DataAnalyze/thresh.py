import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib as pl
path = '../../Data/need_label_2/PCV2'
for patient in pl.Path(path).iterdir():
    for date in patient.iterdir():
        for eye in date.iterdir():
                for layer in eye.iterdir():
                    if layer.stem == '4' or layer.stem == '3':
                                img = cv2.imread(str(layer))
                                img = cv2.resize(img, (304,304))
                                gray_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                normal_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
                                blur1 = cv2.GaussianBlur(normal_img,(1,1),0)
                                blur3 = cv2.GaussianBlur(normal_img,(3,3),0)
                                blur5 = cv2.GaussianBlur(normal_img,(5,5),0)
                                blur7 = cv2.GaussianBlur(normal_img,(7,7),0)
                                # plt.figure(figsize=(10,10))
                                # plt.subplot(2,2,1)
                                # plt.title('blur1')
                                # plt.hist(blur1.ravel(),256,[0,256])
                                # plt.subplot(2,2,2)
                                # plt.title('blur3')
                                # plt.hist(blur3.ravel(),256,[0,256])
                                # plt.subplot(2,2,3)
                                # plt.title('blur5')
                                # plt.hist(blur5.ravel(),256,[0,256])
                                # plt.subplot(2,2,4)
                                # plt.title('blur7')
                                # plt.hist(blur7.ravel(),256,[0,256])


                                # plt.show()


                                clahe_img  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blur3)
                                ret3,th3 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                                # dilated_img = cv2.morphologyEx(clahe_img, cv2.MORPH_DILATE, (7,7))
                                
                                print('ret3',ret3)
                            
                                plt.figure(figsize=(12,12))
                                plt.title(patient.name+' '+date.name+' '+eye.name+' '+layer.stem)
                                plt.subplot(2,2,1)
                                plt.imshow(normal_img, cmap='gray')
                                plt.title('Original')
                                plt.axis('off')  # Remove the axis
                                plt.subplot(2,2,2)
                                plt.hist(normal_img.ravel(), 256, [0, 256])
                                plt.xlabel('Gray Level')
                                plt.ylabel('Frequency')

                                plt.subplot(2,2,3)
                                plt.imshow(th3, cmap='gray')
                                plt.title("Otsu's Threshold")
                                plt.axis('off')  # Remove the axis

                                plt.subplot(2,2,4)
                                plt.hist(gray_img.ravel(), 256, [0, 256])
                                plt.axvline(x=ret3, color='r', linestyle='dashed', linewidth=2)
                                plt.text(ret3+5, 1000, str(ret3), color='r')
                                plt.xlabel('Gray Level')
                                plt.ylabel('Frequency')
                                plt.xlim(0, 256)
                                plt.show()
                                

                                # ret, th1 = cv2.threshold(clahe_img,i,255,cv2.THRESH_TOZERO)
                                # ret, th2 = cv2.threshold(clahe_img,i,255,cv2.THRESH_BINARY)
                                # cv2.imwrite(path+'2/'+patient.name+'/'+date.name+'/'+eye.name+'/TOZERO_'+layer.stem+'_'+str(i)+'.png', th1)
                                # cv2.imwrite(path+'2/'+patient.name+'/'+date.name+'/'+eye.name+'/BINARY_'+layer.stem+'_'+str(i)+'.png', th2)
                                # cv2.imwrite(path+'2/'+patient.name+'/'+date.name+'/'+eye.name+'/morphology_TOZERO_'+layer.stem+'_'+str(i)+'.png', th3)
                                # combined_img = np.hstack((normal_img,blur1,blur3,blur5,blur7))
                                # cv2.imshow('img', combined_img)
                                cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    
