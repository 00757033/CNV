from pickletools import uint4
import timeit
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import glob
import cv2
import numpy as np
import os


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
#from keras import utils as np_utils

#模型建立在UnetModel.py
from UNetModel import *
from ConvMixer import *



#產生訓練資料
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=304):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def load(self, name): #  讀取圖片
        image_path = os.path.join(self.path, "images", name)
        mask_path = os.path.join(self.path, "masks", name)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if image.shape==2: # 轉成3維
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = np.expand_dims(mask, axis=-1)
        # 進行正規化
        image = image / 255.0
        mask = mask / 255.0
        return image, mask
    
    def __getitem__(self, index): # 產生一個batch的資料
        if (index + 1) * self.batch_size > len(self.ids): # 最後一個batch可能不足batch_size
            files = self.ids[index * self.batch_size:] # 最後一個batch
        else: # 其他完整的batch
            files = self.ids[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        masks  = []
        
        for name in files: # 讀取圖片
            image, mask = self.load(name)
            images.append(image)
            masks.append(mask)
            
        images = np.array(images)
        masks  = np.array(masks)
        return images, masks
    
    def on_epoch_end(self): 
        pass

#訓練模型
class train():   
#初始化設定
    def __init__(self, data_class, data_date, model_path, result_path, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold):
        self.data_class   = data_class          #病灶種類
        self.data_date    = data_date           #資料日期
        self.model_path   = model_path          #儲存模型路徑
        self.result_path  = result_path         #儲存結果路徑
        self.image_size   = image_size          #input影像大小
        self.models       = models              #訓練的模型(U-Net、U-Net++...)
        self.batchs       = batchs              #batch size(2、4、8、16)
        self.epochs       = epochs              #epoch size(50、100、200、300、400)
        self.datas        = datas               #使用的訓練資料集，例: train->沒有augment、augmnet5->augmnet5倍...以此類推
        self.lrns         = lrns                #lreaning rate(0.001、0.0001)
        self.filters      = filters             #各層的kernal數量([32, 64, 128, 256, 512])
        self.threshold    = predict_threshold   #預測結果>threshold則判斷為true(1)
        print("Start training")
    
#給外部執行訓練的function
    def run(self, PATH_DATASET, train_signal):  #if(train_signal==true) 才會訓練模型
        for dataset in glob.glob(r"%s/*" %PATH_DATASET): #讀取資料集
            dataset_name = dataset.replace(PATH_DATASET + '\\', "")
            f = open(self.data_class + '_' + self.data_date + '_summary_2.txt', 'a')
            lines = [dataset_name + '\n\n']
            f.writelines(lines)
            f.close()  
            print(dataset_name)
            for data in self.datas: #讀取資料集的訓練資料
                for  model_name in self.models:     #讀取模型 
                    model_path1, result_path1 = self.mkdir(self.result_path, self.model_path, dataset_name, self.data_date, data) #建立模型跟預測結果的資料夾
                    best_model = ''
                    best_score = 0
                    best_score_var = 0 
                    for epoch in self.epochs: #每個模型訓練幾個epoch
                        for batch in self.batchs: #每個模型訓練幾個batch
                            for lrn in self.lrns: #每個模型訓練幾個learning rate
                                train_path   = dataset + '/' + data + '/'
                                valid_path   = dataset + '/' + 'valid' + '/'
                                train_ids    = os.listdir(train_path + 'images/')                        
                                valid_ids = os.listdir(valid_path + 'images/') 

                                train_gen = DataGen(train_ids, 
                                                    train_path, 
                                                    image_size=self.image_size, 
                                                    batch_size=batch)  # 訓練資料
                                valid_gen = DataGen(valid_ids, 
                                                    valid_path, 
                                                    image_size=self.image_size, 
                                                    batch_size=batch) # 驗證資料
                                train_steps = len(train_ids)//batch # 
                                valid_steps = len(valid_ids)//batch
                                feature = '_' + str(epoch) + '_' + str(batch) + '_' + str(lrn)
                                time = 0
                                model = self.getModel(model_name, self.image_size, lrn) # 要訓練的模型
                                model, time = self.fitModel(model, 
                                                        model_name,
                                                        feature,
                                                        epoch, 
                                                        self.filters, 
                                                        train_gen, train_steps, 
                                                        valid_gen, valid_steps, 
                                                        model_path1,
                                                        train_signal)
                                name, ji_score, ji_var, time = self.evaluateModel(model, 
                                                        model_name, 
                                                        feature, 
                                                        model_path1, 
                                                        self.image_size, 
                                                        result_path1, 
                                                        dataset, 
                                                        time, 
                                                        data,
                                                        self.threshold)
                                if ji_score >= best_score:
                                    best_score = ji_score
                                    best_score_var = ji_var
                                    best_name = name
                                    best_time = time
                    # record best model

                    f = open(self.data_class + '_' + self.data_date + '_summary_2.txt', 'a')
                    lines = ['----------Summary:' + best_name + '_' + data +  '----------\n',
                            'Time' + ': ' + str(best_time) + '\n']
                    f.writelines(lines)
                    lines = [' Jaccard index' + ': ' + str(best_score) + ' +- ' + str(best_score_var) + '\n']               
                    f.writelines(lines)
                    lines = ['------------------------------------------\n\n']
                    f.writelines(lines)
                    f.close() 
                
#取得訓練模型(from UnetModel.py)
    def getModel(self, model_name, image_size, learning_rate):
        if model_name == 'UNet':
            myModel = UNet(image_size, learning_rate)
        elif model_name == 'FRUNet':
            myModel = FRUNet(image_size, learning_rate)
        elif model_name == 'UNetPlusPlus':
            myModel = UNetPlusPlus(image_size, learning_rate)
        elif model_name == 'AttentionUNet':
            myModel = AttentionUNet(image_size, learning_rate)
        elif model_name == 'BCDUNet':
            myModel = BCDUNet(image_size, learning_rate)
        elif model_name == 'RecurrentUNet':
            myModel = RecurrentUNet(image_size, learning_rate)        
        elif model_name == 'ResUNet':
            myModel = ResUNet(image_size, learning_rate)    
        elif model_name == 'R2UNet':
            myModel = R2UNet(image_size, learning_rate)
        elif model_name ==  'R2AttentionUNet':
            myModel = R2AttentionUNet(image_size, learning_rate)
        elif model_name == 'DenseUNet':
            myModel = DenseUNet(image_size, learning_rate)          
        #elif model_name == 'ConvMixer':
        #    myModel = ConvMixer(image_size, learning_rate)

        # elif model_name == 'SRF_UNet':
        #     myModel = SRF_UNet(image_size, learning_rate)
        return myModel
#訓練模型，可以調整earlyStop、reduce_lr，回傳模型跟訓練時間
    def fitModel(self, model, model_name, feature, epochs, filters, train_gen, train_steps, valid_gen, valid_steps, model_path, train_signal):
        #Checkpoint:會記錄次訓練的結果，並從中選擇最佳儲存，判斷的評估方式是使用 monitor='val_dice_coef'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath= model_path + model_name + feature + '.h5', monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True)
        #reduce_lr:幾個epoch後(次數:patience)訓練結果(評估方式:monitor)沒有提升降低lreaning rate(下降比例:factor = 0.1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", patience = 20, mode = "auto", factor = 0.1, min_lr = 0.0000001)
        #earlyStop:多少個epoch訓練結果沒有進步則停止訓練
        earlyStop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 40, mode = "auto", verbose = 1)
        
        # 記錄訓練過程

        myModel = model.build_model(filters)
        #model.summary()
        start = timeit.default_timer()
        if train_signal :
            # myModel.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=[checkpoint, reduce_lr], shuffle = True)
            myModel.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=[checkpoint, reduce_lr, earlyStop], shuffle = True)
        stop = timeit.default_timer()
        training_time = round(stop - start, 2)
        return myModel, training_time

#評估訓練結果
    def evaluateModel(self, myModel, model_name, feature, model_path, image_size, result_path, data_path, training_time, data, shreshold):
        print(model_path + model_name + feature + '_1.h5')
        myModel.load_weights(model_path + model_name + feature + '_1.h5')
        #儲存模型
        self.save_results(myModel, image_size, model_name + feature, result_path, data_path, shreshold)
        #紀錄該模型的訓練結果，紀錄於為 病灶_日期_summary.txt
        f = open(self.data_class + '_' + self.data_date + '_summary.txt', 'a')
        lines = ['----------Summary:' + model_name + feature + '_' + data + '_' + data_path[25:] + '----------\n',
                'Time' + ': ' + str(training_time) + '\n']
        f.writelines(lines)
        for result in ['predict']:
            iou_ji, iou_recall, iou_precision, iou_accuracy, iou_dc = self.calculate_jaccard_index(model_name + feature, result_path, result, masks = 'masks')
            ji_score = round(sum(iou_ji) / len(iou_ji), 5)
            ji_var   = round(np.var(iou_ji), 5)
            dc_score = round(sum(iou_dc) / len(iou_dc), 5)
            dc_var   = round(np.var(iou_dc), 5)
            lines = [result + ' Jaccard index' + ': ' + str(ji_score) + ' +- ' + str(ji_var) + '\n',
                    result + ' Dice Coefficient' + ': ' + str(dc_score) + ' +- ' + str(dc_var) + '\n']
                        #' Recall' + ': ' + str(round(sum(iou_recall) / len(iou_recall), 2)) + ' +- ' + str(round(np.var(iou_recall), 2)) + '\n',
                        #' Precision' + ': ' + str(round(sum(iou_precision) / len(iou_precision), 2)) + ' +- ' + str(round(np.var(iou_precision), 2)) + '\n',
                        #' Accuracy' + ': ' + str(round(sum(iou_accuracy) / len(iou_accuracy), 2)) + ' +- ' + str(round(np.var(iou_accuracy), 2)) + '\n']                    
            f.writelines(lines)
        lines = ['------------------------------------------\n\n']
        f.writelines(lines)
        f.close()  
        return  model_name + feature, ji_score, ji_var, training_time

#計算jaccard index
    def calculate_jaccard_index(self, folder_name, result_path, results, masks):
        index = os.listdir(result_path + folder_name + '/images/')
        iou_ji = []
        iou_recall = []
        iou_precision = []
        iou_accuracy = []
        iou_dc = []
        precision_score, recall_score
        for i in index:
            img_true = cv2.imread(result_path + folder_name + '/' + masks + '/' + i, 0)    
            img_true[img_true < 128] = 0
            img_true[img_true >= 128] = 1
            img_pred = cv2.imread(result_path + folder_name + '/' + results + '/' + i, 0)
            img_pred[img_pred < 128] = 0
            img_pred[img_pred >= 128] = 1
            img_true = np.array(img_true).ravel()
            img_pred = np.array(img_pred).ravel()
            iou_dc.append(np.sum(img_pred[img_true==1])*2.0 / (np.sum(img_pred) + np.sum(img_true)))
            iou_ji.append(jaccard_score(img_true, img_pred))
            iou_recall.append(recall_score(img_true, img_pred))
            iou_precision.append(precision_score(img_true, img_pred))
            iou_accuracy.append(accuracy_score(img_true, img_pred))
            iou_accuracy.append(accuracy_score(img_true, img_pred))
        return iou_ji, iou_recall, iou_precision, iou_accuracy, iou_dc
#訓練後的後處理(結果不好，會把血管間的洞補起來)，先closing再opening
    def postproccessing(self, image, kernal_size):
        kernel = np.ones(kernal_size, np.uint8)
        # closing
        dilation1 = cv2.dilate(image, kernel, iterations = 1)
        erosion1 = cv2.erode(dilation1, kernel, iterations = 1)       
        # opening
        erosion2 = cv2.erode(erosion1, kernel, iterations = 1)
        dilation2 = cv2.dilate(erosion2, kernel, iterations = 1)

        return dilation2
#儲存預測結果(predict跟results的結果是一樣的，post是有經過後處理)
    def save_results(self, model, image_size, folder_name, result_path, test_path, shreshold):

        setFloder(result_path + folder_name)
        setFloder(result_path + folder_name + '/images')
        setFloder(result_path + folder_name + '/masks')
        setFloder(result_path + folder_name + '/predict')
        setFloder(result_path + folder_name + '/results')
        setFloder(result_path + folder_name + '/post')
        #儲存所有testing set預測的結果
        test_ids = os.listdir(test_path + '/test/images/')
        test_gen = DataGen(test_ids, test_path + '/test/', image_size=image_size, batch_size=len(test_ids))
        x, y = test_gen.__getitem__(0)
        results = model.predict(x)
        predict = results
        results = results > shreshold
        for j in range(len(results)):
            image = np.reshape(x[j] * 255, (image_size, image_size, 3))
            image = image.astype(np.uint8)
            cv2.imwrite(result_path + folder_name + '/images/' + test_ids[j], image)
            mask = np.reshape(y[j] * 255, (image_size, image_size))
            mask = np.stack((mask,) * 3, -1)
            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(result_path + folder_name + '/masks/' + test_ids[j], mask)
            output_origin = np.reshape(predict[j] * 255, (image_size, image_size))
            output_origin = np.stack((output_origin,) * 3, -1)
            output_origin = output_origin.astype(np.uint8)
            output_origin = cv2.cvtColor(output_origin, cv2.COLOR_BGR2GRAY)

            output = np.reshape(results[j] * 255, (image_size, image_size))
            output = np.stack((output,) * 3, -1)
            output = output.astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output_post = self.postproccessing(output, (3,3))
            cv2.imwrite(result_path + folder_name + '/predict/' + test_ids[j], output_origin)
            cv2.imwrite(result_path + folder_name + '/results/' + test_ids[j], output)
            cv2.imwrite(result_path + folder_name + '/post/' + test_ids[j], output_post)

#建立訓練好的模型跟預測結果的資料夾
    def mkdir(self, result_path, model_path, dataset, time, data):
        floder_path = self.data_class + '_' + time
        setFloder(model_path  + floder_path)
        setFloder(model_path  + floder_path + '/' + dataset)
        setFloder(model_path  + floder_path + '/' + dataset + '/' + data)
        setFloder(result_path + floder_path)
        setFloder(result_path + floder_path + '/' + dataset)
        setFloder(result_path + floder_path + '/' + dataset + '/' + data)
        result_path = result_path + floder_path + '/' + dataset + '/' + data + '/'
        model_path  = model_path  + floder_path + '/' + dataset + '/' + data + '/'
        return model_path, result_path


def setFloder(path):
    if not os.path.isdir(path) : os.mkdir(path)

'''
def main():
    #tf.config.run_functions_eagerly(True)
    data_class   = 'PED'
    data_date    = '0106'    
    image_size   = 304
    models       = ['BCDUNet'] #['FRUNet', 'UNet', 'UNetPlusPlus', 'UNet3Plus', 'ResUNetPlusPlus', 'AttentionUNet']
    batchs       = [2]
    epochs       = [100, 200]
    datas        = ['train'] #['train', 'augment5', 'augment10']
    lrns         = [0.0001]
    filters      = [32, 64, 128, 256, 512]

    PATH_DATASET = '../../Data/' + data_class + '_' + data_date + '/' +'trainset'
    PATH_MODEL   = '../../Model/'    
    PATH_RESULT  = '../../Result/'
    setFloder(PATH_RESULT)
    setFloder(PATH_MODEL)
    Train = train(data_class, data_date, PATH_MODEL, PATH_RESULT, image_size, models, batchs, epochs, datas, lrns, filters)
    Train.run(PATH_DATASET)

if __name__ == '__main__':
    main()
'''
