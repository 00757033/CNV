
import timeit
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import glob
import cv2
import csv
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras.optimizers import *

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# board
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime
#模型建立在UnetModel.py
from UNetModel2 import *
from postprocessing3 import *
import tensorflow as tf
model_classes = {
    'UNet': UNet,
    'FRUNet': FRUNet,
    'UNetPlusPlus': UNetPlusPlus,
    'AttUNet': AttentionUNet,
    'BCDUNet': BCDUNet,
    'RecurrentUNet': RecurrentUNet,
    'ResUNet': ResUNet,
    'R2UNet': R2UNet,
    'R2AttentionUNet': R2AttentionUNet,
    'DenseUNet': DenseUNet,
    'MultiResUNet': MultiResUNet,
    'DCUNet': DCUNet,
    'CARUNet': CARUNet,
}
# ['SegNet','PSPNet','FCN8','FCN32','DeepLabV3Plus','DeepLabV3','DeepLabV2','DeepLabV1']

# 評估指標
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard_index(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

#產生訓練資料 以及測試資料
class DataGenerator(tf.keras.utils.Sequence):
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

    
# 訓練模型
class train(): 
    def __init__(self, data_class, data_date, model_path, result_path, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold):
        self.data_class = data_class # 資料集類別
        self.data_date = data_date # 資料集日期
        self.model_path = model_path # 模型路徑
        self.result_path = result_path # 結果路徑
        self.image_size = image_size # 影像大小
        self.models = models # 模型
        self.batchs = batchs # 批次大小
        self.epochs = epochs # 訓練次數
        self.datas = datas # 資料集
        self.lrns = lrns # 學習率
        self.filters = filters # 濾波器大小
        self.predict_threshold = predict_threshold # 預測閥值

    def record(self, history, time, model_path, model_name, feature):
        # save_model_name = model_name + '_' + dataset_name + '_' + epoch.__str__() + '_' + batch.__str__() + '_' + lrn.__str__()
        [model_name , epoch, batch, lrn] = model_name.split('_')
        model_path = model_path.split('\\')[-2]
        # 紀錄訓練結果 現在日期 時間 模型 資料集  批次大小 訓練次數 學習率  預測閥值 訓練時間 訓練loss 訓練acc 驗證loss 驗證acc 
        print("record :",model_path)
        record_data = [ 
            datetime.datetime.now().strftime("%Y-%m-%d").__str__(), # 現在日期
            model_name+ '_' + feature, # 模型
            model_path, # 資料集
            batch, # 批次大小
            epoch, # 訓練次數
            lrn, # 學習率
            self.predict_threshold, # 預測閥值
            time, # 訓練時間
            history.history['loss'][-1], # 訓練loss
            history.history['dice_coef'][-1], # 訓練dice
            history.history['val_loss'][-1], # 驗證loss
            history.history['val_dice_coef'][-1], # 驗證dice 
            
        ]
        # 紀錄訓練結果
        with open(os.path.join("record", "training.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(record_data)

    # save_results( image_size, folder_name, result_path, test_path, shreshold):
    # def save_results(self, model, test_dataset, result_path):
    def postproccessing(self, image, kernal_size):
        kernel = np.ones(kernal_size, np.uint8)
        # closing
        dilation1 = cv2.dilate(image, kernel, iterations = 1)
        erosion1 = cv2.erode(dilation1, kernel, iterations = 1)       
        # opening
        erosion2 = cv2.erode(erosion1, kernel, iterations = 1)
        dilation2 = cv2.dilate(erosion2, kernel, iterations = 1)
        return dilation2

    def evaluateModel(self,dataset_name, model, test_dataset, result_path,model_path, model_name, feature,predict_threshold = 0.5,postprocess_signal = False):
        print("evaluateModel" , model_name)
        # make folder
        img_path = os.path.join(result_path,model_name+'_' + feature, "images")
        mask_path = os.path.join(result_path,model_name+'_' + feature, "masks")
        predict_path = os.path.join(result_path,model_name+'_' + feature, "predict")
        list = [img_path, mask_path, predict_path]
        if postprocess_signal:
            postprocess = os.path.join(result_path,model_name+'_' + feature, "postcrf")
            post = os.path.join(result_path,model_name+'_' + feature, "post")
            list.append(postprocess)
            list.append(post)
        for path in list:
            if not os.path.isdir(path):
                os.makedirs(path)
        # load test data
        test_path = os.path.join(test_dataset, "test")
        test_ids = os.listdir(os.path.join(test_path, "images"))
        test_dataset = DataGenerator(test_ids, test_path, batch_size=1, image_size=self.image_size)
        # predict
        print(len(test_dataset))

        # load model
        model.load_weights(os.path.join(model_path, model_name+'_' + feature + '.h5'))
        print("load model")
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "predict_jaccard_index", "predict_dice_coefficient","postprocess_jaccard_index", "postprocess_dice_coefficient","postprocess_crf_jaccard_index", "postprocess_crf_dice_coefficient"])
        # evaluate
        iou = []
        dice = []
        post_iou = []
        post_dice = []
        post_crf_iou = []
        post_crf_dice = []
        
        best_id = 0
        worst_id = 0
        best_iou = 0
        worst_iou = 1


        result = model.predict(test_dataset)
        for i, item in enumerate(result):
            img = item[:, :, 0].copy()
            img[img < predict_threshold] = 0
            img[img >= predict_threshold] = 255
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (self.image_size, self.image_size))
            cv2.imwrite(os.path.join(predict_path, test_ids[i]), img)

            img = cv2.imread(os.path.join(predict_path, test_ids[i]), 0)
            img[img < 128] = 0
            img[img >= 128] = 1
            mask = cv2.imread(os.path.join(test_path, "masks", test_ids[i]), 0)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = np.array(mask, dtype=np.uint8)
            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            jc = jaccard_score(mask.ravel(), img.ravel())
            di = dice_coef(mask, img) 
            di = di.numpy()
            # Tensor("Mean_1:0", shape=(), dtype=float32)
            iou.append(jc)
            dice.append(di)

            if jc > best_iou :
                best_iou = jc
                best_id = i
            if jc < worst_iou :
                worst_iou = jc
                worst_id = i

            cv2.imwrite(os.path.join(img_path, test_ids[i]), cv2.imread(os.path.join(test_path, "images", test_ids[i])))
            cv2.imwrite(os.path.join(mask_path, test_ids[i]), cv2.imread(os.path.join(test_path, "masks", test_ids[i])))
            jc_post = 0
            di_post = 0
            jc_postcrf = 0
            di_postcrf = 0

            if postprocess_signal:
                image = cv2.imread(os.path.join(predict_path, test_ids[i]), 0)
                output_post = self.postproccessing(image, (3,3))
                cv2.imwrite(os.path.join(post, test_ids[i]), output_post)
                output_post[output_post < 128] = 0
                output_post[output_post >= 128] = 1
                

                jc_post = jaccard_score(mask.ravel(), output_post.ravel())
                di_post = dice_coef(mask, output_post)
                di_post = di_post.numpy()
                post_iou.append(jc_post)
                post_dice.append(di_post)
                crf_input = item[:, :, 0].copy()
                crf_input = np.array(crf_input, dtype=np.uint8)
                crf_input = cv2.resize(crf_input, (self.image_size, self.image_size))

                postcrf = crf( cv2.imread(os.path.join(img_path, test_ids[i])) ,cv2.imread(os.path.join(predict_path, test_ids[i]),0) ,os.path.join(postprocess, test_ids[i])) 
                postcrf = cv2.imread(os.path.join(postprocess, test_ids[i]), 0)
                postcrf = np.array(postcrf, dtype=np.uint8)
                postcrf[postcrf < 128] = 0
                postcrf[postcrf >= 128] = 1
                jc_postcrf = jaccard_score(mask.ravel(), postcrf.ravel())
                di_postcrf = dice_coef(mask, postcrf)
                di_postcrf = di_postcrf.numpy()
                post_crf_iou.append(jc_postcrf)
                post_crf_dice.append(di_postcrf)
            # record image name, jaccard index, dice coefficient

            record_data = [
                test_ids[i],
                jc,
                di,
                jc_post,
                di_post,
                jc_postcrf,
                di_postcrf
            ]

            with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(record_data)

        # record average jaccard index, dice coefficient  std of jaccard index, dice coefficient
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["mean", np.mean(iou), np.mean(dice), np.mean(post_iou), np.mean(post_dice), np.mean(post_crf_iou), np.mean(post_crf_dice)])
            writer.writerow(["std", np.std(iou, ddof=1), np.std(dice, ddof=1), np.std(post_iou, ddof=1), np.std(post_dice, ddof=1), np.std(post_crf_iou, ddof=1), np.std(post_crf_dice, ddof=1)])
        # record best image name ,jaccard index, dice coefficient worst image name ,jaccard index, dice coefficient
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["best", test_ids[best_id], best_iou, dice[best_id]])
            writer.writerow(["worst", test_ids[worst_id], worst_iou, dice[worst_id]])
        # record postprocess best image name ,jaccard index, dice coefficient worst image name ,jaccard index, dice coefficient
        if postprocess_signal:
            print(os.path.join("record","CRF",dataset_name+'.csv'))
            print(os.path.join("record","Morphology",dataset_name+'.csv'))
            with open(os.path.join("record","CRF",dataset_name+'.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                # 'model_name','original_ji_score','original_ji_var','original_dice_score','original_dice_var','ji_score','ji_var','dice_score','dice_var'
                writer.writerow([model_name,np.mean(iou), np.std(iou, ddof=1),np.mean(dice),np.std(dice, ddof=1),np.mean(post_crf_iou), np.mean(post_crf_dice),np.std(post_crf_dice, ddof=1),np.std(post_crf_iou, ddof=1)])
                
                
            with open(os.path.join("record","Morphology",dataset_name+'.csv'),  'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name,np.mean(iou), np.std(iou, ddof=1),np.mean(dice),np.std(dice, ddof=1),np.mean(post_iou), np.mean(post_dice),np.std(post_dice, ddof=1),np.std(post_iou, ddof=1)])


        return np.mean(iou), np.std(iou, ddof=1), np.mean(dice), np.std(dice, ddof=1)

    def fitModel(self,model,dataset,data,  train_dataset, valid_dataset, epoch, lrn, model_path, model_name, feature):
        print("fitModel",dataset,data)
        # checkpoint
        checkpoint = ModelCheckpoint(os.path.join(model_path, model_name+'_' + feature + '.h5'), monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True)
        # earlystopping
        # earlystopping = EarlyStopping(monitor='val_dice_coef', patience=20, verbose=1, mode='max')
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=10, verbose=1, mode='max')
        # board
        log = os.path.join('logs',dataset,data, model_name+'_'+feature)
        tensorboard = TensorBoard(log_dir=log, histogram_freq=0, write_graph=True, write_images=True)
        tensorboard.set_model(model)
        # 訓練
        start = timeit.default_timer()        # size = len(train_dataset) 
        history = model.fit(train_dataset, epochs=epoch, validation_data=valid_dataset, callbacks=[checkpoint, reduce_lr, tensorboard], verbose=1)
        end = timeit.default_timer()
        time = end - start

        return history, time

    def record_model(self):
        if not os.path.isdir("record"):
            os.makedirs("record")
        if not os.path.isfile(os.path.join("record", "training.csv")):
            with open(os.path.join("record", "training.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["date", "model", "dataset", "batch", "epoch", "learning_rate", "predict_threshold", "time", "loss", "dice_coef", "val_loss", "val_dice_coef"])

				
    def best_model_record(self, best_model, dataset_name, best_model_time, best_iou, best_iou_var, best_dice, best_dice_var):
        print(best_model)
        [model_name , epoch, batch, lrn] = best_model.split('_')
        if not os.path.isfile(os.path.join("record", "best_model.csv")):
            with open(os.path.join("record", "best_model.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["date", "model", "dataset", "batch", "epoch", "learning_rate", "predict_threshold", "time", "best_iou", "best_iou_var", "best_dice", "best_dice_var"])

        with open(os.path.join("record", "best_model.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ 
                datetime.datetime.now().strftime("%Y-%m-%d").__str__(), # 現在日期
                model_name, # 模型名稱
                dataset_name, # 資料集名稱
                batch, # batch size
                epoch, # epoch
                lrn, # learning rate
                self.predict_threshold, # predict threshold
                best_model_time, # 訓練時間
                best_iou, # best iou
                best_iou_var, # best iou var
                best_dice, # best dice
                best_dice_var # best dice var
            ])
        f = open(self.data_class + '_' + self.data_date + '_summary_2.txt', 'a')
        lines = ['----------Summary:' + best_model + '----------\n',
                'Time: '+ str(best_model_time) + '\n',
                'Jaccard index: '+ str(best_iou) + ' +- ' + str(best_iou_var) + '\n',
                'Dice Coefficient: '+ str(best_dice) + ' +- ' + str(best_dice_var) + '\n',
                '----------------------------------------\n\n']
        f.writelines(lines)
        f.close()

    # 訓練模型
    def run(self, PATH_DATASET, train_signal = True, postprocess_signal = False):
        if train_signal:
            self.record_model()

        # 讀取資料集
        for dataset in glob.glob(r"%s/*" %PATH_DATASET): #讀取資料集
            dataset_name = dataset.replace(PATH_DATASET + '\\', "")
            print("dataset_name:", dataset_name)
            if train_signal:
                f = open(self.data_class + '_' + self.data_date + '_summary_2.txt', 'a')
                f.writelines(dataset_name+ '\n')
                f.close()
                                            # 預測
            if postprocess_signal:
                if not os.path.isdir(os.path.join("record","CRF")):
                    os.makedirs(os.path.join("record","CRF"))
                if not os.path.isdir(os.path.join("record","Morphology")):
                    os.makedirs(os.path.join("record","Morphology"))
                if not os.path.isfile(os.path.join("record","CRF",dataset_name+'.csv')):
                # 預測資料
                    with open(os.path.join("record","CRF",dataset_name+'.csv'), 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['model_name','original_ji_score','original_ji_var','original_dice_score','original_dice_var','ji_score','ji_var','dice_score','dice_var'])
                if not os.path.isfile(os.path.join("record","Morphology",dataset_name+'.csv')):
                    with open(os.path.join("record","Morphology",dataset_name+'.csv'), 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(['model_name','original_ji_score','original_ji_var','original_dice_score','original_dice_var','ji_score','ji_var','dice_score','dice_var'])
            for data in self.datas: #讀取資料集的訓練資料
                for  model_name in self.models:  # 讀取模型
                    best_model = ''
                    best_dice = -1
                    best_dice_var = -1
                    best_iou = -1
                    best_iou_var = -1
                    best_model_time = 0
                    for epoch in self.epochs: #每個模型訓練幾個epoch
                        for batch in self.batchs: #每個模型訓練幾個batch
                            for lrn in self.lrns: #每個模型訓練幾個learning rate
                                # 建立模型跟預測結果的資料夾
                                floder_path = self.data_class + '_' + self.data_date
                                model_path = os.path.join(self.model_path,floder_path,dataset_name,data)
                                result_path = os.path.join(self.result_path,floder_path,dataset_name,data)
                                if not os.path.isdir(model_path):
                                    os.makedirs(model_path)
                                if not os.path.isdir(result_path):
                                    os.makedirs(result_path)
                                # 訓練資料集
                                train_path = os.path.join(dataset, "train")
                                valid_path = os.path.join(dataset, "valid")
                                test_path = os.path.join(dataset, "test")
                                train_ids = os.listdir(os.path.join(train_path, "images"))
                                valid_ids = os.listdir(os.path.join(valid_path, "images"))
                                test_ids = os.listdir(os.path.join(test_path, "images"))
                                # 訓練資料
                                train_dataset = DataGenerator(train_ids, train_path, batch_size=batch, image_size=self.image_size)
                                valid_dataset = DataGenerator(valid_ids, valid_path, batch_size=batch, image_size=self.image_size)


                                #設定模型
                                if model_name in model_classes:
                                    getModels = model_classes[model_name](self.image_size ,lrn)
                                else:
                                    raise ValueError(f"Model '{model_name}' is not supported.")
                                model = getModels.build_model(self.filters)
                                # print(model.summary())
                                # # 編譯模型optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss
                                model.compile(optimizer=Adam(learning_rate=lrn), loss=dice_coef_loss, metrics=[dice_coef, jaccard_index])
                                # 訓練模型 
                                save_model_name = model_name + '_' + epoch.__str__() + '_' + batch.__str__() + '_' + lrn.__str__()
                                # 取得字串 用來判斷是否有重複訓練過 去掉最後的_1.h5

                                lists = os.listdir(model_path) #列出資料夾下所有的目錄與檔案
                                count = 0
                                for i in range(0,len(lists)):
                                    # 找名稱完全相同
                                    if lists[i].startswith(save_model_name) and lists[i].endswith('.h5'):
                                        count = count +1
                                
                                if train_signal:
                                    count = count +1 # 紀錄模型訓練次數
                                    history, time = self.fitModel(model,dataset_name,data ,train_dataset, valid_dataset,epoch, lrn ,model_path,save_model_name,count.__str__())

                                    # 紀錄訓練結果
                                    self.record(history, time,model_path,save_model_name,count.__str__())

                                print("evaluate")
                                ji_score, ji_var , dice_score, dice_var = self.evaluateModel(dataset_name,model, dataset, result_path,model_path,save_model_name,count.__str__(),self.predict_threshold,postprocess_signal)
                                print("finish evaluate")
                                if train_signal:
                                    f = open(self.data_class + '_' + self.data_date + '_summary.txt', 'a') # + save_model_name + '----------\n',
                                    lines = ['----------Summary:' + dataset_name + '----------\n',
                                            'Model: '+ save_model_name + '\n',
                                            'Time: '+ str(time) + '\n',
                                            'Jaccard index: '+ str(ji_score) + ' +- ' + str(ji_var) + '\n',
                                            'Dice Coefficient: '+ str(dice_score) + ' +- ' + str(dice_var) + '\n',
                                            '----------------------------------------\n\n']
                                    f.writelines(lines)
                                    f.close()
                                
                                    if ji_score > best_iou :
                                        best_iou = ji_score
                                        best_iou_var = ji_var
                                        best_dice = dice_score
                                        best_dice_var = dice_var
                                        best_model = save_model_name
                                        best_model_time = time
                    
                    if train_signal:  
                        print("best_model:", best_model)        
                        self.best_model_record(best_model, dataset_name, best_model_time, best_iou, best_iou_var, best_dice, best_dice_var)






