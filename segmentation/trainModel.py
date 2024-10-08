
import timeit
import sys
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score , confusion_matrix
import glob
import cv2
import csv
import numpy as np
import os
import pydensecrf.densecrf as dcrf
from keras.utils import to_categorical
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from matplotlib import pyplot as plt
from img2video import img2video
# from tf_explain.core.grad_cam import GradCAM
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# board
from keras.callbacks import TensorBoard
import time
import datetime
#模型建立在UnetModel.py
from UNetModel import *
from postprocessing import *
import tensorflow as tf
import cv2
from Deeplab import *
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
# from transunet import *
from collections import OrderedDict
if sys.version_info[:2] < (3, 7):
    from collections import OrderedDict

model_classes = { # 模型
    'UNet': UNet,
    'FRUNet': FRUNet,
    'UNetPlusPlus': UNetPlusPlus,
    'AttUNet': AttentionUNet,
    'BCDUNet': BCDUNet,
    'RecurrentUNet': RecurrentUNet,
    'ResUNet': ResUNet,
    'R2UNet': R2UNet,
    # 'R2AttentionUNet': R2AttentionUNet,
    'DenseUNet': DenseUNet,
    'MultiResUNet': MultiResUNet,
    'DCUNet': DCUNet,
    'SDUNet': SDUNet,
    'CARUNet' : CARUNet,
    'DeepLabV3Plus101': DeepLabV3Plus101,
    'DeepLabV3Plus50': DeepLabV3Plus50,
    'ResUNetPLUSPLUS' : ResUNetPLUSPLUS,
    # 'TransUNet' : TransUNet,
}

# Clear GPU memory
def clear_gpu_memory(): # 清除GPU記憶體
    physical_devices = tf.config.experimental.list_physical_devices('GPU') # 列出GPU裝置
    if len(physical_devices) > 0: # 如果有GPU裝置
        for device in physical_devices: # 對每個GPU裝置
            tf.config.experimental.set_memory_growth(device, True) # 設定記憶體成長
    else:
        print("No GPU devices found") # 沒有找到GPU裝置


# 評估指標
def dice_coef(y_true, y_pred): # Dice係數
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dic = (2. * intersection + smooth) /(union + smooth)

    return dic

def dice_coef_loss(y_true, y_pred): # Dice損失
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard_index(y_true, y_pred): # Jaccard指數
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jac = (intersection + smooth) / (union + smooth)
    # tf.Tensor to  int
    
    return jac

def jaccard_loss(y_true, y_pred): # Jaccard損失
    return 1.0 - jaccard_index(y_true, y_pred)

def Tversky_similarity_index(y_true, y_pred, alpha=0.3 ): # Tversky相似性指數
    smooth = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred))
    false_pos = tf.reduce_sum((1 - y_true) * y_pred)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def Tversky_loss(y_true, y_pred, alpha=0.5): # Tversky損失
    return 1.0 - Tversky_similarity_index(y_true, y_pred, alpha)


def binary_focal_loss(y_true, y_pred,alpha=0.15, gamma=2): # 二元焦點損失
    # y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    y_true=tf.cast(y_true, tf.float32)
    L=- y_true * alpha * ((1 - y_pred) ** gamma) * tf.math.log(y_pred + 1e-8) - (1 - y_true) * (1 - alpha) * (y_pred ** gamma) * tf.math.log(1 - y_pred + 1e-8)
    return tf.reduce_mean(L)

        
    
     

def focal_tversky(y_true, y_pred, alpha=0.7, gamma=2.0): # 焦點Tversky
    tversky_loss = 1 - Tversky_similarity_index(y_true, y_pred, alpha)
    # return Tversky_loss(y_true, y_pred, alpha) + binary_focal_loss(y_true, y_pred, beta, gamma) 
    return tf.pow(tversky_loss, gamma)

def mean_pixel_accuracy(y_true, y_pred):
    # Count the number of matching pixels
    count_same = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    size = y_true.shape[0] * y_true.shape[1]
    return count_same / size
    
    
    
    

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
# 產生模型


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
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)# , cv2.IMREAD_UNCHANGED
        image = cv2.resize(image, (self.image_size, self.image_size))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
        # print('image.shape',image.shape)
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
    def channels(self):
        return self.channel
    
# 訓練模型
class train(): 
    def __init__(self, data_class, data_date, model_path, result_path, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold,channel):
        self.data_class = data_class # 資料集類別
        self.data_date = data_date # 資料集日期
        self.model_path = model_path # 模型路徑
        self.result_path = result_path # 結果路徑
        self.image_size = image_size # 影像大小
        self.channel = channel # 影像通道
        self.models = models # 模型
        self.batchs = batchs # 批次大小
        self.epochs = epochs # 訓練次數
        self.datas = datas # 資料集
        self.lrns = lrns # 學習率
        self.filters = filters # 濾波器大小
        self.predict_threshold = predict_threshold # 預測閥值
        clear_gpu_memory()
        

    def record(self, history, time, model_path, model_name, feature):
        # save_model_name = model_name + '_' + dataset_name + '_' + epoch.__str__() + '_' + batch.__str__() + '_' + lrn.__str__()
        [model_name , epoch, batch, lrn, filter] = model_name.split('_')
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
            filter, # 濾波器大小
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

    def crf_predict(self,original_image, predict_image, use_2d = True,min_area = 10):
        # 用在血管分割上
        # 將分割結果轉成rgb
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            
        annotated_image = predict_image.copy()
        if(len(annotated_image.shape)>2):
            annotated_image = annotated_image[:,:,0]
        if(len(annotated_image.shape)<3):
            annotated_image = gray2rgb(annotated_image)

        # 轉成uint32
        annotated_image = annotated_image.astype(np.uint32)
        annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)

        # Convert the 32bit integer color to 0,1, 2, ... labels.
        colors, labels = np.unique(annotated_label, return_inverse=True)

        #Creating a mapping back to 32 bit colors
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:,0] = (colors & 0x0000FF)
        colorize[:,1] = (colors & 0x00FF00) >> 8
        colorize[:,2] = (colors & 0xFF0000) >> 16
        
        #Gives no of class labels in the annotated image
        n_labels = len(set(labels.flat)) 
        # print(annotated_image > 0)
        # print("--------------------")

        if use_2d :
            if n_labels > 1:
                # Example using the DenseCRF2D code
                d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

                # 得到一元勢函數 unary potential
                U = unary_from_labels(labels, n_labels, gt_prob=0.8, zero_unsure=False)
                # U = unary_from_softmax(annotated_image, scale=1, clip=0.0001, zero_unsure=False)
                d.setUnaryEnergy(U)
                # 建立二元勢函數 pairwise potential 二元势就引入了邻域像素对当前像素的影响，所以需要同时考虑像素的位置和其观测值
                # sdims = (3, 3)  # 位置特征的scaling参数，决定位置对二元势的影响
                # schan = (0.01,)  # 颜色通道的平滑



                pairwise_energy = create_pairwise_bilateral(sdims=(3,3), schan=(0.1,), img=original_image, chdim=2)
                d.addPairwiseEnergy(pairwise_energy, compat=10)


                Q = d.inference(10)
                final = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

                final = colorize[final,:]
            elif predict_image is not None:
                final = predict_image
                print("only one class")
            else:
                print(predict_image)
                final = predict_image
                print("no class")

        return final
    
    # 評估模型
    def evaluateModel(self,dataset_name, model, test_dataset, result_path,model_path, model_name, feature,predict_threshold = 0.5,postprocess_signal = False,gradcam_signal = False):
        print("evaluateModel" , model_name)
        test_dataset2 = test_dataset
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
        if gradcam_signal:
            self.gradcam(model, test_dataset2, result_path,model_name+'_' + feature)
        
        print("load model")
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name",
                             "predict_jaccard_index", 
                             "predict_dice_coefficient",
                             "predict_Sensitivity",
                             "postprocess_crf_jaccard_index",
                             "postprocess_crf_dice_coefficient",
                             "postprocess_crf_Sensitivity"
                             ])
        # evaluate
        iou = []
        dice = []
        Sensitivitys = []
        post_iou = []
        post_dice = []
        post_Sensitivity = []
        post_crf_iou = []
        post_crf_dice = []
        post_crf_Sensitivity = []
        
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
            Sensitivity = recall_score(mask.flatten(), img.flatten(), average='micro')
            Sensitivitys.append(Sensitivity)

            if jc > best_iou :
                best_iou = jc
                best_id = i
            if jc < worst_iou :
                worst_iou = jc
                worst_id = i

            cv2.imwrite(os.path.join(img_path, test_ids[i]), cv2.imread(os.path.join(test_path, "images", test_ids[i])))
            cv2.imwrite(os.path.join(mask_path, test_ids[i]), cv2.imread(os.path.join(test_path, "masks", test_ids[i])))
            jc_postcrf = 0
            di_postcrf = 0
            Sensitivity_postcrf = 0

            if postprocess_signal:
                img = item[:, :, 0].copy()
                # # image = cv2.imread(os.path.join(predict_path, test_ids[i]), 0)
                # output_post = self.postproccessing(img, (3,3))
                # cv2.imwrite(os.path.join(post, test_ids[i]), output_post)
                # output_post[output_post < 128] = 0
                # output_post[output_post >= 128] = 1
                

                # jc_post = jaccard_score(mask.ravel(), output_post.ravel())
                # di_post = dice_coef(mask, output_post)
                # di_post = di_post.numpy()
                
                crf_input = item[:, :, 0].copy()
                crf_input = np.array(crf_input, dtype=np.uint8)
                crf_input = cv2.resize(crf_input, (self.image_size, self.image_size))

                original = cv2.imread(os.path.join(test_path, "images", test_ids[i]))
                predict= cv2.imread(os.path.join(predict_path, test_ids[i]))
                # cv2.imshow("original", original)
                # cv2.waitKey(0)

                postcrf = self.crf_predict(original,predict) 
                postcrf = cv2.cvtColor(postcrf, cv2.COLOR_BGR2GRAY)
                postcrf = np.array(postcrf, dtype=np.uint8)
               
                # remove small area
                postcrf = remove_small_area(postcrf,50)
                

                cv2.imwrite(os.path.join(postprocess, test_ids[i]), postcrf)
                
                
                postcrf[postcrf < 128] = 0
                postcrf[postcrf >= 128] = 1
                
                mask = cv2.imread(os.path.join(test_path, "masks", test_ids[i]), 0)
                mask = cv2.resize(mask, (self.image_size, self.image_size))
                mask = np.array(mask, dtype=np.uint8)
                mask[mask < 128] = 0
                mask[mask >= 128] = 1
                
            
                jc_postcrf = jaccard_score(mask.ravel(), postcrf.ravel())
                di_postcrf = dice_coef(mask, postcrf)
                di_postcrf = di_postcrf.numpy()
                Sensitivity_postcrf = mean_pixel_accuracy(mask, postcrf)
                post_crf_iou.append(jc_postcrf)
                post_crf_dice.append(di_postcrf)
                post_crf_Sensitivity.append(Sensitivity_postcrf)
            # record image name, jaccard index, dice coefficient

            record_data = [
                test_ids[i],
                jc,
                di,
                Sensitivity,
                jc_postcrf,
                di_postcrf,
                Sensitivity_postcrf
            ]
            

            with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(record_data)

        # record average jaccard index, dice coefficient  std of jaccard index, dice coefficient
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["mean", np.mean(iou), np.mean(dice), np.mean(Sensitivitys), np.mean(post_crf_iou), np.mean(post_crf_dice), np.mean(post_crf_Sensitivity)])
            writer.writerow(["std", np.std(iou, ddof=1), np.std(dice, ddof=1), np.std(Sensitivitys, ddof=1), np.std(post_crf_iou, ddof=1), np.std(post_crf_dice, ddof=1), np.std(post_crf_Sensitivity, ddof=1)])
        # record best image name ,jaccard index, dice coefficient worst image name ,jaccard index, dice coefficient
        with open(os.path.join(result_path,model_name+'_' + feature, "result.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["best", test_ids[best_id], best_iou, dice[best_id],Sensitivitys[best_id]])
            writer.writerow(["worst", test_ids[worst_id], worst_iou, dice[worst_id], Sensitivitys[worst_id]])
        # record postprocess best image name ,jaccard index, dice coefficient worst image name ,jaccard index, dice coefficient
        if postprocess_signal:
            print(os.path.join("record","CRF",dataset_name+'.csv'))
            postprocess_crf = [
                model_name,
                np.mean(iou),
                np.std(iou, ddof=1),
                np.mean(dice),
                np.std(dice, ddof=1),
                np.mean(Sensitivitys),
                np.std(Sensitivitys, ddof=1),
                np.mean(post_crf_iou),
                np.std(post_crf_iou, ddof=1),
                np.mean(post_crf_dice),
                np.std(post_crf_dice, ddof=1),
                np.mean(post_crf_Sensitivity),
                np.std(post_crf_Sensitivity, ddof=1)
                
            ]
            
            with open(os.path.join("record","CRF",dataset_name+'.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                # 'model_name','original_ji_score','original_ji_var','original_dice_score','original_dice_var','ji_score','ji_var','dice_score','dice_var'
                writer.writerow(postprocess_crf)
                
                


        return np.mean(iou), np.std(iou, ddof=1), np.mean(dice), np.std(dice, ddof=1), np.mean(Sensitivitys), np.std(Sensitivitys, ddof=1)

    def fitModel(self,model,dataset,data,  train_dataset, valid_dataset, epoch, lrn, model_path, model_name, feature):
        print("fitModel",dataset,data)
        # checkpoint
        checkpoint = ModelCheckpoint(os.path.join(model_path, model_name+'_' + feature + '.h5'), monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True)
        # earlystopping
        # earlystopping = EarlyStopping(monitor='val_dice_coef', patience=30, verbose=1, mode='max')
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=15, verbose=1, mode='max')
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
                writer.writerow(["date", "model", "dataset", "batch", "epoch", "learning_rate","filter", "predict_threshold", "time", "loss", "dice_coef", "val_loss", "val_dice_coef"])

				
    def best_model_record(self, best_model, dataset_name, best_model_time, best_iou, best_iou_var, best_dice, best_dice_var, best_Sensitivity, best_Sensitivity_var):
        print(best_model)
        [model_name , epoch, batch, lrn, filter] = best_model.split('_')
        if not os.path.isfile(os.path.join("record", "best_model.csv")):
            with open(os.path.join("record", "best_model.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["date", "model", "dataset", "batch", "epoch", "learning_rate", "filter","predict_threshold", "time", "best_iou", "best_iou_var", "best_dice", "best_dice_var", "best_Sensitivity", "best_Sensitivity_var"])

        with open(os.path.join("record", "best_model.csv"), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ 
                datetime.datetime.now().strftime("%Y-%m-%d").__str__(), # 現在日期
                model_name, # 模型名稱
                dataset_name, # 資料集名稱
                batch, # batch size
                epoch, # epoch
                lrn, # learning rate
                filter,
                self.predict_threshold, # predict threshold
                best_model_time, # 訓練時間
                best_iou, # best iou
                best_iou_var, # best iou var
                best_dice, # best dice
                best_dice_var, # best dice var
                best_Sensitivity, # best Sensitivity
                best_Sensitivity_var, # best Sensitivity var
            ])
        f = open(self.data_class + '_' + self.data_date + '_summary_2.txt', 'a')
        lines = ['----------Summary:' + best_model + '----------\n',
                'Time: '+ str(best_model_time) + '\n',
                'Jaccard index: '+ str(best_iou) + ' +- ' + str(best_iou_var) + '\n',
                'Dice Coefficient: '+ str(best_dice) + ' +- ' + str(best_dice_var) + '\n',
                'Sensitivity: '+ str(best_Sensitivity) + ' +- ' + str(best_Sensitivity_var) + '\n',
                '----------------------------------------\n\n']
        f.writelines(lines)
        f.close()

    # 訓練模型
    def run(self, PATH_DATASET, train_signal = True, postprocess_signal = False, gradcam_signal = False):
        set_seed(20)
        
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
                        writer.writerow(['model_name','original_ji_score','original_ji_var','original_dice_score','original_dice_var','original_Sensitivity_score','original_Sensitivity_var','ji_score','ji_var','dice_score','dice_var','Sensitivity_score','Sensitivity_var'])
            for data in self.datas: #讀取資料集的訓練資料
                for model_name in self.models:  # 讀取模型
                    best_model = ''
                    best_dice = -1
                    best_dice_var = -1
                    best_iou = -1
                    best_iou_var = -1
                    best_Sensitivity = -1
                    best_Sensitivity_var = -1
                    best_model_time = 0
                    for epoch in self.epochs: #每個模型訓練幾個epoch
                        for batch in self.batchs: #每個模型訓練幾個batch
                            for lrn in self.lrns: #每個模型訓練幾個learning rate
                                for filter in self.filters:
                                    # 建立模型跟預測結果的資料夾
                                    floder_path = self.data_class + '_' + self.data_date
                                    
                                    if 'concate34OCT' in dataset_name:
                                        print('dataset_name',dataset_name)
                                        self.channel = 4
                                    print("channel",self.channel)
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
                                        getModels = model_classes[model_name]((self.image_size,self.image_size,self.channel),lrn)
                                    else:
                                        raise ValueError(f"Model '{model_name}' is not supported.")
                                    model = getModels.build_model(filter)
                                    print("train model",model_name)
                                    print(model.summary())
                                    # # 編譯模型optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss
                                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrn), loss=focal_tversky, metrics=[dice_coef, jaccard_index])
                                    # 訓練模型 
                                    save_model_name = model_name + '_' + epoch.__str__() + '_' + batch.__str__() + '_' + lrn.__str__() + '_' + str(filter[0])
                                    
                                    
                                    # 取得字串 用來判斷是否有重複訓練過 去掉最後的_1.h5

                                    lists = os.listdir(model_path) #列出資料夾下所有的目錄與檔案
                                    count = 0
                                    for i in range(0,len(lists)):
                                        # 找名稱完全相同
                                        if lists[i].startswith(save_model_name) and lists[i].endswith('.h5'):
                                            count = count +1
                                    time = 0
                                    if train_signal:
                                        count = count +1 # 紀錄模型訓練次數
                                        
                                        history, time = self.fitModel(model,dataset_name,data ,train_dataset, valid_dataset,epoch, lrn ,model_path,save_model_name,count.__str__())

                                        # 紀錄訓練結果
                                        self.record(history, time,model_path,save_model_name,count.__str__())
                                    print("evaluate",save_model_name + '_' + count.__str__())
                                    ji_score, ji_var , dice_score, dice_var , Sensitivity_score, Sensitivity_var = self.evaluateModel(dataset_name,model, dataset, result_path,model_path,save_model_name,count.__str__(),self.predict_threshold,postprocess_signal,gradcam_signal)
                                    
                                    ji_score = ji_score * 100
                                    ji_var = ji_var * 100
                                    dice_score = dice_score * 100
                                    dice_var = dice_var * 100
                                    Sensitivity_score = Sensitivity_score * 100
                                    Sensitivity_var = Sensitivity_var * 100
                                    
                                    # if train_signal:
                                    f = open(self.data_class + '_' + self.data_date + '_summary.txt', 'a') # + save_model_name + '----------\n',
                                    lines = ['----------Summary:' + dataset_name + '----------\n',
                                            'Model: '+ save_model_name + '\n',
                                            'Time: '+ str(time) + '\n',
                                            'Jaccard index: '+ str(ji_score) + ' +- ' + str(ji_var) + '\n',
                                            'Dice Coefficient: '+ str(dice_score) + ' +- ' + str(dice_var) + '\n',
                                            'Sensitivity: '+ str(Sensitivity_score) + ' +- ' + str(Sensitivity_var) + '\n',
                                            '----------------------------------------\n\n']
                                    f.writelines(lines)
                                    f.close()
                                
                                    if ji_score > best_iou :
                                        best_iou = ji_score
                                        best_iou_var = ji_var
                                        best_dice = dice_score
                                        best_dice_var = dice_var
                                        best_Sensitivity = Sensitivity_score
                                        best_Sensitivity_var = Sensitivity_var
                                        best_model = save_model_name
                                        best_model_time = time
                    
                    if train_signal:         
                        self.best_model_record(best_model, dataset_name, best_model_time, best_iou, best_iou_var, best_dice, best_dice_var, best_Sensitivity, best_Sensitivity_var)


    def get_best_model(self,model_data,data, model_name,batch,epoch,lrn,filters,crf= True,channel = 4):# , output_path,model_path, model_name,predict_threshold,postprocess_signal
        self.channel = channel
        getModels = model_classes[model_name]((self.image_size,self.image_size,self.channel) ,lrn)
        model = getModels.build_model(filters)
        save_model_name = model_name + '_' + epoch.__str__() + '_' + batch.__str__() + '_' + lrn.__str__() + '_' + str(filters[0]) 
        print(save_model_name)
        
        
        # load model
        lists = os.listdir(self.model_path)
        folder_path = self.data_class + '_' + self.data_date
        model_path = os.path.join(self.model_path,folder_path,model_data)
        count = 0
        print(model_path)
        for item in os.listdir(os.path.join(model_path)):
            if item.startswith(save_model_name) and item.endswith('.h5'):
                count = count +1

        print(count,os.path.join(model_path, save_model_name + '_'+ count.__str__() + '.h5'))
        model.load_weights(os.path.join(model_path, save_model_name + '_'+ count.__str__() + '.h5'))
        
        # get data 
        data_path = os.path.join(data)
        img_path = os.path.join(data_path, "images")
        mask_path = os.path.join(data_path, "masks")
        test_ids = os.listdir(img_path)
        test_dataset = DataGenerator(test_ids, data_path, batch_size=1, image_size=self.image_size)
        # predict
        predict_path = os.path.join(data_path, "predict")
        print(predict_path)
        if not os.path.isdir(predict_path):
            os.makedirs(predict_path)

        
        # evaluate
        result_path = os.path.join(data_path, "result.csv")
        with open(result_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name",
                             "predict_jaccard_index", 
                             "predict_dice_coefficient",
                             "predict_Sensitivity"
                             ])
        iou = []
        dice = []
        Sensitivitys = []
        for i, item in enumerate(model.predict(test_dataset)):
            img = item[:, :, 0].copy()
            img[img < self.predict_threshold] = 0
            img[img >= self.predict_threshold] = 255
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (self.image_size, self.image_size))
            if crf:
                crf_input = img.copy()
                crf_input = np.array(crf_input, dtype=np.uint8)
                crf_input = cv2.resize(crf_input, (self.image_size, self.image_size)) 
                 
                if "images_original" in os.listdir(data_path):
                    original = cv2.imread(os.path.join(data_path, "images_original", test_ids[i]))
                else:
                    original = cv2.imread(os.path.join(data_path, "images", test_ids[i]))
                
                img = self.crf_predict(original,crf_input) 
                if len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img, dtype=np.uint8)
                
            # remove small area
            img = remove_small_area(img,50)
            
            
            cv2.imwrite(os.path.join(predict_path, test_ids[i]), img)
            
            predict = cv2.imread(os.path.join(predict_path, test_ids[i]), 0)
            mask = cv2.imread(os.path.join(mask_path, test_ids[i]), 0)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = np.array(mask, dtype=np.uint8)
            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            predict[predict < 128] = 0
            predict[predict >= 128] = 1
            jc = jaccard_score(mask.ravel(), predict.ravel())
            di = dice_coef(mask, predict)
            di = di.numpy()
            iou.append(jc)
            dice.append(di)
            Sensitivity = recall_score(mask.flatten(), predict.flatten(), average='micro')
            Sensitivitys.append(Sensitivity)
            record_data = [
                test_ids[i],
                jc,
                di,
                Sensitivity
            ]
            with open(result_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(record_data)
        with open(result_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["mean", np.mean(iou), np.mean(dice), np.mean(Sensitivitys)])
            writer.writerow(["std", np.std(iou, ddof=1), np.std(dice, ddof=1), np.std(Sensitivitys, ddof=1)])
            
        return np.mean(iou), np.std(iou, ddof=1), np.mean(dice), np.std(dice, ddof=1), np.mean(Sensitivitys), np.std(Sensitivitys, ddof=1)
            


    def get_gradcam(self, model, img, layer_name):
        # 載入圖片
        img = cv2.imread(img)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        # 取得模型的最後一層
        layer = model.get_layer(layer_name)
        # 取得模型的權重
        model_gradcam = tf.keras.models.Model([model.inputs], [model.output, layer.output])
        # 取得梯度
        with tf.GradientTape() as tape:
            conv_output, pred = model_gradcam(img)
            class_out = pred[:, :, :, 0]
            grads = tape.gradient(class_out, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap[0]
        return heatmap

    def gradcam(self, model, test_dataset, result_path,model_name):
        # 建立資料夾
        print("gradcam")
        gradcam_path = os.path.join(result_path,model_name, "gradcam")
        print(gradcam_path)
        if not os.path.isdir(gradcam_path):
            os.makedirs(gradcam_path)
        # 取得測試資料


        test_path = os.path.join(test_dataset, "test")
        test_ids = os.listdir(os.path.join(test_path, "images"))
        test_dataset = DataGenerator(test_ids, test_path, batch_size=1, image_size=self.image_size)
        # 取得gradcam
        explainer = GradCAM()
        for i in test_ids:
            for layer_name in [layer.name for layer in model.layers]:
                # index layer_name
                index =  [layer.name for layer in model.layers].index(layer_name)
                save_img_name = i.split('.')[0] 
                gradcam_img_path = os.path.join(gradcam_path, save_img_name, 'img')
                if not os.path.isdir(gradcam_img_path):
                    os.makedirs(gradcam_img_path)
                    
                img = cv2.imread(os.path.join(test_path, "images", i))
                img = cv2.resize(img, (self.image_size, self.image_size))

                # print("layer_name",layer_name)
                if 'reshape' in layer_name or 'concatenate' in layer_name :
                    continue
                else:
                    grid = explainer.explain(((np.expand_dims(img, axis=0)/255.0).astype(np.float32),None), model, class_index=1, layer_name=layer_name)
                    # Resize the heatmap to the original image size
                    
                    heatmap = cv2.resize(grid, (self.image_size, self.image_size))
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                    
                    
                    # Apply colormap to the heatmap
                    fig, ax = plt.subplots()
                    ax.imshow(img, alpha=0.4)
                    colorbar = ax.imshow(heatmap, cmap='hot', alpha=0.6, interpolation='bilinear')
                    plt.axis('off')
                    plt.colorbar(colorbar)
                    plt.savefig(os.path.join(gradcam_img_path,str(index) + '.png'), bbox_inches='tight')
                    # plt.show()
                    plt.close()
            # print("img2video")
            # img to video
            img2video(gradcam_img_path, os.path.join(gradcam_path,save_img_name, save_img_name + '.mp4'))
            
            
def remove_small_area(msk_g, min_area = 10):
    # 刪除小面積
    # 連通域的數目 連通域的圖像 連通域的信息 矩形框的左上角坐標 矩形框的寬高 面積
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(msk_g, connectivity=8)
    # 刪除小面積
    msk_rm_small = msk_g.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            msk_rm_small[labels == i] = 0
    return msk_rm_small