import albumentations as A
import cv2
import os
from segmentation.function import setFolder
from pathlib import Path

class Augment():
    def albumentation(self, image, mask):
        transform = A.Compose([
                A.HorizontalFlip(p=0.5),                                                         #水平翻轉
                A.VerticalFlip(p=0.5)                                                            #垂直翻轉
                #A.OneOf([
                #    # A.IAAAdditiveGaussianNoise(),                                             # 将高斯噪声添加到输入图像
                #    A.GaussNoise(),                                                             # 将高斯噪声应用于输入图像。
                #], p=0.2),                                                                      # 应用选定变换的概率
                #A.OneOf([
                #    A.MotionBlur(p=0.2),                                                        # 使用随机大小的内核将运动模糊应用于输入图像。
                #    A.MedianBlur(blur_limit=3, p=0.1),                                          # 中值滤波
                #    A.Blur(blur_limit=2, p=0.1),                                                # 使用随机大小的内核模糊输入图像。
                #], p=0.2),
                #A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.07, rotate_limit=45, p=0.2), # 随机应用仿射变换：平移，缩放和旋转输入
                #A.RandomBrightnessContrast(p=0.2),                                              # 随机明亮对比度
            ])(image=image,mask=mask)
        image_aug = transform['image']
        mask_aug = transform['mask']
        return image_aug, mask_aug

    def augmentData(self, data_path, aug_path, augment_time):
        setFolder(aug_path)
        setFolder(aug_path + "/images")
        setFolder(aug_path + "/masks")
        for image_name in os.listdir(data_path+"/images"):    
            image = cv2.imread(data_path + "/images/" + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(data_path + "/masks/" + image_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            image_name = image_name.replace(".png", "")
            cv2.imwrite(aug_path + "/images/" + image_name + "_0" + ".png" , image)
            cv2.imwrite(aug_path + "/masks/" + image_name +"_0" +  + ".png" , mask)
            for i in range(0,augment_time):
                image_aug, mask_aug = self.albumentation(image, mask)
                cv2.imwrite(aug_path + "/images/" + image_name + "_" +  str(i) + ".png" , image_aug)
                cv2.imwrite(aug_path + "/masks/" + image_name +"_" + str(i) + ".png" , mask_aug)

if __name__ == '__main__':   
    data_class = 'CNV'
    date = '0324'
    path_base = Path("../../Data") / Path(data_class+ '_' + date + '/') 
    PATH_DATASET = path_base / Path("train")
    #Augment Time
    augment_times = [2, 5, 10]

    for dataset in PATH_DATASET.glob("*"):
        for augment_time in augment_times:
            PATH_DATASET_INPUT = str(dataset / 'train')
            PATH_DATASET_AUG   = str(dataset / Path('augment' + str(augment_time)))
            augment = Augment()
            augment.augmentData(PATH_DATASET_INPUT, PATH_DATASET_AUG, augment_time)