import albumentations as A
import cv2
import tools
import pathlib as pl
import shutil
import os
from matplotlib import pyplot as plt
class Augment():
    def __init__(self,path,layers = {"4":"CC"}):
        self.path = path
        self.layers = layers
        self.train_test = ["train","test","valid"]

    def albumentation(self, path,data_dir,output,image_name):
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(path.replace('images','masks'))
        image_original = None
        if 'images_original' in data_dir:
            image_original = cv2.imread(path.replace('images','images_original'))
            image_original = cv2.resize(image_original, (304,304))
        
        # Define the augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            # A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # A.GridDistortion(p=0.5),
            # A.OpticalDistortion(p=0.5),
            # A.Blur(blur_limit=3, p=0.1),
        ], additional_targets={'image_original': 'image'}, is_check_shapes=False)
        
        inputs = {'image': image, 'mask': mask}
        if image_original is not None:
             inputs['image_original'] = image_original 
        # Augment an image
        augmented = transform(**inputs)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        if image_original is not None:
            aug_image_original = augmented['image_original']
            cv2.imwrite(os.path.join(output,'images_original',image_name + '.png'), aug_image_original)
        # show the augmented image
        # fig, ax = plt.subplots(2, 3, figsize=(10, 10))
        # ax[0, 0].imshow(image)
        # ax[0, 0].axis('off')
        # ax[0, 1].imshow(mask)
        # ax[0, 1].axis('off')
        # if image_original is not None:
        #     ax[0, 2].imshow(image_original)
        #     ax[0, 2].axis('off')
        # ax[1, 0].imshow(aug_image)
        # ax[1, 0].axis('off')
        # ax[1, 1].imshow(aug_mask)
        # ax[1, 1].axis('off')
        # if image_original is not None:
        #     ax[1, 2].imshow(aug_image_original)
        #     ax[1, 2].axis('off')
        # plt.show()

        
        cv2.imwrite(os.path.join(output,'images',image_name + '.png'), aug_image)
        cv2.imwrite(os.path.join(output,'masks',image_name + '.png'), aug_mask)
        
        

    def augumentation(self,input_path,output_name,augment_time = 5):
        data_dir = ['images','masks'] # images_original
        if 'concate' in input_path: 
            data_dir.append('images_original')

        for layer in self.layers:
            layer_input_path = tools.get_label_path(input_path, self.layers[layer])
            layer_output_path = tools.get_label_path(input_path + '_' + output_name, self.layers[layer])
            input_dir = os.path.join(self.path,  layer_input_path)
            output_dir= os.path.join(self.path,  layer_output_path)
            print('input_dir',input_dir)
            print('output_dir',output_dir)
            for train_test in self.train_test:     
                if train_test == 'train' : 
                    input = os.path.join(input_dir,'train')
                    output = os.path.join(output_dir,'train')
                    for  dir in data_dir:
                        tools.makefolder(os.path.join(output,  dir))
                        
                    for image in pl.Path(input + '/'+ 'images').iterdir():
                        if image.suffix == ".png" :
                            image_name = image.stem
                            img_path = str(image)    
                            for dir in data_dir:
                                cv2.imwrite(os.path.join(output, dir, image_name + '.png'), cv2.imread(os.path.join(input, dir, image_name + '.png'), cv2.IMREAD_UNCHANGED))
                                
                            
                            for time in range(augment_time):     
                                self.albumentation(img_path,data_dir,output,image_name + '_' + str(time))

                else :
                    print('copytree')
                    print(input_dir+ '/' +train_test)
                    shutil.copytree(input_dir+ '/' +train_test, output_dir+'/' +train_test)

def make_output_path(path ,output_name, layer_name):
        output_image = os.path.join(path, f"{output_name}_{layer_name}/images")
        output_label = os.path.join(path, f"{output_name}_{layer_name}/masks")
        for output_dir in [output_image, output_label]:
            os.makedirs(output_dir, exist_ok=True)

        return output_image, output_label

if __name__ == "__main__":
    date = '20240524'
    disease = 'PCV'
    PATH = "../../Data/"
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    PATH_LABEL = PATH + "/" + "new_label"
    PATH_IMAGE = PATH + "/" + "OCTA"
    preprocess = Augment(PATH_BASE + '/' + 'trainset')
    augment_time = [1]
    for time in augment_time:
        preprocess.augumentation(disease + "_"+ date +  '_connectedComponent_bil31010_clah0708_concate34OCT_30','aug' + str(time),time)    