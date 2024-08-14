import cv2
import numpy as np
import os
from pathlib import Path
import tools.tools as tools
from matplotlib import pyplot as plt
class reLabel():
    def __init__(self,path,layers = {"4":"CC"}):
        self.path = path
        self.layers = layers

    def relabel(self,output_name,mathod = 'threshold',min_area = 49): # OTSU and then ROI
        data_dir = ['images','masks']

        for layer in self.layers:

            layer_path = tools.get_label_path(output_name, self.layers[layer])
            intput_dir = os.path.join(self.path,  self.layers[layer],'images')
            output_dir = os.path.join(self.path,  layer_path)
            print('intput_dir',intput_dir)
            print('output_dir',output_dir)
            output_image = self.path + '/' + output_name + '_'+self.layers[layer] + '/'+'images'
            output_label = self.path + '/' + output_name + '_'+self.layers[layer] + '/'+'masks'
            tools.makefolder(output_image)
            tools.makefolder(output_label)

            for  dir in data_dir:
                tools.makefolder(os.path.join(output_dir,  dir))
            for images in Path(intput_dir).iterdir():
                image_name = images.name
                img_path = str(images)
                label_path = img_path.replace('images','masks')

                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, (304, 304))
                image_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                image_label = cv2.resize(image_label, (304, 304))
                img_1 = cv2.imread(os.path.join(self.path,  'ALL','1',image_name), cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(os.path.join(self.path,  'ALL','2',image_name), cv2.IMREAD_GRAYSCALE)
                # fig , ax = plt.subplots(1,5)
                # ax[0].imshow(image, cmap = 'gray')
                # ax[1].imshow(image_label, cmap = 'gray')
                # ax[2].imshow(img_1, cmap = 'gray')
                # ax[3].imshow(img_2, cmap = 'gray')
                # ax[4].imshow(img_cut, cmap = 'gray')
                # plt.show()
                if mathod == 'threshold':
                    threshold_label = self.otsuthreshold(image,image_label)
                    cv2.imwrite(os.path.join(output_dir, 'masks', img_path.split('\\')[-1]), threshold_label)
                
                if mathod == 'connectedComponent':
                    # blur = cv2.GaussianBlur(image,(2,2),0)
                    ret, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # mask

                    label = image_label.copy()
                    label[binary_image == 0] = 0

                    # 刪除小面積
                    # 連通域的數目 連通域的圖像 連通域的信息 矩形框的左上角坐標 矩形框的寬高 面積
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=4)

                    # 不同的連通域賦予不同的顏色
                
                    areas = stats[:, cv2.CC_STAT_AREA]
                    without_background = label.copy()
                    
                    output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                    for i in range(1, num_labels):
                        color = [np.random.randint(0, 255) for _ in range(3)]
                        output[labels == i] = color
                        if areas[i] < min_area:
                            without_background[labels == i] = 0
                    
                    cv2.imwrite(os.path.join(output_dir, 'masks', img_path.split('\\')[-1]), without_background)
                cv2.imwrite(output_image + '/'+image_name,image)



    def otsuthreshold(self,image,label_image):
        blur = cv2.GaussianBlur(image,(3,3),0)
        ret, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        label_image[binary_image == 0] = 0
        return label_image


    def detect_blood_vessels(self,image,min_area = 10):

        # otsu threshold
        # ret, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        # minimum area
        contours_img = np.zeros_like(image)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(contours_img, [contour], -1, 255, -1)
                
        return contours_img

    def origin_relabel(self,output):
        for layer in self.layers:
                output_image = self.path + '/' + output + '_'+self.layers[layer] + '/'+'images'
                output_label = self.path + '/' + output + '_'+self.layers[layer] + '/'+'masks'
                tools.makefolder(output_image)
                tools.makefolder(output_label)
                for images in Path(self.path+ '/' + self.layers[layer]+ '/'+ 'images').iterdir():
                    image_name = images.name
                    img_path = str(images)
                    label_path = img_path.replace('images','masks')
                    image = cv2.imread(img_path, 0)
                    print(image_name,image.shape)
                    image_label = cv2.imread(label_path, 0)
                    blur = cv2.GaussianBlur(image,(3,3),0)
                    ret3,image_mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    image_label[image_mask == 0] = 0
                    # show the image
                    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
                    y = 0
                    axs[0,y].imshow(image, cmap="gray")
                    axs[1,y].imshow(blur, cmap="gray")
                    y = 1
                    axs[0,y].imshow(image_mask, cmap="gray")
                    axs[1,y].imshow(image_label, cmap="gray")

                    plt.show()
                    # plt.pause(3)
                    cv2.imwrite(output_image + '/'+image_name,image)
                    cv2.imwrite(output_label + '/'+image_name,image_label)
                    # cv2.imwrite(output_image + '/'+image_name, img)




if __name__ == "__main__":
    date = '20240502'
    disease = 'PCV'
    PATH = "../../Data/"
    FILE = disease + "_"+ date
    PATH_BASE =  PATH + "/" + disease + "_"+ date
    label= reLabel(PATH_BASE)
    # label.relabel('otsu_contour')
    # label.relabel2('OTSU_ROI_contour')
    label.relabel(FILE + '_connectedComponent',mathod='connectedComponent')











