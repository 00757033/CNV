import numpy as np
from skimage.measure import label, regionprops
import cv2
import pathlib as pl
# pip install scikit-image
def region_growing(image, seed_point, threshold):
    labeled_image = label(image)  # 对二值图像进行标记
    regions = regionprops(labeled_image)  # 提取标记区域属性

    seed_label = labeled_image[seed_point]  # 获取种子点的标记值
    target_region = regions[seed_label - 1]  # 获取种子点所属的区域属性

    region_mask = labeled_image == seed_label  # 创建种子点所在的区域掩膜
    while True:
        # 使用区域的平均灰度值作为生长准则
        mean_value = target_region.mean_intensity
        # 获取当前区域周围8邻域内符合条件的像素点
        neighbors = np.argwhere(
            (labeled_image == 0) &
            (np.abs(image - mean_value) <= threshold)
        )
        if len(neighbors) == 0:
            break
        # 将符合条件的像素点添加到区域掩膜中
        for neighbor in neighbors:
            region_mask[neighbor[0], neighbor[1]] = True
            labeled_image[neighbor[0], neighbor[1]] = seed_label
        # 更新当前区域的属性
        target_region = regionprops(region_mask.astype(int))[0]

    return region_mask

# 示例用法
image = cv2.imread(pl.Path('..\\..\\Data\\need_label\\PCV\\00006202\\20220328\\R\4.png').as_posix() )
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (304,304))

# 选择种子点
seed_point = (100, 100)

# 设置阈值
threshold = 10

# 区域生长
result = region_growing(image, seed_point, threshold)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Region Growing Result', result.astype(np.uint8) * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
