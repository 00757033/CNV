from pathlib import Path
import cv2

def setFolder(path_floder):
    path_floder = Path(path_floder)
    if not path_floder.exists():
        path_floder.mkdir()

def otsu(image, kernal_size = (5,5)):
    blur = cv2.GaussianBlur(image,kernal_size,0) #模糊化
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #otsu二值化
    return th3