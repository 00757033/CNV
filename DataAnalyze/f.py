import cv2
import numpy as np
import pathlib as pl

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d'%(x, y) )


img = cv2.imread(pl.Path('..\\..\\Data\\need_label\\PCV\\00006202\\20220328\\R\\4.png').as_posix())
gray_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normal_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
blur = cv2.GaussianBlur(normal_img,(7,7),0)
clahe_img  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blur)
opening = cv2.morphologyEx(normal_img, cv2.MORPH_OPEN, (7,7))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (7,7))
for i in range(0,250,10):
    print(i)
    ret, th3 = cv2.threshold(closing,i,255,cv2.THRESH_TOZERO)

    ret, th1 = cv2.threshold(clahe_img,i,255,cv2.THRESH_TOZERO)
    floodfill =  cv2.floodFill(th3, None, (0,0), 255)
    cv2.imshow('img', floodfill[1])
    # while True:
    #     # mouse click
    #     cv2.namedWindow('Point Coordinates')
    #     cv2.setMouseCallback('Point Coordinates', on_mouse)
    #     cv2.imshow('Point Coordinates', floodfill[1])
    #     if cv2.waitKey(0) == ord('q'):
    #         break



    
    cv2.waitKey(0)
cv2.destroyAllWindows()
