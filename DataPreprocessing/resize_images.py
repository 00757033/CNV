import cv2
from pathlib import Path


PATH_TRAIN_IMAGES = '../Data/' + 'train/images/'
PATH_TRAIN_LABELS = '../Data/'  + 'train/labels/'
PATH_TEST_IMAGES = '../Data/'  + 'test/images/'
PATH_TEST_LABELS = '../Data/'  + 'test/labels/'

def resize_images(path, path_output, size=(304,304)):
    for image in path.glob("*"):
        image = str(image)
        image_name = Path(image).stem
        image = cv2.imread(image)
        cv2.imshow("image", image)
        image = cv2.resize(image, (304,304))
        cv2.imwrite(str(path_output / image_name), image)
    
if __name__ == '__main__':
    resize_images(Path(PATH_TRAIN_IMAGES), Path(PATH_TRAIN_IMAGES))
    resize_images(Path(PATH_TRAIN_LABELS), Path(PATH_TRAIN_LABELS))