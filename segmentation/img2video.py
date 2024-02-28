import cv2
import os
import ffmpeg

# 將圖片轉成影片
def img2video(img_folder ,ouput_video,frame_rate = 4):
    # 影片幀率

    # 取得資料夾下的所有圖片
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]

    # 排序圖片
    images.sort(key=lambda x: int(x.split('.')[0]))

    # 取得第一張圖片的寬高
    img = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = img.shape

    # 建立影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(ouput_video, fourcc, frame_rate, (width,height))

    # 寫入影片
    for image in images:
        video.write(cv2.imread(os.path.join(img_folder, image)))

    # 停留在最後一張圖片一段時間
    for _ in range(frame_rate ):  # 停留兩秒鐘
        video.write(cv2.imread(os.path.join(img_folder, images[-1])))
        
    # 釋放資源
    cv2.destroyAllWindows()
    video.release()
    
if __name__ == "__main__":
    img_folder = './Result/PCV_0205/PCV_0205_bil510_clahe7_concate_42_aug2_CC/train/UNet_150_2_0.001_1/gradcam/00285176_L_20190116/img/'
    
    ouput_video = './Result/PCV_0205/PCV_0205_bil510_clahe7_concate_42_aug2_CC/train/UNet_150_2_0.001_1/gradcam/00285176_L_20190116/00285176_L_20190116.mp4'
    
    img2video(img_folder ,ouput_video)