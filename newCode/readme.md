資料前處理
DataProcessing # 資料夾
- getData.py
- - Errorimage # 判斷是否為錯誤圖片
- - denoise_image # 去除雜訊
- relabels.py # 八連通獲得真實ground Truth
- - relabel # 將label輪廓轉成含有血管
- - contours # 血管輪廓 用在label
- Preprocessing.py
- - clahe_preprocess # CLAHE
- - filter_parameter # bilateral Filter
- - sharp_parameter # SHARP
- ConcateData_eval.py # 影像合併
- splitData_group.py #以病人區分
- splitData.py # 分train test vaild
- AugData.py # 資料擴增
===================================
分割
segmentation # 資料夾
- main.py # 整個分割任務
- - trainModel.py #模型訓練
- - - train
- - - - postproccessing # 後處理
- - - - record # 模型紀錄
- - - - crf_predict # CRF後處理預測結果
- - - evaluateModel #評估模型
- - - DataGenerator #產生訓練資料 以及測試資料
- - - focal_tversky #焦點Tversky
- - - remove_small_area # 刪除小面積
- - - get_best_model # 保留最好的模型
====================================
對位
Alimentation # 資料夾
- feature.py 
- - 變數
- - - expands # 擴張大小
- - - distances # 配對距離
- - - matchers # 比對方法
- - 函式
- - - mean_squared_error_ignore_zeros # 非零MSE
- - - psnr_ignore_zeros # 非零PSNR
- - - ssim_ignore_zeros # 非零SSIM
- - - NCC_ignore_zeros # 非零NCC
- - - feature # 對位
- - - all_evaluate # 評估
- best_align.py  # 僅保留最好的方法到align資料夾中
======================
特徵提取與分類
FeatureExtractionAndClassification #資料夾
-  getCompareData.py # align 的資料轉成compare 格式
-  VesselAnalysis.py # 提取紋理特徵與形態學特徵並轉成變化率
-  ground_truth.py # 以病人視力變化作為分類依據
-  statistic.py #分析治療前後特徵的曼會特寧U檢定
-  classification.py # 分類




