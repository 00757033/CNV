from sklearn.model_selection import GroupKFold
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from UnetModel_pytorch import *
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#產生訓練資料
class CustomDataset(Dataset):
    def __init__(self, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成虛構的圖像和標籤資料（實際情況下，這裡應該載入真實的圖像和標籤）
        image = torch.randn(3, 304, 304)  # 假設彩色圖像，大小為 128x128
        label = torch.randn(1, 304, 304)  # 假設灰度標籤，大小為 128x128
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            
        return image, label

# 定義圖像轉換
transform = transforms.Compose([
    transforms.ToTensor(),  # 將圖像轉換成 Tensor 格式
])

# 創建自定義的訓練資料集
num_samples = 1000  # 設定資料集大小
dataset = CustomDataset(num_samples, transform=transform)

# 使用 DataLoader 進行批次加載
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class train():
    def __init__(self, data_class, data_date, model_path, result_path, image_size, models, batchs, epochs, datas, lrns, filters, predict_threshold):
        self.data_class   = data_class          #病灶種類
        self.data_date    = data_date           #資料日期
        self.model_path   = model_path          #儲存模型路徑
        self.result_path  = result_path         #儲存結果路徑
        self.image_size   = image_size          #input影像大小
        self.models       = models              #訓練的模型(U-Net、U-Net++...)
        self.batchs       = batchs              #batch size(2、4、8、16)
        self.epochs       = epochs              #epoch size(50、100、200、300、400)
        self.datas        = datas               #使用的訓練資料集，例: train->沒有augment、augmnet5->augmnet5倍...以此類推
        self.lrns         = lrns                #lreaning rate(0.001、0.0001)
        self.filters      = filters             #各層的kernal數量([32, 64, 128, 256, 512])

    def getModel(self, model_name, image_size, learning_rate):
        if model_name == 'UNet':
            myModel = UNet(image_size, learning_rate)

    #訓練模型，可以調整earlyStop、reduce_lr，回傳模型跟訓練時間
    def fitModel(self, model, model_name, feature, epochs, filters, train_dataloader, valid_dataloader, model_path, train_signal):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

        # Define your loss function
        criterion = nn.BCEWithLogitsLoss()  # Example loss function

        # Define your callbacks
        checkpoint = None  # PyTorch does not require explicit model checkpointing during training
        reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=0.0000001, verbose=True)
        early_stop = EarlyStopping(patience=40, verbose=True)

    def run(self, PATH_DATASET, train_signal):
        # 訓練模型
        num_epochs = 10
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == '__main__':
    t = train()