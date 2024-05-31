import torch
import config
import importlib
importlib.reload(config)
from config import defaultConfig
from torch import nn
import torch.utils.data
from mydataset import mydataset
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from auto_note import send_email_with_attachment



class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length, dropout) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dropout = dropout

        self.linear = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.linear1 = nn.Linear(self.hidden_size*2, self.output_size)
        # self.linear = nn.Linear(self.hidden_size, self.output_size)  # 简化版

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):  # input(32, 10, 166)
        # batch_size = x.size()[0]
        # seq_len = x.size()[1]
        x = x.to(torch.float32).to(x.device)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))  # output(32, 10, hidden_size)

        predict = self.linear1(F.relu(self.linear(output)))  # predict[32, 10, 1]
        # predict = self.linear(output)  # 简化版

        predict = predict[:, -1, :]  # predict[32, 1]
        return predict


# train..................................................
def train1(epoch_num, criterion, optimizer, module, data_loaders, test_loader, jump):
    train_loss_plot = []
    test_loss_plot = []
    min_loss = 100
    pre_min_loss = 101
    early_stop_counter = 0
    for epoch in range(epoch_num):
        train_loss = 0.
        datasize = 0
        for data_loader in data_loaders:
            for i, (fea_val, tar_val) in enumerate(data_loader):
                fea_val, tar_val = fea_val.to("cuda"), tar_val.to("cuda")
                output = module(fea_val)
                loss = criterion(output, tar_val.to(torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                datasize += 1
            print(f'training...epoch:{epoch+1},({i+1}/{len(data_loader)})')
        epoch_loss = train_loss / datasize
        print(f'Epoch {epoch + 1}: Training loss = {epoch_loss:.4f}')
        print(f'jump:{jump}')
        train_loss_plot.append(epoch_loss)

        test_loss = test1(test_loader, module, criterion)
        test_loss_plot.append(test_loss)
        min_loss = min(test_loss_plot)


        if (epoch + 1) % 5 == 0:
            save_and_plot_results(epoch, module, train_loss_plot, test_loss_plot, batch_size, learning_rate, hidden_size, num_layers, dropout, min_loss, ' ')


#  test..................................................
def test1(test_loader, module, criterion):
    total_loss = 0.0
    total_samples = 0
    module.eval()
    with torch.no_grad():
        for fea_val, tar_val in test_loader:
            fea_val, tar_val = fea_val.to("cuda"), tar_val.to("cuda")
            outputs = module(fea_val)
            loss = criterion(outputs, tar_val)
            total_loss += loss.item() * len(fea_val)
            total_samples += len(fea_val)
    avg_loss = total_loss / total_samples
    print(f'Test MSE Loss: {avg_loss:.4f}')
    module.train()
    return avg_loss

def save_and_plot_results(epoch, module, train_loss_plot, test_loss_plot, batch_size, learning_rate, hidden_size, num_layers, dropout, min_loss, msg):
    # 保存模型
    model_save_path = f'./module/NetV3-{epoch+1}-{train_loss_plot[-1]:.3f}.pth'
    torch.save(module.state_dict(), model_save_path)
    
    # 绘制训练和测试损失曲线
    plt.figure()
    plt.plot(train_loss_plot, label='Train Loss')
    plt.plot(test_loss_plot, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.savefig(f'./img/T5_epoch:{epoch+1}_min_test_loss:{min_loss:.3f}_{batch_size}bs_{learning_rate}lr_{hidden_size}hs_{num_layers}nl_{dropout}dr.png')
    plt.close()

def calculate_r_squared(y_true, y_pred):
    # 计算总平均值
    mean_y_true = np.mean(y_true)

    # 计算总平方和
    ss_total = np.sum((y_true - mean_y_true) ** 2)

    # 计算残差平方和
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # 计算R平方
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared

def test2(epoch_num):
    for epoch in range(epoch_num):
        for i, (fea_val, tar_val) in enumerate(train_loader):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            r2 = calculate_r_squared(tar_val.detach().cpu().numpy().reshape((len(tar_val),)), output.detach().cpu().numpy().reshape((len(output),)))
            # print(np.shape(tar_val.numpy().reshape((len(tar_val),))))
            # print(np.shape(output.detach().numpy().reshape((len(output),))))
            print(f"R-squared score: {r2:.4f}")


opt = defaultConfig()
batch_size = opt.batch_size
epoch_num = opt.epoch_num
learning_rate = opt.learning_rate
input_size = opt.input_size
output_size = opt.output_size
hidden_size = opt.hidden_size
num_layers = opt.num_layers
seq_length = opt.seq_length
trainBool = opt.train
test1Bool = opt.test1
test2Bool = opt.test2
module_path = opt.module_path
loadBool = opt.loadBool
dropout = opt.dropout
# path
features_path1 = opt.features_path1
targets_path1 = opt.targets_path1

features_path2 = opt.features_path2
targets_path2 = opt.targets_path2

features_path3 = opt.features_path3
targets_path3 = opt.targets_path3

features_path4 = opt.features_path4
targets_path4 = opt.targets_path4

features_path5 = opt.features_path5
targets_path5 = opt.targets_path5

features_path6 = opt.features_path6
targets_path6 = opt.targets_path6

features_path7 = opt.features_path7
targets_path7 = opt.targets_path7

features_path8 = opt.features_path8
targets_path8 = opt.targets_path8

test_fpath = opt.features_path
test_tpath = opt.targets_path

# trainLoader
if trainBool:
    print("开始构建trainLoader...")

    features_paths = [features_path1, features_path2, features_path3, features_path4, features_path5, features_path6, features_path7]
    targets_paths = [targets_path1, targets_path2, targets_path3, targets_path4, targets_path5, targets_path6, targets_path7]
    data_loaders = []

    for features_path, targets_path in zip(features_paths, targets_paths):
        dataset = mydataset(features_path, targets_path)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        data_loaders.append(data_loader)

    testData = mydataset(test_fpath, test_tpath)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=True)
    print("trainLoader构建完毕")
        
if test1Bool:
    testData = mydataset(test_fpath, test_tpath)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=True)
    # test1(test_loader)

if test2Bool:
    trainData = mydataset(test_fpath, test_tpath)
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    test2(epoch_num)

if trainBool:
    # 初始化模型
    print(f"{batch_size}bs_{learning_rate}lr_{hidden_size}hs_{num_layers}nl_{dropout}dr")
    module = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length, dropout)
                            
    # 处理多GPU情况
    if torch.cuda.device_count() > 1:
        module.to('cuda')
        module = nn.DataParallel(module)
    else:
        module.to('cuda')
                            
    # 初始化优化器
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
                            
    # 初始化损失函数
    criterion = nn.MSELoss()

    # 开始训练
    print(f'Training started at {time.ctime()}')
    start_time = time.perf_counter()
    train1(epoch_num, criterion, optimizer, module, data_loaders, test_loader, 0)
    end_time = time.perf_counter()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
