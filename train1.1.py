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
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.linear = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.linear1 = nn.Linear(self.hidden_size*2, self.output_size)
        # self.linear = nn.Linear(self.hidden_size, self.output_size)  # 简化版

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):  # input(32, 10, 166)
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.to(torch.float32)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))  # output(32, 10, hidden_size)

        # output = self.linear(output)
        # output = nn.functional.relu(output)
        # output = self.linear1(output)
        # output = nn.functional.relu(output)

        predict = self.linear1(F.relu(self.linear(output)))  # predict[32, 10, 1]
        # predict = self.linear(output)  # 简化版

        predict = predict[:, -1, :]  # predict[32, 1]
        return predict


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
    trainData1 = mydataset(features_path1, targets_path1)
    train_loader1 = torch.utils.data.DataLoader(trainData1, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData2 = mydataset(features_path2, targets_path2)
    train_loader2 = torch.utils.data.DataLoader(trainData2, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData3 = mydataset(features_path3, targets_path3)
    train_loader3 = torch.utils.data.DataLoader(trainData3, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData4 = mydataset(features_path4, targets_path4)
    train_loader4 = torch.utils.data.DataLoader(trainData4, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData5 = mydataset(features_path5, targets_path5)
    train_loader5 = torch.utils.data.DataLoader(trainData5, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData6 = mydataset(features_path6, targets_path6)
    train_loader6 = torch.utils.data.DataLoader(trainData6, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData7 = mydataset(features_path7, targets_path7)
    train_loader7 = torch.utils.data.DataLoader(trainData7, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData8 = mydataset(features_path8, targets_path8)
    train_loader8 = torch.utils.data.DataLoader(trainData8, batch_size=batch_size, shuffle=True, drop_last=True)

    testData = mydataset(test_fpath, test_tpath)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=True)
    print("trainLoader构建完毕")

if loadBool:
    module = torch.load(module_path)  # load module
    if torch.cuda.device_count() > 1:
        module.to('cuda')
        module = nn.DataParallel(module)
else:
    module = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)  # hidden_size,num_layers,batch_size is var
    if torch.cuda.device_count() > 1:
        module.to('cuda')
        module = nn.DataParallel(module)
        

# train..................................................
def train1(epoch_num):
    train_loss_plot = []
    epoch_axis = []
    test_loss_plot = []
    for epoch in range(epoch_num):
        train_loss = 0.
        datasize = 0
        for i, (fea_val, tar_val) in enumerate(train_loader1):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
            end = time.perf_counter()
        print(f'training...epoch:{epoch+1},(1/8),time:{end-start}')
        for i, (fea_val, tar_val) in enumerate(train_loader2):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(2/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader3):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(3/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader4):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(4/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader5):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(5/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader6):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(6/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader7):
            fea_val = fea_val.to("cuda")
            tar_val = tar_val.to("cuda")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch+1},(7/8)')
        # for i, (fea_val, tar_val) in enumerate(train_loader8):
        #     fea_val = fea_val.to("cuda")
        #     tar_val = tar_val.to("cuda")
        #     output = module(fea_val)
        #     output = output.to(torch.float32)
        #     tar_val = tar_val.to(torch.float32)

        #     optimizer.zero_grad()
        #     loss = criterion(output, tar_val)
        #     train_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     datasize += 1
        print(f'training...epoch:{epoch+1},(8/8)')
        train_epoch_avg_loss = train_loss / datasize
        print(f'epoch{epoch + 1:2} loss: {train_epoch_avg_loss}')
        train_loss_plot.append(train_epoch_avg_loss)

        if (epoch) % 5 == 0:
            torch.save(module, f'./module/NetV3-{int(epoch+1)}-{float(train_epoch_avg_loss)}.pth')
            for i in range (0, epoch+1):
                epoch_axis.append(i)
            plt.plot(epoch_axis, train_loss_plot, label='train_loss')
            test_loss_plot, test_avg_mse_loss = test1(test_loader, test_loss_plot)
            epoch_axis = [j for j in epoch_axis if j % 5 == 0]
            plt.plot(epoch_axis, test_loss_plot, label='test_loss')
            plt.plot()
            plt.title("loss Line Plot")
            plt.xlabel("epoch axis")
            plt.ylabel("loss")
            plt.savefig(f"./img/epoch:{epoch+1}_loss:{test_avg_mse_loss}_{batch_size}b_{learning_rate}lr_{hidden_size}hs_{num_layers}nl_{dropout}do.png")
            # send_email_with_attachment(f"./img/epoch:{epoch+1}_loss:{test_avg_mse_loss}_{batch_size}b_{learning_rate}lr_{hidden_size}hs_{num_layers}nl_{dropout}do.png", f"epoch:{epoch+1}_{batch_size}b_{learning_rate}lr_{hidden_size}hs_{num_layers}nl")
            plt.close()
            epoch_axis.clear()
            module.train()

#  test..................................................
def test1(test_loader, test_loss_plot):
    total_mse_loss = 0.0
    total_samples = 0
    mse_loss = nn.MSELoss()
    module.eval()
    with torch.no_grad():  # 在评估阶段，不计算梯度
        for fea_val, tar_val in test_loader:
            fea_val, tar_val = fea_val.to("cuda"), tar_val.to("cuda")
            outputs = module(fea_val)
            
            # 计算MSE损失
            loss = mse_loss(outputs, tar_val)
            total_mse_loss += loss.item() * fea_val.size(0)
            total_samples += fea_val.size(0)

    test_avg_mse_loss = total_mse_loss / total_samples
    test_loss_plot.append(test_avg_mse_loss)
    print(f'test_set_average_MSELoss: {test_avg_mse_loss:.4f}')
    return test_loss_plot, test_avg_mse_loss

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


if test1Bool:
    testData = mydataset(test_fpath, test_tpath)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=True)
    # test1(test_loader)

if test2Bool:
    trainData = mydataset(test_fpath, test_tpath)
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    test2(epoch_num)

if trainBool:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    print(f'training...{time.ctime()}')
    start = time.perf_counter()
    train1(epoch_num)