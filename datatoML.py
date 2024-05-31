import numpy as np
import time
import pandas as pd


def split_array(input_array, chunk_size):
    return [input_array[i:i+chunk_size] for i in range(0, len(input_array), chunk_size)]


def format_numbers(numbers):
    formatted_numbers = []
    for num in numbers:
        num_rounded = round(num, 5)
        formatted_numbers.append(num_rounded)
    return formatted_numbers


def find_nan_indices_3d(arr):
    indices1 = np.where(np.isnan(arr))
    indices2 = np.where(np.isinf(arr))
    return list(zip(indices1[0], indices1[1], indices1[2])), \
           list(zip(indices2[0], indices2[1], indices2[2]))


def find_nan_indices_2d(arr):
    indices1 = np.where(np.isnan(arr))
    indices2 = np.where(np.isinf(arr))

    return list(zip(indices1[0], indices1[1])), list(zip(indices2[0], indices2[1]))


def get_data(part, sequence_length, file_path, year):  # 获取数据集

    dataset = []
    targetset = []
    con = sequence_length
    tot = 0
    #serial = 0
    n = len(part)
    for name in part:  # 遍历股票
        #print("获取数据进度: ", serial / n)
        #print(name)
        #serial += 1
        new_filename = "{0}/alphas/stocks/{1}/{2}.csv".format(file_path, year, name)
        df = pd.read_csv(new_filename)
        # data = []
        # with open(new_filename, 'r') as file:
        # reader = csv.reader(file)
        # for row in reader:
        # data.append(row)
        # print(df.columns)
        cnt = 0
        for i in range(1, len(df.columns)):  # 遍历日期
            traindata = []  # 每个样本
            # print(df.columns[i])
            for forwarddate in df.columns[i:i + con]:  # 遍历某日的接下来con日
                fl = 0
                faclis = df.loc[:, forwarddate].tolist()[:-1]
                for fac in faclis:  # 对每日检测所有因子是否有效， 注意最后一行是涨跌幅，不取
                    # print(df.loc[:, forwarddate])
                    if np.isnan(fac) or np.isinf(fac):  # 有因子无效则跳出循环
                        # nonfactor.add(df.iloc[i, 0])
                        fl = 1
                        # print("不行")
                        break
                if fl:
                    break
                    # print("不完整日期")
                # print("完整日期")
                else:
                    faclis = format_numbers(faclis)  # 保留5位小数
                    traindata.append(faclis)
                # print(faclis)
            # print(len(traindata))

            if len(traindata) == con:  # 如果con日都有效
                # print(traindata)
                try:
                    if not np.isnan(df.iloc[-1, i + con]):  # 不是空值，假定没有Inf
                        targetset.append(round(df.iloc[-1, i + con], 5))  # 第n+1天的涨幅，保留5位小数
                        dataset.append(traindata)
                except:
                    pass  # 出错说明到日期头了，最后10天找不到第11天，跳过即可
                cnt += 1
                # print(cnt)
                continue
        tot += cnt

    print("平均有效样本量:", tot / n)
    #print("样本量:", len(dataset))
    #print("跟踪长度:", len(dataset[0]))
    #print("特征数:", len(dataset[0][0]))
    #print("样本数2:", len(targetset))
    targetset_2d = [[x] for x in targetset]
    return dataset, targetset_2d


def normalize_data(data):
    data = np.array(data)
    min_vals = np.min(data, axis=(0, 1))
    max_vals = np.max(data, axis=(0, 1))
    normalized_data = (data - min_vals) / (max_vals - min_vals + np.finfo(float).eps)
    return normalized_data


def main():
    year = "2023"
    file_path = "/gpfsnyu/home/yh4202/gtja191"
    #file_path = "D:/stock/gtja191"
    # 生成随机样本数据
    # num_samples = 1000
    # num_features = 166
    sequence_length = 10

    start_time = time.time()

    # features = np.load(r"D:\stock\datasets\2019complete_normalized_features.npy")
    # targets = np.load(r"D:\stock\datasets\2019complete_targets.npy")

    f = open("{0}/alphas/stocks/{1}/namelist.txt".format(file_path, year), "r")
    namelist = f.readlines()
    for i in range(len(namelist)):
        namelist[i] = namelist[i][0:6]
    f.close()
    n = len(namelist)  # 原始数组的长度
    chunk_size = 500  # 每个子数组的长度

    split_arrays = split_array(namelist, chunk_size)
    namelist = split_arrays

    for i in range(len(namelist)):
        print("步骤{0}/{1}".format(i+1, len(namelist)))
        features, targets = get_data(namelist[i], sequence_length, file_path, year)
        print("数据已获取")
        print(np.shape(features))
        print(np.shape(targets))
        #features = normalize_data(features)
        np.save("{0}/alphas/datasets/{1}/features{2}".format(file_path, year, i+1), features)
        np.save("{0}/alphas/datasets/{1}/targets{2}".format(file_path, year, i+1), targets)
        nan_indices, inf_indices = find_nan_indices_3d(features)
        for idx in nan_indices:
            print(f"NaN value found at features position: {idx}")
        for idx in inf_indices:
            print(f"Inf value found at features position: {idx}")
        nan_indices, inf_indices = find_nan_indices_2d(targets)
        for idx in nan_indices:
            print(f"NaN value found at targets position: {idx}")
        for idx in inf_indices:
            print(f"Inf value found at targets position: {idx}")
        num_samples = len(features)
        sequence_length = len(features[0])
        num_features = len(features[0][0])

        print("特征集样本数：", num_samples)
        print("序列数：", sequence_length)
        print("特征数：", num_features)
        print("目标集样本数：", len(targets))
        print("目标集二维长度：", len(targets[0]))

    end_time = time.time()
    run_time = end_time - start_time
    print(f"把数据转换成ml可读二进制文件用时: {run_time:.2f} 秒")


main()
