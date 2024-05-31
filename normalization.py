import os
import numpy as np


def normalize_data(data, rangelist):
    max_vals = [pair[0] for pair in rangelist]
    min_vals = [pair[1] for pair in rangelist]
    data = np.array(data)
    max_vals = np.array(max_vals)
    min_vals = np.array(min_vals)
    #print(max_vals)
    #print(np.shape(max_vals))
    #print(type(max_vals))
    normalized_data = (data - min_vals) / (max_vals - min_vals + np.finfo(float).eps)
    return normalized_data


# 循环遍历文件列表
def find_range(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        file_list += [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.startswith('features') and not file_name.endswith('_nor.npy')]
    valuerange = np.zeros((166, 2))
    for file_path in file_list:
        print("文件名: {0}".format(file_path))
        # 构建文件的完整路径
        # 判断当前路径是否为文件
        if os.path.isfile(file_path):
            data = np.load(file_path)
            shape = data.shape

            # 分割成 166 个 (10000, 10) 的子数组
            sub_arrays = np.split(data, shape[2], axis=2)
            for i in range(len(sub_arrays)):
                maxi = np.max(sub_arrays[i])
                mini = np.min(sub_arrays[i])
                #print(np.shape(sub_arrays[i]))
                valuerange[i][0] = max(maxi, valuerange[i][0])
                valuerange[i][1] = min(mini, valuerange[i][1])
                #print("166因子序号{0}存入的最大值:".format(i), valuerange[i][0])
                #print("166因子序号{0}存入的最小值:".format(i), valuerange[i][1])
        else:
            print(file_path, "不是个文件!!!")
            return
    print()
    print("最值寻找完毕")
    print()
    return valuerange


def normalize(folder_paths, rangelist):
    file_list = []
    for folder_path in folder_paths:
        folder_path = "/gpfsnyu/home/yh4202/gtja191/alphas/datasets/2023"
        file_list += [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.startswith('features') and not file_name.endswith('_nor.npy')]
    for file_path in file_list:
        print("文件名: {0}".format(file_path))
        # 构建文件的完整路径
        # 判断当前路径是否为文件
        if os.path.isfile(file_path):
            data = np.load(file_path)
            data = normalize_data(data, rangelist)
            np.save(file_path[:-4] + "_nor.npy", data)
        else:
            print(file_path, "不是个文件!!!")
            return
    print()
    print("归一化完毕")
    print()


def exam(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        file_list = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.startswith('features') and file_name.endswith('_nor.npy')]
    for file_path in file_list:
        print("文件名: {0}".format(file_path))
        data = np.load(file_path)
        print("最大值", np.max(data))
        print("最小值", np.min(data))
        print()


def main():
    #outerpath = "D:/stock/gtja191"
    outerpath = "/gpfsnyu/home/yh4202/gtja191"
    folder_paths = [#'{0}/alphas/datasets/2019'.format(outerpath),
                    #'{0}/alphas/datasets/2020'.format(outerpath),
                    #'{0}/alphas/datasets/2021'.format(outerpath),
                    #'{0}/alphas/datasets/2022'.format(outerpath),
                    '{0}/alphas/datasets/2023_bf'.format(outerpath)]
    rangelist = find_range(folder_paths)
    #print(rangelist)
    normalize(folder_paths, rangelist)
    #exam(folder_path)


main()







