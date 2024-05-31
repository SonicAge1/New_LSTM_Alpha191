import pandas as pd
import math
import csv
import os
import numpy as np
import time
from multiprocessing import Pool

# badalpha = set()
seta = {33, 42, 45, 121, 179, 184}  # 数据量少
setb = {30, 143, 149, 190}  # 数据不能制作
setfile = {21, 25, 30, 33, 35, 42, 45, 55, 121, 127, 143, 149, 166, 179, 180, 181, 184}  # 文档内写的有问题的
# setc = {2, 3, 4, 130, 5, 138, 10, 12, 11, 17, 27, 36, 165, 40, 44, 60, 64, 77, 85, 87, 119}  # 跑出来有问题的
# setd = {37, 6, 7, 171, 14, 46, 20, 52, 117, 94, 31}
setcheck = {64, 138, 87, 25, 33, 113, 171, 165, 166, 181, 85}  # 空值和inf率达到3%的

# badalpha = seta | setb | setc | setd | sete
badalpha = seta | setb | setfile | setcheck
print("无效因子数量：", len(badalpha))


# {64, 33, 101, 38, 103, 154, 138, 117, 87, 56, 25, 26, 123, 184}

def process_part(colname, dates, st_idx, ed_idx):
    global year
    global cnt
    print(colname, cnt)
    cnt += 1
    new_filename = "/gpfsnyu/home/yh4202/gtja191/alphas/stocks/{0}/{1}.csv".format(year, colname)  # 处理完数据输出的文件位置
    # new_filename = r'D:\stock\gtja191\a' + "lphas\stocks" + r"\201" + r"9\000415.csv"  # 处理完数据输出的文件位置
    data = [dates]
    for i in range(1, 192):  # 因子遍历
        if i in badalpha:  # 剔除数据量小的alpha
            continue
        try:
            filename = "/gpfsnyu/home/yh4202/gtja191/alphas/alphas/Alphas191/{0}/alpha{1:03d}.csv".format(year, i)  # 因子数据
            df = pd.read_csv(filename)
        except Exception as e:
            print(e)
            print("!!!!")
            print(i)
            continue
        alpha = [i]
        alpha.extend(list(df.loc[st_idx:ed_idx, colname]))
        # print(len(list(df.loc[st_idx:ed_idx, colname])))
        data.append(alpha)

    try:
        filename = "/gpfsnyu/home/yh4202/gtja191/alphas/data_hfq/{0}.csv".format(colname)  # 股票数据(涨跌幅)
        df = pd.read_csv(filename)
        chg = ["chg"]
    except Exception as e:
        print(e)
        print("!!!!")
        print(i)
        return
    date_series = df['日期']
    for date in dates[1:]:
        try:
            positions = date_series.str.contains(date).tolist()
            idx = positions.index(True)
            chg.append(df.iloc[idx, 9])
        except Exception as e:
            #print(e)
            #print("找不到对应日期！！")
            chg.append(np.nan)
            # print(i)
            continue
    data.append(chg)

    filename = "/gpfsnyu/home/yh4202/gtja191/alphas/data_hfq/{0}.csv".format(colname)

    f = open(new_filename, 'w', newline='')
    writer = csv.writer(f)
    # del_name(data)  # 启用此行则删除了行名，即191个因子
    writer.writerows(data)
    f.close()


def del_name(data):
    for line in data:
        line.pop(0)


cnt = 0

def process_data():
    global year
    alphanumbers = 191
    firstname = '/gpfsnyu/home/yh4202/gtja191/alphas/alphas/Alphas191/{0}/alpha001.csv'.format(year)  # 先拿沪深300的股票代码
    hs300 = pd.read_csv(firstname)
    #st_idx = 242  # 2019年第一个交易日的行数
    st_idx = 329
    #ed_idx = 329
    ed_idx = 10000
    dates = list(hs300.loc[st_idx:ed_idx, 'date'])
    # print(type(dates))
    dates.insert(0, -1)  # 这里要留出，因为第0列下面是Alpha因子的名字

    count = os.cpu_count()
    pool = Pool(count)

    for colname in hs300.columns[1:]:  # 遍历股票代码列表
        # colname = "000415"

        pool.apply_async(process_part, (colname, dates, st_idx, ed_idx))

    pool.close()
    pool.join()

'''
def get_namelist():
    global year
    # 定义文件夹路径
    folder_path = "/gpfsnyu/home/yh4202/gtja191/alphas/stocks/{0}".format(year)

    # 获取文件夹中的所有文件名
    namelist_path = "/gpfsnyu/home/yh4202/gtja191/alphas/stocks/{0}/namelist.txt".format(year)
    if os.path.exists(namelist_path):
        # 删除文件
        os.remove(namelist_path)
    file_names = os.listdir(folder_path)

    f = open("/gpfsnyu/home/yh4202/gtja191/alphas/stocks/{0}/namelist.txt".format(year), "w")
    for file_name in file_names:
        f.write(file_name + "\n")
'''


start_time = time.time()

year = "2023"

process_data()

#get_namelist()

end_time = time.time()
run_time = end_time - start_time
print(f"数据生成运行时间：{run_time:.2f} 秒")

# 要做的事情，暂且不考虑窗口，遍历每个时间点，如果其未来10日数据齐全，生成一个数据集
# 确认因公式问题不存在的因子在所有股票上是一致的，确认数据量较少的因子，用集合筛选掉
# 可能的问题：检查日期本身是否是完整的，日期来源是股票市场还是某个股票的数据