import os
import pandas as pd
from datetime import datetime
import sys
from multiprocessing import Pool
import shutil

cnt = 0
errcnt = 0
def merging(filename):
    global cnt
    global errcnt
    if filename.endswith('.csv'):  # 确保文件是 CSV 文件
        try:
            file_path = os.path.join(day_path, filename)  # 获取完整文件路径
            day_data = pd.read_csv(file_path)
            file_path = os.path.join(min_path, filename)
            min_data = pd.read_csv(file_path, dtype={1: str, 2: str})
        except Exception as e:
            print(f"{filename}无法读取")
            return
    filename = filename[:-4]
    # 使用 iterrows() 扫描 DataFrame 并打印日期列的值
    day_idx = 0
    min_idx = 0
    #stop_idx = len(day_data)-1
    if day_data.loc[0, '日期'] != min_data.loc[0, 'date']:
        print(f"{filename}初始日期不同，跳过")
    while(day_idx < len(day_data)-1):
        #print(min_idx)
        #day_t0 = day_data.loc[day_idx, '日期']  # t0天的日线日期
        #min_t0 = min_data.loc[min_idx, "date"]  # t0天的分钟日期
        day_t1 = day_data.loc[day_idx+1, '日期']  # t1天的日线日期
        min_t1 = min_data.loc[min_idx+48, "date"]  # t1天的分钟日期
        #date = min_row['date']  # 这是分钟数据日期，假设只有分数数据会出现缺失，所以让日线数据跳过 分钟数据缺失的 数据
        if day_t1 != min_t1:  # 数据不匹配
            errcnt += 1
            print(f"{filename}数据存在不匹配!")
            date1 = datetime.strptime(day_t1, "%Y-%m-%d")
            print(day_t1)
            date2 = datetime.strptime(min_t1, "%Y-%m-%d")
            fl = 0
            if date1 < date2:  # 说明分钟数据缺失
                min_idx += 48  # 注意前面检测的是t1数据出现不匹配，所以此时t0和t1肯定跨在缺失数据上，所以要往后一天
                for j in range(2,20):  # 模糊匹配 j=1的时候t0肯定是那个日线多出来的日期，所以从j=2开始
                    try:  # 如果数据缺失出现在数据末尾，可能会有超索引范围的问题，用try模块处理
                        if day_data.loc[day_idx + j, '日期'] == min_data.loc[min_idx, "date"] \
                        and day_data.loc[day_idx + j + 1, '日期'] == min_data.loc[min_idx+48, "date"]:  # 能重新匹配成功
                            #day_data = day_data[:day_idx+1].append(day_data[day_idx+j:])  # 删除不匹配的日线区间
                            #print(j)
                            day_data = day_data.drop(day_data.index[day_idx+1:day_idx+j]).reset_index(drop=True) # 删除日线多余的数据
                            #day_data.to_csv(f'em_merge/{filename}.csv', index=False)
                            
                            day_idx += 1  #  day_idx在多余数据前，也要往后一天
                            fl = 1
                            break
                    except Exception as e:
                        print(f"{filename}数据缺失出现在末尾，跳过")
                        break
            else:
                print(f"{filename}日线数据出现缺失!，跳过")
                break
            if not fl:
                print(f"{filename}模糊重新匹配失败!，跳过")
                break
        

        ###  数据缺失处理与数据替换分割线  ###
        
        #if day_data.loc[day_idx+1, "最高"] == min_data['high'][min_idx+94:min_idx+96].max():  一种可能的算法优化
        #    day_data.loc[day_idx+1, "最高"] = min_data['high'][min_idx+48:min_idx+94].max()  # t1 0930-1450的最高 48+45=93, 注意左闭右开区间

        
        day_data.loc[day_idx+1, "最高"] = min_data['high'][min_idx+47:min_idx+94].max()  # t1 1455-1450的最高 48+45=93, 注意左闭右开区间
        day_data.loc[day_idx+1, "最低"] = min_data['low'][min_idx+47:min_idx+94].min()
        volume_1450 = min_data['volume'][min_idx+47:min_idx+94].sum()  # 1455-1450
        volume_close = volume_1450 + min_data['volume'][min_idx+94:min_idx+96].sum()
        day_data.loc[day_idx+1, "换手率"] = round(day_data.loc[day_idx+1, "换手率"] * (volume_1450 / volume_close),2)  # 需要检查下有没有未来信息引入
        day_data.loc[day_idx+1, "成交量"] = round(volume_1450 / 100, 0)
        day_data.loc[day_idx+1, "成交额"] = round(min_data['amount'][min_idx+47:min_idx+94].sum(), 0)
        day_data.loc[day_idx+1, "开盘"] = min_data.loc[min_idx+46, "close"]  # t1 昨日1450的收盘
        day_data.loc[day_idx+1, "收盘"] = min_data.loc[min_idx+93, "close"]  # t1 1450的收盘
        day_data.loc[day_idx+1, "涨跌幅"] = round((float(min_data.loc[min_idx+93, "close"]) / float(min_data.loc[min_idx+46, "close"]) - 1) * 100, 2)
        #今日1450收盘除以昨日1455收盘
        
        day_idx += 1
        min_idx += 48

    #day_data = day_data.drop(day_data.index[day_idx:])  # 扔掉末尾日期
    day_data = day_data.iloc[1:].reset_index(drop=True)  # 扔掉第一个日期
    day_data.to_csv(f'{res_path}/{filename}.csv', index=False)

    cnt+=1

def merging2(filename):
    print("1")
    global cnt
    global errcnt
    if filename.endswith('.csv'):  # 确保文件是 CSV 文件
        try:
            file_path = os.path.join(day_path, filename)  # 获取完整文件路径
            day_data = pd.read_csv(file_path)
            file_path = os.path.join(min_path, filename)
            min_data = pd.read_csv(file_path)
        except Exception as e:
            print(f"{filename}无法读取")
            return
    filename = filename[:-4]
    # 使用 iterrows() 扫描 DataFrame 并打印日期列的值
    day_idx = 0
    min_idx = 0
    #stop_idx = len(day_data)-1
    if day_data.loc[0, '日期'] != min_data.loc[0, 'date']:
        print(f"{filename}初始日期不同，跳过")
    while(day_idx < len(day_data)):
        #print(min_idx)
        #day_t0 = day_data.loc[day_idx, '日期']  # t0天的日线日期
        #min_t0 = min_data.loc[min_idx, "date"]  # t0天的分钟日期
        day_t1 = day_data.loc[day_idx+1, '日期']  # t1天的日线日期
        min_t1 = min_data.loc[min_idx+48, "date"]  # t1天的分钟日期
        #date = min_row['date']  # 这是分钟数据日期，假设只有分数数据会出现缺失，所以让日线数据跳过 分钟数据缺失的 数据
        if day_t1 != min_t1:  # 数据不匹配
            errcnt += 1
            print(f"{filename}数据存在不匹配!")
            date1 = datetime.strptime(day_t1, "%Y-%m-%d")
            print(day_t1)
            date2 = datetime.strptime(min_t1, "%Y-%m-%d")
            fl = 0
            if date1 < date2:  # 说明分钟数据缺失
                min_idx += 48  # 注意前面检测的是t1数据出现不匹配，所以此时t0和t1肯定跨在缺失数据上，所以要往后一天
                for j in range(2,20):  # 模糊匹配 j=1的时候t0肯定是那个日线多出来的日期，所以从j=2开始
                    try:  # 如果数据缺失出现在数据末尾，可能会有超索引范围的问题，用try模块处理
                        if day_data.loc[day_idx + j, '日期'] == min_data.loc[min_idx, "date"] \
                        and day_data.loc[day_idx + j + 1, '日期'] == min_data.loc[min_idx+48, "date"]:  # 能重新匹配成功
                            #day_data = day_data[:day_idx+1].append(day_data[day_idx+j:])  # 删除不匹配的日线区间
                            #print(j)
                            day_data = day_data.drop(day_data.index[day_idx+1:day_idx+j]).reset_index(drop=True) # 删除日线多余的数据
                            #day_data.to_csv(f'em_merge/{filename}.csv', index=False)
                            
                            day_idx += 1  #  day_idx在多余数据前，也要往后一天
                            fl = 1
                            break
                    except Exception as e:
                        print(f"{filename}数据缺失出现在末尾，跳过")
                        break
            else:
                print(f"{filename}日线数据出现缺失!，跳过")
                break
            if not fl:
                print(f"{filename}模糊重新匹配失败!，跳过")
                break
        

        ###  数据缺失处理与数据替换分割线  ###
        
        #if day_data.loc[day_idx+1, "最高"] == min_data['high'][min_idx+94:min_idx+96].max():  一种可能的算法优化
        #    day_data.loc[day_idx+1, "最高"] = min_data['high'][min_idx+48:min_idx+94].max()  # t1 0930-1450的最高 48+45=93, 注意左闭右开区间

        
        day_data.loc[day_idx+1, "最高"] = min_data['high'][min_idx+47:min_idx+94].max()  # t1 1455-1450的最高 48+45=93, 注意左闭右开区间
        day_data.loc[day_idx+1, "最低"] = min_data['low'][min_idx+47:min_idx+94].min()
        volume_1450 = min_data['volume'][min_idx+47:min_idx+94].sum()  # 1455-1450
        volume_close = volume_1450 + min_data['volume'][min_idx+94:min_idx+96].sum()
        day_data.loc[day_idx+1, "换手率"] = round(day_data.loc[day_idx+1, "换手率"] * (volume_1450 / volume_close),2)  # 需要检查下有没有未来信息引入
        day_data.loc[day_idx+1, "成交量"] = round(volume_1450 / 100, 0)
        day_data.loc[day_idx+1, "成交额"] = round(min_data['amount'][min_idx+47:min_idx+94].sum(), 0)
        day_data.loc[day_idx+1, "开盘"] = min_data.loc[min_idx+46, "close"]  # t1 昨日1450的收盘
        day_data.loc[day_idx+1, "收盘"] = min_data.loc[min_idx+93, "close"]  # t1 1450的收盘
        day_data.loc[day_idx+1, "涨跌幅"] = round((min_data.loc[min_idx+93, "close"] / min_data.loc[min_idx+46, "close"] - 1) * 100, 2)
        #今日1450收盘除以昨日1455收盘
        
        day_idx += 1
        min_idx += 48

    day_data = day_data.drop(day_data.index[day_idx:])  # 扔掉末尾日期
    day_data = day_data.iloc[1:].reset_index(drop=True)  # 扔掉第一个日期
    day_data.to_csv(f'em_merge/{filename}.csv', index=False)

    cnt+=1


# 指定包含 CSV 文件的文件夹路径
day_path = 'day_data'
min_path = '5min_data'

res_path = "em_merge"  # 删除操作！！！注意路径不要写错

# 删除文件夹
shutil.rmtree(res_path)  # 删除操作！！！注意路径不要写错
os.mkdir(res_path)


for file_name in os.listdir(day_path):  # 遍历股票
#for file_name in ["000017.csv"]:  # 遍历股票
    try:
        merging(file_name)
    except Exception as e:
        print(f"{file_name}函数运算中出错！！！")

'''
count = os.cpu_count()
pool = Pool(count)
for file_name in os.listdir(day_path):  # 遍历股票
    try:
        pool.apply_async(merging, (file_name,))
    except Exception as e:
        print(f"{file_name}函数运算中出错！！！")



#filename = "003008.csv"
#merging(filename)
pool.close()
pool.join()
'''
print(cnt)
print("数据不匹配的数量:", errcnt)
