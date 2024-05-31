# import efinance as ef
# import datetime
# import baostock as bs
# import numpy as np

# lg = bs.login()
# # 股票代码
# stock_code = ef.stock.get_realtime_quotes()['股票代码'].to_numpy()
# # 数据间隔时间
# freq0 = 120
# freq1 = 60
# freq2 = 30
# # 目前时间
# timeNow = datetime.datetime.now()
# timeNow = timeNow.strftime("%Y-%m-%d")
# timeNow = '2023-09-15'
# timeNow0 = timeNow + ' 11:30'
# timeNow1 = timeNow + ' 14:00'
# timeNow2 = timeNow + ' 14:30'

# yesterday = '2023-09-14 15:00'
# # 获取最新一个交易日的分钟级别股票行情数据
# j = 0
# startTime = datetime.datetime.now()
# for i in stock_code:
#     df0 = ef.stock.get_quote_history(i, klt=freq0)
#     df1 = ef.stock.get_quote_history(i, klt=freq1)
#     df2 = ef.stock.get_quote_history(i, klt=freq2)
#     df3 = ef.stock.get_quote_history(i, klt=freq0)
#     re0 = df0[df0['日期'] == timeNow0].to_numpy()
#     re1 = df1[df1['日期'] == timeNow1].to_numpy()
#     re2 = df2[df2['日期'] == timeNow2].to_numpy()
#     re3 = df3[df3['日期'] == yesterday].to_numpy()
#     if re0.size != 0 and re1.size != 0 and re2.size != 0 and re3.size != 0:
#         # 开盘3/最高5/最低6/成交量7
#         # 开盘价更新
#         re2[0][3] = re0[0][3]
#         # 最高更新
#         re2[0][5] = max(re0[0][5], re1[0][5], re2[0][5])
#         # 最低更新
#         re2[0][6] = min(re0[0][6], re1[0][6], re2[0][6])
#         # 成交量更新
#         re2[0][7] = re0[0][7] + re1[0][7] + re2[0][7]
#         # 后复权因子
#         yesterdayClose = re3[0][4]
#         todayClose = re2[0][4]
#         fac = yesterdayClose / todayClose
#         j += 1
#         print(f'{j} 名称:{re2[0][0]}||代码:{re2[0][1]}||日期:{re2[0][2]}||开盘:{re2[0][3]}||收盘:{re2[0][4]}||最高:{re2[0][5]}||最低:{re2[0][6]}||成交量:{re2[0][7]}||因子:{fac}')

# endTime = datetime.datetime.now()
# expend = endTime - startTime
# print(expend)


import os

def remove_colon_from_filenames(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名中是否包含“：”
        if ':' in filename:
            # 替换“:”为空字符串
            new_filename = filename.replace(':', '')
            # 获取完整的文件路径
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {filename} -> {new_filename}')

# 使用示例
# 请将 "your_directory_path" 替换为文件夹的实际路径
directory_path = "/home/tmp/lgr/LSTM/stock/img"
remove_colon_from_filenames(directory_path)

