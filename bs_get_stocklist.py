"""
用途：通过baostock接口获得2023-4-18的所有股票名称，写入文件
输入：bs.query_all_stock端口
输出："./stocklist/all_stock.txt" 股票列表文件
输出格式：
code	tradeStatus	code_name
sh.600000	1	浦发银行
sh.600004	1	白云机场
sh.600006	1	东风汽车
sh.600007	1	中国国贸
sh.600008	1	首创环保
sh.600009	1	上海机场
sh.600010	1	包钢股份
sh.600011	1	华能国际
sh.600012	1	皖通高速
"""


import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

#### 获取证券信息 ####
rs = bs.query_all_stock(day="2023-09-27")
print('query_all_stock respond error_code:' + rs.error_code)
print('query_all_stock respond  error_msg:' + rs.error_msg)

#### 打印结果集 ####
data_list = []
data_list2 = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    new = rs.get_row_data()
    if new[0][0:4] == "sh.6" or new[0][0:5] == "sz.30" or new[0][0:4] == "sz.0":  # baostock的数据包含指数，需要只筛选出股票
        data_list.append(new)
    else:
        data_list2.append(new)

result = pd.DataFrame(data_list, columns=rs.fields)
result2 = pd.DataFrame(data_list2, columns=rs.fields)
#### 结果集输出到csv文件 ####
result.to_csv("./stocklist/all_stock_new.txt", sep='\t', encoding='utf-8', index=False)
result2.to_csv("./stocklist/all_stock_new_notstock.txt", sep='\t', encoding='utf-8', index=False)
#print(result)

#### 登出系统 ####
bs.logout()
