import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)


def get_5min_data(code):
    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    # "sh.600000"
    rs = bs.query_history_k_data_plus(code,
        "date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2018-01-01', end_date='2024-03-27',
        frequency="5", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_p++lus respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv(fr"./data_5min/{code}.csv", index=False)
    print(result)


df = pd.read_csv('./stocklist/all_stock_new.txt', sep='\t')  # 获取股票列表
print("股票总数:", len(df))
for i in range(0, len(df)):  # 遍历股票
    print(i)
    stock_code = df.iloc[i][0]
    get_5min_data(stock_code)


#### 登出系统 ####
bs.logout()