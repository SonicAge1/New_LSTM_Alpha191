import akshare as ak
import os
import sys


st_date = '20180101'
ed_date = "20240327"
#st_date = '20210101'  # 仅用于实际操作而非全周期回测

cnt = 0
em_df = ak.stock_zh_a_spot_em()
names = em_df["名称"]
codes = em_df["代码"]
st_cnt = 0
bj_cnt = 0
for i in range(len(codes)):
    if codes[i][0] == "8":
        bj_cnt += 1
        continue
    #if "ST" in names[i] or "退" in names[i]:
        #st_cnt += 1
        #continue
    cnt += 1
    if cnt % 500 == 0:
        print(cnt)
    #df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date='20180101', adjust="hfq")  # 东财
    #df.to_csv(f'em_hfq/{code}.csv')
    df = ak.stock_zh_a_hist(symbol=codes[i], period="daily", start_date=st_date, end_date=ed_date, adjust="")  # 东财
    df.to_csv(f'day_data/{codes[i]}.csv', index=False)  # actual procedure

#print(f"清除北交所股票{bj_cnt}只")
#print(f"清除st或退市股票{st_cnt}只")
path = 'index'
code = "sh000300"
stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=code)
# 创建保存路径
if not os.path.isdir(path):
    os.makedirs(path)
stock_zh_index_daily_df.to_csv(f'{path}/{code}.csv', index=False)


