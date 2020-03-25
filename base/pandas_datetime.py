"""
简介:
  pandas根据时间范围处理数据
------------------------------
测试环境:
  python: 3.7.3
  pandas: 0.24.2
"""
import pandas as pd
import datetime
df = pd.DataFrame(
  {
    'Symbol':['A','A','A', 'B', 'B', 'C', 'C', 'C'] ,
    'Msg':['AAA','AAA','AAC', 'BBB', 'BBC', 'CCC', 'CCC', 'CCD'] ,
    'Date':['02/20/2015','01/15/2016','02/21/2015', '02/24/2015','03/01/2015', '02/22/2015','01/17/2015','03/21/2015']
  }
)
print(df)
df['Date'] =pd.to_datetime(df.Date)
df = df.sort_values(by='Date', ascending=True)
df.index = df['Date']

start = datetime.datetime.strptime(str(df['Date'][0]), '%Y-%m-%d %H:%M:%S')
end = start + datetime.timedelta(days=7)
print(df[start: end])

import math
delta = 7                                             # 处理的时间范围
day_num = (df['Date'].max() - df['Date'].min()).days  # 数据集的时间跨度
loop_num = math.ceil(day_num / delta)                 # 计算循环次数
start = datetime.datetime.strptime(str(df['Date'][0]), '%Y-%m-%d %H:%M:%S')
end = start + datetime.timedelta(days=delta)
for _ in range(loop_num):
	df_period = df[start: end]
	# print('处理%s至%s' % (start, end))
	start = end
	end += datetime.timedelta(days=delta)


df = pd.DataFrame(
  {
    'Symbol':['A','A','B', 'C'] ,
    'Msg':['AAA', 'AAC', 'CCC', 'CCD'] ,
    'Date':['2019-12-04 15:16:41','2019-12-04 15:16:44','2019-12-04 15:16:25', '2019-12-04 15:16:16']
  }
)
df['Date'] =pd.to_datetime(df.Date)
df = df.sort_values(by='Date', ascending=True)
df.index = df['Date']
start = datetime.datetime.strptime(str(df['Date'][0]), '%Y-%m-%d %H:%M:%S')
end = start + datetime.timedelta(seconds=10)            # 除了以天为单位外, 还支持秒和星期等等
print(df[start: end])
