from fredapi import Fred
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

api_key='YOURAPIKEY' ###key to lots of data at https://research.stlouisfed.org/
fred = Fred(api_key=api_key)

wealth={
        'Bottom 50%':'WFRBLB50107',
        '50th to 90th%':'WFRBLN40080',
        '90th to 99th%':'WFRBLN09053',
        '99th to 100th%':'WFRBLT01026',
#         'CPI Urban':'CPIAUCSL',
        }

df_wealth=pd.DataFrame()
for key in wealth:
    try:
        df=fred.get_series(wealth[key])
        df.index=pd.to_datetime(df.index)
        df=df.rename(key).to_frame()    
        if list(wealth.keys()).index(key)==0:
            df_wealth=df
        else:
            df_wealth=pd.merge_asof(df_wealth, df, left_index=True, right_index=True, direction='nearest',tolerance=pd.Timedelta('1M'),)
        ###querry delay to avoid server block
        delay=np.random.rand(1)*10
        print(key, delay)
        time.sleep(delay)
    except:
        pass
    
fig = plt.Figure()
ax = fig.add_subplot(111)
df_wealth.plot.area()

###other forms of the same data: ratio and growth change
# df_wealth.div(df_wealth.sum(axis=1), axis=0).plot.area() #takes percentage of a total for a given period
# df_wealth.pct_change()*100

ax = plt.gca()
ax.relim()
ax.margins(x=0)
plt.ylabel('Millions of Dollars', fontsize=16)
plt.xlabel('Time quaterly', fontsize=16)
plt.title('Total Net Worth Held by the U.S. Population', fontsize=14, loc='left')
plt.xticks(fontsize = 6)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(fontsize=10)
plt.tight_layout()
plt.grid()
plt.savefig('wealth_by_percentile_stacked_TEST.png', dpi=600)
plt.show()
