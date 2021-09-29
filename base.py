import pandas as pd
from matplotlib.pyplot import plot, show

df_train = pd.read_csv('data/b1c3ccf9-0599-4a6d-82e5-8321ff632771.csv')
df_test = pd.read_csv('data/5fda966e-4e30-4a44-a459-838592c9c665.csv')
df_train['日期'] = pd.to_datetime(df_train['日期'])
# df_test['日期'] = pd.to_datetime(df_test['日期'])

df_train['month'] = df_train['日期'].dt.month
df_train['day'] = df_train['日期'].dt.day
df_train['weekday'] = df_train['日期'].dt.weekday

# df_test['weekday'] = df_test['日期'].dt.weekday
tmp1 = df_train.loc[669:669+151]
tmp1['A厂'] = df_train['A厂'].loc[669:669+151].values
tmp1['B厂'] = df_train['B厂'].loc[669:669+151].values

tmp2 = df_train.loc[304:304+150]
tmp2['A厂'] = df_train['A厂'].loc[304:304+150].values
tmp2['B厂'] = df_train['B厂'].loc[304:304+150].values

tmp3 = df_train.loc[:89]
tmp3['A厂'] = df_train['A厂'].loc[:89].values
tmp3['B厂'] = df_train['B厂'].loc[:89].values

tmp = pd.merge(tmp1, tmp2, on=['month', 'day'], how='right')
tmp = pd.merge(tmp, tmp3, on=['month', 'day'], how='left')
tmp.to_csv('tmp.csv')

# plot(df_train['日期'], df_train['A厂'])
# plot(df_train['日期'], df_train['B厂'])
# show()

df_test['A厂'] = tmp[['A厂_x', 'A厂_y', 'A厂']].mean(axis=1).values #* 0.95
df_test['B厂'] = tmp[['B厂_x', 'B厂_y', 'B厂']].mean(axis=1).values #*1.05

# df_test.loc[:80, ['A厂', 'B厂']] = df_test.loc[:80, ['A厂', 'B厂']].values * 0.95
df_test.loc[80:160, ['A厂', 'B厂']] = df_test.loc[80:160, ['A厂', 'B厂']].values * 1.28

ans1 = pd.read_csv('ans_1.csv')
plot(df_test['A厂'])
plot(ans1['A厂'])
show()

df_test.to_csv('ans_2.csv', index=False)