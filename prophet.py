from fbprophet import Prophet
import pandas as pd
from matplotlib.pyplot import plot, show, figure
from chinese_calendar import is_holiday
import time
df_train = pd.read_csv('data/train.csv')[200:].reset_index()
df_test = pd.read_csv('data/test.csv')

# playoffs = pd.DataFrame({
#   'holiday': 'playoff',
#   'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
#                         '2010-01-24', '2010-02-07', '2011-01-08',
#                         '2013-01-12', '2014-01-12', '2014-01-19',
#                         '2014-02-02', '2015-01-11', '2016-01-17',
#                         '2016-01-24', '2016-02-07']),
#   'lower_window': 0,
#   'upper_window': 1,
# })
# superbowls = pd.DataFrame({
#   'holiday': 'superbowl',
#   'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
#   'lower_window': 0,
#   'upper_window': 1,
# })
# holidays = pd.concat((playoffs, superbowls))

df_train['日期'] = pd.to_datetime(df_train['日期'])
df_train['a+b'] = df_train['A厂'] + df_train['B厂']
for name in ['A厂', 'B厂']:
    tmp = df_train[['日期', name]].copy()
    tmp.columns = ['ds', 'y']
    # holi = tmp.copy()
    # holi['is_holiday'] = holi['ds'].map(is_holiday)
    # holi = holi[holi['is_holiday']==True].reset_index()
    # holi['holiday'] = 'holiday'
    # print(tmp)
    model = Prophet(n_changepoints=20,
                    seasonality_prior_scale=60,
                    # changepoint_range=0.4,
                    changepoint_prior_scale=0.1,
                    seasonality_mode='multiplicative')
    # model = Prophet(seasonality_mode='multiplicative')
    model.fit(tmp)
    future = model.make_future_dataframe(periods=151)
    forecast = model.predict(future)
    # print(forecast)
    plot(forecast['yhat'])
    plot(tmp['y'])
    show()
    df_test[name] = forecast['yhat'].values[-151:]
    # model.plot_components(forecast)
    # show()
ans1 = pd.read_csv('ans/ans_prophet_202105062315.csv')
ans2 = pd.read_csv('ans_1.csv')
figure(1)
plot(df_test['A厂'])
plot(ans1['A厂'])
plot(ans2['A厂'])
figure(2)
plot(df_test['B厂'])
plot(ans1['B厂'])
plot(ans2['B厂'])
show()
df_test.to_csv(time.strftime('ans/ans_prophet_%Y%m%d%H%M.csv'), index=False)