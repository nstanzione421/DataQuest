import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns


path = '/Users/nstanzione/Documents/EDU/DataQuest/Data/'
file = 'sphist.csv'

data = pd.read_csv(path+file)
data['Date'] = pd.to_datetime(data['Date'], yearfirst=True)
data.sort_values('Date', inplace=True)

data['FiveDayAvg'] = data['Close'].rolling(5).mean().shift()
data['ThirtyDayAvg'] = data['Close'].rolling(30).mean().shift()
data['TwoHundredDayAvg'] = data['Close'].rolling(200).mean().shift()
data['FiveDayStDev'] = data['Close'].rolling(5).std().shift()
data['ThirtyDayStDev'] = data['Close'].rolling(30).std().shift()
data['TwoHundredDayStDev'] = data['Close'].rolling(200).std().shift()
data['Year'] = data['Date'].apply(lambda x: x.year)
data['TwoDayVolAvg'] = data['Close'].rolling(2).mean().shift()
data['PriorDay'] = data['Close'].shift()

data.dropna(inplace=True)

prep = data[data['Date']<'2013-01-01']
test = data[data['Date']>'2012-12-31']

lr = LinearRegression()

train = prep.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'],axis=1)
features = ['PriorDay','FiveDayStDev','Year','TwoDayVolAvg']
lr.fit(train[features],train['Close']) 
predictions = lr.predict(test[features])
test['predictions'] = predictions

error = mae(predictions, test['Close'])


print(data.info())

print(error)
print(test.loc[:,['Close','predictions']])

corr = data.corr()

print(corr)
