import pandas as pd
df=pd.read_csv(r"C:\Users\welcome\Downloads\KFC Dataset.csv")
print(df.head(10))
print(df.isnull().sum())
print(df.duplicated().sum())
#creating ml model to predict prices
#converting date time to date time format
df['Date']=pd.to_datetime(df['Date'],errors='coerce')
#extracting date time
df['hour']=df['Date'].dt.hour
df['weekday']=df['Date'].dt.weekday
df['Month']=df['Date'].dt.month
#converting ajclose to numeric
df['Adj Close']=pd.to_numeric(df['Adj Close'],errors='coerce')
#spiliting data
x=df[[ 'Open', 'High', 'Low', 'Close', 'Adj Close',  'hour',
       'weekday', 'Month']]
y=df['Volume']
#train test spilt
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#calling model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
#predict
y_pred=model.predict(x_test)
#evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))
