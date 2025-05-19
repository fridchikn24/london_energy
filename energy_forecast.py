import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


df = pd.read_csv("london_energy.csv")
print(df.isna().sum())

#'''aggregates and averages the consumption of energy on each date'''

df_avg_consumption = df.groupby("Date")['KWH'].mean()
df_avg_consumption = pd.DataFrame({"date": df_avg_consumption.index.tolist(), 
                                   "consumption": df_avg_consumption.values.tolist()})
df_avg_consumption["date"] = pd.to_datetime(df_avg_consumption["date"])

print(df_avg_consumption.head())

df_avg_consumption.plot(x='date', y = 'consumption')

df_avg_consumption.query("date > '2012-01-01' & date < '2013-01-01'").plot(x="date", y="consumption")

#'''add in specifics of date time as variables'''

df_avg_consumption["day_of_week"] = df_avg_consumption["date"].dt.dayofweek
df_avg_consumption["day_of_year"] = df_avg_consumption["date"].dt.dayofyear
df_avg_consumption["month"] = df_avg_consumption["date"].dt.month
df_avg_consumption["quarter"] = df_avg_consumption["date"].dt.quarter
df_avg_consumption["year"] = df_avg_consumption["date"].dt.year

print(df_avg_consumption.head())


df_avg_consumption = df_avg_consumption.drop(columns=['date'])#drop date column as it is now redundant

y = df_avg_consumption['consumption']
X = df_avg_consumption.drop(columns = ['consumption'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#'''train xgboost model'''



cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
model = XGBRegressor()
parameters = {
    "max_depth": [3, 4, 6, 5, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": [100, 300, 500, 700, 900, 1000],
    "colsample_bytree": [0.3, 0.5, 0.7]
}


grid_search = GridSearchCV(estimator=model, cv=cv_split, param_grid=parameters)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R² score on test data: {r2:.4f}")




import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal: y = x')

plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()


