import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

data = {
'sqfootage':[900,1200,1500,1800,2000,2200,1600,1400,1700,2100],
'bedroom':[2,3,3,3,4,4,3,2,3,4],
'bathroom':[1,2,2,2,3,3,2,1,2,3],
'location':['rural','rural','suburb','suburb','city','city','suburb','rural','suburb','city'],
'price':[3200000,4200000,7800000,9200000,18500000,21000000,8500000,3500000,9800000,22500000]
}

df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=['location'])

X = df.drop("price",axis=1)
y = df["price"]

model = GradientBoostingRegressor()

model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))