# Ahmed Abdelreheem

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, cross_val_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')
data_train = df_train.fillna(method='bfill')

df_test = pd.read_csv('../input/test.csv')
data_test = df_test.fillna(method='bfill')
data_train.shape
data_train['MSSubClass']=data_train['MSSubClass']/np.max(data_train['MSSubClass'])
data_train['LotFrontage']=data_train['LotFrontage']/np.max(data_train['LotFrontage'])
data_train['LotArea']=data_train['LotArea']/np.max(data_train['LotArea'])
data_train['OverallQual']=data_train['OverallQual']/np.max(data_train['OverallQual'])
data_train['OverallCond']=data_train['OverallCond']/np.max(data_train['OverallCond'])
data_train['YearBuilt']=data_train['YearBuilt']/np.max(data_train['YearBuilt'])
data_train['YearRemodAdd']=data_train['YearRemodAdd']/np.max(data_train['YearRemodAdd'])
data_train['MasVnrArea']=data_train['MasVnrArea']/np.max(data_train['MasVnrArea'])
data_train['BsmtFinSF1']=data_train['BsmtFinSF1']/np.max(data_train['BsmtFinSF1'])
data_train['BsmtFinSF2']=data_train['BsmtFinSF2']/np.max(data_train['BsmtFinSF2'])
data_train['BsmtUnfSF']=data_train['BsmtUnfSF']/np.max(data_train['BsmtUnfSF'])
data_train['TotalBsmtSF']=data_train['TotalBsmtSF']/np.max(data_train['TotalBsmtSF'])
data_train['1stFlrSF']=data_train['1stFlrSF']/np.max(data_train['1stFlrSF'])
data_train['2ndFlrSF']=data_train['2ndFlrSF']/np.max(data_train['2ndFlrSF'])
data_train['LowQualFinSF']=data_train['LowQualFinSF']/np.max(data_train['LowQualFinSF'])
data_train['GrLivArea']=data_train['GrLivArea']/np.max(data_train['GrLivArea'])
data_train['FullBath']=data_train['FullBath']/np.max(data_train['FullBath'])
data_train['BedroomAbvGr']=data_train['BedroomAbvGr']/np.max(data_train['BedroomAbvGr'])
data_train['KitchenAbvGr']=data_train['KitchenAbvGr']/np.max(data_train['KitchenAbvGr'])
data_train['SalePrice']=data_train['SalePrice']/np.max(data_train['SalePrice'])

data_train = pd.get_dummies(data_train)
#data_train = OneHotEncoder(data_train)

#encoded = to_categorical(data,dtype=)

xx_train = data_train.drop(axis=1,columns='SalePrice')
yy_train = data_train['SalePrice']



x_train,x_test,y_train,y_test = train_test_split(xx_train,yy_train,test_size = 0.3,random_state= 0)


xxdata_train.shape

data_test['MSSubClass']=data_test['MSSubClass']/np.max(data_test['MSSubClass'])
data_test['LotFrontage']=data_test['LotFrontage']/np.max(data_test['LotFrontage'])
data_test['LotArea']=data_test['LotArea']/np.max(data_test['LotArea'])
data_test['OverallQual']=data_test['OverallQual']/np.max(data_test['OverallQual'])
data_test['OverallCond']=data_test['OverallCond']/np.max(data_test['OverallCond'])
data_test['YearBuilt']=data_test['YearBuilt']/np.max(data_test['YearBuilt'])
data_test['YearRemodAdd']=data_test['YearRemodAdd']/np.max(data_test['YearRemodAdd'])
data_test['MasVnrArea']=data_test['MasVnrArea']/np.max(data_test['MasVnrArea'])
data_test['BsmtFinSF1']=data_test['BsmtFinSF1']/np.max(data_test['BsmtFinSF1'])
data_test['BsmtFinSF2']=data_test['BsmtFinSF2']/np.max(data_test['BsmtFinSF2'])
data_test['BsmtUnfSF']=data_test['BsmtUnfSF']/np.max(data_test['BsmtUnfSF'])
data_test['TotalBsmtSF']=data_test['TotalBsmtSF']/np.max(data_test['TotalBsmtSF'])
data_test['1stFlrSF']=data_test['1stFlrSF']/np.max(data_test['1stFlrSF'])
data_test['2ndFlrSF']=data_test['2ndFlrSF']/np.max(data_test['2ndFlrSF'])
data_test['LowQualFinSF']=data_test['LowQualFinSF']/np.max(data_test['LowQualFinSF'])
data_test['GrLivArea']=data_test['GrLivArea']/np.max(data_test['GrLivArea'])
data_test['FullBath']=data_test['FullBath']/np.max(data_test['FullBath'])
data_test['BedroomAbvGr']=data_test['BedroomAbvGr']/np.max(data_test['BedroomAbvGr'])
data_test['KitchenAbvGr']=data_test['KitchenAbvGr']/np.max(data_test['KitchenAbvGr'])

data_test = pd.get_dummies(data_test)

data_test.shape
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=289))
model.add(Dense(units=32, activation='relu', input_dim=100))
model.add(Dense(units=1, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='mse')
model.fit(x_train, y_train, epochs=5, batch_size=32)
score = model.evaluate(x_test, y_test, batch_size=128)
rmse = np.sqrt(score)

print(rmse)
fin = model.predict_classes(x_test)
submissions=pd.DataFrame(fin)
submissions.head()
