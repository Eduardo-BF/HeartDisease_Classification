#%% Import Dataset from: https://archive.ics.uci.edu/dataset/45/heart+disease
from ucimlrepo import fetch_ucirepo 
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
# metadata 
print(heart_disease.metadata)   
# variable information 
print(heart_disease.variables) 

#%%
import pandas as pd
import plotly.express as px

#%%
X.head()
# %%
y.head()

# %% Data visualization 
x_plot = pd.concat([X, y], axis=1)
x_plot = x_plot.rename(columns={'num': 'heart disease'})
x_plot['sex'] = x_plot['sex'].replace({0: 'F', 1: 'M'})
x_plot.head()
#%%
fig = px.scatter(
    x_plot,
    x='age',
    y='heart disease',
    color='ca',
    facet_row='sex',
    width=600,   # largura em pixels
    height=400    # altura em pixels
)
fig.show()
#%%
print('Female',len(x_plot.loc[x_plot['sex']== 'F']))
print('Male',len(x_plot.loc[x_plot['sex']== 'M']))

#%% Verify outliers
X.loc[(X['age']>100) | (X['age']<1)]

#%% Missing values: replace with most frequent
X.loc[(X['ca'].isna())] = 0
X.loc[(X['thal'].isna())] = 3

#%% OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# %%
onehotencoder_hd= ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,2,5,6,8,10,11,12] )], remainder='passthrough' )
# %%
X = onehotencoder_hd.fit_transform(X)
X[0:5]

# %% Standardization
from sklearn.preprocessing import StandardScaler
scaler_hd = StandardScaler()
X = scaler_hd.fit_transform(X)
X[0]

# %% Data division
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(f"Training set:   X = {X_train.shape}, y = {y_train.shape}")
print(f"Test set:       X = {X_test.shape},  y = {y_test.shape}")
 
#%%
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()
# %% Save data
import pickle

with open('data_heartdisease.pkl', mode='wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)


# %% --------- Dataset for binary problem ---------
# 0: no diagnosis, 1: one or more diagnoses
yb = y.where(y == 0, 1)
print(yb.value_counts())

# %% Data division
from sklearn.model_selection import train_test_split

X_train, X_test, yb_train, yb_test = train_test_split(X, yb, test_size=0.25, random_state=0)

print(f"Training set:   X = {X_train.shape}, y = {yb_train.shape}")
print(f"Test set:       X = {X_test.shape},  y = {yb_test.shape}")

#%%
yb_train = yb_train.to_numpy().ravel()
yb_test = yb_test.to_numpy().ravel()
# %% Save data

with open('data_heartdisease_bin.pkl', mode='wb') as f:
    pickle.dump([X_train, X_test, yb_train, yb_test], f)
