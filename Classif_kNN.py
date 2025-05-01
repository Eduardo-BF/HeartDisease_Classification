# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# %%
import pickle

with open('data_heartdisease.pkl', mode='rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

X_train.shape, y_train.shape
# %%
knn_credit = KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=2)
knn_credit.fit(X_train,y_train)
# %%
pred = knn_credit.predict(X_test)
pred
#%%
print('\nAccuracy: ',accuracy_score(y_test,pred))
# %%
#%% 
model = KNeighborsClassifier(random_state=0)

# 
param_grid = {
    'n_neighbors': [15, 50, 100, 200],
    'weights' : ['uniform', 'distance'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski', 'chebyshev', 'sqeuclidean', 5],
    'p': [1, 2, 3]  
}


#
grid_search = GridSearchCV(estimator=model, param_grid=param_grid)

#
grid_search.fit(X_train, y_train)

#%%
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)

#%%
print("Best Hyperparameters:", grid_search.best_params_)
print('Accuracy:', accuracy_score(y_test, pred))
print('\n', classification_report(y_test, pred))