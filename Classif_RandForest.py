# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# %%
import pickle

with open('data_heartdisease_bin.pkl', mode='rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

X_train.shape, y_train.shape

#%% 
model = RandomForestClassifier(random_state=0)

# 
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3]  
}


#
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)

#
grid_search.fit(X_train, y_train)

#%%
best_model = grid_search.best_estimator_
pred = best_model.predict(X_test)

#%%
print("Best Hyperparameters:", grid_search.best_params_)
print('Accuracy:', accuracy_score(y_test, pred))
print('\n', classification_report(y_test, pred))
# %% 

randfores = RandomForestClassifier(
    random_state=0,
    criterion='entropy',
    max_depth= 5,
    min_samples_leaf= 3,
    min_samples_split= 2,
    n_estimators= 200 
    )
randfores.fit(X_train,y_train)
pred = randfores.predict(X_test)

print('Accuracy:', accuracy_score(y_test, pred))
print('\n', classification_report(y_test, pred))

# %%
