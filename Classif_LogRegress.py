# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# %%
import pickle

with open('data_heartdisease.pkl', mode='rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

X_train.shape, y_train.shape

#%%
X_train[0]

#%% Best configuration finder

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ['saga','sag','liblinear'],
    "penalty": ['l1', 'l2', 'elasticnet'],
    "l1_ratio": [0.0, 0.5, 1.0], 
    "max_iter": [500,1000,5000],
    "fit_intercept": [True, False]
}


logreg = LogisticRegression()
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best config:", grid.best_params_)

y_pred = best_model.predict(X_test)
print('\nAccuracy: ',accuracy_score(y_test,y_pred))


# %%
logreg_hd = LogisticRegression(
    max_iter=10000, 
    solver='liblinear', 
    penalty='l2',
    C=1.5,
    class_weight='balanced',  
    random_state=2
)

logreg_hd.fit(X_train, y_train)

prev = logreg_hd.predict(X_test)

# %%
print('Accuracy: ',accuracy_score(y_test,prev))
print('\n',classification_report(y_test,prev))

#%%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# %%
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# %%
logreg_hd = LogisticRegression(
                        C= 0.1, fit_intercept= True,       
                        max_iter= 500, 
                        penalty= 'l1', 
                        solver= 'saga'
                    )

logreg_hd.fit(X_train_res, y_train_res)

prev = logreg_hd.predict(X_test)
# %%
print('Accuracy: ',accuracy_score(y_test,prev))
print('\n',classification_report(y_test,prev))
# %%
