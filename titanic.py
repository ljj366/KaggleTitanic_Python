# import packages
import pandas as pd #for data manipuldation
import numpy as np # multidimentional array
import matplotlib.pyplot as plt # visualization
import seaborn as sn #visualization
from sklearn.linear_model import LogisticRegression # ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# read data
train_raw = pd.read_csv('E:/study/ML/Practice/Titanic/train.csv', header=0)
test_raw = pd.read_csv('E:/study/ML/Practice/Titanic/test.csv', header=0)
rlt = pd.read_csv('E:/study/ML/Practice/Titanic/gender_submission.csv', header=0) # the sample provived by Kaggle

# combine train and test
Data = pd.concat([train_raw,test_raw], ignore_index=True)

# check type and number of observations for each var
print(Data.info())

#### Missing Values ####
# number of missing values for each var
Data.isnull().sum()

# drop tick and cabin
Data = Data.drop(['Ticket','Cabin'], axis=1)

# Age #
# use the median of the age of each title to approx   
Data['Title'] = Data['Name'].apply(lambda x: x.split(',')[1].split('.')[0]).str.strip()

# titles of the missing ages
missing_age = pd.unique( Data[Data['Age'].isnull()]['Title'] )

# fill missing ages
for i in missing_age:
    print ("processing: ", i)
    Data.loc[Data.Title == i,'Age'] = Data.loc[Data.Title == i,'Age'].fillna( Data[Data.Title == i].Age.median())
    
# Embarked #
# use the most freq category to fill
Data['Embarked'] = Data['Embarked'].fillna( Data['Embarked'].value_counts().keys()[0] )

# Fare # 
# use median of the Pclass of which the missing value falls in to approx
Data['Fare'] = Data['Fare'].fillna(Data.loc[Data.Pclass== int(Data[Data.Fare.isnull()].Pclass),'Fare'].median())


#### Feature Engineering ####

## create title 
'''def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# summarize infreq types
def title_comb(title):
    if title in ['Mr','Miss','Mrs','Ms','Mlle','Mme']:
        return 'Ordinary'
    elif title in ['Master']:
        return 'Master'
    else:
        return 'Other'
  
Data['Title'] = Data['Name'].apply(get_title).apply(title_comb) 

# summarize all the ordinary people together since these titles either differ in age or sex
Data.loc[ Data.Title.isin (['Mr','Miss','Mrs','Ms','Mlle','Mme']) ,'Title' ] = 'Ordinary'
Data.loc[ -Data.Title.isin(['Ordinary','Master']) ,'Title'] = 'Other'
'''

Data['Title'] = Data['Name'].str.split("[\,\.]").apply(lambda x: x[1]).str.strip()
# aggregate title
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }

Data.Title = Data.Title.map(Title_Dictionary)

# plot survival rate by title
summ = pd.crosstab(Data['Title'], Data['Survived'])
pct = summ.div(summ.sum(1).astype(float), axis=0)
pct.plot( kind='bar', stacked=True, title='Survival rate by title' )
plt.xlabel('Title')
plt.ylabel('Survival Rate')

## Family size 
Data['Fnb'] = Data['SibSp'] + Data['Parch'] + 1
Data[["Fnb", "Survived"]].groupby(['Fnb'],as_index=False).mean()

# Seems family size between 2 and 4 has higher survival rate than other families, so I summarize family size into 3 categories
Data['Fsize'] = 'Singleton'
Data.loc[(Data.Fnb >1) & (Data.Fnb<=4) ,'Fsize'] = 'Small'
Data.loc[Data.Fnb >4, 'Fsize'] = 'Large'

## Age category 
plt.hist( [ Data[Data.Survived ==0].Age,Data[Data.Survived ==1].Age ], stacked = True, bins=30,label=['0','1'])
plt.xlabel('Age');plt.ylabel('Number');plt.legend()

# children below 10 are more likely to survive
Data['Adult'] = 'Adult'
Data.loc[Data['Age'] <= 10, 'Adult'] = 'Child'


#################### Prediction ############################
train = Data[ Data.Survived.notnull() ]
test = Data[np.isnan(Data.Survived)]

formula = 'Survived ~ C(Title) + C(Sex) + Age + C(Pclass) + C(Fsize) + Fare' 
from patsy import dmatrices
from patsy import dmatrix

y,X = dmatrices(formula, data=train, return_type='dataframe')
y = np.asarray(y); y=y.flatten() # y must be 1 dimensional
test_X = dmatrix(formula.split('~')[1], data=test, return_type='dataframe')

#### Use all training data ####
# logistic regression
lg = LogisticRegression()
lg.fit(X,y)
print( lg.score(X,y) ) #in-sample accuracy
ypred = lg.predict(test_X)
sum( rlt['Survived']==ypred )/len(ypred)#similarity with provided sample

# SVM
svm = SVC(kernel='linear').fit(X,y)
ypred = svm.predict(test_X)

# Random Forest
rf = RandomForestClassifier( ).fit(X,y)
print( rf.score(X,y) )
ypred = rf.predict(test_X)
sum( rlt['Survived']==ypred )/len(ypred)

#### Cross validation ####
from sklearn.model_selection import KFold,GridSearchCV

k_fold = KFold( n_splits=10, shuffle=True)
scoring = 'accuracy'

## logistic 
lg = LogisticRegression()

# find optimal param using CV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_search = GridSearchCV(lg, param_grid, scoring=scoring,cv=k_fold)
grid_search.fit(X, y)
print(grid_search.best_score_)
param = grid_search.best_params_

# use the best param to predict
lg_best = LogisticRegression(**param)
lg_best.fit(X,y)
ypred = lg_best.predict(test_X)

## SVM, much slower than other methods
svm = SVC()
param_grid = {'C': [1], 'kernel': ['linear']}
grid_search = GridSearchCV(svm, param_grid, scoring=scoring,cv=k_fold)
grid_search.fit(X,y)
print(grid_search.best_score_)

## random forest
rf = RandomForestClassifier()
param_grid = {"n_estimators"      : [10,20,30],
             "max_features"      : ["auto", "sqrt", "log2"],
             "min_samples_split" : [2,4,8],
             "bootstrap": [True, False]
             }
grid_search = GridSearchCV(rf, param_grid, scoring=scoring,cv=k_fold)
grid_search.fit(X, y)
print(grid_search.best_score_)
param = grid_search.best_params_

rf_best = RandomForestClassifier(**param)
rf_best.fit(X,y)
ypred = rf_best.predict(test_X)


## submit file
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": ypred})
submission.to_csv('E:/study/ML/Practice/Titanic/titanic_pred_Python.csv', index=False)
