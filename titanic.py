# import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# read data
train_raw = pd.read_csv('E:/study/ML/Practice/Titanic/train.csv', header=0)
test_raw = pd.read_csv('E:/study/ML/Practice/Titanic/test.csv', header=0)
rlt = pd.read_csv('E:/study/ML/Practice/Titanic/gender_submission.csv', header=0)

# combine train and test
Data = pd.concat([train_raw,test_raw], ignore_index=True)

# print head
#print(Data.head(3))

# check type
#print (Data.dtypes)

# check type and number of observations for each var
#print(Data.info())


#### Missing Values ####
# number of missing values for each var
Data.isnull().sum()

# drop tick and cabin
Data = Data.drop(['Ticket','Cabin'], axis=1)

# Age #
# use the median of the age of each title to approx   
Data['Title'] = Data['Name'].str.split("[\,\.]").apply(lambda x: x[1]).str.strip()
# or
# Data['Title'] = Data['Name'].apply(lambda x: x.split(',')[1].split('.')[0]).str.strip()

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
'''
Data.loc[ Data.Title.isin (['Mr','Miss','Mrs','Ms','Mlle','Mme']) ,'Title' ] = 'Ordinary'
Data.loc[ -Data.Title.isin(['Ordinary','Master']) ,'Title'] = 'Other'

# plot survival rate by title
summ = pd.crosstab(Data['Title'], Data['Survived'])
pct = summ.div(summ.sum(1).astype(float), axis=0)
pct.plot( kind='bar', stacked=True, title='Survival rate by title' )
plt.xlabel('Title')
plt.ylabel('Survival Rate')

## Family size 
Data['Fnb'] = Data['SibSp'] + Data['Parch'] + 1
Data['Fsize'] = 'Singleton'
Data.loc[(Data.Fnb >1) & (Data.Fnb<=4) ,'Fsize'] = 'Small'
Data.loc[Data.Fnb >4, 'Fsize'] = 'Large'

# calculate survival rate for each size
# Small family has higher survival rate than singleton and large family
train[["Fsize", "Survived"]].groupby(['Fsize'],as_index=False).mean()

## Age category 
Data['Adult'] = 'Adult'
Data.loc[Data['Age'] <= 18, 'Adult'] = 'Child'


###### Prediction ######
train = Data[ Data.Survived.notnull() ]
test = Data[np.isnan(Data.Survived)]

formula = 'Survived ~ C(Title) + C(Sex) + Age + C(Pclass) + C(Fsize) + Fare' 
from patsy import dmatrices
from patsy import dmatrix

y,X = dmatrices(formula, data=train, return_type='dataframe')
y = np.asarray(y); y=y.flatten() #y must be 1 dimensional
test_X = dmatrix(formula.split('~')[1], data=test, return_type='dataframe')

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

'''
## Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0824)
k_fold = KFold( len(train_y), n_folds=10, shuffle=True)
scoring = 'accuracy'

rf = RandomForestClassifier()
results = cross_val_score(rf, X_train,y_train,cv=k_fold,n_jobs=1,scoring=scoring)
grid_search = GridSearchCV(rf)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
'''

## submit file
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": ypred})
submission.to_csv('E:/study/ML/Practice/Titanic/titanic_pred_Python.csv', index=False)
