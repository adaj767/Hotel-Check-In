import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

## reading data
train_hotel = pd.read_csv('train_data_evaluation_part_2.csv')

## data cleaning
train_hotel.drop(['Unnamed: 0','ID'],axis=1,inplace=True)

## data cleaning
train_hotel = train_hotel[train_hotel.Age>=0]
train_hotel['Age'].fillna(train_hotel['Age'].median(),inplace=True)
train_hotel  = train_hotel.reset_index(drop=True)

train_hotel['DaysSinceFirstStay'] = train_hotel['DaysSinceFirstStay'].replace(-1,0)
train_hotel['DaysSinceLastStay'] = train_hotel['DaysSinceLastStay'].replace(-1,0)

#importing standardscaler 
from sklearn.preprocessing import StandardScaler
#creating standardscaler object
cat_var = ['Nationality','DistributionChannel','MarketSegment']
non_cat = train_hotel.drop(cat_var,axis=1)
norm = StandardScaler()
num_var_only = non_cat.drop('SRQuietRoom',axis=1)
#num_var_only = train_hotel.drop(columns=cat_var,axis=1)
leave_last = num_var_only
df_norm = pd.DataFrame(norm.fit_transform(leave_last), columns=leave_last.columns)
df_norm['SRQuietRoom'] = train_hotel.SRQuietRoom
df_norm[cat_var] = train_hotel[cat_var]
enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
df_norm[cat_var] = enc.fit_transform(df_norm[cat_var])

## feature importance 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
rf=RandomForestClassifier(min_samples_split=10)
x = df_norm.drop('SRQuietRoom', axis=1)
y = df_norm['SRQuietRoom']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
rf.fit(x_train,y_train)
features = x.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

l=pd.DataFrame(data= {'features':x.columns, 'importances':rf.feature_importances_*100})
df1 = df_norm.copy().loc[:, ~df_norm.columns.isin(list(l.features[l.importances<1]))]

# Randomly over sample the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote= smote.fit_resample(df1.drop('SRQuietRoom',axis=1), df1.SRQuietRoom)

# Import library for VUF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X,thres=0.5):
    dropped=True
    while dropped:
        variables=X.columns
        dropped=False
        vif = [variance_inflation_factor(X[variables].values,X.columns.get_loc(var)) for var in X.columns]
        max_vif = max(vif)
        if max_vif > thres:
            maxloc = vif.index(max_vif)
            X = X.drop([X.columns.tolist()[maxloc]],axis=1)
            dropped=True
    return X

indep_vars = calc_vif(X_train_smote)

## model creation
#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Neural network
model1 = Sequential()
model1.add(Dense(20, input_dim=11, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(2, activation='softmax'))
model1.compile(optimizer="Adam", 
               loss = "sparse_categorical_crossentropy",
               metrics = ["accuracy"])

x_train, x_test, y_train, y_test = train_test_split(rm1, y_train_smote, test_size=0.3, random_state=1)
history = model1.fit(x_train, y_train, 
           batch_size = 256,
           epochs = 100, 
           validation_data = (x_test.values, y_test.values), validation_batch_size = 128)

nn_pred = model1.predict(x_test).argmax(axis = 1)
# Saving model to disk
pickle.dump(model1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))