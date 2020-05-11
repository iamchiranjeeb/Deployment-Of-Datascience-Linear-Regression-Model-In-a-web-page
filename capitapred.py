import numpy as np
import pandas as pd
from pandas_ods_reader import read_ods
from sklearn import linear_model
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
import pickle
import warnings

warnings.filterwarnings("ignore")
sheetidx = 1
path = "/home/iamchiranjeeb/Desktop/amita/canada_per_capita_income.ods"
df = read_ods(path,sheetidx)

X = df[["year"]]
y=df.capita_income

reg = linear_model.LinearRegression()
reg.fit(X,y)

pickle.dump(reg,open('capita.pkl','wb'))
model = pickle.load(open('capita.pkl','rb'))
