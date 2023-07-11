import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_full = pd.read_csv('C:\\Users\\Admin\\Desktop\\Data Analysis\\house-prices-advanced-regression-techniques\\train.csv',index_col = ['Id'])
test = pd.read_csv('C:\\Users\\Admin\\Desktop\\Data Analysis\\house-prices-advanced-regression-techniques\\test.csv',index_col = ['Id'])
pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
# Filtering Data:
# Check for columns with missing values:
missing_cols = [col for col in train_full.columns if train_full[col].isnull().any()]
highest_missed_vals  = train_full[missing_cols].count().sort_values(ascending=False)
# columns with their number of missing values:
# print("Number of missing values: \n"+ str(highest_missed_vals))
# print(highest_missed_vals.head(13).index)
# We drop columns with more than 70% missing values:
drop_cols =['Electrical', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
       'BsmtExposure', 'BsmtFinType2', 'GarageYrBlt', 'GarageCond',
       'GarageQual', 'GarageFinish', 'GarageType', 'LotFrontage','Alley','MasVnrType','PoolQC','Fence','MiscFeature']
train = train_full.drop(drop_cols,axis = 1)
test = test.drop(drop_cols,axis =1)

# Check if target columns miss any values:
saleprice_miss_val = train.SalePrice.isnull().sum()
# print(saleprice_miss_val)
# SalePrice columns has no missing values

# Number of categorical data columns
cate_cols = train.select_dtypes(include= ['object','category'])

print("-----------------------------------------------------")
# Number of numeric data columns
numeric_data = train.select_dtypes(exclude=['object','category'])
numeric_data_test = test.select_dtypes(exclude=['object','category'])
# Deal with missing values by imputing:
my_imputer = SimpleImputer()
numeric_data_imputed = pd.DataFrame(my_imputer.fit_transform(numeric_data))
numeric_data_imputed.columns = numeric_data.columns
numeric_data_imputed.index =numeric_data_imputed.index +1

categorical  = train.drop(numeric_data,axis = 1)
test = test.drop(numeric_data_test,axis = 1)
train_1 = pd.concat([categorical,numeric_data_imputed],axis = 1)
test = pd.concat([categorical,numeric_data_imputed],axis = 1)

# print(train.select_dtypes(exclude = 'object').describe())
# MiscVal, PoolArea,ScreenPorch,3SsnPorch,EnclosedPorch are columns with 75% of its value are 0
# We have drop the Columns MiscFeatures with too many missing values, therefore MiscValue would also be dropped
# Similar for PoolArea
# sns.scatterplot(x = '3SsnPorch',y = 'SalePrice',data = train)
# sns.scatterplot(x = 'EnclosedPorch',y = 'SalePrice',data = train,legend= True)
# 3SsnPorch and EnclosedPorch has no relations relating to the target => Drop
train_1 = train_1.drop(['MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch'],axis = 1)
test = test.drop(['MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch'],axis = 1)

# Label Encoding:
for colname in train_1.select_dtypes(include = "object"):
    train_1[colname], _ = train_1[colname].factorize()
for colnametest in test.select_dtypes(include = "object"):
    test[colnametest],_ = test[colnametest].factorize()
X = train_1.copy()
y = X.pop('SalePrice')
# Define discrete features for mutual information evaluating
discrete_features = X.dtypes == int
from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
mi_score = make_mi_scores(X,y,discrete_features)

# Display MI scores for each features:
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks= list(scores.index)
    plt.barh(width,scores)
    plt.yticks(width,ticks)
    plt.title("Mutual Information Scores")
plt.figure(figsize=(14,10))
# plot_mi_scores(mi_score)

# The top 10 potential features with high MI scores:
# print(mi_score.head(20))
# All of these features is positively correlated with the target feature
# MothSold is a categorical data type and would need to be further investigated:
# sns.scatterplot(x = 'MoSold',y = 'SalePrice',data = train_1)
# Sale Price does not fluctuate by much through the months => not a potential feature
#Street:
# sns.stripplot(data = train,x = 'Street',y = 'SalePrice')
# Roughly 90% of houses fall in the category Paved Road access and the Sale Price does not differentiate large between the 2.
# Investigate into the second highest correlation feature:
sns.boxplot(data = train, x = 'Neighborhood',y = 'SalePrice')
grouped = train.groupby('Neighborhood').SalePrice.mean().sort_values(ascending=False)

# plt.show()
# print(grouped)
# As we can see, the mean sale price varies largely by neighborhood, with property in the NoRidge, NridgHt, StoneBr sold with the highest mean value.
# Selecting features:
features = []
# print(mi_score)
for i in mi_score.index:
    if mi_score[i] >0.2:
        features.append(i)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
y = train_1.pop('SalePrice')
X = X[features]
X_test = test[features]

X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0)

my_model = XGBRegressor()
my_model.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error
preds = my_model.predict(X_test)
mae = mean_absolute_error(y,preds)
preds = pd.Series(preds)
print("Predictions:")
print(preds)
print("Mean Absolute Error using XGBRegressor: ",mae)







