"""
@author: Aditi
"""

import pandas as pd
import numpy as np
from scipy import stats

#for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#for model development
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

#for model eveluation and refinement
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# IMPORTING DATASET AND BASIC INSIGHTS FROM DATA
file_name='kc_house_data_NaN.csv'
df=pd.read_csv(file_name)

df.head()
#concise summary of dataframe
df.info()
#datatypes present
df.dtypes
#statistical summaryof dataframe
df.describe()


# DATA WRANGLING
#dropping columns that are not required
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

#handling missing values
#checking if there are any missing values present and in which columns
nan_columns = df.columns[df.isnull().any()].tolist()
print('Bedroom null count: ',df['bedrooms'].isnull().sum(),' Bathroom null count: ',df['bathrooms'].isnull().sum())

#replacing null values with average value of the column
for i in nan_columns:
    df[i].replace(np.nan, df[i].mean(), inplace=True)


#binning price into groups: low, medium, high
bins = np.linspace(min(df['price']), max(df['price']),4)
group_names = ['low','medium','high']
df['binned-price']=pd.cut(df['price'], bins, labels=group_names, include_lowest=True)


#EXPLORATORY DATA ANALYSIS
#count of houses by price category
df['binned-price'].value_counts()
#count of houses by number of floors
df['floors'].value_counts()

#analysing price for houses with or without waterfront view
df['waterfront'].value_counts()
df_group_1 = df[['waterfront','price']].groupby(['waterfront'],as_index=True).mean()
new_index = ['No','Yes']
df_group_1.index=new_index
df_group_1 = df_group_1.rename(columns={'price': 'average price'})
df_group_1
#checking for outliers for waterfront view
sns.boxplot(x='waterfront', y='price', data=df)


#correlation of each variable with one another using heatmap
corr_heatmap_plot = sns.heatmap(df.corr(),cmap="YlGnBu", annot=False)
plt.show()

#identifying features that are highly correlated with price
df.corr()['price'].sort_values()

#determining how different features are correlated with price
sns.regplot(x='bedrooms',y='price',data=df)
sns.regplot(x='sqft_living',y='price',data=df)
sns.regplot(x='bathrooms',y='price',data=df)
sns.regplot(x='grade',y='price',data=df)

#pearson correlation for identifying features that are highly correlated with price
int_float_col = list(df.select_dtypes(include=['int64','float64']).columns)
for col in int_float_col:
    pearson_coef, p_value = stats.pearsonr(df[col], df['price'])
    if p_value<0.05 and (pearson_coef>=0.5 or pearson_coef<=-0.5):
        print(col, ' ', pearson_coef, ' ', p_value)


# MODEL DEVELOPMENT
#multiple linear regression
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    

X = df[features]
Y= df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)
yhat_m = lm.predict(X)
Title = 'Distribution  Plot of  Predicted Values vs Actual Values'
DistributionPlot(Y, yhat_m, "Actual Values (Train)", "Predicted Values(Train)", Title)

#Pipeline
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(X,Y)
yhat = pipe.predict(X)

#in-sample evaluation using R^2
pipe.score(X,Y)

#model evaluation using visualisation
Title = 'Distribution  Plot of  Predicted Values vs Actual Values'
DistributionPlot(Y, yhat, "Actual Values (Train)", "Predicted Values(Train)", Title)


# MODEL EVALUATION AND REFINEMENT
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Multiple regression model
lre = LinearRegression()
lre.fit(x_train, y_train)
lre.score(x_train, y_train)
lre.score(x_test, y_test)
yhat_train = lre.predict(x_train)

#plotting distribution of actual training data vs predicted training data
Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Training data'
DistributionPlot(df['price'], yhat_train, "Actual Values (Train)", "Predicted Values(Train)", Title)


#plotting distribution of actual test data vs predcited test data
yhat_tt = lre.predict(x_test)
Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Test data'
DistributionPlot(df['price'], yhat_tt, "Actual Values (Train)", "Predicted Values(Train)", Title)
#overfitting observed

print("Predicted values:", yhat_tt[0:5])
print("True values:", y_test[0:5].values)

#Cross validation
lrc = LinearRegression()
Rcross = cross_val_score(lrc, X,Y, cv=4)
print("The mean of the folds are", Rcross.mean())
print("The standard deviation is" , Rcross.std())
yhat_cross = cross_val_predict(lrc, X,Y, cv=4)

Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Cross Validation'
DistributionPlot(df['price'], yhat_cross, "Actual Values", "Predicted Values", Title)

#Polynomial Regression
#R square test to select the best order
Rsq_test = []
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    Rsq_test.append(poly.score(x_test_pr,y_test))
print(Rsq_test)

plt.plot(order, Rsq_test)
plt.xlim(1, 5)
plt.ylim(-0.50,1)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')

#the best value of R^2 is seen corresponding to order = 2
pr = PolynomialFeatures(degree=2, include_bias=False)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
poly.score(x_train_pr,y_train)
poly.score(x_test_pr,y_test)
#better R^2 observed in case of polynomial regression as compared to linear regression

yhat_poly = poly.predict(x_test_pr)

Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Training Data'
DistributionPlot(df['price'], poly.predict(x_train_pr), "Actual Values(Train)", "Predicted Values(Train)", Title)
Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Test Data'
DistributionPlot(df['price'], yhat_poly, "Actual Values(Test)", "Predicted Values(Test)", Title)
#overfitting-fitting observed


X_pr = pr.fit_transform(X)
poly_cross = LinearRegression()
Rcross_poly = cross_val_score(poly_cross,X_pr,Y,cv=3)
print("The mean of the folds are", Rcross_poly.mean())
print("The standard deviation is" , Rcross_poly.std())
yhat_polycross = cross_val_predict(poly_cross,X_pr,Y,cv=3)
Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Cross Validation'
DistributionPlot(df['price'], yhat_polycross, "Actual Values", "Predicted Values", Title)

print("Predicted values:", yhat_polycross[0:4])
print("True values:", y_test[0:4].values)
#over-fitting is still observed


#Ridge Regression to overcome overfitting
RigeModel = Ridge(alpha=100)
RigeModel.fit(x_train_pr,y_train)
yhat_ridge = RigeModel.predict(x_test_pr)

Title = 'Distribution  Plot of  Predicted Values vs Actual Values Using Ridge Regression'
DistributionPlot(df['price'], yhat_ridge, "Actual Values", "Predicted Values", Title)

RigeModel.score(x_test_pr,y_test)
