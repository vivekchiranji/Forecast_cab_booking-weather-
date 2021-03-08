#!/usr/bin/env python
# coding: utf-8

# # Forecast Cab Booking Demand In City

# In[64]:


#Packages for general funtions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from prettytable import PrettyTable
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
#Linear Analysis
from sklearn.linear_model import LinearRegression,SGDRegressor
#Ensemble Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
#Metrics
from sklearn.metrics import r2_score
#Model Validation
from sklearn.model_selection import GridSearchCV


# In[65]:


df_train_x=pd.read_csv('train.csv')
df_train_y=pd.read_csv('train_label.csv',header=None,names=['Total Bookings'])

df_train=pd.concat([df_train_x,df_train_y],axis=1)
df_train


# In[66]:


df_test_x=pd.read_csv('test.csv')
df_test_y=pd.read_csv('test_label.csv',header=None,names=['Total Bookings'])

df_test=pd.concat([df_test_x,df_test_y],axis=1)


# In[67]:


def timetonum(x):
    (h,m,s)=str(x).split(':')
    return int(h)+(int(m)/60)


# In[68]:


df_train['date']=pd.to_datetime(df_train['datetime']).dt.date#1
df_train['time']=pd.to_datetime(df_train['datetime']).dt.time


# In[69]:


df_train['time']=df_train['time'].map(timetonum)


# In[70]:


df_train


# ###### I am writing a funtion for performing all the data wrangling tasks. Since it needs to be done on both train and test I am just going to pass the train and test for once. Each tasks comment will be written in the next comment section with that particular task number.

# In[71]:


def data_wrangling(dataframe):
    
    # Data Wrangling and Text Removal:
    dataframe['date']=pd.to_datetime(dataframe['datetime']).dt.date#1
    dataframe['time']=pd.to_datetime(dataframe['datetime']).dt.time
    dataframe=dataframe.drop('datetime',axis=1)#2
    
    
    dataframe=pd.concat([dataframe,pd.get_dummies(dataframe.season,drop_first=True)],axis=1)#4.1.1
    dataframe=pd.concat([dataframe,pd.get_dummies(dataframe.weather,drop_first=True)],axis=1)#4.1.2
    
    dataframe=dataframe.drop(['season','weather'],axis=1)
    
    # Visualize data using different visualizations to generate interesting insights.
    dataframe=dataframe.drop(['atemp','holiday'],axis=1)
    dataframe=dataframe.drop('date',axis=1)
    return dataframe
    


# ### Data Wrangling and Text Removal:
# 
# 1) Since date and time are given combined in a column. We need to split into two columns.
# 2)  If we look at the data, the date is not the most important. Because logically its important to find the time and working day than the date. It dosent make sence to predict for a particular, even if we query with a date, we intuitively go and find the day, holiday/working day, season, weather etc. Date just acts an index for all these factors. So were are going to drop the columns with date.
# 
# 3) Preparing data sets for X and Y
# 
# 4) 1) Now in our date, we are left with two text columns. Two deal with this issue one hot encoding is easy approach but it will give too many coulmns. But fortunately we just have <5 catergories in both columns.
#    
#    2) Now we need to drop the original features, now our dataset is completely text free.

# #### 1. Visualize data using different visualizations to generate interesting insights.

# ##### Seaborne HeatMap

# In[72]:


plt.figure(figsize=(15,10))

sns.heatmap(df_train.corr(),annot=True,annot_kws = {'size': 12})
plt.xticks(rotation=0)


# ##### Plot Between temp and atemp:

# In[73]:


plt.scatter(df_train['temp'],df_train['atemp'],alpha=.2)
plt.xlabel('Temp',fontsize=15)
plt.ylabel('atemp',fontsize=15)
plt.title('temp Vs. atemp Scatter Plot')


# In[74]:


plt.figure(figsize=(9,7))
plt.scatter(df_train_x['temp'],df_train['Total Bookings'],alpha=.3,label='temp')
plt.scatter(df_train_x['atemp'],df_train['Total Bookings'],alpha=.1,label='atemp')
plt.xlabel('Temp & atemp',fontsize=15)
plt.ylabel('Total Bookings',fontsize=15)
plt.title('Temp Vs. Total Bookings and aTemp Vs. Total Bookings')
plt.legend()


# Seaborne HeatMap-
# - If we look at the last column of the heat map. Holiday and Working Day are saying the same thing. So having one of them is enough.
# - Same with temp and a temp. We can remove one of them.
# 
# Plots:
# - We can see how temp and atemp are colinear. The coinsiding every where as per scatter plot.
# - And they are almost linearly corelated.

# ##### Now lets look at the variation of Total Booking for Season, Weather and Time.

# In[75]:


plt.figure(figsize=(9,7))

plt.scatter(df_train['time'],df_train['Total Bookings'])
plt.xticks(range(24))
plt.xlabel('Time')
plt.ylabel('Total Bookings')


# ##### This is the plot between time and total bookings. As we can see the number of bookings increases after 6 am and increases as time passes and peaks in the evening time at 5PM-7PM.

# In[76]:


plt.figure(figsize=(9,7))
plt.xlabel('Season')
plt.ylabel('Total Bookings')
plt.scatter(df_train['season'],df_train['Total Bookings'])


# In[77]:


plt.figure(figsize=(9,7))
plt.scatter(df_train['weather'],df_train['Total Bookings'])
plt.xlabel('Weather')
plt.ylabel('Total Bookings')


# ##### There is not much we can infer out of using season and Weather condition. Total bookings is almost same all the time. There is only one outlier during Heavy Rain and Thunderstorms, even though it an outlier but its giving very good indication of the inverse relation between the bookings with that weather condition.

# #### All the data wrangling and the findings in the exploratory data analysis are written the funtion. So the idea is we pass the funtion to both test and train data so the wranglings will be consistent among both of them.

# In[78]:


df_train_wrangle=data_wrangling(df_train)
df_test_wrangle=data_wrangling(df_test)


# In[79]:


df_train_wrangle


# ### 2. Outlier Analysis

# In[80]:



plt.figure(figsize=(20,20))
plt.subplot(4,1,1)
sns.boxplot(x=df_train['windspeed'],y=df_train['Total Bookings'])
plt.xlabel('Windspeed',fontsize=15)
plt.ylabel('Total Booking',fontsize=15)

plt.subplot(4,1,2)
sns.boxplot(x=df_train['temp'],y=df_train['Total Bookings'])
plt.xlabel('temp',fontsize=15)
plt.ylabel('Total Booking',fontsize=15)

plt.subplot(4,1,3)
sns.boxplot(x=df_train['humidity'],y=df_train['Total Bookings'])
plt.xlabel('humidity',fontsize=15)
plt.ylabel('Total Booking',fontsize=15)


# ##### Task 2
# 1. Feature Engineering

# From these plots we can see that there is alot of outliers which needs to be removed from the data. But lets normalize the data so that all the data points represents z score. Then we can eliminate them by applying (<=-3 and >=3)
# 
# But we cant apply normalization because, in our data all our data points are 0--> infinity. But normalization converts it to (-z,0,z).
# So for this data its better to use standard scalar so our data wont go outside of Q1 region.

# In[81]:


def outliers(dataframe):
    wind=dataframe['windspeed']
    temp=dataframe['temp']
    humidity=dataframe['humidity']
    
    wind_upper,wind_lower=np.percentile(wind,75),np.percentile(wind,25)
    temp_upper,temp_lower=np.percentile(temp,75),np.percentile(temp,25)
    humidity_upper,humidity_lower=np.percentile(humidity,75),np.percentile(humidity,25)
    
    wind_iqr=wind_upper-wind_lower
    temp_iqr=temp_upper-temp_lower
    humidity_iqr=humidity_upper-humidity_lower
    
    wind_upper_outlier=wind_upper+(wind_iqr*1.5)
    temp_upper_outlier=temp_upper+(temp_iqr*1.5)
    humidity_upper_outlier=humidity_upper+(humidity_iqr*1.5)
    
    wind_lower_outlier=wind_lower-(wind_iqr*1.5)
    temp_lower_outlier=temp_lower-(temp_iqr*1.5)
    humidity_lower_outlier=humidity_lower-(humidity_iqr*1.5)
    
    dataframe=dataframe[~((dataframe['windspeed']>wind_upper_outlier) |(dataframe['windspeed']<wind_lower_outlier))]
    
    dataframe=dataframe[~((dataframe['temp']>temp_upper_outlier) | (dataframe['temp']<temp_lower_outlier))]
    dataframe=dataframe[~((dataframe['humidity']>humidity_upper_outlier) |(dataframe['humidity']<humidity_lower_outlier))]
    
    return dataframe


# In[82]:


df_train_wrangle_outliers=outliers(df_train_wrangle)
print('Shape of the train data before eliminating outliers:',df_train_wrangle.shape)
print('-'*65)
print('Shape of the train data after eliminating outliers:',df_train_wrangle_outliers.shape)


# Here we cant remove outliers from test data, since test data is the similation of real time data in which outliers are common.

# Now need to scale the data for the range between 0-1, so it what ever model we use it wont create any scalability issues. Please note that data scalling should be done on both train and test data, scaling will not effect data distribution. So its just that we reducing the range to 0-1 with out touching variance and standard deviation as percentage.

# In[83]:


scalar=MinMaxScaler(feature_range=(0, 1))


# In order to apply the above scalar function we need all our data to be in numeric format but we have feature that is an object(i.e date time). So we are going to covert it hours with the below two code snippets.

# In[84]:


def timetofloat(x):
    h=x.hour
    return float('%s' %(h))


# In[85]:


df_train_wrangle_outliers['time']=df_train_wrangle_outliers['time'].map(timetofloat)
df_test_wrangle['time']=df_test_wrangle['time'].map(timetofloat)


# In[86]:


df_train_wrangle_outliers.head()


# As we can see the time feature is a float variable. And now we can apply the fit_transform on this data and I am going to store it in a dataframe with appropriate column names for better interprirtability..

# In[87]:


df_train_y=df_train_wrangle_outliers['Total Bookings']
df_train_wrangle_outliers=df_train_wrangle_outliers.drop(['Total Bookings'],axis=1)

df_test_y=df_test_wrangle['Total Bookings']
df_test_wrangle=df_test_wrangle.drop(['Total Bookings'],axis=1)


# In[88]:


df_train_final=scalar.fit_transform(df_train_wrangle_outliers)
df_train_final=pd.DataFrame(df_train_final,columns=df_train_wrangle_outliers.columns)

df_test_final=scalar.fit_transform(df_test_wrangle)
df_test_final=pd.DataFrame(df_test_final,columns=df_test_wrangle.columns)
df_test_final['Heavy Rain + Thunderstorm']=0


# In[89]:


df_train_final.head()


# In[90]:


df_test_final.head()


# Now our data is ready for application of any kind of ML models on it. Lets first apply the basic linear regression.

# ## LinearRegression

# In[91]:


lin_reg=LinearRegression()


# In[92]:


lin_reg.fit(df_train_final,df_train_y)
lin_reg_predicted=lin_reg.predict(df_test_final)


# In[93]:


print('R2 Score for Linear Regression:',r2_score(df_test_y,lin_reg_predicted))


# As we can see Linear Regression is performing some basic prediction on the test data the R2 score which represents the godness of fit for the model, is very low at around 30%. Now lets look at more advanced regression models with hyper parameter tuning.

# I am going to build a custom funtion for all the repetative tasks below because the work flow is the same as below:
#     
#     (-) Defining a set and range of hyperparameters
#     (-) Performing GridSearchCV or RandomSearchCV on hyperparameters
#     (-) training the data with best parameters
#     (-) predicting the target for test data with the best fit
#     (-) calculating the R2 score for the model
# Since this looks like a loop, the below funtion is intended to perform the above mentioned tasks.

# ### Best Predictor Function

# In[107]:


def best_predictor(predictor,hyperparameter):
    random_cv=GridSearchCV(predictor,hyperparameter,cv=5,n_jobs=-1)
    random_cv.fit(df_train_final.values,df_train_y.values)
    best_hy_para=random_cv.best_params_
    print('This is the best parameters for this predictor:',best_hy_para)
    
    predictor.set_params(**best_hy_para)#imputting the best parameters for the predictor
    predictor.fit(df_train_final.values,df_train_y.values)
    test_predicted=predictor.predict(df_test_final.values)
    print('This is the r2 score for this given predictor with best hyper parameters:',r2_score(df_test_y,test_predicted))
    return best_hy_para,r2_score(df_test_y,test_predicted)
    
    


# ### DecisionTreeRegressor

# Now lets look at Decsion Tree by applying a range of hyper parameters.

# In[97]:


max_depth=[i for i in range(2,10)]
min_samples_split=[i for i in range(2,10)]

hyperparameter=dict(max_depth=max_depth,min_samples_split=min_samples_split)
predictor=DecisionTreeRegressor()


# In[98]:


dec_tree_hy_para,dec_tree_score=best_predictor(predictor,hyperparameter)


# As we can see in the above results the R2 score is more than doubled in this case and I am giving fairly large range of depth and min_splits and CV of 50, its not just a case of one off. Its been tested with different sample datas.

# ### SGDRegressor

# In[99]:


l_rate=[10**i for i in range(-5,0)]
lambda_reg_multi=[10**i for i in range(-3,4)]
eta0=0.01
hyperparameter=dict(alpha=lambda_reg_multi)
predictor=SGDRegressor(learning_rate='adaptive')


# In[100]:


SGD_reg_hy_para,SGD_reg_score=best_predictor(predictor,hyperparameter)


# As we can see Decison Tree Regressor is giving the best performance and linear and SGD regressor are giving similar perfromance indicating that the data is not linearly seperable.

# #### RandomForest Regressor

# In[108]:


predictor=RandomForestRegressor(n_jobs=-1)
n_estimators=[i for i in range(10,300,100)]

min_samples_split=[i for i in range(2,10,2)]
hyperparameter=dict(n_estimators=n_estimators,min_samples_split=min_samples_split)


# In[109]:


rand_forest_hy_para,rand_forest_hy_score=best_predictor(predictor,hyperparameter)


# Now that is very encouraging we worked our way from 30% test accuracy to about 80%, so this clearly shows that the data is better fit when using a non linear model. Decesion Trees and ensemble models with kd tree kind of model as base produces hypercuboids for an n dimensional space(10 dimensions in this case). A linear model is producing a 10 dimensional hyper plane to split the working space. But this will only work when data has some linear relationship with the target variable, which in real wont be the case most of the time. 

# There are some interesting ensemble models which are more advanced and going forward we can expect slight improvement the r2_score.

# #### GradientBoostingRegressor

# In[111]:


predictor=GradientBoostingRegressor()

n_estimators=[i for i in range(10,50,10)]

min_samples_split=[i for i in range(2,5)]


hyperparameter=dict(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)


# In[112]:


grad_reg_hy_para,grad_reg_hy_score=best_predictor(predictor,hyperparameter)


# #### XGBRegressor

# In[113]:


predictor=XGBRegressor(objective='reg:squarederror')

n_estimators=[i for i in range(10,50,10)]

min_samples_split=[i for i in range(2,5)]


hyperparameter=dict(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)


# In[114]:


xgb_hy_para,xgb_hy_score=best_predictor(predictor,hyperparameter)


# ### Summary

# #### The below tabel summarises the performance of the models that were used.

# In[115]:


tabel=PrettyTable()
tabel.field_names=['Model','Hyperparameter','R2 Score']
tabel.add_row(['Linear Regression',np.nan,r2_score(df_test_y,lin_reg_predicted)])
model_names=['SGDRegressor','DecisionTreeRegressor','RandomForestRegressor','GradientBoostingRegressor','XGBRegressor']
final_hy_par=[SGD_reg_hy_para,dec_tree_hy_para,rand_forest_hy_para,grad_reg_hy_para,xgb_hy_para]
final_scores=[SGD_reg_score,dec_tree_score,rand_forest_hy_score,grad_reg_hy_score,xgb_hy_score]

for i,j,k in zip(model_names,final_hy_par,final_scores):
    tabel.add_row([i,j,k])
print(tabel)


# From this tabel we can conclude that ensemble models perform better on this dataset as the variations in this data set is not close to linearirty.

# In[ ]:




