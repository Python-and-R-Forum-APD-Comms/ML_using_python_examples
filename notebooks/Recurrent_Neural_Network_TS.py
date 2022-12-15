
# Recurrent neural network (Keras and Tensorflow)

# ## TENSORFLOW 02 . BUILDING A RNN 21 APR 2021
# #### Install required libraries for data manipulation
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import os

print(tf.version.VERSION)

from tensorflow import keras

# ##### 1.Install neural network types and configurations

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

## Get my working directory
path = os.getcwd()

os.chdir('C:/Pablo UK/46 DATA SCIENCE all/44 Python')

# Sales_clean.to_csv (r'Sales_clean_plot_csv.csv', index = False, header=True)
Sales_clean  = pd.read_csv('Sales_clean_plot_csv.csv') 
pivot = Sales_clean.copy()
pivot.head()

list(pivot.columns.values)

SalesP = pd.pivot_table(Sales_clean, 
                        values = 'Sales', 
                        index=['PERIOD'], 
                        columns = 'Orgname').reset_index()

SalesP['PERIOD'] = pd.to_datetime(SalesP['PERIOD'])

SalesP.set_index("PERIOD",inplace=True)

SalesP.plot()

SalesQ = pivot.copy()

SalesQ.head()

list(SalesQ.columns.values)

SalesQ.reset_index()

list(SalesQ.columns.values)

# #### Subset columns 

SalesR = SalesQ[["PERIOD","Orgname","Sales"]]
SalesR.head()
list(SalesQ.columns.values)
type(SalesQ)

dataTypeSeries = SalesQ.dtypes

print('Data type of each column of Dataframe :')
print(dataTypeSeries)

print (type((SalesQ)))

# #### Subset variable names, select just East Coast branch 
# We subset varibale values by using the double equal sign, first we quaote the dataset and then the variable within the dataset, and then we assign the value we want to filter for, in this case 'East coast branch', using single quotes
# **Saleseast = SalesQ[SalesQ['Orgname'] =='East coast branch']**
# This code abote is how you subset rows based on a variable value

Saleseast = SalesQ[SalesQ['Orgname'] =='East coast branch']
Saleseast.head()
Saleseast

# Check again total number of colums in dataframe
list(SalesQ.columns.values)

# ### Treat PERIOD as date
Saleseast['PERIOD'] = pd.to_datetime(Saleseast['PERIOD'])
Saleseast.set_index("PERIOD",inplace=True)
Saleseast.head()

# ### Subset again columns
Saleseastmod = Saleseast[["Orgname","Sales"]]
Saleseastmod.head()

list(Saleseastmod.columns.values)

Saleseastmod.reset_index()

# ### Rename column names
# We can rename second col and get rid of Orgname
Sales_new = Saleseastmod.rename(columns ={'Sales':'Sales_east'})
Sales_new

# ### 1 Remove null values
Sales_new.isnull().sum()

# Not required this time as there are not any null values
# Check this notebook on how to remove null values from at TS dataframe "TENSORFLOW Data prep and removing null values"
# #### This is the script to remove null values replacing them by same day previous week value
# for row in range(0,len(SalesQ)):
#     SalesQ['Eastbranch']=np.where(
#                           (np.isnan(SalesQ['Eastbranch'])),
#                            SalesQ['Eastbranch'].shift(7),SalesQ['Eastbranch']
#                          )
SalesQ.isnull().sum()


# for row in range(0,len(SalesQ)):
#     SalesQ['Westbranch']=np.where(
#                           (np.isnan(SalesQ['Westbranch'])),
#                            SalesQ['Westbranch'].shift(7),SalesQ['Westbranch']
#                          )
SalesQ.isnull().sum()
SalesQ.head()

# ### 2 Subset again data
Sales_new.head()
Mydata = Sales_new[["Sales_east"]]
Mydata.head()


# ### 3 Plot East Sales 
# Replace any value above 1000 by standard 150 value/ Average
Mydata.plot()

# Enhance this plot
# Using matplotlib import matplotlib.pyplot as plt
Mydata.plot(figsize=(8,5))
plt.grid(True)
## Change scale
plt.title("East coast Sales. 2016.05-2019.05")
plt.gca().set_ylim(0,2000) # This sets vertical range to [0-1]
plt.show()

# We can setup a dark background from matplotlib
plt.style.use('dark_background')

Mydata.plot(figsize=(10,7))
plt.grid(True)
## Change scale
plt.title("East coast Sales. 2016.05-2019.05",fontname ="Times New Roman",fontweight ="bold")
plt.gca().set_ylim(0,2000) 
plt.ylabel('sales')
plt.xlabel('date')
plt.show()

# ### 4 Remove outliers
# We need to remove extreme values
Mydata.head()
type(Mydata)

# ####  4.1 Identify max and max values 
# We use a combination of dataset, variable name and max function to obtain the max value
max_sales = Mydata["Sales_east"].max()
max_sales

# We use a combination of dataset, variable name and min function to obtain the min value
min_sales = Mydata["Sales_east"].min()
min_sales

# ####  4.3 Replace outliers by average value
# We can use the same logic as in the max calculation to get the mean sales value
avg_sales = Mydata["Sales_east"].mean()
avg_sales

# So then we only have to replace that extreme value by the mean. by using the replace function
Mydatab = Mydata.copy()
len(Mydata)

# This is the standard replace function from Pandas
# df.replace(current_value,new_value)
max

Mydatab["Sales_east"].max()
Mydatab["Sales_east"].mean()

max_Mydatab = Mydatab["Sales_east"].max()

max_Mydatab

# We could have replaced outliers by a specific value
# SalesQ.loc[SalesQ.Eastbranch >1000,"Eastbranch"] = 300
# SalesQ.loc[SalesQ.Eastbranch >1000,"Eastbranch"] = 300
# We use this approach

Mydatab.loc[Mydatab.Sales_east >1000,"Sales_east"] = 174

# SalesQ.loc[SalesQ.Westbranch >1000,"Westbranch"] = 300
# **This is a way of replacing values above a certain treshold by the mean value**
Mydatab.loc[Mydatab.Sales_east >1000,"Sales_east"] = Mydata["Sales_east"].mean()

# ####  4.4 Plot new dataset
# Plot resulting series
# We change plot color in the plot statement.(darkorange,coral,gold,dodgerblue)
Mydatab.plot(figsize=(8,5),color ="gold")
plt.grid(True)
## Change scale
plt.title("East coast Sales excluding outliers. 2016.05-2019.05")
plt.gca().set_ylim(0,300) # This sets vertical range to [0-1]
plt.show()

Mydatab.plot(figsize=(8,5),color ="dodgerblue")
plt.grid(True)
## Change scale
plt.title("East coast Sales excluding outliers. 2016.05-2019.05")
plt.gca().set_ylim(0,300) # This sets vertical range to [0-1]
plt.show()

Mydatab.plot(figsize=(8,5),color ="mediumspringgreen")
plt.grid(True)
## Change scale
plt.title("East coast Sales excluding outliers. 2016.05-2019.05")
plt.gca().set_ylim(0,300) # This sets vertical range to [0-1]
plt.show()

Mydatab.plot(figsize=(8,5),color ="orangered")
plt.grid(True)
## Change scale
plt.title("East coast Sales excluding outliers. 2016.05-2019.05")
plt.gca().set_ylim(0,300) # This sets vertical range to [0-1]
plt.show()

# We can also remove the very low values
Mydatab.loc[Mydatab.Sales_east <100,"Sales_east"] = 174

Mydatab.plot(figsize=(10,9),color ="mediumpurple")
plt.grid(True)
## Change scale
plt.title("East coast Sales excluding outliers. 2016.05-2019.05",fontweight="bold")
plt.gca().set_ylim(0,300) # This sets vertical range to [0-1]
plt.show()

# Now the serie is ready to carry on with it
Mydatab.head()

# ## 5 Start modelling using Neural networks 
# Check CUDA installation. New CUDA drivers not installed in this Anaconda version
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# ### 5.1  Split original data into train validation and hold sets
# #### 1. Split main dataset into Train Validation and Hold subsets of data
# #### Train (70%, 84), Validation (20%, 24), Hold (10%, 12). Total rows 120
# * Train (70%, 84), 
# * Validation (20%, 24), 
# * Hold (10%, 12). Total rows 120

len(Mydatab)

Mydatab

# **Tip**: Always check the start and end date of the original dataset before splitting it into Train validation and hols datasets
# When we display the entire dataset, we know that the start date is 2016-05-30 abd the end date is 2019-05-02
# ####Training dataset (70%, 84)
# traindata = Emergency_admissions.iloc[0:84]

# Training dataset (70%, 634 of 906)
traindata = Mydatab.iloc[0:634]
len(traindata)
traindata.head()

traindata

# Validation dataset (20%, 181 of 906)
valdata = Mydatab.iloc[634:815]

# Hold dataset (10%, 90 of 906)
hold = Mydatab.iloc[815:]

hold

# So finally this is the perfect way as we split succcessfully the original dataset into *Train* *Test* and *Hold* datasets

# ## Training dataset (70%, 634 of 906)
# traindata = Mydatab.iloc[0:634]
# ## Validation dataset (20%, 181 of 906)
# valdata = Mydatab.iloc[634:815]
# ## Hold dataset (10%, 90 of 906)
# hold = Mydatab.iloc[815:]

# This is the dataset split
# - **Training dataset (70%, 634 of 906)**
# * traindata = Mydatab.iloc[0:634]
# - **Validation dataset (20%, 181 of 906)**
# * valdata = Mydatab.iloc[634:815]
# - **Hold dataset (10%, 90 of 906)**
# * hold = Mydatab.iloc[815:]

# Training dataset (70%, 634 of 906)
traindata = Mydatab.iloc[0:634]
traindata = Mydatab.iloc[0:634]

plt.figure(figsize=(15,7))
plt.title('Sales. Split into train Validation and Hold datasets',fontweight="bold")
plt.plot(traindata, label = "Training dataset")  # Dataset 
plt.plot(valdata, label = "Validation datasett")  # Dataset 
plt.plot(hold, label = "Hold dataset") # Dataset
plt.legend()
plt.ylabel("value")
plt.xlabel("days")
plt.show()

type(traindata)

traindata_test = traindata.copy()
type(traindata_test)

# ### 6.1 Setup Scaler
#  This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between **zero** and **one**.Transform features by scaling each feature to a given range.
# We will use the The Min Max Scaler library from skelarn 

from sklearn.preprocessing import MinMaxScaler

# The dataset will be scaled between 0  and 1 
# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0,1))

scaler

# ### 6.2 Scale **TRAIN** dataset
# ### 6.2.1. Train dataset scaled using MinMaxScaler. The data will be scaled between 0 and 1
# ### 6.2.2 We apply the MinMaxScaler to the Train dataset. It applies to 2d objects
# The MinMaxScaler function must be aaplied on a DataFrame. It expects a 2d object. So we need to heck for the traindata dataframe whether it is a 2d object by using the .shape() function
# #### Training dataset (70%, 634 of 906)
# #### traindata = Mydatab.iloc[0:634]
# Check first Traindata shape
traindata.shape
print("Length of the original Traindata is:",len(traindata))
Traindata = scaler.fit_transform(traindata)

# ### 6.2.3 Scaled dataset turn into an series 1D object for the Network setup

flat_Traindata = Traindata.flatten()
flat_Traindata.shape
type(flat_Traindata)

# Remember we need to setup the series accordingly to the traindata index, as this is our original traindata data set
Traindata_scaled = pd.Series(flat_Traindata,
                            index=traindata.index)

# ### 6.2.4 Build Target and Features dataset from Traindata_scaled dataset 
Traindata_scaled

# Remember that we just transformed the scaled dataset into a Series object
type(Traindata_scaled)
print("Length of the Series object is:",len(Traindata_scaled))

# Define number of lags (5)
total_lags = 5

# ### 6.2.5 TARGET TRAIN dataset defined as Y_train
# We start by defining the Target dataset from the Traindata scaled dataset
Y_train = Traindata_scaled.iloc[5:,]
Y_train

type(Y_train)

print("Train dataset defined as Y_train length:",len(Y_train))

# This is the result of gettinf from the 5th row onwards data for Train dataset
# ###  6.2.6 FEATURES TRAIN dataset defined as X_train is a  reversed dataframe
# For this step we use the function we build on the P200 Adhoc Functions script 
total_lags = 5

# Remember we defined total_lags as 5 for the above function
# Also the data input to our get_features function, is the traindata we have scaled betwen 0 and 1 earlier
# Traindata_scaled = pd.Series(flat_Traindata,
 #                            index=traindata.index)
# For recurrent neural networks, the features dataframe (X_train) from the Train dataset, must be build in reversed order starting with t-5 and then going all the way down to t-1 t-4,t-3,t-2,t-1 columns. 
# ### For a recurrent neural network the order of theTrain Features X_tran dataframe must be t-5,t-4,t-3,t-2,t-1
# Then we apply the get_features() function

def get_features(data,total_lags):
    columns = [] 
    for each_lag in range(total_lags,0, -1):
        Lag_i = pd.DataFrame(data.shift(each_lag +1,axis=0,fill_value=0))
        columns.append(Lag_i)
    features =pd.concat(columns,axis=1)
    # Include column labels
    labfeatures = features.copy()
    N_cols= len(labfeatures.columns)
    col_list = ['Sales t-' + str(x) for x in range(N_cols,0,-1)]
    labfeatures.columns = col_list
   # remove rows including zero values
    trunc_feat = labfeatures.iloc[total_lags:]
    return trunc_feat

# The data input dataset is the scales Traindata_scaled Series we defined earlier. This is the features dataset X_train that is a reversed dataframe ['Sales t-5', 'Sales t-4', 'Sales t-3', 'Sales t-2', 'Sales t-1']
X_train_rev = get_features(Traindata_scaled,total_lags)
X_train_rev

type(X_train_rev)

# Check number of dimensions for your features X_train transformed dataset
X_train_rev.shape

# Important the TRAIN features dataset (X_train)  must have THREE dimensions
# ###  6.2.6.1  FEATURES train dataset defined as X_train must have **THREE** dimensions
# **Important** We need to reshape input to be 3D structures.reshape input to be 3D (samples, timesteps, features) 
# As we can see in the above code, X_train dataframde has got just two Dimensions
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# We use the reshape command to include an extra dimension to X_train_rev dataframe
X_train = X_train_rev.copy()
X_train
X_train.shape
type(X_train)

# ###  6.2.6.2  FEATURES train dataset must be an Array to include an extra dimension
# To include a new dimensions we need to turn it into an ARRAY

X_train_MODEL = np.array(X_train)
X_train_MODEL.shape

# Now we use it. We keep the first dimension (629) and the second one (5), and we want to include an extra dimension, with value 1 at the end
# ### 629 samples 
# ### 5 sequence lags
# ### features
# This is how we include a 3D We use the reshape command to include an extra dimension to X_train_rev dataframe
# X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
# We want to include an extra dimension, at the end for the **features** parameter
# **Important** The new dimension must be the latest on to the right

X_train_MODEL = X_train_MODEL.reshape((X_train_MODEL.shape[0], X_train_MODEL.shape[1],1))
X_train_MODEL.shape

### Check in your Features TRAIN dataset you have THREE dimensions now
X_train_MODEL_backup = X_train_MODEL.copy()

# Finally these are the two files from the **TRAIN dataset**
# * Y_train_MODEL (Target dataset 1d from TRAIN dataset) 
# * X_train_MODEL (Features dataset 1d from TRAIN dataset). Labels: Sales t-5	Sales t-4	Sales t-3	Sales t-2	Sales t-1
#    This features dataset has got **THREE** dimensions

Y_train_MODEL = Y_train.copy()
Y_train_MODEL.shape
X_train_MODEL.shape

# Now we have the right data shape for the train dataset, we need to proceed in the same way for the Validation dataset
# ### 6.3 Scale **VALIDATION** dataset
# We carry on the same procedure for the Validation dataset. At the end of this 6.3 section we will obtain these two arrays from the original valdata datafram
# * X_val_MODEL
# * Y_val_MODEL
# We start by working with the original valdata data set we created earlier
# - **Validation dataset (20%, 181 of 906)**
# * valdata = Mydatab.iloc[634:815]
valdata.shape

print("Length of the original Valdata is:",len(valdata))
# ### 6.3.1. Validation dataset scaled using MinMaxScaler. The data will be scaled between 0 and 1
# ### 6.3.2 We apply the MinMaxScaler to the Train dataset. It applies to 2d objects
# The MinMaxScaler function must be aaplied on a DataFrame. It expects a 2d object. So we need to heck for the traindata dataframe whether it is a 2d object by using the .shape() function
valdata.shape
Valdata = scaler.fit_transform(valdata)
flat_Valdata = Valdata.flatten()
flat_Valdata.shape
type(flat_Valdata)

Valdata_scaled = pd.Series(flat_Valdata,
                            index=valdata.index)

# ### 1.1 Target Validation dataset (Y_val)
Y_val = Valdata_scaled.iloc[5:,]

# ### 1.2 Features Validation dataset (X_val)
total_lags = 5

def get_features(data,total_lags):
    columns = [] 
    for each_lag in range(total_lags,0, -1):
        Lag_i = pd.DataFrame(data.shift(each_lag +1,axis=0,fill_value=0))
        columns.append(Lag_i)
    features =pd.concat(columns,axis=1)
    # Include column labels
    labfeatures = features.copy()
    N_cols= len(labfeatures.columns)
    col_list = ['Sales t-' + str(x) for x in range(N_cols,0,-1)]
    labfeatures.columns = col_list
   # remove rows including zero values
    trunc_feat = labfeatures.iloc[total_lags:]
    return trunc_feat

X_val_rev = get_features(Valdata_scaled,total_lags)

type(X_val_rev)
X_val_rev.shape
X_val = X_val_rev.copy()

type(X_val)
X_val
X_val_MODEL = np.array(X_val)
type(X_val_MODEL)

# **Important** The new dimension for X_val_MODEL must be the latest on to the right
X_val_MODEL = X_val_MODEL.reshape((X_val_MODEL.shape[0],X_val_MODEL.shape[1],1))
X_val_MODEL.shape
X_val_MODEL_backup = X_val_MODEL.copy()

# Finally these are the two files from the **VALIDATION dataset**
# * Y_validation_MODEL (Target dataset 1d from VALIDATION dataset) 
# * X_validation_MODEL (Features dataset 1d from VALIDATION dataset). Labels: Sales t-5	Sales t-4	Sales t-3	Sales t-2	Sales t-1
#    This features dataset has got **THREE** dimensions

Y_val_MODEL = Y_val.copy()
Y_val_MODEL.shape
X_val_MODEL.shape
X_val_MODEL

# ## 7 Setup RNN Neural Network
# Basic network Structure

from tensorflow.keras.layers import SimpleRNN 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ### 7.1 Setup neural network model structure
# Three layers the first two with 20 neurons on the first two layers
# last layer 1 neuron, because we forecast univariate TS
model1 = Sequential()
model1.add(SimpleRNN(5,input_shape=(5,1), activation='relu'))
model1.add(Dense(1, activation="relu"))

# Display model summary
model1.summary()

# ### 7.2 Compile the model
# Model compile step specifies the loss function and the optimizer to use 
model1.compile(loss='mean_squared_error',optimizer='adam',
             metrics=['accuracy'])

# ### 7.3 Train and evaluate the model using the Train and Validation sets
Y_train_MODEL 
X_train_MODEL
Y_val_MODEL
X_val_MODEL

X_train_MODEL.shape
Y_train_MODEL.shape

X_val_MODEL.shape
Y_val_MODEL.shape

# fit the keras model on the dataset
history = model1.fit(X_train_MODEL, Y_train_MODEL, epochs=10000,
                    validation_data=(X_val_MODEL,Y_val_MODEL))

# ### 7.4 Plot history (10,000) eochs
pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
## Change scale
plt.title("History Accuracy measures")
plt.gca().set_ylim(0,0.050) # This sets vertical range to [0-1]
plt.show()

pd.DataFrame(history.history).plot(figsize=(20,10))
plt.grid(True)
plt.gca().set_ylim(0,0.04) # This sets vertical range to [0-1]
plt.show()

# ### 7.5 Lilst all data in history
print(history.history.keys())

# ### 7.6 Make some predictions with the model
# Using the model we predict future values based on the length of the validation dataset
predictions = model1.predict(X_val_MODEL)
predictions

# #### 7.6.1 Unscale the data
# 
predictions_unscaled=scaler.inverse_transform(predictions)
predictions_unscaled.shape
(20, 1)

# Flatten predictions
predictions_flat = predictions_unscaled.flatten()

# #### 7.6.3 Plot the unscaled predictions
# 

plt.plot(predictions_unscaled)
len(predictions_unscaled)
len(X_val_MODEL)

# #### 7.6.3 Display valdata and predicted values
len(valdata)
valdata.shape

type(valdata)
len(predictions_flat)

type(predictions_flat)

# Note: valdata must be the same length as the predictions array. As the predictions array is 176 length, we need to substract 5 rows from valdata data set to plot them together.
prediction_plot = pd.Series(predictions_flat, 
                                 # index=valdataf.index[5:])
                                index=valdata.index[0:176])

forecastRNN = pd.Series(prediction_plot, index = valdata.index)

# ### 7.7 PLot predictions
forecastRNN.head()
type(forecastRNN)

# As the forecasted values is a Series, we need to rename it prior to turn it into a plot, otherwise the series will not have heading. We introduce a name for this series
RNNforecast = forecastRNN.rename('Training RNN forecast')
RNNforecast.head()

plt.figure(figsize=(15,7))
plt.title('RNN model forecast. Sales 2016-2019')
plt.plot(traindata, label = "Training dataset")  # Dataset 
RNNforecast.plot() # Series 

plt.plot(valdata, label = "Validation dataset") # Dataset
plt.legend()
plt.ylabel("value")
plt.xlabel("days")
plt.show()

plt.figure(figsize=(15,7))
plt.title('RNN model forecast. Sales 2016-2019')
RNNforecast.plot() # Series 
plt.plot(valdata, label = "Validation dataset") # Dataset
plt.legend()
plt.ylabel("value")
plt.xlabel("days")
plt.show()

plt.figure(figsize=(15,7))
plt.title('RNN model forecast. Sales 2016-2019')
RNNforecast.plot() # Series 
plt.legend()
plt.ylabel("value")
plt.xlabel("days")
plt.show()

# ## 8 Compute MAPE value 
type(RNNforecast)

type(valdata)

valdata.isnull().sum()

# Turn forecasted values into a dataframe
RNNForecast_dataframe = pd.DataFrame(RNNforecast)
RNNForecast_dataframe.isnull().sum()

RNNForecast_dataframe

# ### 8.1 Remove null values prior to computing MAPE value 
RNNForecast_nonul =  RNNForecast_dataframe.copy()
RNNForecast_nonul.isnull().sum()

for row in range(0,len(RNNForecast_nonul)):
    RNNForecast_nonul['Training RNN forecast']=np.where(
                          (np.isnan(RNNForecast_nonul['Training RNN forecast'])),
                           RNNForecast_nonul['Training RNN forecast'].shift(7),RNNForecast_nonul['Training RNN forecast']
                         )

RNNForecast_nonul.isnull().sum()

RNNForecast_nonul

plt.figure(figsize=(15,7))
plt.title('RNN model forecast no nulls. Sales 2016-2019')
plt.plot(RNNForecast_nonul)
plt.legend()
plt.ylabel("value")
plt.xlabel("days")
plt.show()

len(RNNForecast_nonul)


len(valdata)

RNNForecast_nonul

valdata

type(valdata)


type(RNNForecast_nonul)

# ### 8.2 Rename columns in both datasets to calculte MAPE value
ValdataNEW = valdata.rename(columns={'Sales_east':'Value'})
RNNForecastNEW = RNNForecast_nonul.rename(columns={'Training RNN forecast':'Value'})

# Both datasets must have the same heading label to be able to use the MAPE formula below
def accuracy_MAPE(ACT,FCAST):
    Value_percent = (abs((ACT-FCAST)/ACT).sum()/len(ACT))*100
    #Mape_value = print(f"RNN model MAPE percent   {Value_percent}")
    #Mape_value_per = print(f"Seasonal ARIMA model MAPE {Value}")
    return Value_percent

accuracy_MAPE(ValdataNEW,RNNForecastNEW)

# **Include some percentage formatting**
def MAPE_value(ACT,FCAST):
  #  Value = abs((ACT-FCAST)/ACT).sum()/len(ACT)
    Value  = (abs((ACT-FCAST)/ACT).sum()/len(ACT))*100
    Mape_value = print(f"RNN model MAPE % {Value}%")
    #Mape_value_per = print(f"Seasonal ARIMA model MAPE {Value}")
    return Mape_value

MAPE_value(ValdataNEW,RNNForecastNEW)



