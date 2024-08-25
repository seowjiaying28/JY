#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[4]:


#directory containing the CSV files

folder_path = './HDB/'


# In[5]:


#List all files in the directory
file_list = os.listdir(folder_path)


# In[6]:


#Initialise an empty list to store DataFrames
hdb_dfs = []


# In[7]:


# Iterate over each file name in the list
for file_name in file_list:
    # Construct the full path to the current file
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the path is a file and has a .csv extension
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_df = pd.read_csv(file_path)
        
        # Append the DataFrame to the list
        hdb_dfs.append(file_df)
    else:
        # Print a message if the file does not exist or is not a CSV file
        print(f"Skipping non-CSV file or directory: {file_path}")


# In[8]:


#concatenate all dataframes into a single dataframe

merged_hdb = pd.concat(hdb_dfs,ignore_index=True)
merged_hdb.head()


# In[9]:


# Separate the 'month' column into 'year' and 'month_number' columns
merged_hdb[['year', 'month_number']] = merged_hdb['month'].str.split('-', expand=True)


# In[10]:


merged_hdb


# In[11]:


# Drop the 'month' column from the DataFrame
merged_hdb.drop(columns=['month'], inplace=True)


# In[12]:


merged_hdb


# In[13]:


# Checks for null (missing) values in the DataFrame df

merged_hdb[merged_hdb.isnull().any(axis=1)]


# In[14]:


#Standardize Column Names
merged_hdb.columns = ['Town', 'Flat Type', 'Block', 'Street Name', 'Storey Range', 'Floor Area (sqm)', 'Flat Model', 'Lease Commence Date', 'Resale Price','Remaining Lease', 
                    'Year', 'Month Number']


# In[15]:


merged_hdb.drop_duplicates(inplace=True)


# In[16]:


#Convert Year and Month Number to integer type
merged_hdb['Year'] = merged_hdb['Year'].astype(int)
merged_hdb['Month Number'] = merged_hdb['Month Number'].astype(int)


# In[17]:


merged_hdb['Flat Type'] = merged_hdb['Flat Type'].str.upper()  # Convert all flat types to uppercase


# In[18]:


merged_hdb


# In[19]:


merged_hdb['Flat Model'] = merged_hdb['Flat Model'].str.upper()  # Convert all flat types to uppercase


# In[20]:


merged_hdb


# In[21]:


current_year = pd.Timestamp.now().year
merged_hdb['Flat Age'] = current_year - merged_hdb['Lease Commence Date']


# In[22]:


merged_hdb


# In[23]:


# Rename columns as required by Prophet: 'ds' for date and 'y' for target variable
merged_hdb = merged_hdb.rename(columns={'year': 'year', 'Resale Price': 'y'})


# In[24]:


pip install prophet


# In[25]:


from prophet import Prophet


# In[26]:


# Generate the 'ds' column by combining 'Year' and 'Month Number'
merged_hdb['ds'] = pd.to_datetime(
    merged_hdb['Year'].astype(str) + '-' +
    merged_hdb['Month Number'].astype(str).str.zfill(2) + '-01'
)

# Rename 'Value' column to 'y' to match Prophet's requirements
merged_hdb = merged_hdb.rename(columns={'Value': 'y'})

# Verify the DataFrame structure
print(merged_hdb.head())
print(merged_hdb.columns)

# Create and configure the Prophet model
my_model = Prophet(
    interval_width=0.95, 
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add custom seasonality
my_model.add_seasonality(
    name='monthly',       # Name of the custom seasonality
    period=30.5,          # Approximate number of days in a month
    fourier_order=8       # Number of Fourier terms to use
)

# Fit the Prophet model with the prepared dataset
my_model.fit(merged_hdb)


# In[27]:


# Create a dataframe with future dates (adjust the period as needed)
future_dates = my_model.make_future_dataframe(periods=12, freq='M')  # Forecasting 12 months ahead
print(future_dates.tail())


# In[28]:


# Generate predictions for the future dates
forecast = my_model.predict(future_dates)

# Print the forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[29]:


# Create a DataFrame with future dates extending 10 years ahead
future_dates = my_model.make_future_dataframe(periods=120, freq='M')  

# Display the last few rows of the future dates DataFrame to verify
print(future_dates.tail())


# In[30]:


# Generate predictions for the future dates
forecast = my_model.predict(future_dates)

# Print the forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[31]:


#Create Time Series Forecasting 
fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})
my_model.plot(forecast);
my_model.plot_components(forecast);  


# In[38]:


# Create a 'Date' column using 'Year'
merged_hdb['Date'] = pd.to_datetime(merged_hdb['Year'].astype(str), format='%Y')

# Set 'Date' as index
merged_hdb.set_index('Date', inplace=True)

# Aggregate resale prices by year (if there are multiple entries per year)
yearly_data = merged_hdb['y'].resample('Y').mean()

# Apply Exponential Smoothing
model = ExponentialSmoothing(yearly_data, trend='add', seasonal=None)  # 'seasonal=None' for yearly data
fit = model.fit()

# Forecasting for the next 10 years
forecast_periods = 10
forecast = fit.forecast(steps=forecast_periods)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(yearly_data.index, yearly_data, label='Original', color='blue')
plt.plot(fit.fittedvalues.index, fit.fittedvalues, label='Fitted', color='orange')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Resale Price')
plt.title('Exponential Smoothing Forecast for 10 Years')
plt.legend()
plt.grid(True)
plt.show()


# In[39]:


# Print the forecasted values year by year
print("Yearly Forecasted Values:")
for year, value in zip(forecast.index.year, forecast):
    print(f"{year}: ${value:,.2f}")


# In[80]:


# Assuming 'merged_hdb' is your DataFrame and 'y' is the resale price column
# Create a 'Date' column using 'Year'
merged_hdb['Date'] = pd.to_datetime(merged_hdb['Year'].astype(str), format='%Y')

# Set 'Date' as index
merged_hdb.set_index('Date', inplace=True)

# Aggregate resale prices by year (if there are multiple entries per year)
yearly_data = merged_hdb['y'].resample('Y').mean()

# Split the data into training and testing sets
train_end = '2020'  # End of training data 
train = yearly_data[:train_end]
test = yearly_data[train_end:]

# Apply Exponential Smoothing
model = ExponentialSmoothing(train, trend='add', seasonal=None)  # 'seasonal=None' for yearly data
fit = model.fit()

# Forecast for the test period
forecast = fit.forecast(steps=len(test))

# Evaluate the model
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = mse**0.5

# Calculate MAPE (mean absolute percentage error)
# Ensure no zero values in test to avoid division by zero
if (test == 0).any():
    mape = np.inf
else:
    mape = (abs(test - forecast) / test).mean() * 100

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(yearly_data.index, yearly_data, label='Original', color='blue')
plt.plot(fit.fittedvalues.index, fit.fittedvalues, label='Fitted', color='orange')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Resale Price')
plt.title('Exponential Smoothing Forecast vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()


# In[83]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




