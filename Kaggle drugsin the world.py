#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv('PrevAllDrugtypes new.csv')  # Replace with the actual file name

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())


# In[5]:


# Read the dataset and skip the first two rows
df = pd.read_csv('PrevAllDrugtypes new.csv', skiprows=2, header=[0, 1])

# Reset column levels
df.columns = df.columns.droplevel(1)

# Replace commas with dots in numeric columns
numeric_columns = df.columns[1:]
df[numeric_columns] = df[numeric_columns].replace(',', '.', regex=True)

# Replace non-numeric values with NaN
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Convert numeric columns to float
df[numeric_columns] = df[numeric_columns].astype(float)

# Display basic information about the cleaned dataset
print(df.info())

# Display the first few rows of the cleaned dataset
print(df.head())


# In[6]:


# Display unique values in the 'Cannabis' column
cannabis_values = df['Cannabis'].unique()
print("Unique values in 'Cannabis' column:")
print(cannabis_values)

# Display unique values in the 'Opioids (opiates and prescription opioids)' column
opioids_values = df['Opioids (opiates and prescription opioids)'].unique()
print("\nUnique values in 'Opioids' column:")
print(opioids_values)

# Display unique values in the 'Opiates' column
opiates_values = df['Opiates'].unique()
print("\nUnique values in 'Opiates' column:")
print(opiates_values)


# In[7]:


# Load the first dataset
file_path1 = 'Preval nac.usodrogasincl.NPS.csv'
df1 = pd.read_csv(file_path1, skiprows=2)

# Display basic information about the first dataset
print(df1.info())

# Display the first few rows of the first dataset
df1.head()


# In[8]:


# Display the first few rows of the "Preval nac.usodrogasincl.NPS.csv" dataset
df1.head()


# In[10]:


# Display the current column names
df_cleaned.columns


# In[13]:


# Corrected list of columns to drop
columns_to_drop = ['Unnamed: 6_level_0', 'Unnamed: 8_level_0', 'Unnamed: 9_level_0', 'Unnamed: 10_level_0', 'Unnamed: 12_level_0', 'Unnamed: 14_level_0', 'Unnamed: 15_level_0', 'Unnamed: 16_level_0', 'Unnamed: 17_level_0', 'Unnamed: 18_level_0']

# Drop unnecessary columns
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Display the cleaned dataset
df_cleaned


# In[14]:


# Rename the columns
df_cleaned.columns = ['Region', 'Cannabis', 'Cannabis_Low', 'Cannabis_High', 'Cannabis_Mean', 'Cannabis_CV', 'Opioids', 'Opioids_Mean', 'Opiates']

# Display the cleaned and renamed dataset
df_cleaned


# In[17]:


# Print the columns of the DataFrame
print(df_cleaned.columns)


# In[25]:


# Extract relevant columns for visualization
visualization_data = df_cleaned[['Region', 'Cannabis', 'Opioids', 'Opiates']]

# Set the region column as the index
visualization_data.set_index('Region', inplace=True)

# Display the visualization data
print(visualization_data)


# In[27]:


import matplotlib.pyplot as plt

# Drop the 'GLOBAL ESTIMATE' row as it may distort the visualization
visualization_data = visualization_data.drop('GLOBAL ESTIMATE', errors='ignore')

# Plot the data
visualization_data.plot(kind='bar', figsize=(12, 8))
plt.title('Drug Use by Region')
plt.ylabel('Prevalence')
plt.xlabel('Region')
plt.legend(title='Drug Type')

# Show the plot
plt.show()


# In[29]:


# Print the index and column names of the DataFrame
print(df_cleaned.index)
print(df_cleaned.columns)


# In[30]:


# Set the 'Region' column as the index
df_cleaned.set_index('Region', inplace=True)

# Filter the data for the top five regions
top_five_regions = df_cleaned.loc[['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']]

# Extract relevant columns for Cannabis visualization
cannabis_data = top_five_regions[['Cannabis']]

# Display the filtered data
print(cannabis_data)


# In[31]:


import matplotlib.pyplot as plt

# Filter the data for the top five regions
top_five_regions = df_cleaned.loc[['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']]

# Extract relevant columns for Cannabis visualization
cannabis_data = top_five_regions[['Cannabis']]

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
cannabis_data.plot(kind='bar', ax=ax, legend=False)
ax.set_ylabel('Cannabis Use (%)')
ax.set_xlabel('Region')
ax.set_title('Cannabis Use in Top Five Regions')

# Show the plot
plt.show()


# In[ ]:




