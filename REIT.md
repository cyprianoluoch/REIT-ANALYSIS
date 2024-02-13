

```python
import os
import pandas as pd
import zipfile

# Define the file path to the ZIP file
zip_file_path = r"C:\Users\adm\Downloads\real_estate_investment_trust_analysis.zip"

# Define the directory where the dataset will be extracted
extracted_dir = r"C:\Users\adm\Downloads\REIT_analysis"

# Extract the contents of the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Check the contents of the extracted directory
print("Contents of the extracted directory:")
print(os.listdir(extracted_dir))

# Load the dataset into a DataFrame
dataset_file_path = os.path.join(extracted_dir, "your_dataset.csv")
if os.path.exists(dataset_file_path):
    df = pd.read_csv(dataset_file_path)
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())
    
    # Display information about the dataset
    print("\nDataset information:")
    print(df.info())
    
    # Generate summary statistics for numerical columns
    print("\nSummary statistics:")
    print(df.describe())
else:
    print("The dataset file does not exist in the specified directory.")

```

    Contents of the extracted directory:
    ['real_estate_investment_trust_analysis', '__MACOSX']
    The dataset file does not exist in the specified directory.
    


```python
import pandas as pd

# Define the file path to the dataset
file_path = r"C:\Users\adm\Documents\real_estate.csv"

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display information about the dataset
print("\nDataset information:")
print(df.info())

# Generate summary statistics for numerical columns
print("\nSummary statistics:")
print(df.describe())

```

    First few rows of the dataset:
       tx_price  beds  baths  sqft  year_built  lot_size  \
    0    295850     1      1   584        2013         0   
    1    216500     1      1   612        1965         0   
    2    279900     1      1   615        1963         0   
    3    379900     1      1   618        2000     33541   
    4    340000     1      1   634        1992         0   
    
                       property_type exterior_walls                 roof  \
    0  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    1  Apartment / Condo / Townhouse          Brick  Composition Shingle   
    2  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    3  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    4  Apartment / Condo / Townhouse          Brick                  NaN   
    
       basement  ...  beauty_spas  active_life  median_age  married  college_grad  \
    0       NaN  ...           47           58          33       65            84   
    1       1.0  ...           26           14          39       73            69   
    2       NaN  ...           74           62          28       15            86   
    3       NaN  ...           72           83          36       25            91   
    4       NaN  ...           50           73          37       20            75   
    
       property_tax  insurance  median_school  num_schools  tx_year  
    0           234         81            9.0            3     2013  
    1           169         51            3.0            3     2006  
    2           216         74            8.0            3     2012  
    3           265         92            9.0            3     2005  
    4            88         30            9.0            3     2002  
    
    [5 rows x 26 columns]
    
    Dataset information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1883 entries, 0 to 1882
    Data columns (total 26 columns):
    tx_price              1883 non-null int64
    beds                  1883 non-null int64
    baths                 1883 non-null int64
    sqft                  1883 non-null int64
    year_built            1883 non-null int64
    lot_size              1883 non-null int64
    property_type         1883 non-null object
    exterior_walls        1660 non-null object
    roof                  1529 non-null object
    basement              1657 non-null float64
    restaurants           1883 non-null int64
    groceries             1883 non-null int64
    nightlife             1883 non-null int64
    cafes                 1883 non-null int64
    shopping              1883 non-null int64
    arts_entertainment    1883 non-null int64
    beauty_spas           1883 non-null int64
    active_life           1883 non-null int64
    median_age            1883 non-null int64
    married               1883 non-null int64
    college_grad          1883 non-null int64
    property_tax          1883 non-null int64
    insurance             1883 non-null int64
    median_school         1883 non-null float64
    num_schools           1883 non-null int64
    tx_year               1883 non-null int64
    dtypes: float64(2), int64(21), object(3)
    memory usage: 382.6+ KB
    None
    
    Summary statistics:
                tx_price         beds        baths         sqft   year_built  \
    count    1883.000000  1883.000000  1883.000000  1883.000000  1883.000000   
    mean   422839.807754     3.420605     2.579926  2329.398832  1982.963887   
    std    151462.593276     1.068554     0.945576  1336.991858    20.295945   
    min    200000.000000     1.000000     1.000000   500.000000  1880.000000   
    25%    300000.000000     3.000000     2.000000  1345.000000  1970.000000   
    50%    392000.000000     3.000000     3.000000  1907.000000  1986.000000   
    75%    525000.000000     4.000000     3.000000  3005.000000  2000.000000   
    max    800000.000000     5.000000     6.000000  8450.000000  2015.000000   
    
               lot_size  basement  restaurants    groceries    nightlife  ...  \
    count  1.883000e+03    1657.0  1883.000000  1883.000000  1883.000000  ...   
    mean   1.339262e+04       1.0    40.210303     4.505045     5.074881  ...   
    std    4.494930e+04       0.0    46.867012     4.491029     8.464668  ...   
    min    0.000000e+00       1.0     0.000000     0.000000     0.000000  ...   
    25%    1.542000e+03       1.0     7.000000     1.000000     0.000000  ...   
    50%    6.098000e+03       1.0    23.000000     3.000000     2.000000  ...   
    75%    1.176100e+04       1.0    58.000000     7.000000     6.000000  ...   
    max    1.220551e+06       1.0   266.000000    24.000000    54.000000  ...   
    
           beauty_spas  active_life   median_age      married  college_grad  \
    count  1883.000000  1883.000000  1883.000000  1883.000000   1883.000000   
    mean     23.416888    15.835369    38.601168    69.091875     65.085502   
    std      25.776916    17.667717     6.634110    19.659767     16.953165   
    min       0.000000     0.000000    22.000000    11.000000      5.000000   
    25%       4.000000     4.000000    33.000000    58.000000     54.000000   
    50%      15.000000    10.000000    38.000000    73.000000     66.000000   
    75%      35.000000    21.000000    43.000000    84.000000     78.000000   
    max     177.000000    94.000000    69.000000   100.000000    100.000000   
    
           property_tax    insurance  median_school  num_schools      tx_year  
    count   1883.000000  1883.000000    1883.000000  1883.000000  1883.000000  
    mean     466.777483   140.454063       6.502921     2.793415  2007.111524  
    std      231.656645    72.929765       1.996109     0.505358     5.196898  
    min       88.000000    30.000000       1.000000     1.000000  1993.000000  
    25%      320.000000    94.000000       5.000000     3.000000  2004.000000  
    50%      426.000000   125.000000       7.000000     3.000000  2007.000000  
    75%      569.000000   169.000000       8.000000     3.000000  2011.000000  
    max     4508.000000  1374.000000      10.000000     4.000000  2016.000000  
    
    [8 rows x 23 columns]
    


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset into a DataFrame
file_path = "C:/Users/adm/Documents/real_estate.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
```

    First few rows of the dataset:
       tx_price  beds  baths  sqft  year_built  lot_size  \
    0    295850     1      1   584        2013         0   
    1    216500     1      1   612        1965         0   
    2    279900     1      1   615        1963         0   
    3    379900     1      1   618        2000     33541   
    4    340000     1      1   634        1992         0   
    
                       property_type exterior_walls                 roof  \
    0  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    1  Apartment / Condo / Townhouse          Brick  Composition Shingle   
    2  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    3  Apartment / Condo / Townhouse    Wood Siding                  NaN   
    4  Apartment / Condo / Townhouse          Brick                  NaN   
    
       basement  ...  beauty_spas  active_life  median_age  married  college_grad  \
    0       NaN  ...           47           58          33       65            84   
    1       1.0  ...           26           14          39       73            69   
    2       NaN  ...           74           62          28       15            86   
    3       NaN  ...           72           83          36       25            91   
    4       NaN  ...           50           73          37       20            75   
    
       property_tax  insurance  median_school  num_schools  tx_year  
    0           234         81            9.0            3     2013  
    1           169         51            3.0            3     2006  
    2           216         74            8.0            3     2012  
    3           265         92            9.0            3     2005  
    4            88         30            9.0            3     2002  
    
    [5 rows x 26 columns]
    


```python
# Summary statistics
print("\nDataset information:")
print(df.info())
```

    
    Dataset information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1883 entries, 0 to 1882
    Data columns (total 26 columns):
    tx_price              1883 non-null int64
    beds                  1883 non-null int64
    baths                 1883 non-null int64
    sqft                  1883 non-null int64
    year_built            1883 non-null int64
    lot_size              1883 non-null int64
    property_type         1883 non-null object
    exterior_walls        1660 non-null object
    roof                  1529 non-null object
    basement              1657 non-null float64
    restaurants           1883 non-null int64
    groceries             1883 non-null int64
    nightlife             1883 non-null int64
    cafes                 1883 non-null int64
    shopping              1883 non-null int64
    arts_entertainment    1883 non-null int64
    beauty_spas           1883 non-null int64
    active_life           1883 non-null int64
    median_age            1883 non-null int64
    married               1883 non-null int64
    college_grad          1883 non-null int64
    property_tax          1883 non-null int64
    insurance             1883 non-null int64
    median_school         1883 non-null float64
    num_schools           1883 non-null int64
    tx_year               1883 non-null int64
    dtypes: float64(2), int64(21), object(3)
    memory usage: 382.6+ KB
    None
    


```python
# Descriptive statistics
print("\nSummary statistics:")
print(df.describe())
```

    
    Summary statistics:
                tx_price         beds        baths         sqft   year_built  \
    count    1883.000000  1883.000000  1883.000000  1883.000000  1883.000000   
    mean   422839.807754     3.420605     2.579926  2329.398832  1982.963887   
    std    151462.593276     1.068554     0.945576  1336.991858    20.295945   
    min    200000.000000     1.000000     1.000000   500.000000  1880.000000   
    25%    300000.000000     3.000000     2.000000  1345.000000  1970.000000   
    50%    392000.000000     3.000000     3.000000  1907.000000  1986.000000   
    75%    525000.000000     4.000000     3.000000  3005.000000  2000.000000   
    max    800000.000000     5.000000     6.000000  8450.000000  2015.000000   
    
               lot_size  basement  restaurants    groceries    nightlife  ...  \
    count  1.883000e+03    1657.0  1883.000000  1883.000000  1883.000000  ...   
    mean   1.339262e+04       1.0    40.210303     4.505045     5.074881  ...   
    std    4.494930e+04       0.0    46.867012     4.491029     8.464668  ...   
    min    0.000000e+00       1.0     0.000000     0.000000     0.000000  ...   
    25%    1.542000e+03       1.0     7.000000     1.000000     0.000000  ...   
    50%    6.098000e+03       1.0    23.000000     3.000000     2.000000  ...   
    75%    1.176100e+04       1.0    58.000000     7.000000     6.000000  ...   
    max    1.220551e+06       1.0   266.000000    24.000000    54.000000  ...   
    
           beauty_spas  active_life   median_age      married  college_grad  \
    count  1883.000000  1883.000000  1883.000000  1883.000000   1883.000000   
    mean     23.416888    15.835369    38.601168    69.091875     65.085502   
    std      25.776916    17.667717     6.634110    19.659767     16.953165   
    min       0.000000     0.000000    22.000000    11.000000      5.000000   
    25%       4.000000     4.000000    33.000000    58.000000     54.000000   
    50%      15.000000    10.000000    38.000000    73.000000     66.000000   
    75%      35.000000    21.000000    43.000000    84.000000     78.000000   
    max     177.000000    94.000000    69.000000   100.000000    100.000000   
    
           property_tax    insurance  median_school  num_schools      tx_year  
    count   1883.000000  1883.000000    1883.000000  1883.000000  1883.000000  
    mean     466.777483   140.454063       6.502921     2.793415  2007.111524  
    std      231.656645    72.929765       1.996109     0.505358     5.196898  
    min       88.000000    30.000000       1.000000     1.000000  1993.000000  
    25%      320.000000    94.000000       5.000000     3.000000  2004.000000  
    50%      426.000000   125.000000       7.000000     3.000000  2007.000000  
    75%      569.000000   169.000000       8.000000     3.000000  2011.000000  
    max     4508.000000  1374.000000      10.000000     4.000000  2016.000000  
    
    [8 rows x 23 columns]
    


```python
# Example visualization: Histogram of transaction prices
plt.figure(figsize=(10, 6))
sns.distplot(df['tx_price'], bins=20, kde=True, color='blue')
plt.title('Distribution of Transaction Prices')
plt.xlabel('Transaction Price ($)')
plt.ylabel('Density')
plt.show()


```


![png](output_5_0.png)



```python
# Example visualization: Scatter plot of square footage vs transaction price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sqft', y='tx_price', data=df, color='green')
plt.title('Square Footage vs Transaction Price')
plt.xlabel('Square Footage')
plt.ylabel('Transaction Price ($)')
plt.show()

```


![png](output_6_0.png)



```python
# Calculate the total number of orders
total_orders = df.shape[0]

# Calculate the number of unique property types
unique_property_types = df['property_type'].nunique()

print("Total number of orders:", total_orders)
print("Number of unique property types:", unique_property_types)
```

    Total number of orders: 1883
    Number of unique property types: 2
    


```python
# Display all column names in the DataFrame
print(df.columns)


```

    Index(['tx_price', 'beds', 'baths', 'sqft', 'year_built', 'lot_size',
           'property_type', 'exterior_walls', 'roof', 'basement', 'restaurants',
           'groceries', 'nightlife', 'cafes', 'shopping', 'arts_entertainment',
           'beauty_spas', 'active_life', 'median_age', 'married', 'college_grad',
           'property_tax', 'insurance', 'median_school', 'num_schools', 'tx_year'],
          dtype='object')
    


```python
# Calculate the total number of unique products (property types)
total_unique_products = df['property_type'].nunique()

# Calculate the total number of unique categories (exterior walls)
total_unique_categories = df['exterior_walls'].nunique()

print("Total number of unique products (property types):", total_unique_products)
print("Total number of unique categories (exterior walls):", total_unique_categories)

```

    Total number of unique products (property types): 2
    Total number of unique categories (exterior walls): 16
    


```python
# Assuming df is the DataFrame containing the dataset

# Group the dataset by product name and calculate total sales and quantity sold
product_sales = df.groupby('property_type').agg({'tx_price': 'sum', 'sqft': 'sum'})

# Sort products based on total sales and quantity sold in descending order
top_products_sales = product_sales.sort_values(by='tx_price', ascending=False).head(10)
top_products_quantity = product_sales.sort_values(by='sqft', ascending=False).head(10)

#Visualize the top 10 products based on total sales
plt.figure(figsize=(12, 6))
top_products_sales['tx_price'].plot(kind='bar', color='skyblue')
plt.title('Top 10 Property Types by Total Sales')
plt.xlabel('Property Type')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize the top 10 products based on quantity sold
plt.figure(figsize=(12, 6))
top_products_quantity['sqft'].plot(kind='bar', color='lightgreen')
plt.title('Top 10 Property Types by Total Square Feet Sold')
plt.xlabel('Property Type')
plt.ylabel('Total Square Feet Sold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

```


![png](output_10_0.png)



![png](output_10_1.png)



```python
# Assuming df is the DataFrame containing the dataset

# Group the dataset by year and calculate the average transaction price
average_price_by_year = df.groupby('tx_year')['tx_price'].mean()

# Plot the average transaction price over the years
plt.figure(figsize=(10, 6))
plt.plot(average_price_by_year.index, average_price_by_year.values, marker='o', linestyle='-')
plt.title('Average Transaction Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Transaction Price ($)')
plt.grid(True)
plt.xticks(average_price_by_year.index, rotation=45)
plt.tight_layout()
plt.show()

```


![png](output_11_0.png)



```python
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is the DataFrame containing the dataset

# Set the style of the seaborn plots
sns.set(style="whitegrid")

# Create box plots for each property type
plt.figure(figsize=(12, 8))
sns.boxplot(x='property_type', y='sqft', data=df)
plt.title('Distribution of Square Footage by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Square Footage')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='property_type', y='beds', data=df)
plt.title('Distribution of Bedrooms by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Number of Bedrooms')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='property_type', y='baths', data=df)
plt.title('Distribution of Bathrooms by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Number of Bathrooms')
plt.show()

```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)



```python
# Assuming df is the DataFrame containing the dataset

# Set the style of the seaborn plots
sns.set(style="ticks")

# Create pair plots to visualize relationships between variables
sns.pairplot(df, vars=['median_school', 'num_schools', 'restaurants', 'cafes', 'shopping'], hue='exterior_walls', diag_kind='kde')
plt.suptitle('Pair Plot of External Factors by Exterior Walls Type', y=1.02)
plt.show()


```


![png](output_13_0.png)



```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df is the DataFrame containing the dataset

# Select the features for clustering
X = df[['median_age', 'married', 'college_grad']]

# Choose the number of clusters (you can adjust this based on your preference)
n_clusters = 4

# Initialize the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Add the cluster labels to the DataFrame
df['cluster'] = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['median_age'], cluster_data['married'], label=f'Cluster {i}')
plt.xlabel('Median Age')
plt.ylabel('Married')
plt.title('Clustering based on Median Age and Marital Status')
plt.legend()
plt.show()

```


![png](output_14_0.png)



```python
import matplotlib.pyplot as plt

# Plot histograms for property taxes and insurance costs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['property_tax'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Property Taxes')
plt.xlabel('Property Tax')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['insurance'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of Insurance Costs')
plt.xlabel('Insurance Cost')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

```


![png](output_15_0.png)



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set_style("whitegrid")

# Group the data by year and calculate average transaction price and total sales
yearly_data = df.groupby('tx_year').agg({'tx_price': 'mean', 'property_type': 'count'}).reset_index()
yearly_data.rename(columns={'property_type': 'total_sales'}, inplace=True)

# Plotting trends over the years
plt.figure(figsize=(12, 6))

# Plot average transaction price over the years
plt.subplot(1, 2, 1)
sns.lineplot(data=yearly_data, x='tx_year', y='tx_price', marker='o', color='blue')
plt.title('Average Transaction Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Transaction Price ($)')

# Plot total sales over the years
plt.subplot(1, 2, 2)
sns.barplot(data=yearly_data, x='tx_year', y='total_sales', color='green')
plt.title('Total Sales Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.show()

```


![png](output_16_0.png)



```python
# Calculate the proportion of properties with basements
basement_presence = df['basement'].value_counts(normalize=True) * 100
print("Percentage of properties with basements:")
print(basement_presence)

```

    Percentage of properties with basements:
    1.0    100.0
    Name: basement, dtype: float64
    


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of transaction prices
plt.figure(figsize=(10, 6))
sns.distplot(df['tx_price'], kde=True, bins=20)
plt.title('Distribution of Transaction Prices')
plt.xlabel('Transaction Price ($)')
plt.show()


```


![png](output_18_0.png)



```python
# Analyze correlation between presence of basements and other features
correlation = df.corr()['basement']
print("Correlation between presence of basements and other features:")
print(correlation)

# Explore relationship between presence of basements and buyer demographics
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='basement', y='median_age')
plt.title('Relationship between Presence of Basements and Median Age')
plt.xlabel('Basement Presence')
plt.ylabel('Median Age')
plt.xticks(ticks=[0, 1], labels=['No Basement', 'With Basement'])
plt.show()

```

    Correlation between presence of basements and other features:
    tx_price             NaN
    beds                 NaN
    baths                NaN
    sqft                 NaN
    year_built           NaN
    lot_size             NaN
    basement             NaN
    restaurants          NaN
    groceries            NaN
    nightlife            NaN
    cafes                NaN
    shopping             NaN
    arts_entertainment   NaN
    beauty_spas          NaN
    active_life          NaN
    median_age           NaN
    married              NaN
    college_grad         NaN
    property_tax         NaN
    insurance            NaN
    median_school        NaN
    num_schools          NaN
    tx_year              NaN
    cluster              NaN
    Name: basement, dtype: float64
    


![png](output_19_1.png)



```python
print(reits_data.columns)



```

    Index(['tx_price', 'beds', 'baths', 'sqft', 'year_built', 'lot_size',
           'property_type', 'exterior_walls', 'roof', 'basement', 'restaurants',
           'groceries', 'nightlife', 'cafes', 'shopping', 'arts_entertainment',
           'beauty_spas', 'active_life', 'median_age', 'married', 'college_grad',
           'property_tax', 'insurance', 'median_school', 'num_schools', 'tx_year'],
          dtype='object')
    


```python
# Select numerical variables for correlation analysis
numerical_variables = ['tx_price', 'beds', 'baths', 'sqft', 'year_built', 'lot_size',
                       'restaurants', 'groceries', 'nightlife', 'cafes', 'shopping',
                       'arts_entertainment', 'beauty_spas', 'active_life', 'median_age',
                       'married', 'college_grad', 'property_tax', 'insurance',
                       'median_school', 'num_schools', 'tx_year']

# Calculate the correlation matrix
correlation_matrix = df[numerical_variables].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

```


![png](output_21_0.png)



```python

```
