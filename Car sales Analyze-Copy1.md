# import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df=pd.read_csv(r"F:\Data analysis Project\car sale Dataset.csv")
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car Make</th>
      <th>Car Mode</th>
      <th>Year</th>
      <th>Mileage</th>
      <th>Price</th>
      <th>Fuel type</th>
      <th>color</th>
      <th>Transmission</th>
      <th>Options/Features</th>
      <th>Condition</th>
      <th>Accident</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hyundai</td>
      <td>Tucson</td>
      <td>2010</td>
      <td>52554</td>
      <td>44143.82</td>
      <td>Hybrid</td>
      <td>Black</td>
      <td>Automatic</td>
      <td>Heated Seats</td>
      <td>Used</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Land Rover</td>
      <td>Range Rover</td>
      <td>2016</td>
      <td>115056</td>
      <td>25414.06</td>
      <td>Diesel</td>
      <td>Silver</td>
      <td>Manual</td>
      <td>GPS</td>
      <td>Used</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



# Useless column removing 


```python
df.columns
```




    Index(['Car Make', 'Year', 'Mileage', 'Price', 'Fuel type', 'Transmission',
           'Condition', 'Accident'],
          dtype='object')




```python
cols_to_drop = ["Color", "Options/Features", "Car Model"]

existing_cols = [col for col in cols_to_drop if col in df.columns]

df.drop(columns=existing_cols, inplace=True)

print("Unused columns removed")

```

    Unused columns removed
    


```python
df.columns
```




    Index(['Car Make', 'Year', 'Mileage', 'Price', 'Fuel type', 'Transmission',
           'Condition', 'Accident'],
          dtype='object')




```python
df.duplicated().sum()
```




    np.int64(0)




```python
df.isna().sum()
```




    Car Make        0
    Year            0
    Mileage         0
    Price           0
    Fuel type       0
    Transmission    0
    Condition       0
    Accident        0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 199 entries, 0 to 198
    Data columns (total 8 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Car Make      199 non-null    object 
     1   Year          199 non-null    int64  
     2   Mileage       199 non-null    int64  
     3   Price         199 non-null    float64
     4   Fuel type     199 non-null    object 
     5   Transmission  199 non-null    object 
     6   Condition     199 non-null    object 
     7   Accident      199 non-null    object 
    dtypes: float64(1), int64(2), object(5)
    memory usage: 12.6+ KB
    


```python
df.shape
```




    (199, 8)



# Detect Outliers


```python
plt.figure(figsize=(6, 5))
sns.boxplot(y=df['Price'])
plt.title('Box Plot of Car Prices')
plt.ylabel('Price')
plt.show()
```


    
![png](output_12_0.png)
    



```python
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]

print(f"Number of outliers in Price: {len(outliers)}")
print(outliers[['Price']])
```

    Number of outliers in Price: 0
    Empty DataFrame
    Columns: [Price]
    Index: []
    


```python
plt.figure(figsize=(6, 5))
sns.boxplot(y=df['Year'])
plt.title('Box Plot of Year')
plt.ylabel('Year')
plt.show()
```


    
![png](output_14_0.png)
    



```python
Q1 = df['Year'].quantile(0.25)
Q3 = df['Year'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Year'] < lower_bound) | (df['Year'] > upper_bound)]

print(f"Number of outliers in Year: {len(outliers)}")
print(outliers[['Year']])
```

    Number of outliers in Year: 0
    Empty DataFrame
    Columns: [Year]
    Index: []
    

# Mean Price and Mean Year


```python
np.mean(df['Price'])
```




    np.float64(23216.93463316583)




```python
df.columns
```




    Index(['Car Make', 'Year', 'Mileage', 'Price', 'Fuel type', 'Transmission',
           'Condition', 'Accident'],
          dtype='object')




```python
target_price = np.float64(23216.934)

matching_cars = df[df['Price'].round(2) == target_price]

if not matching_cars.empty:
    print(matching_cars[['Car Make', 'Price']])
else:
    print("No exact match found for this price.")

```

    No exact match found for this price.
    


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car Make</th>
      <th>Year</th>
      <th>Mileage</th>
      <th>Price</th>
      <th>Fuel type</th>
      <th>Transmission</th>
      <th>Condition</th>
      <th>Accident</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hyundai</td>
      <td>2010</td>
      <td>52554</td>
      <td>44143.820</td>
      <td>Hybrid</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Land Rover</td>
      <td>2016</td>
      <td>115056</td>
      <td>25414.060</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>Used</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Honda</td>
      <td>2022</td>
      <td>18044</td>
      <td>28262.872</td>
      <td>Electric</td>
      <td>Manual</td>
      <td>Like New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kia</td>
      <td>2011</td>
      <td>79251</td>
      <td>28415.848</td>
      <td>Hybrid</td>
      <td>Manual</td>
      <td>New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volkswagen</td>
      <td>2022</td>
      <td>40975</td>
      <td>31509.792</td>
      <td>Electric</td>
      <td>Automatic</td>
      <td>New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Land Rover</td>
      <td>2020</td>
      <td>97842</td>
      <td>6594.720</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>Like New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mazda</td>
      <td>2014</td>
      <td>35192</td>
      <td>23226.450</td>
      <td>Electric</td>
      <td>Manual</td>
      <td>Used</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Audi</td>
      <td>2016</td>
      <td>109975</td>
      <td>22862.648</td>
      <td>Hybrid</td>
      <td>Manual</td>
      <td>New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Audi</td>
      <td>2018</td>
      <td>55830</td>
      <td>45766.520</td>
      <td>Electric</td>
      <td>Automatic</td>
      <td>New</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Chevrolet</td>
      <td>2012</td>
      <td>24753</td>
      <td>10377.264</td>
      <td>Hybrid</td>
      <td>Manual</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Jaguar</td>
      <td>2020</td>
      <td>84382</td>
      <td>32667.872</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>Like New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BMW</td>
      <td>2019</td>
      <td>137924</td>
      <td>17841.368</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>Like New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Audi</td>
      <td>2021</td>
      <td>27106</td>
      <td>22163.768</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kia</td>
      <td>2014</td>
      <td>106558</td>
      <td>18715.696</td>
      <td>Electric</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Audi</td>
      <td>2020</td>
      <td>106335</td>
      <td>17786.610</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Honda</td>
      <td>2021</td>
      <td>27208</td>
      <td>15144.944</td>
      <td>Hybrid</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Volkswagen</td>
      <td>2013</td>
      <td>94583</td>
      <td>38116.056</td>
      <td>Hybrid</td>
      <td>Automatic</td>
      <td>New</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Audi</td>
      <td>2022</td>
      <td>146497</td>
      <td>34064.312</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fiat</td>
      <td>2021</td>
      <td>119733</td>
      <td>5572.928</td>
      <td>Electric</td>
      <td>Manual</td>
      <td>Used</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Jaguar</td>
      <td>2015</td>
      <td>120499</td>
      <td>25775.600</td>
      <td>Electric</td>
      <td>Manual</td>
      <td>New</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.mean(df['Year'])
```




    np.float64(2015.859296482412)




```python
target_Year = np.float64(2016)

matching_cars = df[df['Year'].round(2) == target_Year]

if not matching_cars.empty:
    print(matching_cars[['Car Make', 'Year']])
else:
    print("No exact match found for this Year.")

```

           Car Make  Year
    1    Land Rover  2016
    7          Audi  2016
    24         Ford  2016
    33          Kia  2016
    44      Hyundai  2016
    49        Honda  2016
    75   Land Rover  2016
    96       Subaru  2016
    125     Hyundai  2016
    127    Mercedes  2016
    146       Tesla  2016
    151         BMW  2016
    169       Mazda  2016
    171       Honda  2016
    172       Mazda  2016
    180  Volkswagen  2016
    

# skewness (Price,Mileage,Year)


```python
skew_price = df['Price'].skew()
print(f"Skewness of Price: {skew_price}")
skew_Year = df['Year'].skew()
print(f"Skewness of Year: {skew_Year}")
skew_Mileage = df['Mileage'].skew()
print(f"Skewness of Mileage: {skew_Mileage}")

```

    Skewness of Price: 0.14404610498124987
    Skewness of Year: 0.04961412745034161
    Skewness of Mileage: 0.011359952880605312
    


```python
plt.figure(figsize=(15, 4))

for i, col in enumerate(['Price', 'Year', 'Mileage'], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```


    
![png](output_25_0.png)
    


# Median Price and Median Year


```python
np.median(df['Price'])
```




    np.float64(22695.976)




```python
target_price = np.float64(22695.980)

matching_cars = df[df['Price'].round(2) == target_price]

if not matching_cars.empty:
    print(matching_cars[['Car Make', 'Price']])
else:
    print("No exact match found for this price.")

```

       Car Make      Price
    50    Honda  22695.976
    


```python
np.median(df['Year'])
```




    np.float64(2016.0)




```python
target_Year = np.float64(2016)

matching_cars = df[df['Year'].round(2) == target_Year]

if not matching_cars.empty:
    print(matching_cars[['Car Make', 'Year']])
else:
    print("No exact match found for this Year.")

```

           Car Make  Year
    1    Land Rover  2016
    7          Audi  2016
    24         Ford  2016
    33          Kia  2016
    44      Hyundai  2016
    49        Honda  2016
    75   Land Rover  2016
    96       Subaru  2016
    125     Hyundai  2016
    127    Mercedes  2016
    146       Tesla  2016
    151         BMW  2016
    169       Mazda  2016
    171       Honda  2016
    172       Mazda  2016
    180  Volkswagen  2016
    

# Standard deviation and variation ( Price )


```python
print("Standard Deviation of Price:", df['Price'].std())

```

    Standard Deviation of Price: 10624.574660463337
    


```python
print("Variance of Price:", df['Price'].var())

```

    Variance of Price: 112881586.71575962
    

# Correlation (Price , Year)


```python
df["Price"].corr(df["Year"])
```




    np.float64(0.09252152380221557)



# Correlation (Price , Mileage)


```python
df["Price"].corr(df["Mileage"])
```




    np.float64(-0.02609586513443372)




```python
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_38_0.png)
    



```python

```
