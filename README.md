# Data Processing Guide

[View as notebook here](https://github.com/vineeshvk/ml_data_processing_guide/blob/master/data_processing_guide.ipynb)
[Reference](https://cf-courses-data.static.labs.skills.network/jupyterlite/latest/lab/index.html?notebook_url=https%3A%2F%2Fcf-courses-data.static.labs.skills.network%2FIBM-ML0232EN-SkillsNetwork%2Flabs%2Fmodule%25202%2Flabs%2FData_Cleaning_Lab.jupyterlite.ipynb)

## Contents

-   Initial Analysis
-   Finding Correlation
-   Making it normally distributed
-   Handling Duplicates
-   Handling Missing Data
-   Feature Scaling
-   Handling Outliers

## More operations (Not added yet)

-   Categorical Encoding
-   Feature Engineering
-   Imbalance data

## Initial analysis

Let\'s just analyze the data set first by using `info()` function. Then
Use the `describe()` function to check for individual columns.

The main observation we need to do is the min largely differs from the
25th percentile. Same for the max with 75th percentile If over or under
it, means might not be normally distributed.

``` python
df.info()
df["Column"].describe()
```

We have `describe()` used for numerical data. We can use `value_count()`
for categorical data. It returns the counts of unique values.

``` python
df["categorical data"].value_counts()
```


## Finding Correlation

Next step will be check the correlation of the features to the target
value. We can use the seaborns heatmap to visualize it.

``` python
sns.heatmap(df.corr())
```

### Filtering out non correlated value

``` python
# Not sure if we need to include only the numerical value
df_num = df.select_dtypes(include = ['float64', 'int64']) 

# -1 ignoring the Target to target correlation
df_num_corr = df_num.corr()['Target'][:-1] 

#displays pearsons correlation coefficient greater than 0.5
top_features = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False) 

print("There is {} strongly correlated values with Target:\n{}".format(len(top_features), top_features))
```

Further manually filtering out the lower correlated value by visualizing
a pair plot

``` python
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['Target'])
```


## Making it normally distributed

Plot a normal distribution curve of the target

``` python
sns.distplot(df['Target'])
```

If the distribution has skewness measure it using the `skew()` function

``` python
df['Target'].skew()
```

> **NOTE**: The range of skewness for a fairly symmetrical bell curve
> distribution is between -0.5 and 0.5; moderate skewness is -0.5 to
> -1.0 and 0.5 to 1.0; and highly skewed distribution is \< -1.0 and \>
> 1.0.

If it\'s skewed we can make it normally distributed using `np.log()` or
`np.sqrt()`

More info:
<https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45>

``` python
log_transformed = np.log(df['Target'])
```

## Handling Duplicates

Check for Duplicate Values using `duplicated([])` and drop duplicates
using `drop_duplicates()`

Alternative check for duplicates is `index.is_unique`

``` python
duplicate = df[df.duplicated(['Id'])]
dup_removed = df.drop_duplicates()

df.index.is_unique
```

``` python
```

## Handling Missing Values

#### Find Missing Data

To find and visualize the missing values you can use `isnull()` and sum
and finally plot it in a **bar plot** for vizualizing

``` python
total = df.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)

total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
```

### Ways to handle missing data

-   Remove the data
-   Impute the data: Substittute with mean, median or other estimation
    methods
-   Mask the data - Create a category for the missing values

#### Remove data

Either drop the columns where the value is null for a specfic column\
or\
Drop the whole column if it\'s not necessary

``` python
df.dropna(subset=["Column"])
df.drop("Column", axis=1)
```

#### Impute data

Fill the missing value with median or mean

``` python
median = df["Column"].median()
df["Column"].fillna(median, inplace = True)
```

## Feature Scaling

If all the features are at different scales it is better to equally
scale them. Very important for anything regarding distance calculations
such as K nearest neighbor.

There are mainly two types of scaling:

-   Normalization
-   Standardization

Use the functionality from the sklearn lib

``` python
MinMaxScaler().fit_transform(df)
StandardScaler().fit_transform(df)
```


## Handling Outliers

Handling the outliers, this is a bit tricky. And has many options.
Transformation seems to be the best way. (Log). Even that doesn\'t seem
to be that effective.

And sometimes it\'s real data and you don\'t want to change it. Because
it might affect the desired outcome

#### Handle Outliers

-   Remove the row
-   Substitute value
-   Transform (Log)
-   Predict(from other features, regression)
-   Keep value

#### Unilateral analysis

Use box plot to check if there is any outliers visually

Also use describe to check the percentile

``` python
sns.boxplot(x=housing['Column'])
df['Column'].describe()
```

#### Bilateral analysis

Using a scatter plot to check the outliers and if it also follows the
general trend.If it doens\'t follow the trend then remove it is the
best.

``` python
df.plot.scatter(x="Target")
```

#### Z score analysis

This seems to be the best way to find the outliers

> The usual value should be between -3 to 3 and if it\'s outside that we
> should remove it

------------------------------------------------------------------------

**scipy** has the `zscore()` function

``` python
stats.zscore(df['Column']).describe()
```

