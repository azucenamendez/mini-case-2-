# mini-case-2-

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df.head()
mpg	cylinders	displacement	horsepower	weight	acceleration	model_year	origin	name
0	18.0	8	307.0	130.0	3504	12.0	70	usa	chevrolet chevelle malibu
1	15.0	8	350.0	165.0	3693	11.5	70	usa	buick skylark 320
2	18.0	8	318.0	150.0	3436	11.0	70	usa	plymouth satellite
3	16.0	8	304.0	150.0	3433	12.0	70	usa	amc rebel sst
4	17.0	8	302.0	140.0	3449	10.5	70	usa	ford torino

df.describe()
mpg	cylinders	displacement	horsepower	weight	acceleration	model_year
count	398.000000	398.000000	398.000000	392.000000	398.000000	398.000000	398.000000
mean	23.514573	5.454774	193.425879	104.469388	2970.424623	15.568090	76.010050
std	7.815984	1.701004	104.269838	38.491160	846.841774	2.757689	3.697627
min	9.000000	3.000000	68.000000	46.000000	1613.000000	8.000000	70.000000
25%	17.500000	4.000000	104.250000	75.000000	2223.750000	13.825000	73.000000
50%	23.000000	4.000000	148.500000	93.500000	2803.500000	15.500000	76.000000
75%	29.000000	8.000000	262.000000	126.000000	3608.000000	17.175000	79.000000
max	46.600000	8.000000	455.000000	230.000000	5140.000000	24.800000	82.000000

df.hist("weight")

plt.scatter(x=df.horsepower ,y=df.weight)

sns.pairplot(df)
Based on the calculations done, mpg and weight are the best candidates for linear regression and make a correlation matrix.

df.corr()

sns.heatmap(df.corr())
<ipython-input-17-aa4f4450a243>:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
  sns.heatmap(df.corr())
<Axes: >
There are different linear relations between mpg ans weight, for example and also other variables like acceleration and mpg and acceleration with model_year.

X=df.weight
y=df.mpg

import statsmodels.api as sm
model=sm.OLS(y,X).fit()
model.predict(X)

0      23.641070
1      24.916231
2      23.182282
3      23.162042
4      23.269992
         ...    
393    18.823797
394    14.370856
395    15.484091
396    17.710562
397    18.351516
Length: 398, dtype: float64
