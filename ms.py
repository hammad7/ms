

import csv, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## https://learn.upgrad.com/v/course/1121/session/126403/segment/693349

asd = pd.read_csv("/home/mohd/Downloads/tendulkar_ODI.csv")

asd["Runs"] = asd["Runs"].replace({'\*': ''}, regex=True)
asd["Runs"] = asd["Runs"].replace({'DNB': np.NAN}, regex=True)
plt.hist(asd["Runs"].dropna().apply(int), density=True, bins=range(0, 100, 10))
plt.show()


asd["4s"] = asd["4s"].replace({'-': np.NAN}, regex=True)
plt.hist(asd["4s"].dropna().apply(int), density=True, bins=range(-1, 15, 1))
plt.show()

## preferential attachment - gain by popularity
## univariate analysis - ordered cat var,  mean - avg contribution, median - typical dataset, eg incomes with outlier bill gates
## mode for unordered categorical data. 
## (sd can exxagerate spread, so comunicate 75-25 diff) 

asd = pd.read_csv("/home/mohd/Downloads/popularity.csv")
asd[" shares"].median()
asd.descibe() ####
asd[" shares"].quantile([0.78,0.7,0.8])

## remove outliers
low = .05
high = .95
quant_df = asd.quantile([low, high])
asd.apply(lambda x: x[(x>quant_df.loc[low,x[" shares"]]) & (x < quant_df.loc[high,x[" shares"]])], axis=0)