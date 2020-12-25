

import csv, matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
asd.columns = asd.columns.str.replace(' ', '')
asd["shares"].median()
asd.descibe() ####
asd.info() ####
asd.info() ####
asd["shares"].quantile([0.78,0.7,0.8])
asd.value_counts() #### mode for categorical 
asd.dtypes
asd.numberCol.astype("int32")
asd.str.replace("[+,]","")

#### remove outliers
low = .05
high = .95
quant_df = asd.quantile([low, high])
asd.shares[asd["shares"].apply(lambda x: (x <= quant_df.loc[high,"shares"]))].std()
len(asd.shares[asd["shares"].apply(lambda x: (x <= quant_df.loc[high,"shares"]))])/len(asd.shares)

## segmented univariate- pivot table, correlation is not causation, compare mean, 25,75 percentiles, min,max by segmentation

# 1-d, grouped
#box,hist

# plyr
asd = pd.read_csv("/home/mohd/Downloads/EDA_nas.csv")

grouped = asd[["Watch.TV","Science.."]].dropna().groupby("Watch.TV")
grouped['Science..'].agg([np.sum, np.mean, np.std, "count"])

#### group aggs
# pd.pivot_table(df, values='Runs',columns=['year'], aggfunc=np.max)

asd = pd.read_csv("/home/mohd/Downloads/EDA_census.csv")

asd["ans2"] = asd["8"]/asd["2"]
asd[["DD","ans2"]][(asd["EE"]=="Total") & (asd["1"]=="All ages")].sort_values("ans2")



asd = pd.read_csv("/home/mohd/Downloads/EDA_Gold_Silver_prices.csv")
from datetime import datetime
asd["date"] = asd["Month"].apply(lambda x : x.replace("-","-19") if int(x[4:])>50 else x.replace("-","-20")).apply(lambda s: datetime.strptime(s, '%b-%Y'))
asd.corr()

####  date
asd['Order_Date'] = pd.to_datetime(asd['Order_Date'])

plt.matshow(asd[asd.columns[1:]].corr())
plt.show()


asd = pd.read_csv("/home/mohd/Downloads/nas.csv")
grouped = asd[["Mother.edu","Siblings"]].dropna().groupby(["Mother.edu","Siblings"])
grouped[["Mother.edu"]].agg([ "count"]) ### Categorical bivarite analysis


grouped = asd[["Father.edu","Age","Science.."]].dropna()[asd["Age"]!="11- years"].groupby(["Father.edu","Age"])
grouped['Science..'].agg([ np.mean, np.std, "count"])### Continuous bivarite analysis


asd = pd.read_csv("/home/mohd/Downloads/odi-batting.csv")
##nn
asd[asd["Runs"]>=100][["Runs","Player"]].groupby("Player").agg("count").sort_values("Runs")
asd["SR"] = (asd[asd["Runs"]>=100]["Runs"]*100)/asd[asd["Runs"]>=100]["Balls"]
asd.sort_values("SR").dropna()

## year wih max centuries of indian players
asd["year"] = asd[(asd["Runs"]>=100) & (asd["Country"]=="India")]["MatchDate"].apply(lambda x : datetime.strptime(x,"%d-%m-%Y").year)
asd[["year"]].dropna().groupby("year")["year"].agg("count").sort_values()


asd = pd.read_csv("/home/mohd/Downloads/grades.csv")
grouped = asd["date"].groupby(["date"])

#####
?plt.set_style
help(plt.set_style)

## Investment assignment

import chardet ## to check encoding with confidence
with open("/home/mohd/Downloads/ms/companies.csv",'rb') as rawdata:
    result=chardet.detect(rawdata.read(66366))
    print(result)

companies = pd.read_csv("/home/mohd/Downloads/ms/companies.csv",encoding = 'unicode_escape')
rounds2 = pd.read_csv("/home/mohd/Downloads/ms/rounds2.csv",encoding = 'unicode_escape')

## seems better but used ##
# rounds2 = pd.read_csv("/home/mohd/Downloads/ms/rounds2.csv",engine = 'python')
# companies = pd.read_csv("/home/mohd/Downloads/ms/companies.csv",engine = 'python')

#### for datetime, df.col.dt.year
companies.permalink.dropna().str.lower().unique().size
# 66368
rounds2.company_permalink.dropna().str.lower().unique().size
# 66370

len(set(rounds2.company_permalink.str.lower()).difference(set(companies.permalink.str.lower())))
# 7
# len(set(companies.permalink.str.lower()).difference(set(rounds2.company_permalink.str.lower())))
# 5

## look similar
rounds2.company_permalink[113839].lower()== companies.permalink[65778].lower() ## False 
companies.permalink[42529].lower()==rounds2.company_permalink[73633].lower() ## True but coming in set diff ?
companies.permalink[63486].lower() == rounds2.company_permalink[109968].lower() ## True but coming in set diff ?	

## sanitise
companies.permalink = companies.permalink.str.lower()
rounds2.company_permalink = rounds2.company_permalink.str.lower()

master_frame = pd.merge(companies,rounds2, how='right', left_on='permalink', right_on='company_permalink') ##, on=['permalink', 'company_permalink'])
# 114949

pd.options.display.float_format = "{:,.2f}".format

master_frameG = master_frame[["funding_round_type","raised_amount_usd"]][master_frame["funding_round_type"].isin( ["angel","private_equity","seed","venture"])].dropna()
grouped = master_frameG.groupby(["funding_round_type"])
grouped.describe()

###### removing @5 category wise outliers and checking how many removed
low = .02
high = .98
quant_df = grouped.quantile([low, high])
master_frameG2 = master_frameG[master_frameG.apply(lambda x: (x["raised_amount_usd"] <= quant_df.loc[x["funding_round_type"], high]) & (x["raised_amount_usd"] >= quant_df.loc[x["funding_round_type"], low]) , axis = 1 )["raised_amount_usd"]]
groupedG2 = master_frameG2.groupby(["funding_round_type"])
groupedG2.describe()



# also visible
fig, ax = plt.subplots()
asd=groupedG2.boxplot(column=['raised_amount_usd'], by='funding_round_type', ax=ax)
# for name, group in grouped:
#     ax.plot(group.raised_amount_usd, marker='o', linestyle='', ms=2, label=name)
# plt.ylim(0, 10000000)
plt.suptitle('') 
ax.legend()
plt.show()


groupedtop9 = master_frame[["country_code","raised_amount_usd"]][master_frame["funding_round_type"]=="venture"].dropna().groupby(["country_code"])
top9 = groupedtop9.agg([np.sum, np.mean, np.std, "count"])["raised_amount_usd"].sort_values(["sum"],ascending=False).iloc[0:9,:]

top9["country"] = top9.index

fig, ax = plt.subplots()
top9[["country","sum"]].plot.bar(ax=ax)
# for name, group in grouped:
#     ax.plot(group.raised_amount_usd, marker='o', linestyle='', ms=2, label=name)
# plt.ylim(0, 10000000)
# plt.suptitle('') 
# ax.legend()
plt.show()


mapping = pd.read_csv("/home/mohd/Downloads/ms/mapping.csv")


#https://stackoverflow.com/questions/57861364/melt-multiple-boolean-columns-in-a-single-column-in-pandas  VS https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
# r- melt, dcast, reshape2
#########
mapping["category"] = mapping.dropna().iloc[:,1:10].idxmax(1)
## use right join insead of replace
#mapping_dict = dict(zip(mapping.category_list, mapping.category))
master_frame["category"] = master_frame.category_list.str.split('|').str[0]


master_frame = pd.merge(master_frame, mapping[["category_list","category"]], how='left', left_on='category', right_on='category_list') ##, on=['permalink', 'company_permalink'])
master_frame.drop("category_list_y",axis=1, inplace=True)
master_frame.rename(columns={'category_x': 'primary_sector', 'category_y': 'main_sector'}, inplace=True)


d13 = master_frame[["country_code","funding_round_type","raised_amount_usd","main_sector"]] [ 
(master_frame["country_code"].isin(["USA","GBR","CAN"])) & 
(master_frame["funding_round_type"]=="venture") &
(master_frame["raised_amount_usd"] >= 5000000 ) &
(master_frame["raised_amount_usd"] <= 15000000 ) 
]

grouped = d13[["country_code","main_sector","raised_amount_usd"]].groupby(["country_code","main_sector"])
stats = grouped.agg(["sum","count"])["raised_amount_usd"]

### df.set_index()

stats.loc["USA",:].sum()  
stats.loc["GBR",:].sum()
stats.loc["CAN",:].sum()

stats.sort_values(["country_code","count"],ascending=False)

###### top n in each group
#https://stackoverflow.com/questions/33813419/pandas-return-first-n-rows-of-each-secondary-index-of-dataframe
stats.sort_values("count").groupby(level=0).tail(3).sort_index().sort_values(["country_code","count"],ascending=False)

fig, ax = plt.subplots()
stats.loc["USA",["count"]].plot.pie(y='count',ax=ax)
plt.show()


master_frame [ 
(master_frame["country_code"].isin(["USA"])) & 
(master_frame["main_sector"]=="Cleantech / Semiconductors") &
(master_frame["funding_round_type"]=="venture") &
(master_frame["raised_amount_usd"] >= 5000000 ) &
(master_frame["raised_amount_usd"] <= 15000000 ) 
].dropna().sort_values("raised_amount_usd")[["raised_amount_usd","name"]].groupby("name").agg(["sum","count"])["raised_amount_usd"].sort_values("sum")


# DF made of series not lists, hence uniform type in a series/column
#series
# df.loc[(index tuple), opt column list] ############# general
stats.loc["CAN","sum"]
#df
stats.loc["CAN",["sum","count"]]
stats.loc[["CAN"],"sum"]

stats.loc[("CAN","Others"),"sum"]

## by index, same
stats.loc[("CAN","Others")]   ##  for multiple; stats.loc[[("CAN","Others"),("USA","Others")]] OR df.query
stats.loc["CAN","Others"]  #-- only tuple, not list, default in python - "," separated is tuple
#stats.loc[["CAN","Others"]] ---- wrong

#same
stats.loc[stats["sum"]>420000000]
stats[stats["sum"]>420000000]

#same
stats[0:2] ## rows
# df.iloc[row number list, opt column list] ############# general
stats.iloc[0:5]

stats[["sum","count"]] ## column list


stats.iloc[0:5,0:2]

# update 
data.loc[(data.Year < 2015), ['Mileage']] = 22

df.reset_index(inplace=True,drop=True) ### for continues index, eg after dropping


### Plotting
sns.set_style("dark")
#OR
plt.style.use("ggplot") #####
plt.style.use("dark_background") #####
#### continues, 1d
df.col.plot.box()   
plt.boxplot(df.col)
 
#### continues, 1d
df.col.plot.hist()  
#OR
sns.distplot(inp1.Rating) ## pdf, gaussian density, ~ histogram
plt.show()




###### categorical, 1d , asif value_counts() 
sns.countplot(inp1["Android Ver"]) #1
# https://colab.research.google.com/drive/1qh5jGes0fO2Cj3fJgH7fa2erMZyQE1-k#scrollTo=7xahaSI5xnjF
inp1[‘Content Rating’].value_counts().plot.bar() #2
inp1["Content Rating"].value_counts().plot.pie() #3

import scipy.stats as stats
# continuos, 2d -scatter plot
sns.jointplot(inp1.Size,inp1.Rating, stat_func = stats.pearsonr )
sns.jointplot(x=inp1.Size,y=inp1.Rating,kind="reg") ## with regressio line


## pair plot, both kind vars
sns.pairplot(inp1)
