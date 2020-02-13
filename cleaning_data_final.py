
#GROUP H Code for Final Project

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xlrd
from openpyxl import load_workbook
import sklearn
import array as arr
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
%matplotlib inline

#Calling googleplaystore sheet - -- will need to update this to where it's saved on your drive
xls = pd.ExcelFile(r'C:\Users\206581774\Documents\DataScienceClass\google_play_store.xlsx')
df_googleplaystore = pd.read_excel(xls, 'googleplaystore')


#taking out incidences without ratings
df_googleplaystore.dropna(inplace = True)

#call the columns that we want Price, Category, Content Rating
df_googleplaystore = df_googleplaystore[['App','Category','Rating','Price','Content Rating']]

# Get names of indexes for which column Price has value grater than 10
# Same for ratings above 5
indexNames = df_googleplaystore[ df_googleplaystore['Price'] > 10 ].index
indexRNames = df_googleplaystore[ df_googleplaystore['Rating'] > 5 ].index
#df_googleplaystore.count()


 
# Delete these row indexes from dataFrame for price and index
df_googleplaystore.drop(indexNames , inplace=True)
df_googleplaystore.drop(indexRNames , inplace=True) 
#df_googleplaystore.count()
  
#convert content rating into integrers to separate in groups
dist_ratings = df_googleplaystore['Content Rating'].unique() #.unique gives unique values of the series or array
#print(dist_ratings)

ratingDict = {} #create empty dictionary, dictoranies are indexed by keys
for i in range(len(dist_ratings)):
    ratingDict[dist_ratings[i]] = i
    
#print(ratingDict)
    
#df.map() = maps values of series or dict according to input correspondance 
#.astype = define type of map value 
df_googleplaystore['New Content Rating'] = df_googleplaystore['Content Rating'].map(ratingDict).astype(int)


print('\n\nContent Ratings with represented numbers:\n')
print(ratingDict)

#write new dataframe to excel file
df_googleplaystore.to_excel("Clean_Data.xlsx", index = False)
print("\n Finished writting in excel file")

################################ Exploratory Data analysis ################################


#read in cleaned data file
df_clean = pd.read_excel("Clean_Data.xlsx")
#print(df_clean)

#print histogram of all of our variables: Rating, New Content Rating, Price
Rating = df_clean[['Rating']]
Rating1 = Rating.values

New_Content_Rating = df_clean[['New Content Rating']]
New_Content_Rating1 = New_Content_Rating.values

Price = df_clean[['Price']]
Price1 = Price.values


#Plot historgrams
plt.hist(Rating1,10)
plt.hist(New_Content_Rating1,10)
plt.hist(Price1,10)


#check distribution of price when remove free apps
PricePaid = Price1[Price1 >0]
plt.hist(PricePaid,10)


#remove free apps - $0 Price
df_clean_new = df_clean[df_clean.Price > 0]
#print(df_clean_new)

#re-define variables

Rating = df_clean_new[['Rating']]
Rating2 = Rating.values

New_Content_Rating = df_clean_new[['New Content Rating']]
New_Content_Rating2 = New_Content_Rating.values

Price = df_clean_new[['Price']]
Price2 = Price.values




#print graphs with titles use appropiarte title for hist  - before and after 
#price
plt.hist(Price1,10)
plt.xlabel("Price", fontsize=16)  
plt.title("Price before free apps where removed", fontsize = 16)
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()

plt.hist(PricePaid,10)
plt.xlabel("Price", fontsize=16)  
plt.title("Price after free apps where removed", fontsize = 16)
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()

#app rating
plt.hist(Rating1,10)
plt.xlabel("App Rating", fontsize=16)  
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()

plt.hist(Rating2,10)
plt.xlabel("New App Rating", fontsize=16)  
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()


#content rating
plt.hist(New_Content_Rating1,10)
plt.xlabel("Content Rating", fontsize=16)  
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()

plt.hist(New_Content_Rating2,10)
plt.xlabel("New Content Rating", fontsize=16)  
plt.ylabel("# of Occurance", fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.show()




#export new table without $0 prices to csv
df_clean_new.to_csv("clean_data_CSV.csv", index = False)



################################ EDA OF UPDATED DATA ######################################



#data in CSV format
data_CSV = pd.read_csv('clean_data_CSV.csv')
data_CSV.head()
data_CSV.info()


#get summary statistics
data_CSV.describe()

#print boxplot
plt.boxplot(data_CSV['Rating'])

#check visual correlation
sns.pairplot(data_CSV)

#check numerical correlation
data_CSV.corr()

#check distribution
sns.distplot(data_CSV['Price'])
sns.distplot(data_CSV['Rating'])


#setting the dependent and independent variables
X = data_CSV[['Price', 'New Content Rating']]
y = data_CSV['Rating']

#training data using 60% of training data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.40, random_state=100)
lm = LinearRegression()
lm.fit(X_train, y_train)
       
predictions = lm.predict(X_test)

#plot y_test vs y_pred
plt.scatter(y_test,predictions)    
plt.xlim([3, 5])
plt.ylim([3, 5]) 
plt.xlabel("Test Data", fontsize=16)  
plt.ylabel("Predictions", fontsize=16)

np.subtract(predictions,y_test)

       
#check accruacy
print("Training set score: {:.3f}".format(lm.score(X_train, y_train)))
print("Test set score: {:.3f}".format(lm.score(X_test, y_test)))

#Get MSE
mse(y_test,predictions)


#DETERMINE OUR APP RATING BASED ON OUR CURRENT CONTENT RATING AND PRICE

#Calling googleplaystore sheet that has our app data -- will need to update this to where it's saved on your drive
xls = pd.ExcelFile(r'C:\Users\206581774\Documents\DataScienceClass\google_play_store.xlsx')
df_app = pd.read_excel(xls, 'app')

#assign x and y
x_test = df_app.iloc[:,[3,4]]
print(x_test)
y_test = df_app.iloc[:,2]
print(y_test)

#predict our app score
y_pred = lm.predict(x_test)
print(y_pred)


np.subtract(y_pred,y_test)
perc_diff = ((y_pred / y_test)-1)*100
print(perc_diff)



