import numpy as np
import time
import pandas as pd
import scipy
import os
import matplotlib.pyplot as plot
from scipy import signal
import seaborn as sns
# 1. Download data file
#creditCardRawData = os.system('/bin/bash -c "curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"')

# 2. Load the datasets into numpy objects
np.set_printoptions(suppress=True) #genfrom text now will give int values rather than scientific notation
original_credit_card = np.genfromtxt('creditcard.txt', skip_header = 2)

# 3. make a panda object to use it in task2 and try in linear regration
def load_data(path, header):
    df = pd.read_csv(path, delimiter='\t',header=1)
    return df
if __name__ == "__main__":
# load the data from the file
    data = load_data("creditcard.txt", None)
    print(data)

# 4. Do not need to clean the data because there arent any missing values

#Prepare data to be used for statistics, we use lastcol to see the label
lastcol = original_credit_card[:,-1]
minValue = np.count_nonzero(lastcol == 1) #count of 1's at lastcol
maxValue = np.count_nonzero(lastcol == 0) #count of 0's at lastcol

# 5. Perform some statistics
#show histogram of age col
agecol = original_credit_card[:,-20]
targetCol = original_credit_card[:,-1]
histogramOfAgeColumn= plot.hist(agecol, color='red')
plot.show() 
histogramOfTargetCol = plot.hist(targetCol, color='blue')
plot.show() 
#correlation of the credit_Card_Data
dataForCorrelation = original_credit_card[:,:-1] #everything but the last column
corr = data.corr()
sns.heatmap(corr)
plot.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.88:
            print(i,j)
            
        else:
            #print(i)
            continue
#Correlation with output variable
cor_target = abs(corr.iloc[:,-1])#Selecting highly correlated features
relevant_features = cor_target[cor_target<0.05]
print(relevant_features)

fig, a = plot.subplots(3,3)
a[0][0].scatter(data.iloc[:,12], data.iloc[:,13])
a[0][1].scatter(data.iloc[:,12], data.iloc[:,14])
a[0][2].scatter(data.iloc[:,13], data.iloc[:,14])
a[1][0].scatter(data.iloc[:,13],data.iloc[:,15])
a[1][1].scatter(data.iloc[:,14],data.iloc[:,15])
a[1][2].scatter(data.iloc[:,14],data.iloc[:,16])
a[2][0].scatter(data.iloc[:,15],data.iloc[:,16])
a[2][1].scatter(data.iloc[:,15],data.iloc[:,17])
a[2][2].scatter(data.iloc[:,16],data.iloc[:,17])


plot.show()

sns.distplot(data.iloc[:,15])
plot.show()

sns.boxplot(data=data.iloc[:,15])
plot.show()