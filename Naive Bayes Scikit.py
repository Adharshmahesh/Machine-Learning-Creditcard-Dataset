import numpy as np
import pandas as pd
import scipy
#from loaddata import data
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def load_data(path, header):
    df = pd.read_excel(path, header=header)
    return df


if __name__ == "__main__":
# load the data from the file
	data = load_data("credit.xls", None)

data = data.drop(data.index[0])
data = data.drop(data.index[0])

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
y=y.astype('int')



from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(x_train,y_train)
train_score=lasso.score(x_train,y_train)
test_score=lasso.score(x_test,y_test)
coeff_used = np.sum(lasso.coef_==0)


print("coeff_used is", coeff_used)