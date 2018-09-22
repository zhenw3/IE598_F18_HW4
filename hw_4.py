# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:08:57 2018

@author: zhenw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score 

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

#the class and functions that used to show the relation between R^2 and iteration numbers
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            #print(X.T.shape)
            #print(errors.shape)
            self.w_[1:] += self.eta *np.dot(X.T,errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


#read data
X = pd.read_csv('make_regressionX.csv',
                 header=None)
x=X

Y = pd.read_csv('make_regressionY.csv',
                 header=None)
data=pd.concat([X,Y],axis=1)

#calculate correlations between real-valued attributes
corMat1 = pd.DataFrame(data.corr())
#visualize correlations using heatmap
plt.figure()
plt.rcParams['figure.figsize']=[5,4]
plt.pcolor(corMat1)
plt.title('corr heatmap')
plt.show()

#select variables
sign=corMat1.iloc[:-1,-1]
sign=list(sign[sign**2>0.09].index)
X=X.iloc[:,sign]


#plot corrheat of y and selected variables
for i in range(len(sign)):
    sign[i] = 'var'+str(sign[i])
sign.append('y')
data=pd.concat([X,Y],axis=1)
data.columns=sign

#pair plot
sns.pairplot(data, size=1.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()

#calculate correlations between real-valued attributes
corMat = pd.DataFrame(data.corr())
#visualize correlations using heatmap
plt.figure
sns.heatmap(corMat,annot=True)
plt.title('corr heatmap')
plt.show()

#before fitting linear regression, standardize the data first
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X_std = sc_x.fit_transform(X)
#Y_std = sc_y.fit_transform(Y)

#split sample
X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state=42)


#cross val n fold
n=10

#linear regression
print()
print('Linear Regression')

#plot the relation between R^2 and iteration times
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X_std = sc_x.fit_transform(X)
#y_std=Y.values
#y_std = sc_y.fit_transform(y_std).flatten()
lr = LinearRegressionGD()
lr.fit(np.array(X), np.array(Y).flatten())
plt.figure()
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.tight_layout()
#plt.savefig('images/10_05.png', dpi=300)
plt.show()


slr = LinearRegression()
print('cross_val_score = '+str(np.mean(cross_val_score(slr,X,Y,cv=n))))
slr.fit(X_train, y_train)
y_fit = slr.predict(X_train)
y_pred=slr.predict(X_test)
for i in range(len(slr.coef_[0])):
   print('Slope %.3f' % i,' =',str(slr.coef_[0][i]))
print('Intercept: %.3f' % slr.intercept_)
print("R^2 for training sample: {}".format(slr.score(X_train, y_train)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_train,y_fit))
print("Root Mean Squared Error training sample: {}".format(rmse))
print("R^2 for testing sample: {}".format(slr.score(X_test, y_test)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error testing sample: {}".format(rmse))
#plot the performance
plt.scatter(y_fit,  y_fit - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_pred,  y_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.title('Linear Regression')
plt.hlines(y=0, xmin=y_train.min(), xmax=y_train.max(), color='black', lw=2)
#plt.xlim([-10, 50])
plt.tight_layout()
plt.show()



#test different c
a=range(-5,3)
c=[]
sc=[]
for i in a:
    c.append(10**i)
    sc.append(str(10)+'^'+str(i))

#Lasso Regression
print()
print('Lasso')
score1=[]#put score in it
for j in c:
    lasso = Lasso(alpha=j)
    score1.append(np.mean(cross_val_score(lasso,X,Y,cv=n)))
#draw R^2 with c
plt.figure()
plt.plot(sc,score1)
plt.xlabel('hyperparameter')
plt.ylabel('cross_val_R^2 for Lasso')
plt.show()

#show the best Lasso
lasso_c=c[score1.index(max(score1))]
lasso_a=a[score1.index(max(score1))]
print('The Best c: 10 ^ %.3f' % lasso_a)

lasso = Lasso(alpha=lasso_c)
#print the R^2 for cross_val_sample for the best c
print('cross_val_score = '+str(np.mean(cross_val_score(lasso,X,Y,cv=n))))

lasso.fit(X_train, y_train)
y_fit = lasso.predict(X_train)
y_pred = lasso.predict(X_test)
for i in range(0,len(lasso.coef_)):
   print('Slope %.3f' % i,' =',str(lasso.coef_[i]))
print('Intercept: %.3f' % lasso.intercept_)
print("R^2 for training sample: {}".format(lasso.score(X_train, y_train)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_train,y_fit))
print("Root Mean Squared Error training sample: {}".format(rmse))
print("R^2 for testing sample: {}".format(lasso.score(X_test, y_test)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error testing sample: {}".format(rmse))
#plot the performance
plt.scatter(y_fit,  y_fit - np.array(y_train).flatten(),
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_pred,  y_pred - np.array(y_test).flatten(),
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.title('Lasso')
plt.hlines(y=0, xmin=y_train.min(), xmax=y_train.max(), color='black', lw=2)
#plt.xlim([-10, 50])
plt.tight_layout()
plt.show()



#Ridge Regression
print()
print('Ridge')
score2=[]#put score in it
for j in c:
    ridge = Ridge(alpha=j)
    score2.append(np.mean(cross_val_score(ridge,X,Y,cv=n)))
#draw R^2 with c
plt.figure()
plt.plot(sc,score2)
plt.xlabel('hyperparameter')
plt.ylabel('cross_val_R^2 for Ridge')
plt.show()

#show the best Ridge
ridge_c=c[score2.index(max(score2))]
ridge_a=a[score2.index(max(score2))]
print('The Best c: 10 ^ %.3f' % ridge_a)

ridge = Ridge(alpha=ridge_c)
#print the R^2 for cross_val_sample for the best c
print('cross_val_score = '+str(np.mean(cross_val_score(ridge,X,Y,cv=n))))

ridge.fit(X_train, y_train)
y_fit = ridge.predict(X_train)
y_pred = ridge.predict(X_test)
for i in range(0,len(ridge.coef_[0])):
   print('Slope %.3f' % i,' =',str(ridge.coef_[0][i]))
print('Intercept: %.3f' % ridge.intercept_)
print("R^2 for training sample: {}".format(ridge.score(X_train, y_train)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_train,y_fit))
print("Root Mean Squared Error training sample: {}".format(rmse))
print("R^2 for testing sample: {}".format(ridge.score(X_test, y_test)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error testing sample: {}".format(rmse))
#plot the performance
plt.scatter(y_fit,  y_fit - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_pred,  y_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.title('Ridge')
plt.hlines(y=0, xmin=y_train.min(), xmax=y_train.max(), color='black', lw=2)
#plt.xlim([-10, 50])
plt.tight_layout()
plt.show()



#ElasticNet
print()
print('ElasticNet')
nn=2#the number of alpha
a=range(-5,nn)
c = [[] for i in range(len(a))]
sc=[[] for i in range(len(a))]
score3=[[] for i in range(len(a))]#put score in it
maxi=0
for i in range(0,len(a)):
    for j in range(0,i+1):
        #print(i,j)
        c[i].append(10**a[j])
        sc[i].append('alpha: 10^'+str(a[i])+', l1: 10^'+str(a[j]))
        en = ElasticNet(alpha=10**a[i],l1_ratio=10**a[j])
        scor=np.mean(cross_val_score(en,X,Y,cv=n))
        #find the maximum score and optimized alpha and l1
        if scor>maxi:
            maxi=scor
            alp=a[i]
            l1=a[j]
        score3[i].append(scor)
#draw R^2 with c
#plt.figure()
#plt.plot(sc,score3)
#plt.xlabel('hyperparameter for ElasticNet')
#plt.ylabel('cross_val_R^2 for ElasticNet')
#plt.show()

#show the best ElasticNet
en_alp=10**alp
en_l1=10**l1
print('The Best alpha: 10 ^ %.3f' % alp)
print('The Best l1: 10 ^ %.3f' % l1)

en = ElasticNet(alpha=en_alp,l1_ratio=en_l1)
#print the R^2 for cross_val_sample for the best c
print('cross_val_score = '+str(np.mean(cross_val_score(en,X,Y,cv=n))))

en.fit(X_train, y_train)
y_fit = en.predict(X_train)
y_pred = en.predict(X_test)
for i in range(0,len(en.coef_)):
   print('Slope %.3f' % i,' =',str(en.coef_[i]))
print('Intercept: %.3f' % en.intercept_)
print("R^2 for training sample: {}".format(en.score(X_train, y_train)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_train,y_fit))
print("Root Mean Squared Error training sample: {}".format(rmse))
print("R^2 for testing sample: {}".format(en.score(X_test, y_test)))
#rmse = np.sqrt(np.mean((Y-y_pred)**2))[0]
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error testing sample: {}".format(rmse))
#plot the performance
plt.scatter(y_fit,  y_fit - np.array(y_train).flatten(),
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_pred,  y_pred - np.array(y_test).flatten(),
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.title('ElasticNet')
plt.hlines(y=0, xmin=y_train.min(), xmax=y_train.max(), color='black', lw=2)
#plt.xlim([-10, 50])
plt.tight_layout()
plt.show()


print("My name is Zhen Wang")
print("My NetID is: zhenw3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")