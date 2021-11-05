import numpy as np
import sklearn
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats 
import linearRegression

def mse(y_pred, y_test):
	cost = 0.0
	for i in range(len(y_pred)):
		diff = y_pred[i] - y_test[i]
		cost += diff**2

	cost = cost/len(y_pred)
	print("Mean squared error: ", cost)


features,y = load_boston(return_X_y=True)

X = features[:, np.newaxis,5]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1234)

x = X_train.flatten()
intercept, slope,_,_,_ = stats.linregress(x,y_train)
print ("intercept",intercept,"Slope: ",slope)

theta, bias, cost_history, epoch = linearRegression.stochasticGradientDescent(X_train, y_train, learningRate = 0.00015, epoch=1000)
y_pred = linearRegression.predict(X_test, theta, bias)

print("Final Cost: ", cost_history[-1])

line = X_train * theta + bias
print("X_train *", theta, "+", bias)
plt.figure(figsize=(10,20))
blue = plt.scatter(X_train,y_train,c='b')
green = plt.scatter(X_test,y_test,c='g')
plt.xlabel("RM\nNumber of rooms vs Cost of house")
plt.ylabel("Cost")
plt.plot(X_train,line,c='r')
plt.legend((blue, green), ("Training Data", "Testing Data"), loc = 'lower right')
plt.show()

fig,ax = plt.subplots(figsize=(12,8))
ax.set_ylabel('Cost funtion')
ax.set_xlabel('Iterations')
_=ax.plot(range(epoch),cost_history,'b.')
plt.show()

mse(y_pred, y_test)
