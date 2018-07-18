import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from sklearn import linear_model
style.use('fivethirtyeight')

def graph(features, target, pred):
    # graphing dataset
    plt.plot(features,target,'bo')
    plt.plot(features, pred, color='blue', linewidth=3)
    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    plt.axis([0,14,0,18])
    plt.show()

def cost_function(predictions, target):
    # cost function to determine performance of theta_zero and theta_one
    cost = 0
    cost = (target - predictions) ** 2
    cost = cost.sum()
    cost = cost / (2 * len(target))

    return cost

def predict(features, coefficients):
    # theta_zero is y-intercept, theta_one is slope
    predictions = np.zeros((features.shape[0],1))

    for x in range(0,features.shape[0]):
        val = 0
        for y in range(0, features.shape[1]):
            if(features.shape[1]>1):
                val = val + (features[x,y]*coefficients[y+1])
            else:
                val = val + (features[x] * coefficients[y+1])
        val = val + coefficients[0]
        predictions[x] = val

    return predictions

def gradient_descent(coefficients, learning_rate, target, features):
    # derivative
    predictions = predict(features, coefficients)

    # intercept derivative
    cost_deriv = target - predictions
    cost_deriv = cost_deriv.sum()

    updates = coefficients

    # update coefficient values
    for i in range(len(coefficients)):

        if(i==0):
            change = (learning_rate * ((-1 / len(target)) * cost_deriv))
        else:
            cost_deriv_one = (target - predictions)
            feat = np.reshape(features[:,i-1],(-1,1))
            cost_deriv_one = cost_deriv_one * feat
            cost_deriv_one = cost_deriv_one.sum()
            change = (learning_rate * ((-1 / len(target)) * cost_deriv_one))

        temp_other = coefficients[i] - change
        updates[i] = temp_other

    coefficients = updates

    return coefficients

def compute_coefficients(target, features):
    intercept_col = np.ones((features.shape[0],1))
    features_full = np.column_stack([intercept_col,features])
    trans = np.transpose(features_full)
    result = np.dot(trans,features_full)
    result_inv = np.linalg.inv(result)
    coefficients = np.dot(trans,target)
    coefficients = np.dot(result_inv, coefficients)

    return coefficients

def main():
    # linear regression equation (univariate)
    learning_rate = 0.01
    i=0

    # fake dataset
    dataset = np.array(([1,2,2],[2,2,4],[4,5,8],[8,12,16]))

    # split data into inputs and outputs (column 0 is inputs, column 1 is outputs)
    features = dataset[:, :dataset.shape[1]-1]
    target = dataset[:, dataset.shape[1]-1:]

    target = np.reshape(target, (-1, 1))

    # create starting coeff array
    num_coeff = features.shape[1]+1
    coefficients = np.zeros((num_coeff,1))
    # predictions
    predictions = predict(features, coefficients)
    cost = cost_function(predictions, target)

    while i < 5000:
       coefficients = gradient_descent(coefficients, learning_rate, target, features)
       predictions = predict(features, coefficients)
       cost = cost_function(predictions, target)
       i=i+1

       if(i%1000==0):
          graph(features, target, predictions)
          #print(coefficients)
          #print(cost)

    print("GD Coefficients:")
    print(coefficients)
    print('\n')

    # sklearn coeff for reference
    regr = linear_model.LinearRegression()
    regr.fit(features, target)
    pred = regr.predict(features)

    print("Sklearn Coefficients:")
    print(regr.coef_)
    print(regr.intercept_)
    print('\n')

    # normal equations
    coeff_comp = compute_coefficients(target,features)
    print("Calculate Coefficients:")
    print(coeff_comp)

    # see ending slope and intercept
    # graphing dataset
    graph(features,target, pred)

main()