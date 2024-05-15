#!/usr/bin/env python
import numpy as np 
import matplotlib.pylab as plt 
import pytest

mycolors = {"blue": "steelblue", "red":"#a76c6e",  "green":"#6a9373", "smoke": "#f2f2f2"}

def eval_RSS(X, y, b0, b1):
    rss = 0 
    for ii in range(len(df)):
        xi = df.loc[ii, "x"]
        yi = df.loc[ii, "y"]
        rss += (yi - (b0 + b1 * xi)) ** 2
    return rss

def plotsurface(X, y, bhist=None):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-1, 5, 300))
    Z = np.zeros((xx.shape[0], yy.shape[0]))
    for ii in range(X.shape[0]):
        Z += (y[ii] - xx - yy * X[ii,1]) ** 2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    levels = [125, 200] + list(range(400,2000,400))
    CS = ax.contour(xx, yy, Z, levels=levels)
    ax.clabel(CS, CS.levels, inline=True, fontsize=10)
    ax.set_xlim([-3,3])
    ax.set_ylim([-1,5])
    ax.set_xlabel(r"$\beta_0$", fontsize=20)
    ax.set_ylabel(r"$\beta_1$", fontsize=20)
    if bhist is not None:
        for ii in range(bhist.shape[0]-1):
            x0 = bhist[ii][0]
            y0 = bhist[ii][1]
            x1 = bhist[ii+1][0]
            y1 = bhist[ii+1][1]
            ax.plot([x0, x1], [y0, y1], color="black", marker="o", lw=1.5, markersize=5)
            
def dataGenerator(n, sigsq=1.0, random_state=1236):
    np.random.seed(random_state)
    x_train = np.linspace(-1, 1, n)
    x_valid = np.linspace(-1, 1, int(n / 4))
    y_train = 1 + 2 * x_train + np.random.randn(n)
    y_valid = 1 + 2 * x_valid + np.random.randn(int(n / 4))
    return x_train, x_valid, y_train, y_valid 

def sgd(X, y, beta, eta=0.1, num_epochs=100):
    """
    Peform Stochastic Gradient Descent 
    
    :param X: matrix of training features 
    :param y: vector of training responses 
    :param beta: initial guess for the parameters
    :param eta: the learning rate 
    :param num_epochs: the number of epochs to run 
    """
    
    # initialize history for plotting 
    bhist = np.zeros((num_epochs+1, len(beta)))
    bhist[0,0], bhist[0,1] = beta[0], beta[1]
    
    # perform steps for all epochs 
    for epoch in range(1, num_epochs+1):
        
        # shuffle indices (randomly)
        shuffled_inds = list(range(X.shape[0]))
        np.random.shuffle(shuffled_inds)
        
        # TODO: loop over training examples, update beta (beta[0] and beta[1]) as per the above formulas
        # your code here
        # loop over training examples
        # for instance in shuffled_inds:
        #     # get the current training example
        #     xi = X[instance, :]
        #     yi = y[instance]
        #     # prediction
        #     y_pred = np.dot(xi, beta)
        #     # error
        #     error = y_pred - yi
        #     # update beta
        #     beta[0] = beta[0] - eta * error
        #     beta[1] = beta[1] - eta * error * xi[1]
        for instance in shuffled_inds:
            xi = X[instance]
            yi = y[instance]
            
            # Predict value
            pred = beta[0] + beta[1] * xi[1]
            
            # Compute gradients
            grad_b0 = 2 * (pred - yi)
            grad_b1 = 2 * (pred - yi) * xi[1]
            
            # Update parameters
            beta[0] -= eta * grad_b0
            beta[1] -= eta * grad_b1
            
        # save history 
        bhist[epoch, :] = beta
        
    return bhist 




if __name__=="__main__":
    # generate some data
    x_train, x_valid, y_train, y_valid = dataGenerator(100)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.scatter(x_train, y_train, color="steelblue", s=100, label="train")
    ax.scatter(x_valid, y_valid, color="#a76c6e", s=100, label="valid")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.legend(loc="upper left", fontsize=16);
    
    #Since we're going to be implementing things ourselves, we're going to want to prepend the data matrices with a column of ones so we can fit a bias term. We can do this using numpy's column_stack function.
    X_train = np.column_stack((np.ones_like(x_train), x_train))
    X_valid = np.column_stack((np.ones_like(x_valid), x_valid))
    
    # Finally, let's fit a linear regression model with sklearn's LinearRegression class and print the coefficients so we know what we're shooting for.
    from sklearn.linear_model import LinearRegression 
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)
    print("sklearn says the coefficients are ", reg.coef_)
    
    # visualize the surface of the RSS,
    plotsurface(X_train, y_train)
    
    # SGD Test for 2 features
    np.random.seed(42)

    mock_X = np.array([[ 1., -1.], [ 1., -0.97979798], [ 1., -0.95959596], [ 1., -0.93939394]])
    mock_y = np.array([-1.09375848, -2.65894663, -0.51463485, -2.27442244])
    mock_beta_start = np.array([-2.0, -1.0])

    mock_bhist_exp = np.array([[-2., -1.], [-2.01174521, -0.98867152], [-2.02304238, -0.97777761], [-2.03400439, -0.96720934]])
    mock_bhist_act = sgd(mock_X, mock_y, beta=mock_beta_start, eta=0.0025, num_epochs=3)

    for exp, act in zip(mock_bhist_exp, mock_bhist_act):
        assert pytest.approx(exp, 0.0001) == act, "Check sgd function"
    
    # Start at (-2,1)
    beta_start = np.array([-2.0, -1.0])

    # Training 
    bhist = sgd(X_train, y_train, beta=beta_start, eta=0.0025, num_epochs=1000) # old = 0.0025

    # Print and Plot 
    print("beta_0 = {:.5f}, beta_1 = {:.5f}".format(bhist[-1][0], bhist[-1][1]))
    plotsurface(X_train, y_train, bhist=bhist)