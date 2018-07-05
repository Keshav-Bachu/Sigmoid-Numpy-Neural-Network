# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:53:08 2018

@author: Keshav Bachu
"""

import numpy as np
import matplotlib.pyplot as plt
import Helper_Math_Functions

def init_Params(layers = None, scaling_factor = 0.01):
    #layers is interpreted as:
    #                           0th entry is the number of X input features
    #                           1st thru nth entries are the size of the hidden layers
    if layers == None:
        return None
    parameters = {}
    L= len(layers)
    for itterator in range(0 , L - 1):
       parameters["W" + str(itterator + 1)] = np.random.randn(layers[itterator + 1], layers[itterator]) * scaling_factor
       parameters["b" + str(itterator + 1)] = np.zeros((layers[itterator + 1], 1))
    return parameters

def calculate_Z(W, b, A_prev):
    cache = (W, b, A_prev)
    Z_new = np.dot(W, A_prev) + b
    return Z_new, cache

#somewhere here is the error!
def forward_Prop_Single(W, b, A_prev, activation):
    #A: The activation function output, from inputting Z
    #A_cache: The values used for W, b, A_prev; needed for back_propogation
    #B_cache: Contains the value of Z, needed for back_prop
    #Z: Output of calculating the Z from the calculate_Z function
    #A, A_cache, B_cache, Z = None
    #print("pre enter")
    if activation == "Sigmoid":
        Z, A_cache = calculate_Z(W, b, A_prev)
        A, B_cache = Helper_Math_Functions.Sigmoid(Z)
        
    elif activation == "ReLu":
        Z, A_cache = calculate_Z(W, b, A_prev)
        A, B_cache = Helper_Math_Functions.ReLu(Z)
    
        
    cache = (A_cache, B_cache)
    return A, cache

def forward_Prop_Multiple(X, parameters):
    L = int(len(parameters) / 2)
    A = X
    cache_return = []
    for l in range(1,L):
        A_prev = A
        A, cache = forward_Prop_Single(parameters["W" + str(l)], parameters["b" + str(l)], A_prev, "ReLu")
        cache_return.append(cache)
    
    A_final, cache = forward_Prop_Single(parameters["W" + str(L)], parameters["b" + str(L)], A, "Sigmoid")
    cache_return.append(cache)
    return A_final, cache_return
    
def compute_Cost(Y, A_final):
    m = Y.shape[1]
    cost = Helper_Math_Functions.LogBasedCost(Y, A_final, m)
    return cost

def linear_Backprop(dZ, A_cache):
    W, b, A_prev = A_cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1)
    db = db.reshape(b.shape)
    dA_prev = np.dot(W.T, dZ)
    
    return dW, db, dA_prev

def linear_Activation_Backprop(dA, cache, activation):
    
    if activation == "Sigmoid":
        dZ = Helper_Math_Functions.Sigmoid_Backwards(dA, cache[1])
        dW, db, A_prev = linear_Backprop(dZ, cache[0])
        
    elif activation == "ReLu":
        dZ = Helper_Math_Functions.ReLu_Backwards(dA, cache[1])
        dW, db, A_prev = linear_Backprop(dZ, cache[0])
        
        
    return dW, db, A_prev

def backward_Prop_Multiple(A_final, Y, cache):
    grads = {}
#    m = Y.shape[1]
    epsilon = 0.000000001
    L = int(len(cache))
    dA_final = - (np.divide(Y, A_final + epsilon) - np.divide(1 - Y, 1 - A_final + epsilon))
    #dA_final = A_final - Y
    current_cache = cache[L - 1]
    grads["dW" + str(L)], grads["db" + str(L)], grads["dA" + str(L - 1)] = linear_Activation_Backprop(dA_final, current_cache, "Sigmoid")
    
    for itterator in reversed(range(L - 1)):
        current_cache = cache[itterator]
        dW_hold, db_hold, dA_prev_hold = linear_Activation_Backprop(grads["dA" + str(itterator + 1)], current_cache, "ReLu")
        grads["dA" + str(itterator)] = dA_prev_hold
        grads["dW" + str(itterator + 1)] = dW_hold
        grads["db" + str(itterator + 1)] = db_hold
        
    return grads
def update_Parameters(parameters, grads, learning_rate = 0.1):
    L = int(len(parameters)/2)
    
    for i in range(L):
        parameters["W" + str(i+1)] -= grads["dW" + str(i+1)] * learning_rate
        parameters["b" + str(i+1)] -= grads["db" + str(i+1)] * learning_rate
    return parameters

def Make_Sigmoid_NeuralNetwork(X, Y, layers, itterations = 15000):
    layers.insert(0, X.shape[0])
    
    #make the parameters of the neural network with random initialization
    parameters = init_Params(layers)
    costPrev = 10000000
    paramPrev = parameters
    learning_rate = 0.1
    
    for itterator in range(0,itterations + 1):
        A_final, cache_return = forward_Prop_Multiple(X, parameters)
        cost = compute_Cost(Y, A_final)
        if(costPrev < cost):
            learning_rate = learning_rate * 0.95
            parameters = paramPrev
            print("Learning Rate Anomaly Detected, lowering learning rate")
        costPrev = cost
        paramPrev = parameters
        
        grads = backward_Prop_Multiple(A_final, Y, cache_return)
        parameters = update_Parameters(parameters, grads, learning_rate)
        
        if((itterator) % 100) == 0:
            #print("Cost after itteration ", itterator, " is: ", cost)
            print(cost)
        

    print("Final Cost is: ", cost)  
    return parameters

def Predictions(X, Y, params):
    A_final, cache = forward_Prop_Multiple(X, params)
    
    prediction = A_final > 0.5
    errors = np.sum(np.abs(Y - prediction))
    total = Y.shape[1]
    print("Accuracy: ", 100 - errors/total * 100, "%")
    return None
