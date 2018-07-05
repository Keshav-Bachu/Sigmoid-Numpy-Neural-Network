# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:16:07 2018

@author: Keshav Bachu
"""
import numpy

def Sigmoid(Z):
    #represents the sigmoid activation function being greater than certian values
    return 1/(1 + numpy.exp(-1 * Z)), Z

def ReLu(Z):
    value = numpy.maximum(Z, 0)
    return value , Z

def LogBasedCost(Y, A_final, m):
    epsilon = 1/100000000
    cost = -1/m * numpy.sum(Y * numpy.log(A_final + epsilon) + (1- Y) * numpy.log(1 - A_final + epsilon))
    cost = numpy.squeeze(cost)
    return cost

def Sigmoid_Backwards(dA, B_cache):
    #derivative of a sigmoid function is: sigmoid * (1 - sigmoid)
    #dZ[l]=dA[l]∗g′(Z[l])
    Z = B_cache
    derivative = Sigmoid(Z)[0]
    derivative = derivative * (1 - derivative)
    
    return dA * derivative

def ReLu_Backwards(dA, B_cache):
    #derivative of a Relu function is: sigmoid * (1 - sigmoid)
    #dZ[l]=dA[l]∗g′(Z[l])
    Z = B_cache
    derivative = Z > 0
    
    return dA * derivative

def Tanh_Backwards(dA, B_cache):
    Z = b_cache
    derivative = numpy.arctanh
    return 1 - np.power(A1, 2)