#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:20:52 2021

@author: mder
"""
from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor as _GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as _RBF
from sklearn.gaussian_process.kernels import WhiteKernel as _WhiteKernel
from sklearn.gaussian_process.kernels import Matern as _Matern
from sklearn.svm import SVC as _SVC

def GP_rbf_classifier():
    kernel = 1.0 * _RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + _WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    
    gpr = _GaussianProcessClassifier(kernel=kernel,random_state=None, \
                                     n_restarts_optimizer=1)
    return gpr
    
def GP_rbf_regression():
    kernel = 1.0 * _RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + _WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))

    gpr = _GaussianProcessRegressor(kernel=kernel,alpha=0,random_state=None, \
                                    n_restarts_optimizer=1)
    return gpr

def GP_matern_classifier():
    kernel = 1.0 * _Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + _WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    
    gpr = _GaussianProcessClassifier(kernel=kernel,random_state=None, \
                                     n_restarts_optimizer=1)
    return gpr

def GP_matern_regression():
    kernel = 1.0 * _Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + _WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))

    gpr = _GaussianProcessRegressor(kernel=kernel,random_state=None, \
                                   n_restarts_optimizer=1)
    return gpr


def svm():
    svm = _SVC(C=1, gamma=0.001, class_weight='balanced', kernel='rbf')
    return svm





