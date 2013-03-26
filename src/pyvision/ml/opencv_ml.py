'''
This module includes some helper functions for training OpenCV's machine learning algorithms.
Created on Mar 25, 2013

@author: David S. Bolme
Oak Ridge National Laboratory
'''
import pyvision as pv
import cv2
import numpy as np


def svc_rbf(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC )
    model = cv2.SVM()
    model.train_auto(data,responses,None,None,params)
    return model
    
    
def svc_linear(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC)
    model = cv2.SVM()
    model.train_auto(data,responses,None,None,params)
    return model
    
    
def svr_rbf(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_EPS_SVR , p=1.0)
    model = cv2.SVM()
    model.train_auto(data,responses,None,None,params)
    return model
    
def svr_linear(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.SVM()
    model.train_auto(data,responses,None,None,params)
    return model
    
    
def random_forest(data,responses,n_trees=100):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict(max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.RTrees()
    model.train(data,cv2.CV_ROW_SAMPLE,responses,params=params)
    return model
    
    
def boost(data,responses,n_trees=100):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict(max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.Boost()
    model.train(data,cv2.CV_ROW_SAMPLE,responses,params=params)
    return model
    
def gbtrees(data,responses,n_trees=100):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict(max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.GBTrees()
    model.train(data,cv2.CV_ROW_SAMPLE,responses,params=params)
    return model
    
    
if __name__ == '__main__':
    #print "IRIS_DATA:",pv.IRIS_DATA
    #print pv.IRIS_LABELS
    labels = np.float32((pv.IRIS_LABELS=='versicolor') + 2*(pv.IRIS_LABELS=='virginica'))
    
    model = svc_rbf(pv.IRIS_DATA[0::2,:],labels[0::2])
    print "Prediction:",np.float32([model.predict(s) for s in pv.IRIS_DATA[1::2,:]])
    print "Prediction:",model.predict_all(pv.IRIS_DATA[1::2,:])
    assert 0












