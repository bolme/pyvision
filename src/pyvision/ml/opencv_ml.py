'''
This module includes some helper functions for training OpenCV's machine learning algorithms.
Created on Mar 25, 2013

@author: David S. Bolme
Oak Ridge National Laboratory
'''
import pyvision as pv
import cv2
import numpy as np
import tempfile
import os

class StatsModelWrapper(object):
    '''
    This class wraps an opencv stats model to support pickling and other 
    pythonic features, etc.
    '''
    
    def __init__(self,model):
        '''
        Init the wrapper with the model.
        '''
        self.model = model
        
    def predict(self,*args,**kwarg):
        '''
        Wrapper for the predict function.
        '''
        return self.model.predict(*args,**kwarg)

    def predict_prob(self,*args,**kwarg):
        '''
        Wrapper for the predict function.
        '''
        return self.model.predict_prob(*args,**kwarg)
        
    def predict_all(self,*args,**kwarg):
        '''
        Wrapper for the predict function.
        '''
        return self.model.predict_all(*args,**kwarg)
        
    def __getstate__(self):
        ''' Save the state for pickling '''
        state = {}
        state['model_class'] = str(self.model.__class__).split("'")[-2]
        filename = tempfile.mktemp(suffix='.mod', prefix='tmp')
        self.model.save(filename)
        data = open(filename,'rb').read()
        state['model_data'] = data
        
        for key,value in self.__dict__.iteritems():
            if key != 'model':
                state[key] = value
        
        return state
    
    
    def __setstate__(self,state):
        ''' Load the state for pickling. '''
        model_class = 'cv2.ml.'+state['model_class'][7:]
        self.model = eval(model_class+"_create()")
        filename = tempfile.mktemp(suffix='.mod', prefix='tmp')
        open(filename,'wb').write(state['model_data'])
        self.model.load(filename)
        os.remove(filename)
        for key,value in state.iteritems():
            if key not in ('model_data','model_class'):
                setattr(self,key,value)
                
    def save(self,*args,**kwargs):
        self.model.save(*args,**kwargs)


def svc_rbf(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    data = np.float32(data)
    responses = np.int32(responses) 
    params = dict( kernel_type = cv2.ml.SVM_RBF, svm_type = cv2.ml.SVM_C_SVC )
    model = cv2.ml.SVM_create()
    model.train(data,cv2.ml.ROW_SAMPLE,responses)
    return StatsModelWrapper(model)
    
    
def svc_linear(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_C_SVC)
    model = cv2.ml.SVM_create()
    model.train(data,responses,None,None,params)
    return StatsModelWrapper(model)
    
    
def svr_rbf(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.ml.SVM_RBF, svm_type = cv2.ml.SVM_EPS_SVR , p=1.0)
    model = cv2.ml.SVM_create()
    model.train(data,responses,None,None,params)
    return StatsModelWrapper(model)
    
def svr_linear(data,responses):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict( kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_EPS_SVR , p=1.0 )
    model = cv2.ml.SVM_create()
    model.train_auto(data,responses,None,None,params)
    return StatsModelWrapper(model)
    
    
def random_forest(data,responses,n_trees=100):
    '''
    Auto trains an OpenCV SVM.
    '''
    np.float32(data)
    np.float32(responses) 
    params = dict(max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_EPS_SVR , p=1.0 )
    model = cv2.ml.RTrees_create()
    model.train(data,responses,params=params)
    return StatsModelWrapper(model)
    
    
def boost(data,responses,weak_count=100,max_depth=20,boost_type=None):
    '''
    Auto trains an OpenCV SVM.
    '''
    if boost_type is None:
        try:
            # opencv 2.6
            boost_type=cv2.ml.BOOST_DISCRETE
        except:
            # opencv 2.4
            boost_type=cv2.BOOST_DISCRETE
        
    np.float32(data)
    np.float32(responses) 
    params = dict(boost_type=boost_type,weak_count=weak_count,max_depth=max_depth)
    model = cv2.ml.Boost_create()
    model.train(data,responses,params=params)
    return StatsModelWrapper(model)
    
def gbtrees(data,responses,n_trees=100):
    '''
    Auto trains an OpenCV SVM.
    '''
    raise NotImplementedError("This was removed for opencv 3.0.")
    #np.float32(data)
    #np.float32(responses) 
    #params = dict(max_num_of_trees_in_the_forest=n_trees,termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    #model = cv2.ml.GBTrees_create()
    #model.train(data,responses,params=params)
    #return StatsModelWrapper(model)
    
    
if __name__ == '__main__':
    #print "IRIS_DATA:",pv.IRIS_DATA
    #print pv.IRIS_LABELS
    labels = np.float32((pv.IRIS_LABELS=='versicolor') + 2*(pv.IRIS_LABELS=='virginica'))
    
    model = svc_rbf(pv.IRIS_DATA[0::2,:],labels[0::2])
    
    import cPickle as pkl
    buf = pkl.dumps(model)
    model = pkl.loads(buf)
    print "Prediction:",np.float32([model.predict(s) for s in pv.IRIS_DATA[1::2,:]])
    print "Prediction:",model.predict_all(pv.IRIS_DATA[1::2,:])
    assert 0












