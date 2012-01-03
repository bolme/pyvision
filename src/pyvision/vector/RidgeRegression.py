from pyvision.vector.VectorClassifier import *
from numpy import array,zeros,eye,dot,exp
from numpy.linalg import pinv,inv
from pyvision.analysis.Table import Table
import pyvision
import random


class RBF:
    '''
    Basic Radial Basis Function kernel
    '''
    
    def __init__(self,gamma=1.0,**kwargs):
        self.gamma = gamma
        
    def __call__(self,X,y):
        #TODO: print X.shape,y.shape
        tmp = X-y
        return exp(-self.gamma*(tmp*tmp).sum(axis=0))

    def __str__(self):
        return "RBF(%f)"%self.gamma
        
class LINEAR:
    '''
    Basic linear kernel.
    '''
    
    def __init__(self,**kwargs):
        pass
        
    def __call__(self,X,y):
        return dot(X.transpose(),y).flatten()
    
    def __str__(self):
        return "LINEAR()"
        
        
class RidgeRegression(VectorClassifier):
    ''' 
    Ridge Regression algorithm based on Max Welling's Tutorial, 
    Univerity of Toronto.
    
    http://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf
    
    Implemented by David Bolme.
    
    '''
    def __init__(self,lam=0.1,**kwargs):
        self.lam = lam
        
        VectorClassifier.__init__(self,TYPE_REGRESSION,**kwargs)
        
    def trainClassifer(self,labels,vectors,verbose=False,ilog=None):
        '''
        Do not call this function instead call train.
        '''
        self.training_size = len(labels)
        
        c = len(labels)
        r = len(vectors[0])
        
        y = array(labels,'d')
        
        X = zeros((r,c),'d')
        
        for i in range(len(vectors)):
            X[:,i] = vectors[i]

        tmp1 = inv(self.lam*eye(r) + dot(X,X.transpose()))
        
        tmp2 = dot(y,X.transpose())
        
        self.w = w = dot(tmp1,tmp2)

    def predictValue(self,data,ilog=None):
        '''
        Please call predict instead.
        '''
        x = array(data,'d')

        return dot(self.w,x)

                            
class KernelRidgeRegression(VectorClassifier):
    ''' 
    Ridge Regression algorithm based on Max Welling's Tutorial, 
    Univerity of Toronto.
    
    http://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf
    
    Implemented by David Bolme.
    
    '''
    def __init__(self,
                 training_size=0.67, # The fraction of the data to use for training
                 validation_size=None, # The fraction of the data to use for training
                 kernels= [RBF(gamma=2**i) for i in range(-15,4)], 
                 lams   = [2.0**i for i in range(-8,9)],
                 random_seed = None,
                 **kwargs):

        if isinstance(lams,list):
            self.lams = lams
        else:
            self.lams = [lams]
            
        if isinstance(kernels,list):
            self.kernels = kernels
        else:
            self.kernels = [kernels]
            
        self.training_size = training_size
        self.validation_size = validation_size
        
        # set durring training
        self.mse           = None
        self.lam           = None
        self.kernel        = None
        self.training_info = []
        
        self.rng = random.Random(random_seed)
        
        VectorClassifier.__init__(self,TYPE_REGRESSION,**kwargs)
        
    def trainClassifer(self,labels,vectors,verbose=False,ilog=None):
        '''
        Do not call this function instead call train.
        '''
        
        if len(self.lams) > 1 or len(self.kernels) > 1:
            # optimize kernel and lambda using a grid search
            
            if self.training_size <= 1.0:
                self.training_size = int(self.training_size*len(labels))
            else:
                self.training_size = int(self.training_size)
            
            if self.validation_size == None:
                self.validation_size = len(labels) - int(self.training_size)
            else:
                self.validation_size = int(self.validation_size)
            
            train_labels   = []
            train_vectors  = []
            verify_labels  = []
            verify_vectors = []
            
            order = list(range(len(labels)))
            self.rng.shuffle(order)
            for i in order[:self.training_size]:
                train_labels.append(labels[i])
                train_vectors.append(vectors[i])
            for i in order[self.training_size:self.training_size+self.validation_size]:
                verify_labels.append(labels[i])
                verify_vectors.append(vectors[i])
            
            best_mse = None
            best_lam = None
            best_kernel = None

            self.training_size = len(train_labels)
            
            c = len(train_labels)
            r = len(train_vectors[0])
            
            self.c = c
            self.r = r
            y = array(train_labels,'d')
            
            X = zeros((r,c),'d')
            for i in range(len(train_vectors)):
                X[:,i] = train_vectors[i]
    
            self.X = X
            
            for kernel in self.kernels:
                self.kernel = kernel
                
                kernel_matrix = zeros((c,c),'d')
                for i in range(c):
                    kernel_matrix[i,:] = self.kernel(X,X[:,i:i+1])

                
                for lam in self.lams:
                    self.lam = lam

                    self.w = w = dot(y,inv(kernel_matrix + self.lam*eye(c)))
                    
                    n = len(verify_labels)
                    mse = 0.0
                    for i in range(n):
                        e = verify_labels[i] - self.predictValue(verify_vectors[i])
                        mse += e*e
                    mse = mse/n
                    
                    self.training_info.append([lam,kernel,mse])
                    
                    if verbose: print "KRR Trianing: %s %10.5f %s %10.5f %s"%(kernel_matrix.shape,lam,kernel,mse,best_mse)
                    #   %s   %10.5f   %10.5f"%(lam,kernel,mse,best_mse)
                    if best_mse == None or best_mse > mse:
                        best_mse    = mse
                        best_lam    = lam
                        best_kernel = kernel
            
            self.mse    = best_mse
            self.lam    = best_lam
            self.kernel = best_kernel
            
        else:
            self.lam    = self.lams[0]
            self.kernel = self.kernels[0]
        
        self.trainRidgeRegression(labels,vectors,verbose=verbose)

    def trainRidgeRegression(self,labels,vectors,verbose=False):   
        self.training_size = len(labels)
        
        c = len(labels)
        r = len(vectors[0])
        
        self.c = c
        self.r = r
        y = array(labels,'d')
        
        X = zeros((r,c),'d')
        for i in range(len(vectors)):
            X[:,i] = vectors[i]

        self.X = X
        
        kernel_matrix = zeros((c,c),'d')
        for i in range(c):
            kernel_matrix[:,i] = self.kernel(X,X[:,i:i+1])

        self.w = w = dot(y,inv(kernel_matrix + self.lam*eye(c)))
                            

    def predictValue(self,data,ilog=None):
        '''
        Please call predict instead.
        '''
        x = array(data)
        x = x.reshape((self.r,1))
        k = self.kernel(self.X,x)

        return dot(self.w,k)



class _TestKRR(unittest.TestCase):
    ''' Unit tests for SVM '''
    
    def setUp(self):
        pass
    
        
    def test_kernel_regression_linear1(self):
        rega = KernelRidgeRegression(lams=0.1,kernels=LINEAR())
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','regression.dat')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[0]))
            vectors.append([float(datapoint[3]),float(datapoint[4]),float(datapoint[5])])
            
        for i in range(50):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        
        mse = 0.0
        total = 0
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
        mse = mse/(len(labels)-50)
        #print "Regression Error:",mse
        
        self.assertAlmostEqual(mse,0.24301122718491874,places=4)

    
    def test_kernel_regression_linear2(self):
        rega = KernelRidgeRegression(lams=0.1,kernels=LINEAR())
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_lin.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        
        mse = 0.0
        ase = 0.0
        total = 0
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            
        mse = mse/(len(labels)-50)
        ase = ase/(len(labels)-50)
        #print "Regression Error:",mse
        
        self.assertAlmostEqual(mse,0.041430725415995143,places=7)
        self.assertAlmostEqual(ase,0.00029750577054177876,places=7)
        
    def test_kernel_regression_linear3(self):
        rega = KernelRidgeRegression(lams=0.1,kernels=LINEAR())
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_cos.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        
        mse = 0.0
        ase = 0.0
        total = 0
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            
        mse = mse/(len(labels)-50)
        ase = ase/(len(labels)-50)
        #print "Regression Error:",mse
        
        self.assertAlmostEqual(mse,0.54150787637715125,places=7)
        self.assertAlmostEqual(ase,0.50489208165416299,places=7)
        
    def test_kernel_regression_rbf1(self):
        rega = KernelRidgeRegression(lams=0.1,kernels=RBF())
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_cos.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        
        mse = 0.0
        ase = 0.0
        total = 0
        table = Table()
        for i in range(100,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            table.setElement(i,'x',vectors[i][0])
            table.setElement(i,'measure',labels[i])
            table.setElement(i,'truth',truth[i])
            table.setElement(i,'pred',p)
            table.setElement(i,'e',e)
            table.setElement(i,'a',a)
            
            
        mse = mse/(len(labels)-100)
        ase = ase/(len(labels)-100)
        #print "Regression Error:",mse
        #print table
        #table.save("../../rbf1.csv")
        
        self.assertAlmostEqual(mse,0.0462477093266263,places=7)
        self.assertAlmostEqual(ase,0.0067187523452463625,places=7)
        
    def test_kernel_regression_rbf2(self):
        rega = KernelRidgeRegression(lams=0.1,kernels=RBF())
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_mix.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train()
        #print rega.w
        
        mse = 0.0
        ase = 0.0
        total = 0
        table = Table()
        for i in range(100,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            table.setElement(i,'x',vectors[i][0])
            table.setElement(i,'measure',labels[i])
            table.setElement(i,'truth',truth[i])
            table.setElement(i,'pred',p)
            table.setElement(i,'e',e)
            table.setElement(i,'a',a)
            
            
        mse = mse/(len(labels)-100)
        ase = ase/(len(labels)-100)
        #print "Regression Error:",mse
        #print table
        #table.save("../../rbf2.csv")
        
        self.assertAlmostEqual(mse,0.563513669235162,places=7)
        self.assertAlmostEqual(ase,0.51596869146460422,places=7)
        
    def test_kernel_regression_rbf3(self):
        rega = KernelRidgeRegression(random_seed=28378)
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_mix.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train(verbose=False)
        #print rega.w
        
        mse = 0.0
        ase = 0.0
        total = 0
        table = Table()
        for i in range(100,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            table.setElement(i,'x',vectors[i][0])
            table.setElement(i,'measure',labels[i])
            table.setElement(i,'truth',truth[i])
            table.setElement(i,'pred',p)
            table.setElement(i,'e',e)
            table.setElement(i,'a',a)
            
            
        mse = mse/(len(labels)-100)
        ase = ase/(len(labels)-100)
        #print "Regression Error:",mse
        #print table
        #table.save("../../rbf3.csv")
        
        self.assertAlmostEqual(mse,0.047179521440302921,places=7)
        self.assertAlmostEqual(ase,0.0052453297596735905,places=7)
        
    def test_kernel_regression_rbf4(self):
        rega = KernelRidgeRegression(random_seed=28378)
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','synth1_quad.txt')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        truth = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[1]))
            vectors.append([float(datapoint[0])])
            truth.append(float(datapoint[2]))
            
        for i in range(100):
            rega.addTraining(labels[i],vectors[i])
        rega.train(verbose=False)
        #print rega.w
        
        mse = 0.0
        ase = 0.0
        total = 0
        table = Table()
        for i in range(100,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            mse += e*e
            a = p - truth[i]
            ase += a*a
            table.setElement(i,'x',vectors[i][0])
            table.setElement(i,'measure',labels[i])
            table.setElement(i,'truth',truth[i])
            table.setElement(i,'pred',p)
            table.setElement(i,'e',e)
            table.setElement(i,'a',a)
            
            
        mse = mse/(len(labels)-100)
        ase = ase/(len(labels)-100)
        #print "Regression Error:",mse
        #print table
        #table.save("../../rbf4.csv")
        
        self.assertAlmostEqual(mse,0.063883259792847411,places=7)
        self.assertAlmostEqual(ase,0.028811752673991175,places=7)
        
    def test_regression_linear(self):
        # synthetic linear regression
        rega = RidgeRegression()
        filename = os.path.join(pyvision.__path__[0],'data','synthetic','regression.dat')
        reg_file = open(filename,'r')
        labels = []
        vectors = []
        for line in reg_file:
            datapoint = line.split()
            labels.append(float(datapoint[0]))
            vectors.append([float(datapoint[3]),float(datapoint[4]),float(datapoint[5])])

        for i in range(50):
            rega.addTraining(labels[i],vectors[i])
        rega.train()

        mse = 0.0
        total = 0
        table = Table()
        for i in range(50,len(labels)):
            p = rega.predict(vectors[i])
            e = p - labels[i]
            table.setElement(i,'truth',labels[i])
            table.setElement(i,'pred',p)
            table.setElement(i,'Residual',e)
            #print labels[i],p,e
            mse += e*e
            
        #print table
        #table.save('../../tmp.csv')
        
        mse = mse/(len(labels)-50)
        self.assertAlmostEqual(mse,0.24301122718491874,places=4)
        

 