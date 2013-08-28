'''
This package contains some standard tests that can be performed with
the test data distributed with pyvision.
'''
import pyvision as pv
import os
from pyvision.analysis.FaceAnalysis.EyesFile import EyesFile

SCRAPS_EYES = EyesFile(os.path.join(pv.__path__[0],'data','csuScrapShots','coords.txt'))
''' The path to a text file containing coordinates to the scrapshots dataset. '''



# Common test images
AIRPLANE  = os.path.join(pv.__path__[0],'data','misc','airplane.jpg')
''' The path to a commonly use test image of a jet airplane.'''

BABOON    = os.path.join(pv.__path__[0],'data','misc','baboon.jpg')
''' The path to a commonly use test image of a baboon face.'''

FRUITS    = os.path.join(pv.__path__[0],'data','misc','fruits.jpg')
''' The path to a commonly use test image of fruit.'''

LENA      = os.path.join(pv.__path__[0],'data','misc','lena.jpg')
''' The path to a commonly used "Lena" test image.'''

LOGO      = os.path.join(pv.__path__[0],'data','misc','logo.jpg')
''' The path to a commonly use test image of a butterfly used as the pyvision logo.'''

TAZ_IMAGE = os.path.join(pv.__path__[0],'data','test','TAZ_0010.jpg')
''' The path to a commonly use test image of a butterfly used as the pyvision logo.'''

TAZ_VIDEO = os.path.join(pv.__path__[0],'data','test','TazSample.m4v')
''' The path to a test video of a Loony Tunes Taz stuffed animal.'''

CAR_VIDEO = os.path.join(pv.__path__[0],'data','test','toy_car.m4v')
''' The path to a test video of a small moving car.'''

BUGS_VIDEO = os.path.join(pv.__path__[0],'data','test','BugsSample.m4v')
''' The path to a test video sequence of bugs crawling through weeds.'''


FONT_ARIAL = os.path.join(pv.__path__[0],'config','Arial.ttf')
''' The path to a file containing the Arial true type font. '''


def genderClassifier(clsfy, ilog=None):
    '''
    genderClassifier takes a classifier as an argument and will use the 
    csuScrapShot data to perform a gender classification test on that 
    classifier.
    
    These three functions will be called::
    
        for im in training_images:
            clsfy.addTraining(label,im,ilog=ilog)
        
        clsfy.train(ilog=ilog)
        
        for im in testing_images:
            clsfy.predict(im,ilog=ilog)
    
    label = 0 or 1 (0=Female,1=Male)
    
    im is a 64x64 pyvision image that is normalized to crop the face
    
    Output of predict should be a class label (0 or 1)
    
    @returns: the success rate for the testing set.
    '''
    filename = os.path.join(pv.__path__[0],'data','csuScrapShots','gender.txt')
    f = open(filename,'r')
    image_cache = []
    examples = []
    for line in f:
        im_name, class_name = line.split()
        if class_name == 'F':
            class_name = 0
        else:
            class_name = 1
        long_name = os.path.join(pv.__path__[0],'data','csuScrapShots',im_name)
        leye,reye = SCRAPS_EYES.getEyes(im_name)[0]
        im = pv.Image(long_name)
        image_cache.append(im)
        im = pv.AffineFromPoints(leye,reye,pv.Point(22,27),pv.Point(42,27),(64,64)).transformImage(im)
        #im = pv.Image(im.asPIL().resize((64,64)))
        examples.append([class_name,im,im_name])
    
    training = examples[:103]
    testing = examples[103:]     
              
    for each in training[:103]:
        clsfy.addTraining(each[0],each[1],ilog=ilog)
        
    clsfy.train(ilog=ilog)
    
    table = pv.Table()
    #values = {0:[],1:[]}

    correct = 0
    total = 0
    for each in testing:
        label = clsfy.predict(each[1],ilog=ilog)
        total += 1
        if label == each[0]:
            correct += 1
        
    rate = float(correct)/total

    if ilog: ilog.table(table)
    return rate

if __name__ == "__main__":
    from pyvision.vector.SVM import SVM
    
    svm = SVM(kernel='LINEAR',random_seed=30)
    ilog = pv.ImageLog()
    print "SVM rate:",genderClassifier(svm,ilog=None)
    
    svm = SVM(kernel='RBF',random_seed=30)
    ilog = pv.ImageLog()
    print "SVM rate:",genderClassifier(svm,ilog=None)
    
    ilog.show()
    
