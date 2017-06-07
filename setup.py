from distutils.core import setup
import os

try:
    import cv2
    cv_version_data = cv2.__version__.split('.')
    cv_version_data = [int(cv_version_data[0]),int(cv_version_data[1])]
    if cv_version_data[0] > 2:
        pass
    else:
        assert cv_version_data[0] == 2
        assert cv_version_data[1] == 4
    
except:
    print "ERROR: Could not load OpenCV.  Make sure the python bindings for opencv 2.4.9 or later are installed."
    raise

package_data = {}


# add package data for the data dir
package_data['pyvision.data'] = []
DATA_DIR = 'src/pyvision/data/'
for path,dirs,files in os.walk(DATA_DIR):
    for filename in files:
        pathname = os.path.join(path,filename)
        
        # strip off the front
        pathname = pathname[len(DATA_DIR):]
        base,ext = os.path.splitext(pathname)
        if ext not in ['.jpg','.png','.pgm','.txt','.pdf','.csv','.data','.tiff','.dat','.m4v','.mov']:
            #print 'skipping',pathname
            continue
        
        package_data['pyvision.data'].append(pathname)


# add package data for the config dir
package_data['pyvision.config'] = []
DATA_DIR = 'src/pyvision/config/'
for path,dirs,files in os.walk(DATA_DIR):
    for filename in files:
        pathname = os.path.join(path,filename)
        
        # strip off the front
        pathname = pathname[len(DATA_DIR):]
        base,ext = os.path.splitext(pathname)
        if ext not in ['.xml','.ttf','.fel']:
            #print 'skipping',pathname
            continue
        
        package_data['pyvision.config'].append(pathname)

#pyvision/config

setup(name='PyVision',
      version='1.1.0',
      description='A computer vision library for python.',
      author='David Bolme',
      author_email='dbolme@gmail.com',
      url='https://github.com/bolme/pyvision',
      packages=['pyvision',
                'pyvision.vector',
                'pyvision.point',
                'pyvision.analysis',
                'pyvision.analysis.R',
                'pyvision.analysis.FaceAnalysis',
                'pyvision.analysis.classifier',
                'pyvision.config',
                'pyvision.surveillance',
                'pyvision.face',
                'pyvision.features',
                'pyvision.data',
                'pyvision.edge',
                'pyvision.optimize',
                'pyvision.util',
                'pyvision.gui',
                'pyvision.segment',
                'pyvision.tools',
                'pyvision.testsuite',
                'pyvision.other',
                'pyvision.beta',
                'pyvision.types',
                'pyvision.ml',
                ],
      package_dir = {'': 'src'},
      package_data = package_data,
)
