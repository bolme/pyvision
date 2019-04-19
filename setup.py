from distutils.core import setup
import os


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

setup(name='pyvision_toolkit',
      version='1.3.2',
      description='A computer vision library for python.',
      author='David Bolme',
      author_email='dbolme@gmail.com',
      url='https://github.com/bolme/pyvision',
      keywords = ["machine learning", "vision", "image"],
         classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        ],
    long_description = "A computer vision library for python.",

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
      install_requires=[
          'opencv-python',
          'Pillow',
          'numpy',
          'scipy',
          'scikit-image',
          'scikit-learn',
      ],
)
