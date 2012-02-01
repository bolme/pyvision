'''
This package contains "beta" code that is not quite ready to de included in 
the main pyvision distribution.  The package consists of a set of "install"
commands that add functionality to the pyvision namespace.  This is to allow
easy access to new classes and methods while under development and to make
it easy to update code when that funtionality gets added to the main library.
These install functions will be deleted without notice when the code is 
include in pyvision.  

For example, the video task manager can be installed like this:

import pyvision as pv
pv.beta.installVideoTaskManager()
pv.VideoTaskMangager()

When the interface for the video task manager is finalized and it is added to
the main library the install command will be deleted from the beta package and
any external code that was using the beta code can just delete the insall
command.  The rest of the external code should run little or no modification.

Created Jan 27, 2012
@author: bolme
'''

import pyvision as pv

def installVideoTaskManager():
    '''
    Install the video task manager beta code into the pyvision namespace.
    '''
    from vtm import VideoTask, VideoTaskManager
    from videotasks import FaceDetectorVT
    pv.VideoTask = VideoTask
    pv.VideoTaskManager = VideoTaskManager
    pass