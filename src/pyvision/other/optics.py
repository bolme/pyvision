'''
Created on Mar 22, 2013

@author: David S. Bolme
Oak Ridge National Laboratory
'''

def diffractionLimit(distance,wavelength,aperture):
    '''
    This function computes the Diffraction limit of an optical system.  It
    returns the smallest resolvable pattern at a given wavelength and 
    aperture. 
    
    @param distance: distance to the target in meters.
    @param wavelength: the wavelength of the light in nanometers
    @param aperture: the size of the aperture in meters.
    @returns: the resolution limit in meters
    '''
    # Convert the wavelength of the 
    wavelength = 1.0e-9*wavelength
    
    # Compute the resolution
    resolution = distance * 1.220*(wavelength/aperture)
    return resolution

def apertureComputation(distance,wavelength,resolution):
    '''
    This function computes the Diffraction limit of an optical system.  It
    returns the smallest resolvable pattern at a given wavelength and 
    aperture. 
    
    @param distance: distance to the target in meters.
    @param wavelength: the wavelength of the light in nanometers
    @param resolution: the resolution on target in metes.
    @returns: the aperture size in meters.
    '''
    # Convert the wavelength of the 
    wavelength = 1.0e-9*wavelength
    
    # Compute the resolution
    aperture = (distance * 1.220* wavelength) / resolution 
    
    return aperture

def fNumber(focal_length,aperture):
    N=focal_length/aperture
    return N
    
def depthOfField(hyperfocal,distance):
    '''
    
    '''
    H = hyperfocal
    s = distance
    Dn = (H*s)/(H+s)
    Df = (H*s)/(H-s)
    return Dn,Df,Df-Dn
    
def hyperFocalDistance(focal_length,fnumber,circle_of_confusion,definition=2):
    '''
    http://en.wikipedia.org/wiki/Hyperfocal_distance
    
    Definition 1: The hyperfocal distance is the closest distance at which a 
    lens can be focused while keeping objects at infinity acceptably sharp. 
    When the lens is focused at this distance, all objects at distances from 
    half of the hyperfocal distance out to infinity will be acceptably sharp.
    
    Definition 2: The hyperfocal distance is the distance beyond which all 
    objects are acceptably sharp, for a lens focused at infinity.
    '''        
    return (focal_length**2)/(fnumber*circle_of_confusion)

