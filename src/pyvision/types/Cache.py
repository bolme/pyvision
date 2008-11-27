

class Cache:
    '''
    This is a simple object that acts like a dictionary.  It can be used to 
    store objects that may or may not be reused in the future.  One example
    is the generation of image filters in the Fourier domain.  The filters can
    be generated once and stored in the cache for future use.  This will save 
    time if the same size image is filtered over and over again such as in 
    video processing.  If multiple image sizes are used only a maximum number
    are stored in the cache.
    
    The Cache is used like a simple dictionary. An object (obj) is stored in 
    the cache using a simple key (key) using the following command:
    
    cache[key] = obj
    
    The object can be retrieved using the command:
    
    obj = cache[key]
    
    Each of these commands will reset that objects last access time.
    
    If the number of objects in the cache exceeds the purge number, the objects
    that have not been accessed recently will be removed until the size is reduced
    down to the specified size of the cache.
    '''
    
    def __init__(self, size = 50, purge = 100):
        '''
        @param size: the number of items to keep in the Cache
        @param purge: when the  number of items exceeds this size the
                      oldest items in the cache will be purged.
        '''
        raise NotImplementedError("This function is not yet implemented.")
