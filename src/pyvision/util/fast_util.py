import numpy as np
from scipy import weave


class LocalMaximumDetector:
    def __init__(self,max_length=1000000):
        self.max_length = max_length
        self.maxes = np.zeros((max_length,2),dtype=np.int)
        self.vals = np.zeros((max_length,),dtype=np.float)
    
    def __call__(self, mat, threshold = None, sort_results = True):
        '''
        All any local maximum that are greater than threshhold up to a total of 
        max_length.
        
        To save time arrays that hold the maxes and vals that are created 
        once and reused for each call.  This means that local maximum detection
        is not thread safe. If using this class with threads create an instance
        for each thread.
        
        @param mat: 2d Real Matrix input.
        @param threshold: Mininum value of local maxima.
        @param sort_results: set to False to save time and return an unorderd list.
        
        @returns: maxes,vals
        '''
        maxes = self.maxes
        vals = self.vals
        r,c = mat.shape
        max_length = self.max_length
        
        if threshold != None:
            count = weave.inline(
                '''  
                int count = 0;
                
                for( int i = 1; i < r-1 ; i++){
                    for(int j = 1; j < c-1 ; j++){
                        // Check if the current location meets the threshold
                        
                        if (mat(i,j) > threshold    &&
                            mat(i,j) > mat(i,j-1)   &&
                            mat(i,j) > mat(i,j+1)   &&
                            mat(i,j) > mat(i-1,j-1) &&
                            mat(i,j) > mat(i-1,j)   &&
                            mat(i,j) > mat(i-1,j+1) &&
                            mat(i,j) > mat(i+1,j-1) &&
                            mat(i,j) > mat(i+1,j)   &&
                            mat(i,j) > mat(i+1,j+1)){
                        
                            // This is a local max
                            maxes(count,0) = i;
                            maxes(count,1) = j;
                            vals(count) = mat(i,j);
                            count += 1;
                            
                            if(count == max_length){
                                i = r;
                                j = c;
                            }
                        }   
                    }
                }
    
                return_val = count;
                ''',
                arg_names=['mat','maxes','vals','max_length','threshold','r','c'],
                type_converters=weave.converters.blitz,
            )
        else:
            count = weave.inline(
                '''  
                int count = 0;
                
                for( int i = 1; i < r-1 ; i++){
                    for(int j = 1; j < c-1 ; j++){
                        // Check if the current location meets the threshold
                        
                        if (mat(i,j) > mat(i,j-1)   &&
                            mat(i,j) > mat(i,j+1)   &&
                            mat(i,j) > mat(i-1,j-1) &&
                            mat(i,j) > mat(i-1,j)   &&
                            mat(i,j) > mat(i-1,j+1) &&
                            mat(i,j) > mat(i+1,j-1) &&
                            mat(i,j) > mat(i+1,j)   &&
                            mat(i,j) > mat(i+1,j+1)){
                        
                            // This is a local max
                            maxes(count,0) = i;
                            maxes(count,1) = j;
                            vals(count) = mat(i,j);
                            count += 1;
                            
                            if(count == max_length){
                                i = r;
                                j = c;
                            }
                        }   
                    }
                }
    
                return_val = count;
                ''',
                arg_names=['mat','maxes','vals','max_length','r','c'],
                type_converters=weave.converters.blitz,
            )
        
        if sort_results == False:
            return maxes[:count,:].copy(),vals[:count].copy()
        
        order = np.argsort(vals[:count])[::-1]
        maxes = maxes[order]
        vals = vals[order]
        
        #print vals
        #print maxes
        
        return maxes,vals
        
        

