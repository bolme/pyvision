import time
import pyvision as pv

class Timer:
    '''
    A simple timer class used to measure run times.
    '''
    def __init__(self):
        self.table = pv.Table()
        self.started = False
        self.i = 0
        self.mark("Timer Created")
                
    def mark(self,event,notes=None):
        current = time.time()
        if self.started == False:
            self.started = True
            self.prev = self.start = current
        self.table[self.i,"event"] = event
        self.table[self.i,"time"] = current
        self.table[self.i,"current"] = current - self.prev
        self.table[self.i,"total"] = current - self.start
        self.table[self.i,"notes"] = notes
        self.prev = current
        self.i += 1
        
    def __str__(self):
        return self.table.__str__()
    
    def save(self,filename):
        self.table.save(filename)
        
            
            
        

