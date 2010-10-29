import time
import pyvision as pv

class Timer:
    '''
    A simple timer class used to measure and record run times.  Each timer has a
    member variable named table which keeps a log of timing data.
    
    Usage:
    timer = pv.Timer()
    
    ... Do Some Stuff ...
    
    timer.mark("Event 1")
    
    ... Do Some Stuff ...
    
    timer.mark("Event 2")
    
    print timer
    -- or --
    ilog(timer,"TimingData")
    
    '''
    
    def __init__(self):
        '''
        Create and setup the timer.  Also creates a mark titled "Timer Created".
        '''
        self.table = pv.Table()
        self.started = False
        self.i = 0
        self.mark("Timer Created")
                
    def mark(self,event,notes=None):
        '''
        
        @param event: a short text description of the event being marked.
        @param notes: additional notes for this event.
        @returns: 6-tuple of times in seconds: Wall Clock Time, Time since last mark, Time since creation, CPU time, CPU time since last mark, CPU time since creation
        '''
        current = time.time()
        cpu = time.clock()
        if self.started == False:
            self.started = True
            self.prev = self.start = current
            self.cpu_prev = self.cpu_start = cpu
        self.table[self.i,"event"] = event
        self.table[self.i,"time"] = current
        self.table[self.i,"current"] = current - self.prev
        self.table[self.i,"total"] = current - self.start
        self.table[self.i,"cpu_time"] = cpu
        self.table[self.i,"cpu_current"] = cpu - self.cpu_prev
        self.table[self.i,"cpu_total"] = cpu - self.cpu_start
        self.table[self.i,"notes"] = notes
        
        rt = current
        ct = current - self.prev
        tot = current - self.start
        crt = cpu
        cct = cpu - self.cpu_prev
        ctot = cpu - self.cpu_start
        
        self.prev = current
        self.cpu_prev = cpu
        self.i += 1
        
        return rt,ct,tot,crt,cct,ctot
        
    def __str__(self):
        '''Render the timing log to a string.'''
        
        return self.table.__str__()
    
    def save(self,filename):
        '''Save the timing log to a csv file.'''
        self.table.save(filename)
        
            
            
        

