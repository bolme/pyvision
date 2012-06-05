import sys
import time

def secToHMS(seconds):
    hours = int(seconds/3600.0)
    seconds -= hours*3600.0
    minutes = int(seconds/60.0)
    seconds -= minutes*60.0
    seconds = int(seconds)
    return hours,minutes,seconds

class ProgressBar:
    '''
    Modified from: http://code.activestate.com/recipes/168639-progress-bar-class/
    '''
    def __init__(self, minValue = 0, maxValue = 10, totalWidth=60):
        self.start_time = time.time()
        self.progBar = "[]"   # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue
        self.width = totalWidth
        self.amount = 0       # When amount == max, we are 100% done 
        self.updateAmount(0)  # Build progress bar string

    def updateAmount(self, newAmount = None):
        '''
        Update the progress bar.  By default this will increment the amount by 1.
        
        After calling this make sure to call show() to display the progress bar. 
        '''
        if newAmount == None: newAmount = self.amount + 1
        if newAmount < self.min: newAmount = self.min
        if newAmount > self.max: newAmount = self.max
        self.amount = newAmount

        # Figure out the new percent done, round to an integer
        diffFromMin = float(self.amount - self.min)
        fracDone = diffFromMin / float(self.span)
        percentDone = (diffFromMin / float(self.span)) * 100.0
        percentDone = round(percentDone)
        percentDone = int(percentDone)

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2
        numHashes = (percentDone / 100.0) * allFull
        numHashes = int(round(numHashes))

        # build a progress bar with hashes and spaces
        self.progBar = "[" + '#'*numHashes + ' '*(allFull-numHashes) + "]"

        # figure out where to put the percentage, roughly centered
        percentPlace = (len(self.progBar) / 2) - len(str(percentDone)) 
        percentString = str(percentDone) + "%"

        # slice the percentage into the bar
        self.progBar = self.progBar[0:percentPlace-1] + " " + percentString + " " + self.progBar[percentPlace+len(percentString)+1:]

        # compute time remaining in hours, min, and sec
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if fracDone > 0.01:
            remaining_time = elapsed_time * (1-fracDone) / fracDone
            hours,minutes,seconds = secToHMS(remaining_time)
            self.time_string = "%02dh %02dm %02ds"%(hours,minutes,seconds)
        else: 
            self.time_string = "--h --m --s"
            
        
    def __str__(self):
        return str(self.progBar)
    
    def show(self):
        '''Displays the progress bar on stdout.'''
        sys.stdout.write(self.__str__()+"  " + self.time_string + "\r")
        sys.stdout.flush()
    
    def finish(self):
        '''Conducts Cleanup and new line.'''
        self.updateAmount(self.max)
        elapsed_time = time.time() - self.start_time
        hours,minutes,seconds = secToHMS(elapsed_time)

        sys.stdout.write(self.__str__()+"  %02dh %02dm %02ds"%(hours,minutes,seconds) + "\n")
        sys.stdout.flush()
    
if __name__ == '__main__':
    prog = ProgressBar(0, 100, 60)
    print dir(prog)
    for i in xrange(101):
        prog.updateAmount(i)
        prog.show()
        time.sleep(.05)
    prog.finish()
    
    