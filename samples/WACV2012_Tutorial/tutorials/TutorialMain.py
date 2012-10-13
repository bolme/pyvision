def add(a,b):
    return a + b

class Add:
    def __init__(self,a,b):
        self.val = a+b

    def getValue(self): 
        return self.val

if __name__ == '__main__':
    # execute this code if this
    # is the main script
    print "Hello World!!!"
    print "2 + 3 =",add(2,3)
    my_obj = Add(2,3)
    print "Add(2,3)=", my_obj.getValue()