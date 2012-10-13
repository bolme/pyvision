def foo(a,b):
    ''' A function that adds two numbers '''
    return a + b

# Count from 0 to 9

i = 0
while i < 10:

    print "Number:",i,

    if i % 2 == 0:
        print "even"
    else:
        print "odd"

    i = foo(i, 1)
