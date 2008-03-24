Blobs: Synthetic data for testing denoising functions.

Regression: synthetic data for testing linear regression.  Generated using: 
	import random
	for i in range(100):
	    x = random.uniform(10.0,50.0)    
	    y = random.uniform(10.0,50.0)
	    z = random.uniform(10.0,50.0)
	    a = 0.25*x + .54*y + .98*z + 32.2 + random.normalvariate(0.0,0.5)
	    b = 5.0*x + 5.4*y + 0.4*z + 12.2 + random.normalvariate(0.0,2.0)
	    c = 3.3*x + 2.1*y + 3.0*z + 0.0 + random.normalvariate(0.0,3.0)
	    print "%0.5f %0.5f %0.5f %0.5f %0.5f %0.5f"%(a,b,c,x,y,z)
    
