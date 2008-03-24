plotprediction <- function(truth,predicted){

	msd = max(sd(truth),sd(predicted))
	mt = mean(truth)
	mp = mean(predicted)
	minval = min(truth,predicted)
	maxval = max(truth,predicted)
	
	print(msd) 
	print(mt) 
	print(mp) 
	print(minval)
	print(maxval)
	
	quartz("RegressionScatter",10,10)
	plot(c(mt-2*msd,mt+2*msd),c(mp-2*msd,mp+2*msd),type='n',xlab='Truth',ylab='Predicted',asp=1.0)
	points(truth,predicted)
	lines(c(minval,maxval),c(minval,maxval))
	lines(c(minval,maxval),c(minval+1,maxval+1),col='gray')
	lines(c(minval,maxval),c(minval+2,maxval+2),col='gray')
	lines(c(minval,maxval),c(minval+3,maxval+3),col='gray')
	lines(c(minval,maxval),c(minval+4,maxval+4),col='gray')
	lines(c(minval,maxval),c(minval-1,maxval-1),col='gray')
	lines(c(minval,maxval),c(minval-2,maxval-2),col='gray')
	lines(c(minval,maxval),c(minval-3,maxval-3),col='gray')
	lines(c(minval,maxval),c(minval-4,maxval-4),col='gray')

	quartz("RegressionResidual",10,10)
	residuals = predicted-truth
	rsd = sd(residuals)
	plot(c(mt-2*msd,mt+2*msd),c(0-2*rsd,0+2*rsd),type='n',xlab='Truth',ylab='Predicted',asp=1.0)
	points(truth,residuals)
	lines(c(mt-2*msd,mt+2*msd),c(0,0))
	
}