plottraining <- function(training){
	quartz(width=10,height=10)
	wireframe(training$error ~ log(training$gamma) * -log(training$C))
	quartz(width=10,height=10)
	wireframe(log(training$error) ~ log(training$gamma) * -log(training$C))
	quartz(width=10,height=10)
	wireframe(training$predict ~ log(training$gamma) * -log(training$C))
	}