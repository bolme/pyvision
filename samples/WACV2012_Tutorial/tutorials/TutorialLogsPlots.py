
import pyvision as pv

ilog = pv.ImageLog()
im = pv.Image("baboon.jpg")
ilog(im,"Baboon")

table = pv.Table()
table[1,"image"] = im.filename
table[1,"width"] = im.size[0]
table[1,"height"] = im.size[1]
ilog(table,"ImageData")
print(table)

plot = pv.Plot(title="Some Dots and Lines");
plot.points([[3.5,7.1],[1,1],[5.5,2]],shape=2)
plot.lines([[5.5,7.5],[2,3],[3.3,7]])
ilog(plot,"MyPlot")

ilog.show()