from knn import *

if __name__ == "__main__":
    import pyvision as pv
    
    plot = pv.Image(np.zeros((500,500)))
    
    data =   np.array([[ 0.89761049,  0.31978809],
                       [ 0.08168021,  0.75605386],
                       [ 0.67596172,  0.94886192],
                       [ 0.8283411 ,  0.53639021],
                       [ 0.50589098,  0.64003199],
                       [ 0.66290861,  0.45572   ],
                       [ 0.34614808,  0.16191715],
                       [ 0.49566747,  0.83423913],
                       [ 0.32471352,  0.20317006],
                       [ 0.42948424,  0.78900121],
                       [ 0.017235  ,  0.99522359],
                       [ 0.21276987,  0.15219815],
                       [ 0.84833654,  0.87647   ],
                       [ 0.99716754,  0.47017644],
                       [ 0.51667204,  0.63936825],
                       [ 0.370152  ,  0.06977327],
                       [ 0.16250232,  0.42129633],
                       [ 0.59071007,  0.48371244],
                       [ 0.70240547,  0.72759716],
                       [ 0.21276305,  0.76596722]])
    
    data = np.random.random((100,2))
    
    for x,y in data[:,:2]:
        plot.annotatePoint(500.0*pv.Point(x,y),color='gray')
    
    x = np.array([[ 0.57097488,  0.33239627],
                  [ 0.65494268,  0.31132802],
                  [ 0.58122984,  0.69620259]])  
    x = np.random.random((7,2))

    knn = KNearestNeighbors(data)

    dist,dist_sort = knn.query(x,k=5,p=np.inf)

    #print dist
    #print dist_sort

    for i in range(dist_sort.shape[0]):
        sx,sy = x[i,:2]
        color = ["red","green","blue",'orange','purple',"cyan",'magenta'][i]
        plot.annotatePoint(500.0*pv.Point(sx,sy),color=color)  
        for tx,ty in knn.data[dist_sort[i],:2]:
            plot.annotateLine(500.0*pv.Point(sx,sy),500.0*pv.Point(tx,ty),color=color)

    plot.show()
    
        
