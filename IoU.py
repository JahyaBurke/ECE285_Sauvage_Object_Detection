import numpy as np

def intersects(x_coord,y_coord):
    '''Both inputs are tuples of form (x0,x1,dx0,dx1)'''
    x = np.zeros(2)
    y = np.zeros(2)
    dx = np.zeros(2)
    dy = np.zeros(2)
    x[0],x[1],dx[0],dx[1] = x_coord
    y[0],y[1],dy[0],dy[1] = y_coord
    c_x = x+dx/2
    c_y = y+dy/2
    return (np.abs(c_x-c_y) < (dx+dy)/2).all()
    