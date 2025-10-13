import numpy as np


def rk4(fRHS,x0,y0,dx):
    #???????? from here
    k1 = dx*fRHS(x0,y0,dx)
    k2 = dx*fRHS(x0 + dx/2,y0 + k1/2,dx)
    k3 = dx*fRHS(x0 + dx/2,y0 + k2/2,dx)
    k4 = dx*fRHS(x0 + dx/1,y0 + k3/1,dx)
    
    y = y0 + (k1+k4)/6 + (k2+k3)/3
    #???????? to here
    return y,1
