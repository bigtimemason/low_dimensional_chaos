import numpy as np

def rk4(fRHS,t,s0,dt):

    k1 = dt*fRHS(t,s0,dt)
    k2 = dt*fRHS(t + dt/2,s0 + k1/2,dt)
    k3 = dt*fRHS(t + dt/2,s0 + k2/2,dt)
    k4 = dt*fRHS(t + dt/1,s0 + k3/1,dt)
    
    s = s0 + (k1+k4)/6 + (k2+k3)/3

    return s,1
