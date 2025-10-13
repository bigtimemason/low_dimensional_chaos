import numpy as np

def ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep):

    nvar    = y0.size                      # number of ODEs
    x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
    y       = np.zeros((nvar,nstep+1))     # result array 
    y[:,0]  = y0                           # set initial condition
    dx      = (x[1]-x[0])                  # step size
    it      = np.zeros(nstep+1)
    for k in range(1,nstep+1):
        y[:,k],it[k] = fORD(fRHS,x[k-1],y[:,k-1],dx)
    return x,y,it
