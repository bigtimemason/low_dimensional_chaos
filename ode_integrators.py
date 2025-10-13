import numpy as np

def ode_ivp(fRHS,fORD,t0,s0,t1,nstep):

    nvar    = s0.size                      # number of ODEs
    t       = np.linspace(t0,t1,nstep+1)   # generates equal-distant support points
    s       = np.zeros((nvar,nstep+1)) 
    s[:,0]  = s0                         
    dt      = 1e-3                         # ???? step size - IMPLAMENT ADAPTIVE STEP SIZE
    it      = np.zeros(nstep+1)
    for k in range(1,nstep+1):
        s[:,k],it[k] = fORD(fRHS,t[k-1],s[:,k-1],dt)
    return t,s,it
