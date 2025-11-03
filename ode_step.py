import numpy as np

def rk4(fRHS,t,s0,dt):

    k1 = dt*fRHS(t,s0,dt)
    k2 = dt*fRHS(t + dt/2,s0 + k1/2,dt)
    k3 = dt*fRHS(t + dt/2,s0 + k2/2,dt)
    k4 = dt*fRHS(t + dt/1,s0 + k3/1,dt)
    
    s = s0 + (k1+k4)/6 + (k2+k3)/3

    return s,1

def rk45single(fRHS,t,s0,dt):
    a         = np.array([0.0,0.2,0.3,0.6,1.0,0.875]) # weights for x
    b         = np.array([[0.0           , 0.0        , 0.0          , 0.0             , 0.0         ],
                          [0.2           , 0.0        , 0.0          , 0.0             , 0.0         ],
                          [0.075         , 0.225      , 0.0          , 0.0             , 0.0         ],
                          [0.3           , -0.9       , 1.2          , 0.0             , 0.0         ],
                          [-11.0/54.0    , 2.5        , -70.0/27.0   , 35.0/27.0       , 0.0         ],
                          [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]])
    c         = np.array([37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0])
    dc        = np.array([2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25])
    dc        = c-dc
    n         = s0.size
    ds        = np.zeros(n)        # updates (arguments in f(x,y))
    dsdt      = np.zeros((6,n))    # derivatives (k1,k2,k3,k4,k5,k6)
    sout      = s0                 # result
    serr      = np.zeros(n)        # error
    dsdt[0,:] = dt*fRHS(t,s0,dt)  # first guess
    for i in range(1,6):           # outer loop over k_i 
        ds[:]     = 0.0
        for j in range(i):         # inner loop over y as argument to fRHS(x,y)
            ds = ds + b[i,j]*dsdt[j,:]
        dsdt[i,:] = dt*fRHS(t+a[i]*ds,s0+ds,a[i]*dt)
    for i in range(0,6):           # add up the k_i times their weighting factors
        sout = sout + c [i]*dsdt[i,:]
        serr = serr + dc[i]*dsdt[i,:]

    return sout,serr

def rk45bare(fRHS,t,s0,dt):
    sout,serr = rk45single(fRHS,t,s0,dt)
    return sout,1
