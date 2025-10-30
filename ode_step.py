import numpy as np

def rk45(fRHS,t,s0,dt):

    ai    = [0, 0.2, 0.3, 0.6, 1, 7 / 8]
    bij   = [[], [0.2], [3 / 40, 9 / 40], [0.3, -0.9, 1.2], [-11 / 54, 2.5, -70 / 27, 35 / 27], 
             [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096]]
    ci    = [37/378, 0, 250 / 621, 125 / 594, 0, 512 / 1771]
    ci_st = [2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 0.25]
    #Kuttas
    k1    = dt*fRHS(t,s0,dt)
    k2    = dt*fRHS(t + (ai[1] * dt),s0 + (bij[1][0]*k1),dt)
    k3    = dt*fRHS(t + (ai[2] * dt),s0 + (bij[2][0]*k1) + (bij[2][1]*k2),dt)
    k4    = dt*fRHS(t + (ai[3] * dt),s0 + (bij[3][0]*k1) + (bij[3][1]*k2) + (bij[3][2]*k3),dt)
    k5    = dt*fRHS(t + (ai[4] * dt),s0 + (bij[4][0]*k1) + (bij[4][1]*k2) + (bij[4][2]*k3) + (bij[4][3]*k4),dt)
    k6    = dt*fRHS(t + (ai[5] * dt),s0 + (bij[5][0]*k1) + (bij[5][1]*k2) + (bij[5][2]*k3) + (bij[5][3]*k4) + (bij[5][4]*k5),dt)
    #General Form
    s_gen = s0 + (ci[0] * k1) + (ci[1] * k2) + (ci[2] * k3) + (ci[3] * k4) + (ci[4] * k5) + (ci[5] * k6)
    #Embedded Form
    s_emb = s0 + (ci_st[0] * k1) + (ci_st[1] * k2) + (ci_st[2] * k3) + (ci_st[3] * k4) + (ci_st[4] * k5) + (ci_st[5] * k6)

    return s_gen,s_emb,1

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
