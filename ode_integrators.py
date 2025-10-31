import numpy as np

def ode_ivp(fRHS,fORD,t0,s0,t1,nstep):

    nvar    = s0.size                      # number of ODEs
    t       = np.linspace(t0,t1,nstep+1)   # generates equal-distant support points
    s       = np.zeros((nvar,nstep+1)) 
    s[:,0]  = s0                         
    dt      = 1e-1                    # ???? step size - IMPLAMENT ADAPTIVE STEP SIZE
    it      = np.zeros(nstep+1)
    
    for k in range(1,nstep+1):
        s_gen,s_emb,_ = fORD(fRHS,t[k-1],s[:,k-1],dt)
        err = np.max(np.abs(s_gen - s_emb))
        tol = np.max(1e-9 * (np.abs(s_gen) + (dt * fRHS(t[k-1],s[:,k-1],dt))))
        if err == 0:
            err = tol / 10e6
        h1 = dt * ((tol / err) ** 0.2)
        if err <= tol:
            dt = dt
            _,s[:,k],it[k] = fORD(fRHS,t[k-1],s[:,k-1],dt)
        else:
            adapt = 0
            max_adapt = 1000
            while err > tol and adapt < max_adapt:
                dt = h1
                s_gen,s_emb,_ = fORD(fRHS,t[k-1],s[:,k-1],dt)
                err = np.max(np.abs(s_gen - s_emb))
                tol = np.max(1e-9 * (np.abs(s_gen) + (dt * fRHS(t[k-1],s[:,k-1],dt))))
                if err == 0:
                    break
                h1 = dt * ((tol / err) ** 0.2)
                adapt += 1
                if adapt == max_adapt:
                    dt = 1e-8
                    break
            _,s[:,k],it[k] = fORD(fRHS,t[k-1],s[:,k-1],dt)
    return t,s,it
