import numpy as np


def rk4(fRHS,x0,y0,dx):
    #???????? from here
    #Cash-Karp Parameters
    def rk(fRHS,x0,y0,dx):
        ai    = [0, 0.2, 0.3, 0.6, 1, 7 / 8]
        bij   = [[], [0.2], [3 / 40, 9 / 40], [0.3, -0.9, 1.2], [-11 / 54, 2.5, -70 / 27, 35 / 27], 
                 [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096]]
        ci    = [37/378, 0, 250 / 621, 125 / 594, 0, 512 / 1771]
        ci_st = [2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 0.25]
        #Kuttas
        k1    = dx*fRHS(x0,y0,dx)
        k2    = dx*fRHS(x0 + (ai[1] * dx),y0 + (bij[1][0] * k1),dx)
        k3    = dx*fRHS(x0 + (ai[2] * dx),y0 + (bij[2][0]*k1) + (bij[2][1]*k2),dx)
        k4    = dx*fRHS(x0 + (ai[3] * dx),y0 + (bij[3][0]*k1) + (bij[3][1]*k2) + (bij[3][2]*k3),dx)
        k5    = dx*fRHS(x0 + (ai[4] * dx),y0 + (bij[4][0]*k1) + (bij[4][1]*k2) + (bij[4][2]*k3) + (bij[4][3]*k4),dx)
        k6    = dx*fRHS(x0 + (ai[5] * dx),y0 + (bij[5][0]*k1) + (bij[5][1]*k2) + (bij[5][2]*k3) + (bij[5][3]*k4) + (bij[5][4]*k5),dx)
        #General Form
        y_gen = y0 + (ci[0] * k1) + (ci[1] * k2) + (ci[2] * k3) + (ci[3] * k4) + (ci[4] * k5) + (ci[:,5] * k6)
        #Embedded Form
        y_emb = y0 + (ci_st[0] * k1) + (ci_st[1] * k2) + (ci_st[2] * k3) + (ci_st[3] * k4) + (ci_st[4] * k5) + (ci_st[5] * k6)
        return y_gen,y_emb
    y_gen,y_emb = rk(fRHS(x0,y0,dx)
    err   = np.abs(y_emb - y_gen)
    h1    = dx - (tol / err) ** 0.2
    tol   = 10e2 * (np.abs(emb0) + (dx * np.abs(fRHS)))
    if err <= tol:
        dx = dx
        y_gen,y_emb = rk(fRHS,x0,y0,dx)
    else:
        dx = h1
        y_gen,y_emb = rk(fRHS,x0,y0,dx)
    #???????? to here
    return y_emb,1
