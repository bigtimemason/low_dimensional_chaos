import numpy as np
import matplotlib.pyplot as plt
import ode_integrators as odeint
import ode_step as step
from mpl_toolkits import mplot3d

# initial 
sig = 10
b = 8/3

# ========================================
# s = array containing x(t), y(t), z(t)
# r is variable and defined in main
# returns and array containing time derivatives of x,y,z
# ----------------------------------------
def dsdt(s, r):
    
    x = s[0]
    y = s[1]
    z = s[2]
    
    dsdt = np.zeros(3)
    
    dsdt[0] = sig * (y - x)
    dsdt[1] = r*x - y - x*z
    dsdt[2] = x*y - b*z
    
    return dsdt

def ode_init():
          
    fRHS    = dsdt     
    fINT    = odeint.ode_ivp   
    fORD    = step.rk4                    

    return fINT,fORD,fRHS

def main():
    
    nstep = 1000
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    
    fINT, fORD, fRHS = ode_init()
    fBVP = 0
    s0 = np.array(x0, y0, z0)
    
    x,y,z,it = fINT(fRHS, fORD, fBVP, x0, y0, z0, nstep)
    
    
    return x, y, z, it

main()
