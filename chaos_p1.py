import numpy as np
import matplotlib.pyplot as plt
import ode_integrators as odeint
import ode_step as step

# Constants
sig = 10.0
b = 8/3
r = 212

# ========================================
# s = array containing x(t), y(t), z(t)
# r is variable and defined in main
# returns and array containing time derivatives of x,y,z
# ----------------------------------------
def dsdt(t, s, dt):
    
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
    
    nstep = 10000
    t0 = 0.0
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    s0 = np.array([x0, y0, z0])
    t1 = 30.0
    
    fINT, fORD, fRHS = ode_init()
    
    t,s,it = fINT(fRHS, fORD, t0, s0, t1, nstep)
    
    plt.figure(num=1,figsize=(10,15),dpi=300,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(s[0], s[1], s[2], 'blue') # plotting x, y, z
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
    # x vs t
    plt.figure(num=1,figsize=(20,10),dpi=100,facecolor='white')
    plt.subplot(311)
    plt.plot(t,s[0])
    plt.xlabel('t')
    plt.ylabel('x')
    # y vs t
    plt.subplot(312)
    plt.plot(t,s[1])
    plt.xlabel('t')
    plt.ylabel('y')
    # z vs t
    plt.subplot(313)
    plt.plot(t,s[2])
    plt.xlabel('t')
    plt.ylabel('z')
    plt.show()
    
    x = s[0]
    y = s[1]
    z = s[2]

    
    return s, it

main()
