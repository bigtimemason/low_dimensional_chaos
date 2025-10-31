import numpy as np
import matplotlib.pyplot as plt
import ode_integrators as odeint
import ode_step as step

# Constants
sig = 10.0
b = 8/3
r = 166.83

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
    fORD    = step.rk45                   

    return fINT,fORD,fRHS

def main():
    
    nstep = 20000
    t0 = 0.0
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    s0 = np.array([x0, y0, z0])
    t1 = 10.0
    
    fINT, fORD, fRHS = ode_init()
    
    t,s,it = fINT(fRHS, fORD, t0, s0, t1, nstep)
    
    plt.figure(num=1,figsize=(10,15),dpi=300,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(s[0], s[1], s[2], '#4B9CD3') # plotting x, y, z
    ax.set_xlabel('x',fontsize=18)
    ax.set_ylabel('y',fontsize=18)
    ax.set_zlabel('z',fontsize=18)
    ax.view_init(azim=60, elev=40)
    plt.show()
    
    # x vs t
    plt.figure(num=1,figsize=(20,10),dpi=100,facecolor='white')
    plt.subplot(311)
    plt.plot(t,s[0],'black')
    plt.ylabel('x',fontsize=18)
    # y vs t
    plt.subplot(312)
    plt.plot(t,s[1], 'black')
    plt.ylabel('y',fontsize=18)
    # z vs t
    plt.subplot(313)
    plt.plot(t,s[2], 'black')
    plt.ylabel('z',fontsize=18)
    plt.xlabel('t',fontsize=18)
    plt.show()
    
    
    x = s[0]
    y = s[1]
    z = s[2]
    
    dx = sig * (y - x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    
    zmax_ary = []
    
    tol = 0.0001*nstep
    for i in range(1, len(dz)):
        
        if dz[i-1] > 0 and dz[i] <= 0:
            diff = (dz[i-1] - dz[i])
            
            if abs(diff) <= 0 + tol:
                z_max = max(z[i-1], z[i])
            else:
                a = dz[i-1] / diff

                z_max = z[i-1] + a * (z[i] - z[i-1])
            zmax_ary.append(z_max)
    
    zi_ary  = zmax_ary[:-1]
    zi1_ary = zmax_ary[1:]

   
    plt.scatter(np.array(zi_ary), np.array(zi1_ary), s=3)
    plt.xlabel('zn')
    plt.ylabel('zn+1')
    plt.title(f'Lorenz Map for r = {r}')
    plt.show()

        
    
    
 
    return t, s, it

main()
