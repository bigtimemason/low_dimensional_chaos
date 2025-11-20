import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ode_integrators as odeint
import ode_step as step
from matplotlib.colors import LinearSegmentedColormap

# Constants
sig = 10.0
b = 8/3
r = 24

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
    fINT    = odeint.ode_ivpp1
    fORD    = step.rk45                   

    return fINT,fORD,fRHS

def main():
    
    nstep = 30000
    t0 = 0.0
    x0 = 10
    y0 = 10
    z0 = 10
    s0 = np.array([x0, y0, z0])
    
    fINT, fORD, fRHS = ode_init()
    
    t,s,it = fINT(fRHS, fORD, t0,s0, nstep)
    
    x = s[0]
    y = s[1]
    z = s[2]
    
    dx = sig * (y - x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    
    plt.figure(num=1,figsize=(10,15),dpi=300,facecolor='w')
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color = 'black', s=2) # plotting x, y, z
    ax.plot3D(x, y, z, 'black', linewidth=0.5)
    ax.set_xlabel('x',fontsize=18, color='black')
    ax.set_ylabel('y',fontsize=18, color='black')
    ax.set_zlabel('z',fontsize=18, color='black')
    ax.view_init(azim=60, elev=40)
    plt.gca().set_facecolor('w')
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
    
    # Build Lorenz map pairs (z_n, z_{n+1})
    zi_ary  = zmax_ary[:-1]
    zi1_ary = zmax_ary[1:]

   
    plt.scatter(np.array(zi_ary), np.array(zi1_ary), s=3, color='k')
    plt.xlabel('zn')
    plt.ylabel('zn+1')
    plt.title(f'Lorenz Map for r = {r}')
    plt.show()
    
    plt.figure(num=1,figsize=(10,10),dpi=300,facecolor='w')
    ax = plt.axes(projection='3d')
    ax.scatter3D(dx, dy, dz, color = 'black', s=0.5) 
    ax.plot3D(dx, dy, dz, 'black', linewidth=0.5)
    ax.set_xlabel('dx/dt',fontsize=18, color='black')
    ax.set_ylabel('dy/dt',fontsize=18, color='black')
    ax.set_zlabel('dz/dt',fontsize=18, color='black')
    ax.view_init(azim=0, elev=0)
    plt.gca().set_facecolor('w')
    plt.show()
    
    v_mag = np.zeros(len(dx))
    for i in range(len(v_mag)):
        v_mag[i] = np.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2)
    
    v_norm = v_mag/max(v_mag)
    
    plt.plot(t, v_norm,'k', linewidth = 0.8)
    plt.xlabel('t (s)')
    plt.ylabel('Speed of Particle')
    plt.title(f'Speed of particle vs time (r = {r})')
    plt.show()
    
    fig = plt.figure(num=1,figsize=(10,10),dpi=300,facecolor='w')
    ax2 = plt.axes(projection='3d')
    
    sc = ax2.scatter3D(x, y, z, c=v_norm, cmap='magma', s=0.5)

    #cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.1)
    #cbar.set_label("Normalized Speed", fontsize=14)
    ax2.plot3D(x, y, z, 'gray', linewidth=0.5)

    
    ax2.set_xlabel('x',fontsize=18, color='black')
    ax2.set_ylabel('y',fontsize=18, color='black')
    ax2.set_zlabel('z',fontsize=18, color='black')
    ax2.set_title(f'r = {r}, x0 = {x0}, y0 = {y0}, z0 = {z0}',fontsize=18)
    plt.show()
    
    plt.plot(t[0:4000],dz[0:4000],'k')
    plt.axhline(0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('t (s)')
    plt.ylabel('dz/dt')
    plt.title('dz/dt vs Time (s)')
    plt.show()
    
    print(x[-1],y[-1],z[-1])


    # ==========ANIMATION CODE===============
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # line, = ax.plot([], [], [], lw=0.5, color='k')
    
    # # Set limits based on your data
    # ax.set_xlim(np.min(x), np.max(x))
    # ax.set_ylim(np.min(y), np.max(y))
    # ax.set_zlim(np.min(z), np.max(z))
    # frames = range(0, len(t), 10)   # every 10th point → ~2000 frames instead of 20000
    
    # def init():
    #     line.set_data([], [])
    #     line.set_3d_properties([])
    #     return line,
    
    # def update(i):
    #     # i is already the actual index
    #     line.set_data(x[:i], y[:i])
    #     line.set_3d_properties(z[:i])
    #     # --- rotating camera ---
    #     angle = 0.05 * i    # adjust rotation speed here
    #     ax.view_init(elev=40, azim=angle)
    #     ax.set_title(f"t = {t[i]:.2f}")
    #     return line,
    
    
    # ani = FuncAnimation(fig, update, frames=frames,
    #                 init_func=init, interval=20, blit=False)
    
    # # Save instead of showing live
    # ani.save("/Users/masonmiller/Desktop/lorenz.gif", writer="pillow", fps=1)
    
    
 
    return t, s, it

main()
