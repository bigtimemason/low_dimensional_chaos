# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import ode_integrators as odeint
import ode_step as step

# Constants
sig = 10.0
b = 8/3
r = 24

def generate_random_binary_sequence(n):
    return [np.random.randint(low=0, high=2) for _ in range(n)]

try1 = generate_random_binary_sequence(10)
# ========================================
# s = array containing x(t), y(t), z(t), u(t), v(t), w(t)
# r is variable and defined in main
# returns and array containing time derivatives of x,y,z,u,v,w
# x,y,z is sender ; u,v,w is reciever
# u,v,w will equal x,y,z 
# ----------------------------------------

def dsdt(t, s, dt):
    
    t = float(np.atleast_1d(t)[0])  # ensure scalar time
    A = 0.3       
    Tb = 0.5      
    n_bits = 20   # number of bits in the message
    binary_seq = generate_random_binary_sequence(n_bits)


    bit_index = int((t // Tb) % n_bits)  # wrap around when t > total duration
    current_bit = binary_seq[bit_index]

    # Convert bit (0 or 1) into amplitude-modulated signal
    sound_signal1 = A * current_bit
    
    
    x = s[0]
    y = s[1]
    z = s[2]
    u = s[3]
    v = s[4]
    w = s[5]
    
    dsdt = np.zeros(6)
    
  # X_t = x#X_t = x + s, where s(t) is audio signal - > s = X - u
    X_t = x+sound_signal1
    
    
    dsdt[0] = sig * (y - x) #dx/dt
    dsdt[1] = r*x - y - x*z #dy/dt
    dsdt[2] = x*y - b*z #dz/dt
    dsdt[3] = sig * (v-u) #du/dt
    dsdt[4] = r * X_t - v - X_t * w #dv/dt
    dsdt[5] = X_t * v - b*w #dw/dt
    
    return dsdt

def ode_init():
          
    fRHS    = dsdt   
    fINT    = odeint.ode_ivp   
    fORD    = step.rk45bare                  
    return fINT,fORD,fRHS

def main():
    
    nstep = 200000
    t0 = 0.0
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    u0 = 1
    v0 = 1
    w0 = 1
    
    s0 = np.array([x0, y0, z0,u0,v0,w0])
    t1 = 10.0
    
    fINT, fORD, fRHS = ode_init()
    
    t,s,it = fINT(fRHS, fORD, t0, s0, t1, nstep)
    
    #x, y, z
    plt.figure(num=1,figsize=(10,15),dpi=300,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(s[0], s[1], s[2], '#4B9CD3') # plotting x, y, z
    ax.set_xlabel('x',fontsize=18)
    ax.set_ylabel('y',fontsize=18)
    ax.set_zlabel('z',fontsize=18)
    ax.view_init(azim=60, elev=40)
    plt.show()
    
    #u, v, w
    plt.figure(num=1,figsize=(10,15),dpi=300,facecolor='white')
    ax = plt.axes(projection='3d')
    ax.plot3D(s[3], s[4], s[5], '#4B9CD3') # plotting x, y, z
    ax.set_xlabel('u',fontsize=18)
    ax.set_ylabel('v',fontsize=18)
    ax.set_zlabel('w',fontsize=18)
    ax.view_init(azim=60, elev=40)
    plt.show()
    
    # x and u against time
    plt.figure(num=1,figsize=(20,10),dpi=100,facecolor='white')
    plt.subplot(511)
    plt.plot(t,s[0],'black', label = 'x (sender)')
    plt.plot(t,s[3],'red', label = 'u (reciever)')
    plt.ylabel('x and u',fontsize=18)
    
    # |u-x| vs. time
    plt.subplot(512)
    plt.plot(t,np.abs(s[3]-s[0]), 'blue')
    plt.ylabel('|u-x|',fontsize=18)
    
    # |v-y| vs. time
    plt.subplot(513)
    plt.plot(t,np.abs(s[4]-s[1]), 'blue')
    plt.ylabel('|v-y|',fontsize=18)
    
    # |w-z| vs. time
    plt.subplot(514)
    plt.plot(t,np.abs(s[5]-s[2]), 'blue')
    plt.ylabel('|w-z|',fontsize=18)
    
    # time versus the sound signal (change to binary signal)
    sound_signal1 = 0.5 * np.sin(2 * np.pi * t / 0.2)
    plt.subplot(515)
    plt.plot(t,sound_signal1, 'black')
    plt.ylabel('s(t)',fontsize=18)
    plt.xlabel('t',fontsize=18)
    plt.legend()
    plt.show()
    
    return t, s, it

main()
