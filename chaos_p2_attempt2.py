#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:57:49 2025

@author: masonmiller
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import ode_integrators as odeint
import ode_step as step
import wave

dt = 0.001
A_SIG  = 0.5
TB_SIG = None
BITS   = None   # will hold the bit sequence once
N_BITS = None
INPUT_MODE = "rand"   # "wav" or "rand"


# Constants
sig = 10.0
b = 8/3
r = 20

def generate_random_binary_sequence(n):
    return [np.random.randint(low=0, high=2) for _ in range(n)], 0.01

def generate_song_signal():

    with wave.open('Classicals.de - Satie - Gymnopedie No. 1.wav', "rb") as w:
        n_channels   = w.getnchannels()
        sample_width = w.getsampwidth()
        frame_rate   = w.getframerate()
        n_frames     = w.getnframes()
        raw_data     = w.readframes(n_frames)

    # Decode raw audio bytes
    dtype = np.int16 if sample_width == 2 else np.uint8
    data = np.frombuffer(raw_data, dtype=dtype)

    # If stereo, average to mono
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    data = data.astype(np.float32) / np.iinfo(dtype).max
    
    return data, frame_rate

try1 = generate_random_binary_sequence(10)
# ========================================
# s = array containing x(t), y(t), z(t), u(t), v(t), w(t)
# r is variable and defined in main
# returns and array containing time derivatives of x,y,z,u,v,w
# x,y,z is sender ; u,v,w is reciever
# u,v,w will equal x,y,z 
# ----------------------------------------

def dsdt(t, s, dt):
    global BITS, N_BITS, TB_SIG, A_SIG, i
    
    t = float(np.atleast_1d(t)[0])

     # --- robust bit index from time ---
    if N_BITS is None or N_BITS == 0 or not np.isfinite(TB_SIG) or TB_SIG <= 0:
        current_bit = 0.0
    else:
        # clamp t to >= 0, use a floor index, then wrap into [0, N_BITS-1]
        idx = int(np.floor(max(t, 0.0) / TB_SIG)) % N_BITS
        current_bit = float(BITS[idx])   # scalar 0.0 or 1.0
        #current_bit = BITS[i]

    sound_signal1 = A_SIG * current_bit

    x, y, z, u, v, w = s
    dsdt = np.zeros(6)
    X_t = x + sound_signal1
    dsdt[0] = sig*(y - x)
    dsdt[1] = r*x - y - x*z
    dsdt[2] = x*y - b*z
    dsdt[3] = sig*(v - u)
    dsdt[4] = r*X_t - v - X_t*w
    dsdt[5] = X_t*v - b*w
    
    return dsdt


def ode_init():
    fRHS    = dsdt   
    fINT    = odeint.ode_ivp   
    fORD    = step.rk45bare               
    return fINT,fORD,fRHS


def setup_input(mode, dt):
    global BITS, N_BITS, TB_SIG

    if mode == "wav":
        BITS, fs = generate_song_signal()     # BITS is your WAV sample stream (float array)
        N_BITS   = len(BITS)
        TB_SIG   = 1.0 / fs                   # one sample per “bit” in sim time
        M        = N_BITS                     # play once
        t1       = M * TB_SIG
        return t1

    elif mode == "rand":
        # Use your existing random function; it returns (list, 3), we take the list
        rand_list, _ = generate_random_binary_sequence(200000)  
        BITS   = np.asarray(rand_list, dtype=np.uint8)
        N_BITS = len(BITS)
        TB_SIG = 1 * dt                     
        M      = N_BITS
        t1     = M * TB_SIG
        return t1

def main():
    global dt, sound_data, BITS, N_BITS, TB_SIG
    
    t1 = setup_input(INPUT_MODE, dt)

    t0 = 0.0
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    u0 = 1.0
    v0 = 1.0
    w0 = 1.0
    
    nstep = int(np.floor((t1 - t0)/dt))
    
    s0 = np.array([x0, y0, z0,u0,v0,w0])
    
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
    plt.legend()
    
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
    
    
    
    A = A_SIG
    Tb = TB_SIG
    
    if N_BITS == 0:
        s_on_t = np.zeros_like(t)
    else:
        idx = ((t // Tb) % N_BITS).astype(int)
        s_on_t = A * BITS[idx]              # original injected message
    
    # Now recovered signal from integrated data:
    recovered = (s[0] + s_on_t) - s[3]      # X = x + s(t), so X - u = s(t)
    
    # --- Plot both ---
    plt.subplot(515)
    plt.plot(t, recovered, 'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.show()
    
    return t, s, it

main()
