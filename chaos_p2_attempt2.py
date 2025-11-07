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
A_SIG  = 5.0
TB_SIG = None
BITS   = None   # will hold the bit sequence once
N_BITS = None


# Constants
sig = 10.0
b = 8/3
r = 166.83

def generate_random_binary_sequence(n):
    return [np.random.randint(low=0, high=2) for _ in range(n)]

def generate_song_signal():

    with wave.open('Vintage-1960s-drum-groove-loop-88-bpm.wav', "rb") as w:
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
    global BITS, N_BITS, TB_SIG, A_SIG
    
    t = float(np.atleast_1d(t)[0])

     # --- robust bit index from time ---
    if N_BITS is None or N_BITS == 0 or not np.isfinite(TB_SIG) or TB_SIG <= 0:
        current_bit = 0.0
    else:
        # clamp t to >= 0, use a floor index, then wrap into [0, N_BITS-1]
        idx = int(np.floor(max(t, 0.0) / TB_SIG)) % N_BITS
        current_bit = float(BITS[idx])   # scalar 0.0 or 1.0

    sound_signal1 = A_SIG * current_bit

    x, y, z, u, v, w = s
    out = np.zeros(6)
    X_t = x + sound_signal1
    out[0] = sig*(y - x)
    out[1] = r*x - y - x*z
    out[2] = x*y - b*z
    out[3] = sig*(v - u)
    out[4] = r*X_t - v - X_t*w
    out[5] = X_t*v - b*w
    return out

def ode_init():
          
    fRHS    = dsdt   
    fINT    = odeint.ode_ivp   
    fORD    = step.rk45bare                  
    return fINT,fORD,fRHS

def main():
    global dt, sound_data, BITS, N_BITS, TB_SIG
    BITS, fs = generate_song_signal()
    N_BITS = len(BITS)
    
    # Choose how many WAV samples per bit. Pick k so you get ~10 ODE steps per bit
    TB_SIG = 1.0 / fs                           # <<< single source of truth for bit timing
    
    # Simulate exactly M bits (<= N_BITS) so time domains match nicely
    M = min(N_BITS, N_BITS)                   # pick how many bits you want to use
    t1 = M * TB_SIG
    
    nstep = 10000
    t0 = 0.0
    x0 = 10.0
    y0 = 10.0
    z0 = 10.0
    u0 = 1.0
    v0 = 1.0
    w0 = 1.0
    
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
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.show()
    
    return t, s, it

main()