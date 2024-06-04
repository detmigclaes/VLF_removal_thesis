import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_data(n, p):
    # n: number of data points
    # p: probability of getting 1
    data = np.random.choice([0, 1], size=n, p=[1-p, p])
    return data

""" def phase_function(data, T, fs, offset=0):
    #t = np.arange(0, (len(data))*T, 1/fs)
    t = np.linspace(0, (len(data))*T, int(len(data)*T*fs), endpoint=False) 
    data_nrz = data.copy()
    data_nrz[data_nrz == 0] = -1
    phase = np.zeros_like(t)
    for n in range(len(data_nrz)):
        t1 = int(n*T*fs)
        t2 = int((n+1)*T*fs)+1
        phase[t1:t2] = phase[t1] + data_nrz[n]*np.pi/(2*T)*(t[t1:t2]-n*T)
    phase = np.mod(phase, 2*np.pi)
    return phase, t """

def phase_function(data, t, T, fs):
    # data: input data
    # t: time vector
    # T: symbol duration
    # fs: sampling frequency

    data_nrz = data.copy()
    data_nrz[data_nrz == 0] = -1
    phase = np.zeros_like(t)
    for n in range(len(data_nrz)):
        t1 = int(n*T*fs)
        t2 = int((n+1)*T*fs)+1
        phase[t1:t2] = phase[t1] + data_nrz[n]*np.pi/(2*T)*(t[t1:t2]-n*T)
    
    phase = np.mod(phase, 2*np.pi)
    
    return phase 

def generate_msk(N, p, T, fc, fs, offset=0, phase_offset=0, amplitude=1):
    # N: number of data points
    # p: probability of getting 1
    # T: symbol duration
    # fc: carrier frequency
    # fs: sampling frequency
    # offset: offset measured in seconds
    # phase_offset: phase offset

    padding =  np.random.randint(2,5)
    data = generate_data(N +padding, p) # +padding to avoid errors in the phase function
    #data = generate_data(N, p) 
    t = np.linspace(0, (len(data))*T, int(len(data)*T*fs), endpoint=False)
    phase = phase_function(data, t, T, fs)
    phase = np.roll(phase, int(fs*offset))


    phase = phase[(padding-1)*int(fs*T):-int(fs*T)]
    data = data[padding-1:-1]
   
    t = np.linspace(0, (len(data))*T, int((len(data))*T*fs), endpoint=False)

    s = amplitude * np.cos(2*np.pi*fc*(t-offset) + phase + phase_offset)
  
    return s, t, data, phase

def generate_msk_data(data1, T, fc, fs, offset=0, phase_offset=0, amplitude=1):
    # data1: input data
    # T: symbol duration
    # fc: carrier frequency
    # fs: sampling frequency
    # offset: offset measured in seconds
    # phase_offset: phase offset

    data = data1.copy()

    data_extended = np.concatenate((np.ones(1), np.zeros(1),data, np.zeros(1), np.ones(1)))
 
    t = np.linspace(0, (len(data_extended))*T, int(len(data_extended)*T*fs), endpoint=False)
    phase= phase_function(data_extended, t, T, fs)
    phase = np.roll(phase, int(fs*offset))

  
    phase = phase[2*int(fs*T):-2*int(fs*T)]
    t = np.linspace(0, (len(data))*T, int(len(data)*T*fs), endpoint=False)

    s = amplitude * np.cos(2*np.pi*fc*(t-offset) + phase + phase_offset)

    return s, t, phase 

def amplitude_estimation(r):
    # r: received signal
    # N: number of samples per symbol
    # L0: number of symbols

    a_hat = np.mean(np.abs(r.real))
    a_hat *= np.pi/2 
    a_hat *= 2 # because of the half after downconversion

    #print(a_hat, np.mean(np.abs(r.imag)*np.pi))
    a_hat = (a_hat + np.mean(np.abs(r.imag)*np.pi))/2

    return a_hat

def saw(x,T):
    # x : input signal
    # T : upper and lower limits of the function
    return ((x+T/2) % (2*T/2))- T/2

def phase_and_time_sync(r,N,T,fs,L0):
    # r : baseband input signal
    # T : symbol period
    # fs : sampling frequency
    # N : number of samples per symbol
    # L0 : number of symbols
    
    
    q = np.linspace(0, 0.5, int(fs*T), endpoint=False)
    q = np.concatenate((q, np.array([0.5]), np.flip(q)))
    h0 = np.sin(np.pi*q)
    x = signal.convolve(r, h0,'valid')**2 # atomatically chooses the fastest method, signal for speed
    lambda_2 = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(0,L0-2):
            lambda_2[k] += (-1)**n*x[int(N*n+k)]
    lambda_2 = lambda_2/L0

    k = np.arange(N)
    tau_hat = np.sum((np.abs(lambda_2[k]**2)*np.exp(-1j*2*np.pi*k/N)))
    tau_hat = -T/(2*np.pi)*np.angle(tau_hat)

    
    #k_hat = np.argmin(np.abs(saw(tau_hat-k*T/N,T)))

    theta_hat = 0.5*(np.angle(lambda_2[int(tau_hat*fs)])) # simpler solution 
    #theta_hat = 0.5*np.angle(lambda_2[k_hat])
 
    return r, x, tau_hat, theta_hat


def viterbi_metric(r, symbols, new_symbol, phase_start):
    # r : received signal
    # symbols : symbols to test
    # phase_start : phase of the signal at beginning of sequence

    phase = 0.5*np.pi*np.sum(symbols) + np.pi*new_symbol*np.linspace(0, 0.5, len(r), endpoint=False)+phase_start
    
    return np.sum(np.real(np.exp(-1j*phase)*r))

def viterbi_decoder(s, N):
    # s : received signal
    # T : symbol period
    # N : number of symbols

    # initial phase and symbols
    p0s = np.array([0,.5*np.pi,np.pi,1.5*np.pi]) 
    p0_max = 0
    for p in p0s:
        metric1 = viterbi_metric(s[:N], [], np.array([-1]), p)
        metric2 = viterbi_metric(s[:N], [], np.array([1]), p)
        if max(metric1, metric2) > p0_max:
            p0_max = max(metric1, metric2)
            phase_start = p

    path1 = np.array([-1])
    path2 = np.array([1])
    #pm1 = viterbi_metric(s[:N], [], path1, p) # check if the phase is correct
    #pm2 = viterbi_metric(s[:N], [], path2, p)
    pm1 = viterbi_metric(s[:N], [], path1, phase_start) 
    pm2 = viterbi_metric(s[:N], [], path2, phase_start)
    
    for i in range(1,len(s)//N):

        pm1_0 = pm1 + viterbi_metric(s[i*N:(i+1)*N], path1,-1, phase_start)
        pm1_1 = pm1 + viterbi_metric(s[i*N:(i+1)*N], path1, 1, phase_start)
        pm2_0 = pm2 + viterbi_metric(s[i*N:(i+1)*N], path2,-1, phase_start)
        pm2_1 = pm2 + viterbi_metric(s[i*N:(i+1)*N], path2, 1, phase_start)
        
        
        if (pm1_0>pm2_0):
            path1_new = np.concatenate((path1,np.array([-1])))
        else:
            path1_new = np.concatenate((path2,np.array([-1])))

        if (pm1_1>pm2_1):
            path2_new = np.concatenate((path1,np.array([1])))
        else:
            path2_new = np.concatenate((path2,np.array([1])))

        
        pm1 = max(pm1_0, pm2_0)
        pm2 = max(pm1_1, pm2_1)

        (path1,path2) = (path1_new,path2_new)

    path1[path1 == -1] = 0
    path2[path2 == -1] = 0
    if (pm1 > pm2):
        return path1, phase_start
    else:
        return path2, phase_start
    
def add_zeros(data, zero_time, interval_time, fs):
    # Add zero intervals to the data
    # data: data to add zeros to
    # zero_time: length of zeros in seconds
    # interval_time: interval between zeros in seconds
    # fs: sampling frequency

    data_zeros = np.copy(data)

    for i in range(int(len(data)/int(interval_time*fs))):
        T1 = i*int(interval_time*fs)
        T2 = T1 + int(zero_time*fs)
        data_zeros[T1:T2] = 0

    return data_zeros