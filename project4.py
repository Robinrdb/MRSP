import numpy as np
import matplotlib.pyplot as plt
from sound import *
import warnings
warnings.filterwarnings('ignore')
import scipy.signal as sp
import scipy.optimize as opt

########################## Project 4 ###########################
audio = "Track32.wav";
s,rate = wavread(audio);
# print(len(s))
N = 16 #block length
L = len(s)//N+1 #number of blocks
# print(L)
ch0_pad = np.pad(s[:, 0], (0, L * N - len(s)), 'constant', constant_values=0)
# print(len(ch0_pad))
blocks=np.empty([N,L])
blocks_fft=np.empty([N,L])

################------------time/freq representation with fft---------------##################
for i in range(L):
    blocks[:,i]= ch0_pad[i * N:(i + 1) * N]   ##divide the whole signal to 16 blocks
    blocks_fft[:,i]=np.fft.fft(blocks[:,i])   ##FFT to each block
# print(blocks.shape)
# print(blocks_fft.shape)

plt.figure()
for i in range(N):
    plt.subplot(4,4,i+1)
    # [freq,resp]=sp.freqz(blocks_fft[i,:])
    # plt.plot(freq/(2*np.pi), 20*np.log10(np.abs(resp)+1e-6))
    plt.plot(blocks_fft[i,:])
plt.suptitle('a time/frequency representation with fft')
plt.show()
print("time index is",L)
print("frequency index is",N)
############------------Frequency response of each equivalent FFT filter----------------####################
plt.figure()
T=np.fft.fft(np.eye(N,dtype=int))
for i in range(N):
    freq,resp=sp.freqz(np.flipud(T[:,i]),whole=True)
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.axis([0,2*np.pi,-5,25])
plt.title('freq response of equivalent FFT filters')
plt.xlabel('freq')
plt.ylabel('dB')
plt.show()

#############---------------reconstruct--------------###########
blocks_ifft=np.empty([N,L])
rec=np.empty(N*L)
for i in range(L):
    blocks_ifft[:,i]=np.fft.ifft(blocks_fft[:,i])
    rec[i*N:(i+1)*N]=blocks_ifft[:,i]

plt.figure()
plt.plot(ch0_pad)
plt.plot(rec)
plt.legend(labels=['original','reconstructed'])
plt.title('original vs, reconstructed signal')
plt.xlabel('samples')
plt.ylabel('value')
plt.show()



################################# Homework 3 ######################################
SB=8
N=32
n=np.arange(N)

rand_window = np.random.rand(N)

wstop=0.17
h_desired=np.sin(wstop*(n-(N-1)/2))/(np.pi*(n-(N-1)/2))
hfilter=h_desired*rand_window

wstopb=0.17
h_desiredb=np.sin(wstopb*(n-(N-1)/2))/(np.pi*(n-(N-1)/2))
hfilterb=h_desiredb*rand_window


def functionerror(hi):
   samples = 1024
   passband = int(samples/5)
   transitionband = int(samples/1024)
   w, h = sp.freqz(hi, 1, samples)
   h_ideal = np.concatenate((np.ones(passband), np.zeros(samples - passband)))
   weight = np.concatenate((np.ones(passband), np.zeros(transitionband), 1000 * np.ones(samples-passband-transitionband)))
   err = np.sum(np.abs(h - h_ideal)*weight)
   return err

minout = opt.minimize(functionerror, hfilter)
opt_filter=minout.x

bandminout = opt.minimize(functionerror, hfilterb)
opt_bandfilter=minout.x


opt_filters = np.zeros((N,SB))
opt_filters[:,0] = opt_filter

opt_high=opt_filter*np.cos(np.pi*n)
opt_filters[:,SB-1] = opt_high

for i in range(1,SB-1):
    opt_filters[:,i] = opt_bandfilter*np.cos(np.pi*i/7*n)*2/3*np.pi

filters = np.zeros((N,SB))
filters[:,0] = hfilter

hhigh=hfilter*np.cos(np.pi*n)
filters[:,SB-1] = hhigh

for i in range(1,SB-1):
    filters[:,i] = hfilterb*np.cos(np.pi*i/7*n)*np.pi*2.0/3.0


####plot the difference between PR3 and PR4
fig3 = plt.figure()
plt.subplot(2,1,1)
for i in range(SB):
    [freq, response] = sp.freqz(opt_filters[:, i])
    plt.plot(freq, 20*np.log10(np.abs(response)+1e-6))
plt.xlabel('Normalized frequency')
plt.ylabel('Magnitude of Frequency Response in dB')
plt.title('Magnitude of Frequency Response for 8-band filters built with optimized vs. random window method')
plt.subplot(2,1,2)
T=np.fft.fft(np.eye(16,dtype=int))
for i in range(16):
    freq,resp=sp.freqz(np.flipud(T[:,i]),whole=True)
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.axis([0,2*np.pi,-5,25])
plt.title('freq response of equivalent FFT filters')
plt.xlabel('freq')
plt.ylabel('dB')
plt.show()