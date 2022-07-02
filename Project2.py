import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
# from freqz_plot import *

########################+++++++++++++++++++build of filters+++++++++++++++++++++++#############################
SB=8
N=32  #length of LP
n=np.arange(N)
w=1/SB
ideallowpass=np.sin(w*(np.pi)*(n-(N-1)/2))/(np.pi*(n-(N-1)/2))
omega,H=sp.freqz(ideallowpass)

def windowmodel(name='Rect',L=32):
    if name == 'Sine':
        window = np.sin(np.pi*(n+0.5)/L)
    elif name == 'Hanning':
        window = 0.5-0.5*np.cos(2*np.pi*(n+0.5)/N)
    elif name == 'Rect':
        window = np.ones(L)
    elif name == 'Kaiser':
        window = np.kaiser(L,8)
    elif name == 'Vorbis':
        window = np.sin(np.pi/2*np.sin(np.pi/L*(n+0.5))**2)
    return window

plt.figure()

rectangular= windowmodel('Rect') * ideallowpass
omega, rect=sp.freqz(rectangular)
plt.plot(omega, 20 * np.log10(np.abs(rect)),label='rectangular')

Vorbis=windowmodel('Vorbis')*ideallowpass
omega, V=sp.freqz(Vorbis)
plt.plot(omega, 20 * np.log10(np.abs(V)),label='Vorbis')

Sine=windowmodel('Sine')*ideallowpass
omega, sin=sp.freqz(Sine)
plt.plot(omega, 20 * np.log10(np.abs(sin)),label='Sine')

Hanning=windowmodel('Hanning')*ideallowpass
omega, hann=sp.freqz(Hanning)
plt.plot(omega, 20 * np.log10(np.abs(hann)),label='Hanning')

Kaiser=windowmodel('Kaiser')*ideallowpass
omega, K=sp.freqz(Kaiser)
plt.plot(omega, 20 * np.log10(np.abs(K)),label='Kaiser')


plt.legend()
plt.xlabel('Normalized frequency')
plt.ylabel('Magnitude of Frequency Response in dB')
plt.title('Magnitude of Frequency Response for different window methods on ideal LPF')
plt.show()


####Filter Design with the Window Method
transition=0.1
stopwidth_result=1/SB+transition
# Theresulting stopband starts at the stopband frequency of the
#ideal frequency response (cutoff frequency) plus the
#frequency of the start of the stopband of the window
#function (adding the transition band).

L=32 ##length of window
beta=8  ##beta of kaiser window
nf=np.arange(L)
h_kaiser=np.kaiser(L,beta)

#we can see bandwidth is 0.13 of kaiser window
stopwidth_kaiser=0.17
stopwidth_ideal=stopwidth_result-stopwidth_kaiser
wstop=stopwidth_ideal*np.pi  #wstop is the cutoff of our sinc funktion

#print(stopwidth_ideal)
h_desired=np.sin(wstop*(nf-(L-1)/2))/(np.pi*(nf-(L-1)/2))
hfilter=h_desired*h_kaiser  #mutiple the window and ideal LP to get result LP

filters = np.zeros((L,SB))
filters[:,0] = hfilter

##modulated to highpass
hhigh=hfilter*np.cos(np.pi*nf)
filters[:,SB-1] = hhigh

###modulated to bandpass,need find a new L to let the bandwidth same as LP
L_band=32        #length of the bandwidth is half as LP because it will be doubel when shifted it
n_band=np.arange(L_band)
h_kaiserb=np.kaiser(L_band,beta)
##we can see the cutoff of L=128 is 0.13
transitionb=0.045
stopwidth_resultb=1/(SB*2)+transitionb
stopwidth_kaiserb=0.13/np.pi
stopwidth_idealb=stopwidth_resultb-stopwidth_kaiserb
wstopb=stopwidth_idealb*np.pi
#print(stopwidth_idealb)
h_desiredb=np.sin(wstopb*(n_band-(L_band-1)/2))/(np.pi*(n_band-(L_band-1)/2))
hfilterb=h_desiredb*h_kaiserb  ##all thos are same as LP just change the length and set a new transition band

for i in range(1,SB-1):
    filters[:,i] = hfilterb*np.cos(np.pi*i/7*n_band)*2/3*np.pi

fig1 = plt.figure(1)
for i in range(SB):
    [freq, response] = sp.freqz(filters[:, i])
    plt.plot(freq, 20*np.log10(np.abs(response)+1e-6))
plt.xlabel('Normalized frequency')
plt.ylabel('Magnitude of Frequency Response in dB')
plt.title('Magnitude of Frequency Response for 8-band filters built with Kaiser window method')
plt.show()

#####################################+++++++++++++++++++++++++++process of audio++++++++++++++++++++++################################

from sound import *

audio, fs = wavread('Track32.wav')
# sound(audio, fs)

ch1 = audio[:, 0]
# print('playing original channel 1 ...')
# sound(ch1, fs)

# pad ch1 with 0s in the end until can be divided by SB for downsampling
ch1 = np.pad(ch1, (0, len(ch1)//SB*SB+SB-len(ch1)), 'constant', constant_values = (0, 0))

subbands = np.zeros((len(ch1), SB))
print(len(ch1))
for i in range(SB):
    subbands[:, i] = sp.lfilter(filters[:, i], 1, ch1)

# print('playing 1st subband ...')
# sound(subbands[:, 0], fs)

######################----------downsampling method1--------------############################
subbands_ds = np.zeros((len(ch1)//SB, SB))

for i in range(SB):
    subbands_ds[:, i] = subbands[::SB, i]

# print('playing downsampled 1st subband ...')
# sound(subbands_ds[:, 0], fs)

######################----------upsampling--------------############################
subbands_up = np.zeros((len(ch1), SB))

for i in range(SB):
    subbands_up[::SB, i] = subbands_ds[:, i]

######################----------filter after upsampling--------------############################
subbands_re = np.zeros((len(ch1), SB))

for i in range(SB):
    subbands_re[:, i] = sp.lfilter(filters[:, i], 1, subbands_up[:, i])

recon = np.zeros(len(ch1))

for i in range(SB):
    recon += subbands_re[:, i]

# print('playing reconstructed sound ...')
# sound(recon, fs)

fig3=plt.figure(3)
plt.plot(recon/np.amax(recon)*np.amax(ch1), label = 'reconstructed')
plt.plot(ch1, label = 'original ch1')
plt.legend()
plt.title('Original signal vs. reconstructed signal after normalising')
plt.xlabel('samples')
plt.ylabel('values')
plt.show()