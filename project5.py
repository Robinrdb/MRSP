import math
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import os

####################################
#####----- task1 setting 1 -----#####
####################################

######################----------build of DCT4 filter banks--------------############################
K = 8
N = 8
T = np.zeros((N, K))

for n in range(N):
    for k in range(K):
        T[n, k] = np.sqrt(2 / N) * np.cos((np.pi / N) * (n + 0.5) * (k + 0.5))

################ DCT4 the equivalent impulse responses for the analysis #############
plt.figure()
for k in range(K):
    freq,resp=sp.freqz(np.flipud(T[:,k]))   ###time-inversed the colums of DCT4 function to get equivalent analysis
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.title('equivalent impulse responses for the analysis')
plt.xlabel('freq')
plt.ylabel('dB')
# plt.show()

################ DCT4 the equivalent impulse responses for the synthesis #############
plt.figure()
for n in range(N):
    freq,resp=sp.freqz(T[n,:]) #####identical to colums of DCT4 Function
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.title('equivalent impulse responses for the synthesis')
plt.xlabel('freq')
plt.ylabel('dB')
# plt.show()

############# Test this filter banks with ramp function ###########
Fs = 8
f1 = 5
timePoints = np.linspace(0, 1, Fs)
ramp = 5*sp.sawtooth(2 * np.pi * f1 * timePoints)
ramp_analysis=np.dot(ramp,T)
remp_synthesis=np.dot(ramp_analysis,T)

fig=plt.figure()
plt.plot(ramp, label = 'original ramp')
plt.plot(remp_synthesis/np.amax(remp_synthesis)*np.amax(ramp), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal after normalising')
plt.xlabel('samples')
plt.ylabel('values')
# plt.show()

######################----------apply filter banks to audio--------------############################
from sound import *

audio, fs = wavread('Track32.wav')
# sound(audio, fs)

ch1 = audio[:, 0]
# print('playing original channel 1 ...')
# sound(ch1, fs)

# pad ch1 with 0s in the end until can be divided by SB for downsampling
ch1 = np.pad(ch1, (0, len(ch1)//K*K+K-len(ch1)), 'constant', constant_values = (0, 0))

L = len(ch1)//K
subbands = np.zeros((len(ch1), K))
# subbands = np.zeros((L, K))
                                 ############# anaysis ##############
for i in range(K):
    subbands[:, i] = sp.lfilter(T[:, i], 1, ch1)
    # subbands[:,i] = ch1[L*i:L*(i+1)]

# subbands = np.dot(subbands,T)
# print('playing 1st subband ...')
# sound(subbands[:, 0], fs)

subbands_ds = np.zeros((len(ch1)//K, K))
# subbands_ds = np.zeros((L//K+1, K))

for i in range(K):
    subbands_ds[:, i] = subbands[::K, i]

# print('playing downsampled 1st subband ...')
# sound(subbands_ds[:, 0], fs)

subbands_up = np.zeros((len(ch1), K))
# subbands_up = np.zeros((L, K))

for i in range(K):
    subbands_up[::K, i] = subbands_ds[:, i]

         ######################--------- synthesis--------------############################
subbands_re = np.zeros((len(ch1), K))
# subbands_re = np.dot(subbands_up,T)
# subbands_re = np.dot(subbands,T)

for i in range(K):
    subbands_re[:, i] = sp.lfilter(T[:, i], 1, subbands_up[:, i])

recon = np.zeros(len(ch1))

for i in range(K):
    recon += subbands_re[:, i]
    # recon[L*i:L*(i+1)] = subbands_re[:,i]
wavwrite(recon,fs,'reconstructed.wav')
# print('playing reconstructed sound ...')
# sound(recon, fs)

fig=plt.figure()
plt.plot(ch1, label = 'original ch1')
plt.plot(recon/np.amax(recon)*np.amax(ch1), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal in DCT4 setting1')
plt.xlabel('samples')
plt.ylabel('values')
# plt.show()

####################################
#####----- task1 setting 2 -----#####
####################################

##########keep only the first two subbands, set the others to zero#######
subbands_ex = np.zeros(subbands.shape)
for i in range(2):
    subbands_ex[:,i] = subbands[:,i]
    # subbands_ex[:,i] = sp.lfilter(T_ex[:, i], 1, ch1)
# subbands_ex = np.dot(subbands_ex,T)

subbands_ex_ds = np.zeros(subbands_ds.shape)

for i in range(K):
    subbands_ex_ds[:, i] = subbands_ex[::K, i]

subbands_ex_up = np.zeros(subbands_up.shape)

for i in range(K):
    subbands_ex_up[::K, i] = subbands_ex_ds[:, i]

# subbands_ex_re = np.dot(subbands_ex_up, T)

subbands_ex_re = np.zeros((len(ch1), K))
for i in range(K):
    subbands_ex_re[:, i] = sp.lfilter(T[:, i], 1, subbands_ex_up[:, i])
    # subbands_ex_re[:, i] = sp.lfilter(T_ex[:, i], 1, subbands_ex_up[:, i])

recon_ex = np.zeros(len(ch1))

for i in range(K):
    recon_ex += subbands_ex_re[:, i]
    # recon_ex[L*i:L*(i+1)] = subbands_ex_re[:,i]
wavwrite(recon_ex,fs,'reconstructed extracted.wav')
# print('playing reconstructed sound ...')
# sound(recon_ex, fs)

fig=plt.figure()
plt.plot(ch1, label = 'original ch1')
plt.plot(recon_ex/np.amax(recon_ex)*np.amax(ch1), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal in DCT4 setting2')
plt.xlabel('samples')
plt.ylabel('values')
# plt.show()
##########################Save them and compare how much is the compression ratio########
size0 = os.path.getsize('Track32.wav')
size1 = os.path.getsize('reconstructed.wav')
size2 = os.path.getsize('reconstructed extracted.wav')
print('compression ratio 1 in setting1: ',size1/size0)
print('compression ratio 2 in setting2: ',size2/size0)
########################## 0 also take spaces #######################
####################################
#####----- task2 -----#####
####################################

L=16
############### sine window or baseband prototype ##################
h=np.sin(np.pi/L*(np.arange(L)+0.5))

############### MDCT impluse responses for the analysis ################
hk=np.empty([L,N])

for k in range(N):
    hk[:,k]=math.sqrt(2/N)*np.cos(np.pi/N*(np.arange(L)+0.5+N/2)*(k+0.5))*h

plt.figure()
for n in range(N):
    freq,resp=sp.freqz(hk[:,n])
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.axis([0, np.pi, -70, 15])
plt.title('MDCT impluse responses for the analysis')
plt.xlabel('freq')
plt.ylabel('dB')
# plt.show()
############### MDCT impluse responses for the synthesis ###############
gk=np.empty([L,N])

for k in range(N):
    gk[:,k]=math.sqrt(2/N)*np.cos(np.pi/N*(np.arange(L)+0.5-N/2)*(k+0.5))*h
plt.figure()
for n in range(N):
    freq,resp=sp.freqz(gk[:,n])
    plt.plot(freq,20*np.log10(np.abs(resp)+1e-6))
plt.axis([0, np.pi, -70, 15])
plt.title('MDCT impluse responses for the synthesis')
plt.xlabel('freq')
plt.ylabel('dB')
# plt.show()

##############__________________setting 1______________##############
##########__________ramp____________###########
# print(gk.shape)
Fs = 16*8
f1 = 5
timePoints = np.linspace(0, 1, Fs)
ramp = 5*sp.sawtooth(2 * np.pi * f1 * timePoints)
# ramp = ramp.reshape((16,8))
# print(ramp.shape)
# ramp_analysis=np.dot(hk,ramp)
# remp_synthesis=np.dot(ramp_analysis,gk)
ramp_analysis=np.zeros((16*8,8))
ramp_synthesis=np.zeros((16*8,8))
for i in range(8):
    ramp_analysis[:,i]=sp.lfilter(hk[:, i],1,ramp)
for i in range(8):
    ramp_synthesis[:,i]=sp.lfilter(gk[:, i],1,ramp_analysis[:,i])
ramp_rec=np.zeros(16*8)
for i in range(8):
    ramp_rec+=ramp_synthesis[:,i]

fig=plt.figure()
plt.plot(np.flipud(ramp), label = 'original ramp')
plt.plot(ramp_rec/np.amax(ramp_rec)*np.amax(ramp), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal in MDCT ramp')
plt.xlabel('samples')
plt.ylabel('values')
# plt.show()

######_______audio________########
subbands_mdct = np.zeros((len(ch1), K))
for i in range(K):
    subbands_mdct[:, i] = sp.lfilter(hk[:, i], 1, ch1)

subbands_mdct_ds = np.zeros((len(ch1)//K, K))

for i in range(K):
    subbands_mdct_ds[:, i] = subbands_mdct[::K, i]

subbands_mdct_up = np.zeros((len(ch1), K))

for i in range(K):
    subbands_mdct_up[::K, i] = subbands_mdct_ds[:, i]

subbands_mdct_re = np.zeros((len(ch1), K))

for i in range(K):
    subbands_mdct_re[:, i] = sp.lfilter(gk[:, i], 1, subbands_mdct_up[:, i])

recon_mdct = np.zeros(len(ch1))

for i in range(K):
    recon_mdct += subbands_mdct_re[:, i]

wavwrite(recon_mdct,fs,'reconstructed_mdct.wav')
# print('playing reconstructed sound ...')
# sound(recon_mdct, fs)

fig=plt.figure()
plt.plot(ch1, label = 'original ch1')
plt.plot(recon_mdct/np.amax(recon_mdct)*np.amax(ch1), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal in MDCT setting1')
plt.xlabel('samples')
plt.ylabel('values')
# plt.show()

#############________Task2_setting 2__________##########
subbands_ex_mdct = np.zeros(subbands_mdct.shape)
for i in range(2):
    subbands_ex_mdct[:,i] = subbands_mdct[:,i]

subbands_ex_mdct_ds = np.zeros(subbands_ds.shape)

for i in range(K):
    subbands_ex_mdct_ds[:, i] = subbands_ex_mdct[::K, i]

subbands_ex_mdct_up = np.zeros(subbands_up.shape)

for i in range(K):
    subbands_ex_mdct_up[::K, i] = subbands_ex_mdct_ds[:, i]

subbands_ex_mdct_re = np.zeros((len(ch1), K))
for i in range(K):
    subbands_ex_mdct_re[:, i] = sp.lfilter(gk[:, i], 1, subbands_ex_mdct_up[:, i])

recon_ex_mdct = np.zeros(len(ch1))

for i in range(K):
    recon_ex_mdct += subbands_ex_mdct_re[:, i]
wavwrite(recon_ex_mdct,fs,'reconstructed extracted_mdct.wav')
# print('playing reconstructed sound ...')
# sound(recon_ex, fs)

#size0 = os.path.getsize('Track32.wav')
size3 = os.path.getsize('reconstructed_mdct.wav')
size4 = os.path.getsize('reconstructed extracted_mdct.wav')
print('compression ratio 3 in MDCT setting1: ',size3/size0)
print('compression ratio 4 in MDCT setting2: ',size4/size0)

fig=plt.figure()
plt.plot(ch1, label = 'original ch1')
plt.plot(recon_ex/np.amax(recon_ex)*np.amax(ch1), label = 'reconstructed')
plt.legend()
plt.title('Original signal vs. reconstructed signal in MDCT Setting2')
plt.xlabel('samples')
plt.ylabel('values')
plt.show()
