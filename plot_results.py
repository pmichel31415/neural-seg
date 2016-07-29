from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread


def load(fname):
    ret = np.loadtxt(fname, delimiter=',').T
    ret = ret[:, np.argsort(ret[0])]
    return ret


def plot(x, text, color):
    plt.plot(x[2], x[1], '-o', linewidth=8,
             label=text, color=color, markersize=12)
    # for i, txt in enumerate(x[0]):
    #     plt.annotate(txt,xy=(x[2,i]+0.005,x[1,i]+0.005),fontsize=18,color=color)


def diplay_baseline(x, y, label):
    plt.plot(x, y, 'o', color='purple', markersize=12,label=label)
    # plt.annotate(
    #     label,
    #     xy=(x, y), xytext=(10, 40),
    #     textcoords='offset points', ha='center', va='top',
    #     bbox=dict(boxstyle='round,pad=0.2', fc='orange', alpha=0.5),
    #     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
    #     fontsize=15
    # )


def plot_boundaries(x, c, phn):
    low=min(0,np.min(x))
    up=np.max(x)
    for bound in c:
        plt.plot([bound, bound], [low, up],
                 '--', color='#d62d20', linewidth=3)
    plt.xticks((c[:-1]+c[1:])/2.0, phn, size=22)
    plt.yticks(size=22)

ns=''

res_kmeans = load('results/results_crossent_'+ns+'tot.txt')
res_mse = load('results/results_mse_'+ns+'tot.txt')
res_st = load('results/results_markov_'+ns+'tot.txt')
res_per = load('results/results_periodic_tot.txt')
t = np.arange(0.05, 1, 0.005)
for a in np.arange(0.1, 0.99, 0.01):
    plt.plot(t, a*t/(2*t-a+0.000000000001), '--', color='grey', linewidth=0.5)
    t = t[1:]

# plt.xlim(0.4, 1)
# plt.ylim(0.3, 0.9)

plt.xlim(0.3, 1)
plt.ylim(0.4, 1)
# P/R plot
print(res_st)
plot(res_st, 'Markov', '#ffa700')
plot(res_mse, 'RNN (continuous)', '#0057e7')
plot(res_kmeans, 'RNN (categorical)', '#008744')
plot(res_per, 'Periodic boundaries', '#d62d20')

# diplay_baseline( 0.910,0.575, 'Boundaries every 5ms')
# diplay_baseline(0.752, 0.668, 'Dusan, Rabiner (2006)')
# diplay_baseline(0.74, 0.7, 'Rasanen (2014)')
# diplay_baseline(0.762, 0.764, 'Lee, Glass (2012)')


plt.legend(fontsize=20)
plt.xlabel('Recall', size=25)
plt.ylabel('Precision', size=25, rotation='vertical')
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid()
plt.show()

if True:

    # Features visualization
    c = (np.loadtxt('../../resources/TIMIT/phn_gold/test_dr3_mcsh0_si2179.syldet')
         * 100).astype(int)/100


    phn = np.loadtxt('../../resources/TIMIT/si2179.phn', dtype=str)[:, -1]
    mfcc = np.load('../../resources/TIMIT/mfcc_npy_13/test_dr3_mcsh0_si2179.npy')
    states = np.load('../../resources/TIMIT/kmeans_8/test_dr3_mcsh0_si2179.npy')
    f, wav = wavread('../../resources/TIMIT/wav/test_dr3_mcsh0_si2179.wav')
    wav = (wav/np.max(np.abs(wav)))#[int(160*10):int(160*120)]
    end = len(wav)/f

    plt.subplot(311)
    plt.plot(np.linspace(0, end, len(wav)), wav, color='#0057e7')
    plot_boundaries(wav, c, phn)
    plt.yticks(size=20)
    # plt.title('Waveform',size=20)
    plt.xlim(0.1, 1.20)
    plt.legend(fontsize=30)

    plt.subplot(312)
    plt.pcolormesh(np.linspace(0, end, len(mfcc)), np.arange(mfcc.shape[1]+1), mfcc.T)
    plot_boundaries(mfcc.shape[1], c, phn)
    plt.yticks(size=20)
    # plt.title('MFCCs',size=20)
    plt.xlim(0.1, 1.20)
    plt.legend(fontsize=30)

    plt.subplot(313)
    plt.pcolormesh(np.linspace(0, end, len(states)), np.arange(states.shape[1]+1), states.T)
    plot_boundaries(states.shape[1], c, phn)
    plt.yticks(size=20)
    # plt.title('States',size=20)
    plt.xlim(0.1, 1.20)
    plt.legend(fontsize=30)

    plt.show()

    # Visual comparison

    y_st = np.load('output/markov/test_dr3_mcsh0_si2179_loss.npy')
    y_old = np.load(
        'output/timit_stateful_biglstm_10_proper_baseline_reset/test_dr3_mcsh0_si2179_loss.npy')
    y = np.load('output/out4/test_dr3_mcsh0_si2179_loss.npy')
    y_st[0] = 0
    y_old[0] = 0
    y[:7] = 0
    y_st[:7] = 0
    y_old[:7] = 0
    c = (np.loadtxt(
        '../../resources/TIMIT/phn_gold/test_dr3_mcsh0_si2179.syldet')*100).astype(int)
    #np.insert(c, 0, 0)
    phn = np.loadtxt('../../resources/TIMIT/si2179.phn', dtype=str)[:, -1]

    plt.subplot(311)
    plt.plot(y_st, label='Markov chain',
             color='#ffa700', linewidth=5)
    plot_boundaries(y_st, c, phn)
    plt.legend(fontsize=18)
    plt.xlim(10, 120)
    plt.ylim(0,y_st[10:120].max()*1.05)
    plt.subplot(312)
    plt.plot(y, label='RNN (cat.)',
             color='#008744', linewidth=5)
    plot_boundaries(y, c, phn)
    plt.legend(fontsize=18)
    plt.xlim(10, 120)
    plt.ylim(0,y[10:120].max()*1.05)
    plt.subplot(313)
    plt.plot(y_old, label='RNN (cont.)',
             color='#0057e7', linewidth=5)
    plot_boundaries(y_old, c, phn)
    plt.legend(fontsize=18)
    plt.xlim(10, 120)
    plt.ylim(0,y_old[10:120].max()*1.05)

    plt.show()
