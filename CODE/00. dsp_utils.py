import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#  Chuyển biên độ → dB 
def amplitude_to_db(amplitude):
    a = np.abs(amplitude)
    with np.errstate(divide='ignore'):
        return np.where(a > 0, 20*np.log10(a), -300)


#       Vẽ FFT (x_max cố định)
def plot_fft(freqs, Y1, Y2=None, fs=44000, cutoff=None,
             label1="Tín hiệu 1", label2="Tín hiệu 2",
             title="So sánh phổ FFT"):

    N = len(Y1)
    x_max = min(freqs.max()/2,fs/2)               
    mag_Y1 = amplitude_to_db(Y1[:N//2])

    mag_Y2 = amplitude_to_db(Y2[:N//2]) if Y2 is not None else None

    # Lấy cùng scale biên độ để so sánh chính xác
    if mag_Y2 is not None:
        y_min = min(np.nanmin(mag_Y1), np.nanmin(mag_Y2))
        y_max = max(np.nanmax(mag_Y1), np.nanmax(mag_Y2))
    else:
        y_min, y_max = np.nanmin(mag_Y1), np.nanmax(mag_Y1)

    fig, ax = plt.subplots(1, 2 if Y2 is not None else 1,
                           figsize=(13,5), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)

    # ------------ FFT 1 ------------
    ax[0].plot(freqs[:N//2], mag_Y1, lw=1.2)
    ax[0].set_title(f"{label1} (dB)")
    ax[0].set_xlabel("Tần số (Hz)")
    ax[0].set_ylabel("Biên độ (dB)")
    ax[0].set_xlim(0, x_max)            
    ax[0].set_ylim(y_min, y_max)
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # ------------ FFT 2 nếu có ------------
    if Y2 is not None:
        ax[1].plot(freqs[:N//2], mag_Y2, lw=1.2, color='tab:red')
        ax[1].set_title(f"{label2} (dB)")
        ax[1].set_xlabel("Tần số (Hz)")
        ax[1].set_xlim(0, x_max)       
        ax[1].set_ylim(y_min, y_max)
        ax[1].grid(True, linestyle='--', alpha=0.6)

    if cutoff:
        for a in ax:
            a.axvline(x=cutoff, color='gray', ls=':', lw=1)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


#        Spectrogram (có fmax)
def plot_spectrogram(y1, y2, fs, fmax=None,
                     title1="Tín hiệu 1", title2="Tín hiệu 2",
                     suptitle="So sánh Spectrogram",
                     label="Mật độ phổ (PSD)"):

    N = max(len(y1), len(y2))
    nfft = choose_nfft(N)
    noverlap = nfft//2
    if fmax is None: 
        fmax = fs/2 

    fig = plt.figure(figsize=(14,6))
    gs  = gridspec.GridSpec(1,3,width_ratios=[1,1,0.06])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    cax = plt.subplot(gs[2])

    P1,f1,t1,im1 = ax1.specgram(y1,NFFT=nfft,Fs=fs,noverlap=noverlap,cmap='inferno')
    ax1.set_title(title1)
    ax1.set_xlabel("Thời gian (s)")
    ax1.set_ylabel("Tần số (Hz)")
    ax1.set_ylim(0, fmax)       # scale Y theo f_plot_max

    P2,f2,t2,im2 = ax2.specgram(y2,NFFT=nfft,Fs=fs,noverlap=noverlap,cmap='inferno')
    ax2.set_title(title2)
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylim(0, fmax)       # scale Y theo f_plot_max

    # đồng bộ màu
    vmin=min(im1.get_clim()[0],im2.get_clim()[0])
    vmax=max(im1.get_clim()[1],im2.get_clim()[1])
    im1.set_clim(vmin,vmax); im2.set_clim(vmin,vmax)

    fig.colorbar(im1,cax=cax,label=label)
    fig.suptitle(suptitle,fontsize=14,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


#    Chọn NFFT tự động
def choose_nfft(L, prefer=1024):
    """ tự động chọn NFFT phù hợp """
    if L >= prefer:
        return prefer
    P = 2**int(np.log2(L)) if L>=1 else 256
    return max(256,P)
