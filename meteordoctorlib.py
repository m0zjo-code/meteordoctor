from scipy.io.wavfile import read as wavfileread
import math, string, random
import numpy as np
from scipy.misc import imresize, imsave
from scipy import signal
from scipy.fftpack import fft, ifft, fftn
from scipy.signal import stft

LOG_IQ = True
LOG_SPEC = True
NOVERLAP = 0.8

## From https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
def power_bit_length(x):
    x = int(x)
    return 2**((x-1).bit_length()-1)

def load_wav_file(input_filename):
    """Read .wav file containing IQ data from SDR#"""
    fs, samples = wavfileread(input_filename)
    length = len(samples)
    print("File loaded FS: %i, LEN: %i (%f s)" % (fs, length, length/fs))
    return fs, length, samples

def import_buffer(input_file,fs,start,end):
    """Extract buffer from array and balance the IQ streams"""  
    input_frame = input_file[int(start):int(end)]
    #input_frame_iq = np.zeros((int(input_frame.shape[0]/2), 2), dtype=np.complex)
    #input_frame_iq = input_frame
    return input_frame

def save_buffer(channel_dict, output_format = 'npy', output_folder = 'logs/'):
    """
    Write IQ data into npy file
    The MATLAB *.mat file can also be used
    """
    filename = id_generator()
    if LOG_IQ:
        if (output_format == 'npy'):
            np.savez(output_folder+filename+".npz",channel_iq=channel_dict['data'], fs=channel_dict['fs'])
        else:
            savemat(output_folder+filename+".mat", {'channel_iq':channel_dict['data'], 'fs':channel_dict['fs']})
    if LOG_SPEC:
        #features = generate_features(fs, channel_iq)
        imsave(output_folder+filename+".png", np.flipud(channel_dict['magnitude'])) ## Flip as something seems upside down... 

def calculate_nseg(buffer_len):
    ## Its a log.square root relationship... Not sure why....
    ## This is different from the SDL as we are optimising for a different fractional bandwidth (1/8 rather than 1/1)
    NFFT = math.pow(2, int(math.log(math.sqrt(buffer_len), 2) + 3 )) #Arb constant (that seems to work...)... Need to analyse TODO
    return NFFT

## From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate random string
    """
    return ''.join(random.choice(chars) for _ in range(size))

def normalise_spectrogram(input_array, newx=256, newy=256):
    """
    Interpolate NxN array into newx and newy array
    """
    arr_max = input_array.max()
    arr_min = input_array.min()
    input_array = (input_array-arr_min)/(arr_max-arr_min)

    return imresize(input_array, (newy, newx))



def generate_features(local_fs, iq_data, spec_size=256, roll = True, plot_features = False, overlap=NOVERLAP):
    """
    Generate classification features
    """
    
    ## Generate normalised spectrogram
    NFFT =calculate_nseg(len(iq_data)) #Arb constant... Need to analyse TODO
    #print("NFFT:", NFFT)
    f, t, Zxx_cmplx = signal.stft(iq_data, local_fs, noverlap=NFFT*overlap, nperseg=NFFT, return_onesided=False)

    Zxx_mag = np.abs(np.power(Zxx_cmplx, 2))
    
    Zxx_mag_fft = np.abs(fftn(Zxx_mag))
    Zxx_mag_fft = np.fft.fftshift(Zxx_mag_fft)
    Zxx_mag_log = log_enhance(Zxx_mag, order=2)
    
    diff_array0 = np.diff(Zxx_mag_log, axis=0)
    diff_array1 = np.diff(Zxx_mag_log, axis=1)
    
    diff_array0_rs = normalise_spectrogram(diff_array0, spec_size, spec_size)
    diff_array1_rs = normalise_spectrogram(diff_array1, spec_size, spec_size)
    
    Zxx_phi = np.abs(np.unwrap(np.angle(Zxx_cmplx), axis=0))
    Zxx_cec = np.abs(np.corrcoef(Zxx_mag_log, Zxx_mag_log))
    
    
    Zxx_mag_rs = normalise_spectrogram(Zxx_mag_log, spec_size, spec_size)
    Zxx_phi_rs = normalise_spectrogram(Zxx_phi, spec_size, spec_size)
    Zxx_cec_rs = normalise_spectrogram(Zxx_cec, spec_size*2, spec_size*2)
    
    Zxx_cec_rs = blockshaped(Zxx_cec_rs, spec_size, spec_size)
    Zxx_cec_rs = Zxx_cec_rs[0]
    
    # We have a array suitable for NNet
    ## Generate spectral info by taking mean of spectrogram ##
    PSD = np.mean(Zxx_mag_rs, axis=1)
    Varience_Spectrum = np.var(Zxx_mag_rs, axis=1)
    Differential_Spectrum = np.sum(np.abs(diff_array1_rs), axis=1)
    
    Min_Spectrum = np.min(Zxx_mag_rs, axis=1)
    Max_Spectrum = np.max(Zxx_mag_rs, axis=1)
    
    Zxx_mag_hilb = np.abs(signal.hilbert(Zxx_mag.astype(np.float), axis=1))
    
    if plot_features:
        nx = 3
        ny = 4
        plt.subplot(nx, ny, 1)
        plt.title("Magnitude Spectrum")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.pcolormesh(Zxx_mag_rs) ## +1 to stop 0s
        
        plt.subplot(nx, ny, 2)
        plt.title("Max Spectrum")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.plot(Max_Spectrum)
        
        plt.subplot(nx, ny, 3)
        plt.title("PSD")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.plot(PSD)
        
        plt.subplot(nx, ny, 4)
        plt.title("Autoorrelation Coefficient (Magnitude)")
        plt.pcolormesh(Zxx_cec_rs)
        
        plt.subplot(nx, ny, 5)
        plt.title("Frequency Diff Spectrum")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.pcolormesh(diff_array0)
        
        plt.subplot(nx, ny, 6)
        plt.title("Time Diff Spectrum")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.pcolormesh(diff_array1)
        
        plt.subplot(nx, ny, 7)
        plt.title("Varience Spectrum")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.plot(Varience_Spectrum)
        
        plt.subplot(nx, ny, 8)
        plt.title("Differential Spectrum")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.plot(Differential_Spectrum)
        
        plt.subplot(nx, ny, 9)
        plt.title("Min Spectrum")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.plot(Min_Spectrum)
        
        plt.subplot(nx, ny, 10)
        plt.title("FT Spectrum")
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.pcolormesh(Zxx_mag_fft)
        
        plt.subplot(nx, ny, 11)
        plt.title("Hilbert Spectrum")
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.pcolormesh(Zxx_mag_hilb)
        
        mng = plt.get_current_fig_manager() ## Make full screen
        mng.full_screen_toggle()
        plt.show()
    
    output_dict = {}
    output_dict['magnitude'] = Zxx_mag_rs
    output_dict['phase'] = Zxx_phi_rs
    output_dict['corrcoef'] = Zxx_cec_rs
    output_dict['psd'] = PSD
    output_dict['variencespectrum'] = Varience_Spectrum
    output_dict['differentialspectrumdensity'] = Differential_Spectrum
    output_dict['differentialspectrum_freq'] = diff_array0_rs
    output_dict['differentialspectrum_time'] = diff_array1_rs
    output_dict['min_spectrum'] = Min_Spectrum
    output_dict['max_spectrum'] = Max_Spectrum
    output_dict['fft_spectrum'] = normalise_spectrogram(Zxx_mag_fft, spec_size, spec_size)
    output_dict['hilb_spectrum'] = normalise_spectrogram(Zxx_mag_hilb, spec_size, spec_size)

    return output_dict

def log_enhance(input_array, order=1):
    input_array_tmp = input_array
    for i in range(0, order):
        min_val = np.min(input_array_tmp)
        input_array_shift = input_array_tmp-min_val+1
        input_array_tmp = np.log2(input_array_shift)
    return input_array_tmp


## From: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def generate_mag_feature(fs, in_frame, spec_xy = 256, norm_aspect = 1):
    
    MINF = 500
    MAXF = 1500
    
    length = power_bit_length(len(in_frame))
    
    NFFT = calculate_nseg(length)

    #print("NFFT,", NFFT)
    f, t, Zxx = stft(in_frame, fs=fs, nperseg=NFFT, noverlap=NFFT*NOVERLAP)
    mag = np.abs(Zxx*Zxx.conj())
    
    minf = NFFT*MINF/fs
    maxf = NFFT*MAXF/fs
    mag_norm = normalise_spectrogram(mag[int(minf):int(maxf), ...], newx=int(spec_xy*norm_aspect), newy=int(spec_xy))
    
    return mag_norm