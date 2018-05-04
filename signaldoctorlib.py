"""
Jonathan Rawlinson 2018
Imperial College EEE Department
"""
import numpy as np
import math
import string
import random
import csv

from scipy import signal
from scipy.io.wavfile import read as wavfileread
from scipy.io import savemat
from scipy.misc import imresize, imsave
from scipy.stats import zscore
from numpy.fft import fftshift

from detect_peaks import detect_peaks

## FFT Config
import pyfftw
from scipy.fftpack import fft, ifft, fftn

## SETUP
import os.path
import pickle

## DEBUG
import matplotlib.pyplot as plt

## TODO ADD TO CFG FILE/CMD LINE OPTIONS
energy_threshold = -5
peak_threshold = 0.005
peak_threshold = 1

#fs = 2e6
#fs = 8000
MaxFFTN = 22
iq_buffer_len = 2000 ##ms
OSR = 1.5
plot_features = False
plot_peaks = False
#IQ_FS_OVERRIDE = True
#IQ_FS = fs
smooth_no = 1

## Spectrogram Calculation Ratio
spectrogram_size = 256
SPEC_OVERLAP_RATIO = 2

## Smoothing Parameters - for 1e6 Hz 
resample_ratio = 256
smooth_stride_hz = 5e3

## Bandwidth Calculation Parameters
BW_CALC_VAR = 4

## Logging
LOG_IQ = True
LOG_SPEC = True
LOG_MODE = 'npz'

## Debug BW Override
BW_OVERRIDE = False
BW_OVERRIDE_VAL = 5000/resample_ratio ##Currently its the RS BW


## Help from -->> https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list
def npmax(l):
    """ 
    Find Max value and index of wall
    """
    max_idx = int(np.argmax(l))
    max_val = l[max_idx]
    return (max_idx, max_val)


def save_IQ_buffer(channel_dict, output_format = 'npy', output_folder = 'logs/'):
    """
    Write IQ data into npy file
    The MATLAB *.mat file can also be used
    """
    filename = id_generator()
    if LOG_IQ:
        if (output_format == 'npy'):
            np.savez(output_folder+filename+".npz",channel_iq=channel_dict['iq_data'], fs=channel_dict['local_fs'])
        else:
            savemat(output_folder+filename+".mat", {'channel_iq':channel_dict['iq_data'], 'fs':channel_dict['local_fs']})
    if LOG_SPEC:
        #features = generate_features(fs, channel_iq)
        imsave(output_folder+filename+".png", channel_dict['magnitude'])

## From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate random string
    """
    return ''.join(random.choice(chars) for _ in range(size))

def load_IQ_file(input_filename):
    """Read .wav file containing IQ data from SDR#"""
    fs, samples = wavfileread(input_filename)
    return fs, len(samples), samples

def import_buffer(iq_file,fs,start,end):
    """Extract buffer from array and balance the IQ streams"""
    #fs, samples = scipy.io.wavfile.read(input_filename)
    #print("\nFile Read:", input_filename, " - Fs:", fs)
    #print(len(samples))
    input_frame = iq_file[int(start):int(end)]
    
    try:
        print("No Channels:", input_frame.shape[1])
        REAL_DATA = False
    except IndexError:
        print("####Single Channel Audio####")
        REAL_DATA = True
    
    if not REAL_DATA:
        input_frame_iq = np.zeros(input_frame.shape[:-1], dtype=np.complex)
        input_frame_iq.real = input_frame[..., 0]
        input_frame_iq.imag = input_frame[..., 1]
        # Balance IQ file
        input_frame_iq = remove_DC(input_frame_iq)
    elif REAL_DATA:
        input_frame_iq = np.zeros((int(input_frame.shape[0]/2), 2), dtype=np.complex)
        input_frame_iq = input_frame
        #input_frame = np.reshape(input_frame, (int(input_frame.shape[0]/2), 2))
        #input_frame_iq = input_frame[...,0]# + 1j*input_frame[...,1] ### This makes reflections in the negative freq plane (obviously) - need to do fix, freq shift and decimate
        #input_frame_iq.imag = input_frame[...,1] 
    
    return input_frame_iq, fs
   


## From https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
def power_bit_length(x):
    x = int(x)
    return 2**((x-1).bit_length()-1)

def process_iq_file(filename, LOG_IQ, pubsocket=None, metadata=None):

    #loaded_model, index_dict = get_spec_model(MODEL_NAME)

    fs, file_len, iq_file = load_IQ_file(filename)
    #if IQ_FS_OVERRIDE:
        #fs = IQ_FS
    print("###Loaded IQ File###, Len: %i, Fs %i" % (file_len, fs))

    ##Calculate buffer length
    length = (fs/1000)*iq_buffer_len
    length = power_bit_length(length)

    buf_no = int(np.floor(file_len/(length)))

    print("Length of buffer: ", length/fs, "s")
    for i in range(0, buf_no):
        print("Processing buffer %i of %i" % (i+1 , buf_no))
        ## Read IQ data into memory
        in_frame, fs = import_buffer(iq_file, fs, i*length, (i+1)*length)
        print("IQ Len: ", len(in_frame))
        classify_buffer(in_frame, fs=fs, LOG_IQ=LOG_IQ,  pubsocket=pubsocket, metadata=metadata)


def classify_buffer(buffer_data, fs=1, LOG_IQ = True, pubsocket=None, metadata=None):
    """
    This function generates the features and acts as an entry point into the processing framework
    The features are generated, logged (if required) and then published to the next block
    """
    extracted_iq_dict = process_buffer(buffer_data, fs, tx_socket = pubsocket[1], metadata=metadata)

    # We now have the features and iq data
    if LOG_IQ:
        print("Logging.....")
        for iq_channel in extracted_iq_dict:
            save_IQ_buffer(iq_channel, output_format=LOG_MODE)
    send_features(extracted_iq_dict, pubsocket[0], metadata=metadata)
    
def send_features(extracted_features, pubsocket, metadata=None):
    """
    Publish the feature arrays over the network
    The ZMQ socket is passed from the calling function
    """
    for feature_dict in extracted_features:
        #data = np.asarray(data)
        feature_dict['iq_data'] = np.asarray(feature_dict['iq_data'])
        feature_dict['metadata'] = metadata
        pubsocket.send_pyobj(feature_dict)
        #print(data)



def classify_spectrogram(input_array, model, index):
    """
    This function takes the input features and runs them through the classification networks
    """
    print("LEN:", len(input_array))
    for i in range(0, len(input_array)):
        tmpspec = input_array[i][0]
        tmppsd = input_array[i][1]
        
        #tmpspec = tmpspec.reshape((1, tmpspec.shape[0], tmpspec.shape[1], 1))
        tmpspec_z = np.zeros((1, tmpspec.shape[0], tmpspec.shape[1], 3))
        for j in range(0, 3):
            tmpspec_z[0,:,:,j] = tmpspec

        
        tmpspec_z = tmpspec_z.astype('float32')
        tmpspec_z /= np.max(tmpspec_z)
        prediction_spec = model[0].predict(tmpspec_z)

        idx_spec = npmax(prediction_spec[0])[0]
        print("Classified signal (spec) -->>", index[idx_spec])
        
        tmppsd = tmppsd.astype('float32')
        tmppsd /= np.max(tmppsd) 
        tmppsd = tmppsd.reshape((1, tmppsd.shape[0]))
        prediction_psd= model[1].predict(tmppsd)

        idx_psd = npmax(prediction_psd[0])[0]
        print("Classified signal (psd) -->>", index[idx_psd])
        #prediction = prediction.flat[0]

def fft_wrap(iq_buffer, mode = 'pyfftw'):
    """
    Compute FFT - either using pyfftw (default) or scipy
    """
    if mode == 'pyfftw':
        return pyfftw.interfaces.numpy_fft.fft(iq_buffer)
    elif mode == 'scipy':
        return fft(iq_buffer) #### TODO CLEAN

def ifft_wrap(iq_buffer, mode = 'pyfftw'):
    """
    Compute IFFT - either using pyfftw (default) or scipy
    """
    if mode == 'pyfftw':
        return pyfftw.interfaces.numpy_fft.ifft(iq_buffer)
    elif mode == 'scipy':
        return ifft(iq_buffer)
    

def process_buffer(buffer_in, fs=1, tx_socket=None ,metadata=None):
    """
    Analyse input buffer, extract signals and pass onto the classifiers
    """
    buffer_len = len(buffer_in)
    print("Processing signal. Len:", buffer_len)
    #Do FFT - get it out of the way!
    buffer_fft = fft_wrap(buffer_in, mode = 'scipy')
    
    ## Will now unroll FFT for ease of processing
    buffer_fft_rolled = np.roll(buffer_fft, int(len(buffer_fft)/2))
    #Is there any power there?
    buffer_abs = np.abs(buffer_fft_rolled*buffer_fft_rolled.conj())
    buffer_energy = np.log10(buffer_abs.sum()/buffer_len)

    #If below threshold return nothing
    if (buffer_energy<energy_threshold):
        print("Below threshold - returning nothing")
        return None
    
    ## Resampling 
    buffer_abs = np.reshape(buffer_abs, (resample_ratio, int(len(buffer_abs)/resample_ratio)), order='F')
    buffer_abs = np.mean(buffer_abs, axis=0)
    
    ## If we want to compute the logarithm of the data for peak finding, remove all zeros
    #buffer_abs[buffer_abs == 0] = 0.01 #Remove all zeros
    
    smooth_stride = int((buffer_len*smooth_stride_hz)/(resample_ratio*fs))
    
    
    buffer_log2abs = buffer_abs
    print("Smoothing, stride: %i Hz, No %i" % (int(smooth_stride*resample_ratio*fs/buffer_len), smooth_no))
    for i in range(0,smooth_no):
        buffer_log2abs = smooth(buffer_log2abs, window_len=smooth_stride, window ='hanning')
    
    ## Normalise
    buffer_log2abs = zscore(buffer_log2abs)
    #buffer_log2abs = signal.detrend(buffer_log2abs)
    buffer_log2abs = buffer_log2abs - np.min(buffer_log2abs)
    
    
      
    buffer_peakdata = detect_peaks(buffer_log2abs, mph=peak_threshold , mpd=2, edge='rising', show=plot_peaks)
    #Search for signals of interest
    output_signals = []
    for peak_i in buffer_peakdata:
        #Decimate the signal in the frequency domain
        #Please note that the OSR value is from the 3dB point of the signal - if there is a long roll off (spectrally) some of the signal may be cut
        output_signal_fft, bandwidth = fd_decimate(buffer_fft_rolled, resample_ratio, buffer_log2abs, peak_i, OSR)
        #print("Bandwidth--->>>", bandwidth)
        buf_len = len(buffer_fft_rolled)
        if len(output_signal_fft) ==0:
            continue
        #Compute IFFT and add to list
        td_channel = ifft_wrap(output_signal_fft, mode = 'scipy')
        output_signals.append([td_channel, peak_i*resample_ratio])
    
    
    print("We have %i signals!" % (len(output_signals)))
    
    ## Generate Features ##
    output_signal_data = []
    for i in output_signals:
        buf = i[0]
        local_fs = fs * len(buf)/buffer_len
        feature_dict = generate_features(local_fs, buf, plot_features=plot_features)
        feature_dict['local_fs'] = local_fs
        feature_dict['iq_data'] = buf
        feature_dict['offset'] = (fs * (i[1]-buffer_len/2)/buffer_len) + metadata['cf']
        output_signal_data.append(feature_dict)
    
    globaltx_dict = {}
    globaltx_dict['recent_psd'] = buffer_log2abs.tolist()
    globaltx_dict['cf'] = metadata['cf']
    globaltx_dict['fs'] = fs
    globaltx_dict['buf_len'] = buffer_len
    globaltx_dict['resampling_ratio'] = resample_ratio
    
    tx_socket.send_pyobj(globaltx_dict) ## Send global psd
    
    return output_signal_data

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

def generate_features(local_fs, iq_data, spec_size=spectrogram_size, roll = True, plot_features = False):
    """
    Generate classification features
    """
    
    ## Generate normalised spectrogram
    NFFT = math.pow(2, int(math.log(math.sqrt(len(iq_data)), 2) + 0.5)) #Arb constant... Need to analyse TODO
    #print("NFFT:", NFFT)
    f, t, Zxx_cmplx = signal.stft(iq_data, local_fs, noverlap=NFFT/SPEC_OVERLAP_RATIO, nperseg=NFFT, return_onesided=False)
    f = fftshift(f)
    #f = np.roll(f, int(len(f)/2))
    Zxx_cmplx = fftshift(Zxx_cmplx, axes=0)
    if roll:
        Zxx_cmplx = np.roll(Zxx_cmplx, int(len(f)/2), axis=0)

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

def smooth(x,window_len=12,window='flat'):
    ## From http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    Smooth the data using a window with variable size
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[int(window_len/2-1):int(-window_len/2)]

def normalise_spectrogram(input_array, newx, newy):
    """
    Interpolate NxN array into newx and newy array
    """
    arr_max = input_array.max()
    arr_min = input_array.min()
    input_array = (input_array-arr_min)/(arr_max-arr_min)

    return imresize(input_array, (newx, newy))

def remove_DC(IQ_File):
    """
    Remove DC offset from input data file
    """
    DC_Offset_Real = np.mean(np.real(IQ_File))
    DC_Offset_Imag = np.mean(np.imag(IQ_File))
    return IQ_File - (DC_Offset_Real + DC_Offset_Imag * 1j)

def fd_decimate(fft_data, resample_ratio, fft_data_smoothed, peakinfo, osr):
    """
    Decimate Buffer in Frequency domain to extract signal from 3db Bandwidth
    """
    #Calculate 3dB peak bandwidth
    #TODO
    
    cf = peakinfo
    bw = find_3db_bw_JR_single_peak(fft_data_smoothed, peakinfo)
    if BW_OVERRIDE:
        bw = int(BW_OVERRIDE_VAL)
    slicer_lower = (cf-(bw/2)*osr)*resample_ratio #- resample_ratio*smooth_stride*smooth_no ##Offset fix? The offset is variable with changes to the smoothing filter - convolution shift?
    slicer_upper = (cf+(bw/2)*osr)*resample_ratio #- resample_ratio*smooth_stride*smooth_no
    #print("Slicing Between: ", slicer_lower, slicer_upper, "BW: ", bw, "CF: ", cf, peakinfo)
    #Slice FFT
    output_fft = fft_data[int(slicer_lower):int(slicer_upper)]
    return output_fft, bw

def find_3db_bw_JR_single_peak(data, peak):
    """ 
    Find bandwidth of single peak 
    """
    max_bw = find_3db_bw_max(data, peak)
    min_bw = find_3db_bw_min(data, peak)
    bw1 = (max_bw - peak)*2
    bw2 = (peak - min_bw)*2
    bw = min(bw1, bw2)
    #print("BW max min:", bw, max_bw, min_bw)
    return bw


def find_3db_bw_max(data, peak, step=10):
    """ 
    Find 3dB point going up the array 
    """
    max_height = data[peak]
    min_global = np.min(data)
    thresh = max_height - (np.abs(max_height - min_global))/BW_CALC_VAR
    previous_value = max_height
    for i in range(peak, len(data)-1, step):
        if (data[i] < thresh):
            return i
        elif (data[i] > previous_value): ## Need to Interpolate Thresh BW here
            return i
        previous_value = data[i]
    return len(data)-1 ##Nothing found - should probably raise an error here. TODO check error

def find_3db_bw_min(data, peak, step=10):
    """ 
    Find 3dB point going down the array
    """
    max_height = data[peak]
    min_global = np.min(data)
    thresh = max_height - (np.abs(max_height - min_global))/BW_CALC_VAR
    previous_value = max_height
    for i in range(peak, 0-1, -step):
        if (data[i] < thresh):
            return i
        elif (data[i] > previous_value):
            return i
    return 0 ##Nothing found - should probably raise an error here

