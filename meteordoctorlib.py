from scipy.io.wavfile import read as wavfileread
import math, string, random
import numpy as np
from scipy.misc import imresize, imsave

LOG_IQ = False
LOG_SPEC = True

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
