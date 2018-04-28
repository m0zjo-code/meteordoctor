import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import meteordoctorlib as mdl

## Settings
plot_spec = False

## Some constants
target_buffer_len = 8000
NOVERLAP = 0.8
spec_xy = 256
norm_aspect = 1.5

## Filename
fn = "sample_meteor_data_1hr.wav"

### Load main file
def process_wav_file(fn): 
    
    fs, file_len, main_buffer = mdl.load_wav_file(fn)
    
    length = (fs/1000)*target_buffer_len
    length = mdl.power_bit_length(length)

    buf_no = int(np.floor(file_len/(length)))

    print("Length of buffer: ", length/fs, "s")
    print("%i buffers filled" % buf_no)
    for i in range(0, buf_no*2-1):
        ## We're going to do buffer overlap to ensure that nothing is missed!
        print("Processing buffer %i of %i" % (i+1 , buf_no*2))
        ## Read audio data into memory
        shifter = i/2 ## 1/2 1/2 buffer overlap
        in_frame = mdl.import_buffer(main_buffer, fs, shifter*length, (shifter+1)*length)
        #print("Data Len: ", len(in_frame))
        ## Lets do stuff!
        NFFT = mdl.calculate_nseg(length)

        #print("NFFT,", NFFT)
        f, t, Zxx = stft(in_frame, fs=fs, nperseg=NFFT, noverlap=NFFT*NOVERLAP)
        mag = np.abs(Zxx)
        
        signal_dict  = {}
        signal_dict['data'] = in_frame
        signal_dict['fs'] = fs
        minf = NFFT*500/fs
        maxf = NFFT*1500/fs
        signal_dict['magnitude'] = mdl.normalise_spectrogram(mag[int(minf):int(maxf), ...], newx=int(spec_xy*norm_aspect), newy=int(spec_xy))
        mdl.save_buffer(signal_dict)
        if plot_spec:
            plt.subplot(2, 1, 1)
            plt.pcolormesh(t, f, mag, cmap='viridis')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim((500,1500))
            
            plt.subplot(2, 1, 2)
            mag = np.square(mag)
            plt.pcolormesh(t, f, mag, cmap='viridis')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.ylim((500,1500))
            plt.show()
        
process_wav_file(fn)
