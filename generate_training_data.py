import numpy as np
import meteordoctorlib as mdl
import os
import glob
import matplotlib.pyplot as plt

## Takes training IQ data and generates training features

SPEC_SIZE = 256 ## NxN input tensor size
input_folder = "/home/jonathan/meteordoctor_dataset_1"
NOVERLAP = 0.8


def load_npz(filename):
    data = np.load(filename)
    iq_data = data['channel_iq']
    fs = data['fs']
    return fs, iq_data

def feature_gen(file_list, spec_size):
    output_list_spec = [] ## This is going to be very big!
    output_list_psd = []
    output_list_phi = []
    output_list_cec = []
    
    for filename in file_list:
        fs, iq_data = load_npz(filename)
        #print("%i samples loaded at a rate of %f" % (len(iq_data), fs))
        #print("We have %f s of data" % (len(iq_data)/fs))
        print("Reading-->>",filename)

        feature_dict = mdl.generate_features(fs, iq_data, spec_size, overlap = NOVERLAP, plot_features = False)
        
        #print(feature_dict['magnitude'].shape)
        #print(feature_dict['phase'].shape)
        #print(feature_dict['corrcoef'].shape)
        #print(feature_dict['differentialspectrum_freq'].shape)
        #print(feature_dict['differentialspectrum_time'].shape)
        
        #tmp_spec = np.stack((feature_dict['magnitude'], feature_dict['phase'], feature_dict['corrcoef'], feature_dict['differentialspectrum_freq'], feature_dict['differentialspectrum_time']), axis=-1)
        output_list_spec.append(feature_dict['magnitude'])
        
        #print(feature_dict['psd'].shape)
        #print(feature_dict['variencespectrum'].shape)
        #print(feature_dict['differentialspectrumdensity'].shape)
        #print(feature_dict['min_spectrum'].shape)
        #print(feature_dict['min_spectrum'].shape)
        
        #tmp_psd = np.stack((feature_dict['psd'], feature_dict['variencespectrum'], feature_dict['differentialspectrumdensity'], feature_dict['min_spectrum'], feature_dict['min_spectrum']), axis=-1)
        output_list_psd.append(feature_dict['psd'])

        #plt.pcolormesh(Zxx_dat)
        #plt.show()
    return [output_list_spec, output_list_psd]


def clear_dir(path):
    file_list = glob.glob(path)
    for f in file_list:
        try:
            os.remove(f)
        except:
            print("DIR")

## Delete old data
clear_dir('nnetsetup/specdata/*')
clear_dir('nnetsetup/psddata/*')
clear_dir('nnetsetup/*')

for sig in os.listdir(input_folder):
    sig_input_folder = input_folder + "/" + sig
    if os.path.isdir(sig_input_folder):
        print("Processing from --->>>>>>", sig_input_folder)
        filename_list = []
        for fileZ in os.listdir(sig_input_folder):
            #print(fileZ)
            if fileZ.endswith(".npz"):
                filename_list.append(os.path.join(sig_input_folder, fileZ))

        data_list = feature_gen(filename_list, SPEC_SIZE)

        spec_array = np.asarray(data_list[0])
        psd_array = np.asarray(data_list[1])
        np.save('nnetsetup/specdata/' + sig + '.npy', spec_array)
        np.save('nnetsetup/psddata/' + sig + '.npy', psd_array)


## Splits training data into training and test data

## $$$$$$$ SPECTROGRAM CNN $$$$$$$$

train_testsplit = 0.8

input_folder = "nnetsetup/specdata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/spec_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()

y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))

np.savez("nnetsetup/SpecTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("######## SPECTROGRAM ########")
print("Train:", X_train.shape)
print("Test:", X_test.shape)


## $$$$$$$ PSD $$$$$$$$

train_testsplit = 0.8 ##

input_folder = "nnetsetup/psddata"

filename_list = []
y_train = []
X_train = []
y_test = []
X_test = []

for filez in os.listdir(input_folder):
    if filez.endswith(".npy"):
        filename_list.append(os.path.join(input_folder, filez))

index = 0
index_data = []
for filename in filename_list:
    x = np.load(filename)
    x_len = len(x)
    #np.random.shuffle(x)
    training_tmp, test_tmp = x[:int(x_len*train_testsplit),:], x[int(x_len*train_testsplit):,:]
    for i in training_tmp:
        X_train.append(i)
        y_train.append(index)

    for i in test_tmp:
        X_test.append(i)
        y_test.append(index)

    line = filename + ",%s"%(index)
    index_data.append(line)
    print(line)
    index = index + 1

thefile = open('nnetsetup/psd_data_index.csv', 'w')
for i in index_data:
    thefile.write("%s\n" % i)
thefile.close()


y_train = np.asarray(y_train)

X_train = np.asarray(X_train)
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

y_test = np.asarray(y_test)

X_test = np.asarray(X_test)
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

np.savez("nnetsetup/PsdTrainingData.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("######## PSD ########")
print("Train:", X_train.shape)
print("Test:", X_test.shape)
