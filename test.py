import numpy as np

data1 = np.load("output_decoding_times_speculative_20250225_182212.npy")
data2 = np.load("output_decoding_times_speculative.npy")

for i in range(len(data1)):
    print((data1[i]-data2[i])/data1[i])

