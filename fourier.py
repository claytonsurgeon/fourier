import matplotlib.pyplot as plt
import csv
import numpy as np
# from scipy.fft import fft, fftfreq
# from math import floor, ceil

accU = []
raw = []
with open("./AC190520.csv", "r") as infile:
    reader = csv.reader(infile, delimiter=",")

    for row in reader:
        raw.append(row)

    i = 0
    while i < len(raw):
        if i % 2051 == 0:
            i += 3
            accU.append([])

        else:
            accU[-1].append(float(raw[i][-1]))
            i += 1


# print(accU[0])
print(accU[0][-1])

accU_0 = accU[0]

# forward smooth outliers: outliers are smoothed by previous n values
# reverse smooth outliers: outliers are smoothed by next n values
# average together

# change 'last_n' to '1' to simply extend over outliers as Kai described


accU_0_std = np.std(accU_0)
accU_0_mean = np.mean(accU_0)

lower = accU_0_mean - accU_0_std * 3
upper = accU_0_mean + accU_0_std * 3

print(accU_0_std)
print(accU_0_mean)
outliers = [x for x in accU_0 if x < lower or x > upper]


last_n = 10

last_n_mean = [accU_0_mean]  # incase first value is outlier...
forward_smooth = []
for x in accU_0:
    if x < lower or x > upper:
        print('before ', x)
        x = np.mean(last_n_mean)
        print('after  ', x)

    forward_smooth.append(x)

    if len(last_n_mean) >= last_n:
        last_n_mean.pop(0)
    last_n_mean.append(x)


outliers_2 = [x for x in forward_smooth if x < lower or x > upper]

print(">>", len(outliers_2))


last_n_mean = [accU_0_mean]  # incase last value is outlier...
reverse_smooth = []

accU_0.reverse()        # note mutation

for x in accU_0:
    if x < lower or x > upper:
        print('before ', x)
        x = np.mean(last_n_mean)
        print('after  ', x)

    reverse_smooth.append(x)

    if len(last_n_mean) >= last_n:
        last_n_mean.pop(0)
    last_n_mean.append(x)


outliers_3 = [x for x in reverse_smooth if x < lower or x > upper]

print(">>", len(outliers_3))


accU_0.reverse()       # reverse the reversal
reverse_smooth.reverse()  # now in same order as forward smooth

accU_0_smoothed = []
i = 0
for x1, x2 in zip(forward_smooth, reverse_smooth):
    accU_0_smoothed.append(
        np.mean([x1, x2])
    )
    if x1 != x2:
        print(x1, x2, np.mean([x1, x2]))


outliers_4 = [x for x in accU_0_smoothed if x < lower or x > upper]

print(">>", len(outliers_4))


accU_0_smoothed_mean = np.mean(accU_0_smoothed)
accU_0_smoothed_mean_adjusted = np.array(
    accU_0_smoothed) - accU_0_smoothed_mean


print(accU_0_smoothed[0:10])
print(accU_0_smoothed_mean_adjusted[0:10])


# fs is sampling frequency
duration = 20*60    # 20 minutes
samples = 2048
fs = (20*60)/2048
time = np.linspace(0, duration, samples, endpoint=False)
wave = np.sin(np.pi*time * 2) + np.cos(np.pi*time * 0.020)

print(time)
print(len(time))
# quit()

plt.plot(time, accU_0_smoothed_mean_adjusted)
# plt.plot(time, wave)
plt.xlim(0, duration)
plt.xlabel("time (second)")
plt.title('Original Signal in Time Domain')


plt.show()


accU_0_fft = np.fft.fft(accU_0_smoothed_mean_adjusted)
# accU_0_fft = np.fft.fft(wave)
accU_0_fft_spec = accU_0_fft[0:len(accU_0_fft)//2]
accU_0_fft_magnitude = np.abs(np.abs(accU_0_fft_spec))
# print(accU_0_fft)

# fft_fre = np.fft.fftfreq(
#     n=accU_0_smoothed_mean_adjusted.size // 2, d=duration/samples)
fft_fre = np.fft.fftfreq(
    n=accU_0_smoothed_mean_adjusted.size, d=duration/samples)


# plt.plot(fft_fre, accU_0_fft_magnitude)
plt.plot(fft_fre[0:1024], accU_0_fft_magnitude)
# plt.plot(time[0:1024], accU_0_fft_magnitude)
# plt.xlim(-2, 2)
plt.xlabel("frequency (Hz)")
plt.title('Frequency Domain')
plt.show()


"""
accU_spec =
    accU Smooth - accU Mean,                         smooth outliers, then zero mean
    FFT,                                             
    [0, accU Length / 2] Range,                      Longest wave = sampling period / 2
    Magnitude                                        magnitude of complex numbers


"""
