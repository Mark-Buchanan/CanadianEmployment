# program reads in historical Canadian employment data and analyses
# it primarily using detrending and frequency filtering

# necessary imports and setup
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import iirnotch
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# define constants
EMPLOYMENT_CONV_FAC = 1000
MONTHS_PER_YEAR = 12
YEAR_START = 1976
YEAR_END = 2020
MONTH_END = 4
MONTH_START = 480
DURATION_MONTHS = 48
DURATION_YEARS = DURATION_MONTHS / MONTHS_PER_YEAR
MONTHS_OFFSET = 144

# return the detrended signal and its fourier transform of input y
# independent variable x and time step dt
def signal_process(x, y, dt):
    # calculate linear fit and remove it
    fit_coef = np.polyfit(x, y, 1)
    fit_line = fit_coef[0]*x + fit_coef[1]
    y_det = y - fit_line
    
    # calculate signal FT and respective frequencies
    f = np.fft.fftshift(np.fft.fftfreq(len(y_det), dt))
    y_ft = np.fft.fftshift(np.fft.fft(y_det)) * dt
    
    # frequency filter out everything > 2 cycles per month
    for i in range(len(f)):
        if abs(f[i]) > 2:
            y_ft[i] = 0
    return y_det, y_ft

# import data into arrays
with open('employment.csv', 'r') as file:
    reader = csv.reader(file)
    i = 0
    for row in reader:
        if i == 7:
            date = np.asarray(row[2:])
        if i == 10:
            male = np.asarray(row[2:])
        if i == 11:
            female = np.asarray(row[2:])
        i += 1

# preprocess data
for j in range(len(male)):
    male[j] = float(male[j].replace(',', ''))
    female[j] = float(female[j].replace(',', ''))

# convert data into number of people employed
male = np.asarray(male, dtype=np.float64) * EMPLOYMENT_CONV_FAC
female = np.asarray(female, dtype=np.float64) * EMPLOYMENT_CONV_FAC

# time step
dt = MONTHS_PER_YEAR ** -1                                    # years
# time array from start to end of data period
t = np.arange(YEAR_START + dt, YEAR_END + MONTH_END * dt, dt) # years
# time array for the duration of the samples in the overlaid processed datasets
t_dur = np.arange(dt, DURATION_YEARS + dt, dt)                # years
# time offset array for the cross-correlations of the processed datasets
t_off = np.arange(-DURATION_YEARS + dt, DURATION_YEARS, dt)   # years

# detrend the data sets and frequency filter them
####################### 2016-2020 #######################
start_2016 = MONTH_START
end_2016 = start_2016 + DURATION_MONTHS
m_det_2016, m_ft_2016 = signal_process(t[start_2016 : end_2016], male[start_2016 : end_2016], dt)
f_det_2016, f_ft_2016 = signal_process(t[start_2016 : end_2016], female[start_2016 : end_2016], dt)
t_det_2016 = t[start_2016 : end_2016]

####################### 2004-2008 #######################
start_2004 = MONTH_START - MONTHS_OFFSET
end_2004 = start_2004 + DURATION_MONTHS
m_det_2004, m_ft_2004 = signal_process(t[start_2004 : end_2004], male[start_2004 : end_2004], dt)
f_det_2004, f_ft_2004 = signal_process(t[start_2004 : end_2004], female[start_2004 : end_2004], dt)
t_det_2004 = t[start_2004:end_2004]

####################### 1992-1996 #######################
start_1992 = MONTH_START - 2 * MONTHS_OFFSET
end_1992 = start_1992 + DURATION_MONTHS
m_det_1992, m_ft_1992 = signal_process(t[start_1992 : end_1992], male[start_1992 : end_1992], dt)
f_det_1992, f_ft_1992 = signal_process(t[start_1992 : end_1992], female[start_1992 : end_1992], dt)
t_det_1992 = t[start_1992 : end_1992]

####################### 1980-1984 #######################
start_1980 = MONTH_START - 3 * MONTHS_OFFSET
end_1980 = start_1980 + DURATION_MONTHS
m_det_1980, m_ft_1980 = signal_process(t[start_1980 : end_1980], male[start_1980 : end_1980], dt)
f_det_1980, f_ft_1980 = signal_process(t[start_1980 : end_1980], female[start_1980 : end_1980], dt)
t_det_1980 = t[start_1980 : end_1980]

# calculate a specific sample's raw fourier transform
frq_2016 = np.fft.fftshift(np.fft.fftfreq(len(m_det_2016), dt))
m_ft_raw_2016 = np.fft.fftshift(np.fft.fft(m_det_2016)) * dt
f_ft_raw_2016 = np.fft.fftshift(np.fft.fft(f_det_2016)) * dt

# male cross correlations
ccm16_16 = np.correlate(np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, mode='full')
ccm16_04 = np.correlate(np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(m_ft_2004)).real / dt, mode='full')
ccm16_92 = np.correlate(np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(m_ft_1992)).real / dt, mode='full')
ccm16_80 = np.correlate(np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(m_ft_1980)).real / dt, mode='full')

# female cross correlations
ccf16_16 = np.correlate(np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, mode='full')
ccf16_04 = np.correlate(np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(f_ft_2004)).real / dt, mode='full')
ccf16_92 = np.correlate(np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(f_ft_1992)).real / dt, mode='full')
ccf16_80 = np.correlate(np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, 
                        np.fft.ifft(np.fft.ifftshift(f_ft_1980)).real / dt, mode='full')

# plot results
plt.figure(1)
plt.plot(t, male, label='Male')
plt.plot(t, female, label='Female')
plt.xlabel('Time (years)')
plt.ylabel('Employed (count)')
plt.title('Number of Employed Canadians Aged 15+')
plt.legend()
plt.savefig("raw_tot.pdf")
plt.show()

plt.figure(2)
plt.plot(frq_2016, m_ft_raw_2016.real, label='Male Re')
plt.plot(frq_2016, m_ft_raw_2016.imag, label='Male Im')
plt.plot(frq_2016, f_ft_raw_2016.real, label='Female Re')
plt.plot(frq_2016, f_ft_raw_2016.imag, label='Female Im')
plt.xlabel('Frequency (cycles/month)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform of Detrended Employment Data from\n2016-2020')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig('det_ft_2016.pdf')
plt.show()

plt.figure(3)
plt.plot(t_det_2016, m_det_2016, label='Male')
plt.plot(t_det_2016, f_det_2016, label='Female')
plt.xticks([2016, 2017, 2018, 2019, 2020])
plt.xlabel('Time (years)')
plt.ylabel('Detrended Employed (count)')
plt.title('Number of Employed Canadians Aged 15+\nfrom 2016-2020 (Detrended)')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("det_2016.pdf")
plt.show()

plt.figure(4)
plt.plot(t_det_2016, np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, label='Male')
plt.plot(t_det_2016, np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, label='Female')
plt.xticks([2016, 2017, 2018, 2019, 2020])
plt.xlabel('Time (years)')
plt.ylabel('Detrended Employed (count)')
plt.title('Frequency Filtered Number of Employed Canadians\nAged 15+ from 2016-2020 (Detrended)')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("det_ff_2016.pdf")
plt.show()

plt.figure(5)
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(m_ft_2016)).real / dt, label='2016-20')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(m_ft_2004)).real / dt, label='2004-08')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(m_ft_1992)).real / dt, label='1992-96')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(m_ft_1980)).real / dt, label='1980-84')
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('Time (years)')
plt.ylabel('Detrended Employed (count)')
plt.title('Frequency Filtered Employed Male Canadians\nAged 15+ (Overlaid Datasets & Detrended)')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("det_ff_m_tot.pdf")
plt.show()

plt.figure(6)
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(f_ft_2016)).real / dt, label='2016-20')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(f_ft_2004)).real / dt, label='2004-08')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(f_ft_1992)).real / dt, label='1992-96')
plt.plot(t_dur, np.fft.ifft(np.fft.ifftshift(f_ft_1980)).real / dt, label='1980-84')
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('Time (years)')
plt.ylabel('Detrended Employed (count)')
plt.title('Frequency Filtered Employed Female Canadians\nAged 15+ (Overlaid Datasets & Detrended)')
plt.legend(loc='upper center')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("det_ff_f_tot.pdf")
plt.show()

plt.figure(7)
plt.plot(t_off, ccm16_16, label='2016-20')
plt.plot(t_off, ccm16_04, label='2004-08')
plt.plot(t_off, ccm16_92, label='1992-96')
plt.plot(t_off, ccm16_80, label='1980-84')
plt.xlabel('Time offset (years)')
plt.ylabel('Cross Correlation Amplitude')
plt.title('Cross Correlations of the 2016-2020 Male Dataset with\nItself & Other Frequency Filtered Data')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.legend()
plt.savefig('cc_m_2016.pdf')
plt.show()

plt.figure(8)
plt.plot(t_off, ccf16_16, label='2016-20')
plt.plot(t_off, ccf16_04, label='2004-08')
plt.plot(t_off, ccf16_92, label='1992-96')
plt.plot(t_off, ccf16_80, label='1980-84')
plt.xlabel('Time offset (years)')
plt.ylabel('Cross Correlation Amplitude')
plt.title('Cross Correlations of the 2016-2020 Female Dataset with\nItself & Other Frequency Filtered Data')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.legend()
plt.savefig('cc_f_2016.pdf')
plt.show()
