import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy import signal

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

def fft_ifft_picture(input):
    input = input.cpu()
    #sequence_len为采样频率
    sequence_len = input.shape[1]
    y = input[0, :sequence_len, 0].view(-1)
    y = y.detach().numpy()
    x = [i for i in range(sequence_len)]
    #原始波形
    plt.figure()
    plt.plot(x, y)
    plt.title('原始波形')
    #fft
    fft_y = fft(y)
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    plt.figure()
    plt.plot(x, abs_y)
    plt.title('双边振幅谱（未归一化）')
    #ifft
    ifft_y = ifft(fft_y)
    plt.figure()
    plt.plot(x, ifft_y)
    plt.title('从频域经过ifft变回时域')
    plt.show()
def apart_frequence(cnt, count, input):
    """

    :param cnt: encoder的层数
    :param count: 第几层
    :param input:
    :return:
    """
    input = input.cpu()
    # sequence_len为采样频率
    fs = input.shape[1]
    interval = fs//cnt
    y = input[0, :fs, 0].view(-1)
    y = y.detach().numpy()

    fl = count * interval
    fh = fl + interval
    #滤波
    wn1 = 2 * fl / fs
    wn2 = 2 * fh / fs
    if count==0:
        b, a = signal.butter(8, wn2, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
        y_out = signal.filtfilt(b, a, y)  # data为要过滤的信号
    elif count==cnt-1:
        b, a = signal.butter(8, wn1, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
        y_out = signal.filtfilt(b, a, y)  # data为要过滤的信号
    else:
        b, a = signal.butter(8, [wn1,wn2], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
        y_out = signal.filtfilt(b, a, y)  #data为要过滤的信号
    return y_out#滤波后的序列


# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
# x = np.linspace(0, 1, 1400)
#
# # 设置需要采样的信号，频率分量有200，400和600
# y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)
# plt.figure()
# plt.plot(x, y)
# plt.title('原始波形')
#
# fft_y=fft(y)                          #快速傅里叶变换
# print(len(fft_y))
# print(fft_y[0:5])
# N = 1400
# x = np.arange(N)  # 频率个数
#
# abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
#
# # plt.figure()
# # plt.plot(x, abs_y)
# # plt.title('双边振幅谱（未归一化）')
#
# ifft_y = ifft(fft_y)
# plt.figure()
# plt.plot(x, ifft_y)
# plt.title('双边频谱ifft',fontsize=9, color='blue')
#
# fs = 1400
# fl = 500
# fh = 650
# wn1 = 2*fl/fs
# wn2 = 2*fh/fs
# b, a = signal.butter(8, wn1, 'highpass')   #配置滤波器 8 表示滤波器的阶数
# y_out = signal.filtfilt(b, a, y)  #data为要过滤的信号
#
# plt.figure()
# plt.plot(x, y_out)
# plt.title('带阻双边振幅谱（未归一化）')
#
# yout_fft = fft(y_out)
# abs_yout = np.abs(yout_fft)
# plt.figure()
# plt.plot(x, abs_yout)
# plt.title('双边振幅谱（未归一化）')
# plt.show()
#
# ifft_y = ifft(yout_fft)
# plt.figure()
# plt.plot(x, ifft_y)
# plt.title('双边频谱ifft',fontsize=9, color='blue')
# plt.show()

# ifft_half_y = ifft(fft_y[range(int(N/2))])
# half_x = x[range(int(N/2))]
# plt.figure()
# plt.plot(half_x,ifft_half_y)
# plt.title('单边频谱ifft', fontsize=9, color='red')



# normalization_y=abs_y/N            #归一化处理（双边频谱）
# plt.figure()
# plt.plot(x,normalization_y,'g')
# plt.title('双边频谱(归一化)',fontsize=9,color='green')
# plt.show()
#
# half_x = x[range(int(N/2))]                                  #取一半区间
# normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
# plt.figure()
# plt.plot(half_x,normalization_half_y,'b')
# plt.title('单边频谱(归一化)',fontsize=9,color='blue')
# plt.show()



