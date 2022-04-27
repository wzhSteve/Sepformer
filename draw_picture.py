import matplotlib.pyplot as plt
import numpy as np
import torch
plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False#显示负号

def draw_picture(true, pred, str, dic):
    """
    :param true: 真实值
    :param pred: 预测值
    :param str: 标题
    :return:
    """
    plt.clf()
    true = true.cpu()
    pred = pred.cpu()
    # sequence_len = 50
    sequence_len = true.shape[1]
    # y1 = true[0, :50, 0].view(-1)
    # y2 = pred[0, :50, 1].view(-1)
    y1 = true[0, :sequence_len, 0].view(-1)
    y2 = pred[0, :sequence_len, 0].view(-1)
    y1 = y1.detach().numpy()
    y2 = y2.detach().numpy()
    x = [i for i in range(sequence_len)]
    plt.figure(figsize=(10,5))
    #plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()

    plt.plot(x, y1, color='red',label="true",linewidth=1.5)
    plt.plot(x, y2, color="blue",label="pred",linewidth=1.5)
    #plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks([])
    #plt.title("true pred", fontsize=12, fontweight='bold')  # 默认字体大小为12
    #plt.xlabel("pred_len", fontsize=13, fontweight='bold')
    #plt.ylabel("MSE", fontsize=13, fontweight='bold')
    #plt.xlim(0, 3)  # 设置x轴的范围
    #plt.ylim(0.2, 1.2)
    plt.legend(loc=4, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()

    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('figure/' + dic + '/' +str + '.jpg')
    plt.cla()
    #plt.show()

# true = torch.ones((32,10,4))
# pred = torch.rand((32, 10, 4))
# draw_picture(true, pred, 'nidie')