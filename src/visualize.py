import numpy as np
import matplotlib.pyplot as plt


def plot_loss_precision_task3(n_epochs, params_SGD, params_MomentumSGD, B):
    """SGDとMomentumSGDのloss・precisionの変化を可視化"""
    epoch_list = range(1, n_epochs+1)
    SGD_train_loss = []
    SGD_valid_loss = []
    SGD_train_precision = []
    SGD_valid_precision = []
    MSGD_train_loss = []
    MSGD_valid_loss = []
    MSGD_train_precision = []
    MSGD_valid_precision = []

    for params in params_SGD:
        SGD_train_loss.append(params[0])
        SGD_train_precision.append(params[1])
        SGD_valid_loss.append(params[2])
        SGD_valid_precision.append(params[3])

    for params in params_MomentumSGD:
        MSGD_train_loss.append(params[0])
        MSGD_train_precision.append(params[1])
        MSGD_valid_loss.append(params[2])
        MSGD_valid_precision.append(params[3])

    # 図を並べて表示する
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2, 2, 1) 
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_title('SGD (B: {})'.format(B), fontsize=20)
    ax1.plot(epoch_list, SGD_train_loss, color='red')
    ax1.plot(epoch_list, SGD_valid_loss, color='blue')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    ax1.legend(['train loss', 'valid loss'])
    ax1.set_xlim([0, 101])
    ax1.set_ylim([0.5, 2.3])

    ax2.set_title('MomentumSGD (B: {})'.format(B), fontsize=20)
    ax2.plot(epoch_list, MSGD_train_loss, color='red')
    ax2.plot(epoch_list, MSGD_valid_loss, color='blue')
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'loss')
    ax2.legend(['train loss', 'valid loss'])
    ax2.set_xlim([0, 101])
    ax2.set_ylim([0.5, 2.3])

    ax3.plot(epoch_list, SGD_train_precision, color='red')
    ax3.plot(epoch_list, SGD_valid_precision, color='blue')
    ax3.set_xlabel(r'epoch')
    ax3.set_ylabel(r'precision')
    ax3.legend(['train precision', 'valid precision'])
    ax3.set_xlim([0, 101])
    ax3.set_ylim([0.0, 1.0])

    ax4.plot(epoch_list, MSGD_train_precision, color='red')
    ax4.plot(epoch_list, MSGD_valid_precision, color='blue')
    ax4.set_xlabel(r'epoch')
    ax4.set_ylabel(r'precision')
    ax4.legend(['train precision', 'valid precision'])
    ax4.set_xlim([0, 101])
    ax4.set_ylim([0.0, 1.0])

    plt.savefig('../results/SGD_MomentumSGD_{}.png'.format(B), bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_loss_precision_task4(n_epochs, params_Adam, B):
    """Adamのlossとprecisionを可視化"""
    epoch_list = range(1, n_epochs+1)
    train_loss = []
    valid_loss = []
    train_precision = []
    valid_precision = []
    for params in params_Adam:
        train_loss.append(params[0])
        train_precision.append(params[1])
        valid_loss.append(params[2])
        valid_precision.append(params[3])

    # 図を並べて表示する
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(10,5))

    ax1.set_title('loss (B: {})'.format(B), fontsize=20)
    ax1.plot(epoch_list, train_loss, color='red')
    ax1.plot(epoch_list, valid_loss, color='blue')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    ax1.legend(['train loss', 'valid loss'])
    ax1.set_xlim([0, 101])
    ax1.set_ylim([0.5, 2.3])

    ax2.set_title('precision (B: {})'.format(B), fontsize=20)
    ax2.plot(epoch_list, train_precision, color='red')
    ax2.plot(epoch_list, valid_precision, color='blue')
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'precision')
    ax2.legend(['train precision', 'valid precision'])
    ax2.set_xlim([0, 101])
    ax2.set_ylim([0.0, 1.0])

    plt.savefig('../results/Adam_{}.png'.format(B), bbox_inches='tight', pad_inches=0)
    plt.show()