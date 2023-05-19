import os
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from denoise import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, ReLU, BatchNorm1d
from sklearn.metrics import roc_curve, auc
from model import get_model
from dataset import get_data


class DrawConfusionMatrix:
    def __init__(self, labels_name):
        """

        :param num_classes: 分类数目
        """
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def draw(self):
        per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
        for i in range(self.num_classes):
            self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比

        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(len(self.labels_name)), self.labels_name)  # y轴标签
        plt.xticks(range(len(self.labels_name)), self.labels_name, rotation=45)  # x轴标签

        for x in range(len(self.labels_name)):
            for y in range(len(self.labels_name)):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('../../assets/img/cnn-mel-v2-confusion_maxtrix.png', bbox_inches='tight')
        plt.show()
        # plt.savefig("imgs/confusion_maxtrix_5")


def draw_ROC_curve(y_true, y_predict):
    # 绘制ROC曲线图
    fpr0, tpr0, thresholds0 = roc_curve(y_true[0], y_predict[0])
    fpr1, tpr1, thresholds1 = roc_curve(y_true[1], y_predict[1])
    fpr2, tpr2, thresholds2 = roc_curve(y_true[2], y_predict[2])
    fpr3, tpr3, thresholds3 = roc_curve(y_true[3], y_predict[3])
    fpr4, tpr4, thresholds4 = roc_curve(y_true[4], y_predict[4])

    roc_auc0 = auc(fpr0, tpr0)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)

    roc_aou = (roc_auc0 + roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4) / 5
    print("roc_aou ", roc_aou)

    plt.figure(figsize=(10, 8), dpi=300)
    # plt.grid(zorder=-1)
    # plt.plot(fpr, tpr, label='ROC curve', color='#1F77B4', zorder=1)

    plt.plot(fpr0, tpr0, color='#6D8CECFF', linestyle='-', linewidth=3, markerfacecolor='none',
             label=u'AS_AUC = %0.5f' % roc_auc0)
    plt.plot(fpr1, tpr1, '-', color='#756DECFF', linestyle='-', linewidth=3, markerfacecolor='none',
             label=u'MR_AUC = %0.5f' % roc_auc1)
    plt.plot(fpr2, tpr2, color='#F88E61FF', linestyle='-', linewidth=3, markerfacecolor='none',
             label=u'MS_AUC = %0.5f' % roc_auc2)
    plt.plot(fpr3, tpr3, '-', color='#F8D791FF', linestyle='-', linewidth=3, markerfacecolor='none',
             label=u'MVP_AUC = %0.5f' % roc_auc3)
    plt.plot(fpr4, tpr4, '-', color='#A0F1C0FF', linestyle='-', linewidth=3, markerfacecolor='none',
             label=u'Normal_AUC = %0.5f' % roc_auc4)

    # plt.fill_between(fpr, 0, tpr, color='#CBE3F5', zorder=1, label="AUC")  # 填充两个函数之间的区域，本例中填充(0和Y+1之间的区域)
    # plt.scatter(fpr_, tpr_, color="#E55D13", zorder=3, label="Current classifier")
    # plt.plot([0, 1], [0, 1], '--', color="#4E86EFFF")

    plt.title("CNN-Mel Secondary Classification ROC Curve")
    plt.legend()
    # plt.plot(fpr, tpr, label='ROC')
    # plt.title('ROC曲线')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def evaluate():
    weights_path = "saved_dict/cnn-mel-v1-2023-05-16-02-15-20.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = get_model()
    model_ft.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model_ft = model_ft.to(device)
    model_ft.eval()
    _, _, test_loader = get_data()

    test_label = []
    yaseen_y_predict = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_true_yaseen = [[], [], [], [], []]
    y_predict_yaseen = [[], [], [], [], []]
    normal_num = 0
    abnormal_num = 0

    for idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        print('data ', data.shape, labels.shape)
        output = model_ft(data)
        predict = torch.argmax(output, dim=-1)
        print('predict ', predict.shape, predict)
        assert len(labels) == len(predict)

        for i in range(len(labels)):
            test_label.append(labels[i])
            yaseen_y_predict.append(predict[i])
            # 0, 表示abnormal, 1 表示normal
            if labels[i] == 1:
                normal_num += 1
            else:
                abnormal_num += 1

            # for j in range(5):
            #     if test_label[i] == j:
            #         y_true_yaseen[j].append(1)
            #         y_predict_yaseen[j].append(output[j])
            #     else:
            #         y_true_yaseen[j].append(0)
            #         y_predict_yaseen[j].append(output[j])

            if (predict[i] == labels[i] and labels[i] == 1):  # 预测正确，实际是正例
                TP += 1
            elif (predict[i] == labels[i] and labels[i] != 1):  # 预测正确，实际是负例
                TN += 1
            elif (predict[i] != labels[i] and labels[i] == 1):  # 预测错误，实际是正例
                FN += 1
            elif (predict[i] != labels[i] and labels[i] != 1):  # 预测错误，实际是负例
                FP += 1

    # 首先计算数值
    # 准确率
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 灵敏度=召回率 表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力
    Sensitivity = Recall = TP / (TP + FN)
    # 精确率、精度
    Precision = TP / (TP + FP)
    # 特异度 表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力
    Specificity = TN / (TN + FP)
    # F1 Score 可知F1综合了P和R的结果，当F1较高时则能说明试验方法比较有效
    F1 = 2 * Recall * Precision / (Recall + Precision)
    # 真正率
    TPR = Sensitivity
    # 假正率
    FPR = 1 - Specificity

    print('====== cnn-mel-v1 evaluate ======')
    print('abnormal num= {}'.format(abnormal_num))
    print('normal num= {}'.format(normal_num))
    print('准确度Accuracy= {}'.format(Accuracy))
    print('召回率Recall= {}'.format(Recall))
    print('灵敏度Sensitivity={}'.format(Sensitivity))
    print('特异度Specificity={}'.format(Specificity))
    print('F1 Score= {}'.format(F1))
    print('TPR= {}'.format(TPR))
    print('FPR= {}'.format(FPR))

    # draw_ROC_curve(y_true_yaseen, y_predict_yaseen)
    #
    # 画混淆矩阵
    labels_name = ['AS', 'MR', 'MS', 'MVP', 'Normal']
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)  # 实例化
    print(yaseen_y_predict)
    print(test_label)
    drawconfusionmatrix.update(yaseen_y_predict, test_label)  # 将新批次的predict和label更新（保存）
    drawconfusionmatrix.draw()  #


if __name__ == '__main__':
    evaluate()
