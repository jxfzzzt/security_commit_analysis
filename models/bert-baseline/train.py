"""
训练网络模型
tensorboard --logdir=./logs_train
"""
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import torch
from torch import nn
from model import SecurityCommitModel
from dataset import get_loader
from config import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterion, optimizer, scheduler, train_loader, valid_loader, device):
    model.to(device)
    # 绘图tensorboard全局变量
    avg_train_acc_list = []
    avg_train_loss_list = []

    global_train_acc_list = []
    global_train_loss_list = []

    avg_valid_acc_list = []
    avg_valid_loss_list = []

    global_valid_acc_list = []
    global_valid_loss_list = []

    best_val_loss = float('inf')
    best_val_acc = 0

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    print('begin training ... ')
    for cur_epoch in range(epoch):
        print("----------------- Epoch ", cur_epoch + 1, " -----------------")
        print("               Training phase")

        # 模型训练阶段
        model.train()
        total_train_loss = 0.
        total_train_acc = 0.
        step = 0
        correct_nums = 0
        for iter_num, (message, code, labels) in enumerate(train_loader):
            message_input_ids, message_mask = message['input_ids'].to(device), message['attention_mask'].to(device)
            code_input_ids, code_mask = code['input_ids'].to(device), code['attention_mask'].to(device)
            outputs = model(message_input_ids, message_mask, code_input_ids, code_mask)

            _, predicted = torch.max(outputs.data, 1)
            # print('train predicted : ', predicted)
            loss = criterion(outputs, labels)
            cur_correct_nums = (outputs.argmax(dim=1) == labels).sum().cpu().item()
            correct_nums += cur_correct_nums
            acc = cur_correct_nums / len(labels)
            total_train_acc += acc
            total_train_loss += loss.item() * labels.size(0)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(
                    'Epoch [{}/{}], step [{}/{}], train acc: {}, train loss: {}'.format(cur_epoch + 1, epoch, step + 1,
                                                                                        len(train_loader),
                                                                                        '%.4f' % acc, '%.4f' % loss))

        # 平均数据
        avg_train_acc = total_train_acc / step
        avg_train_loss = total_train_loss / step

        avg_train_acc_list.append(avg_train_acc)
        avg_train_loss_list.append(avg_train_loss)

        global_train_acc = correct_nums / len(train_loader.dataset)
        global_train_acc_list.append(global_train_acc)

        global_train_loss_list.append(total_train_loss)

        writer.add_scalar("train_loss", total_train_loss, cur_epoch)
        writer.add_scalar("train_accuracy", global_train_acc, cur_epoch)

        print("Epoch [{}/{}], avg train acc: {}, global train acc: {}, avg train loss: {}".format(cur_epoch + 1,
                                                                                                  epoch, avg_train_acc,
                                                                                                  global_train_acc,
                                                                                                  avg_train_loss))

        print("============== Validating phase ==============")
        # 模型验证阶段
        with torch.no_grad():
            model.eval()
            # 计算全局值
            correct = 0
            total = 0
            val_loss = 0
            step = 0

            total_valid_acc = 0
            for message, code, labels in valid_loader:
                message_input_ids, message_mask = message['input_ids'].to(device), message['attention_mask'].to(device)
                code_input_ids, code_mask = code['input_ids'].to(device), code['attention_mask'].to(device)
                outputs = model(message_input_ids, message_mask, code_input_ids, code_mask)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                # print('valid predicted : ', predicted)

                total += labels.size(0)
                cur_correct = (predicted == labels).sum().item()
                cur_acc = cur_correct / len(labels)
                total_valid_acc += cur_acc
                correct += cur_correct
                step += 1

            avg_valid_acc = total_valid_acc / step
            avg_valid_acc_list.append(avg_valid_acc)

            avg_valid_loss = val_loss / step
            avg_valid_loss_list.append(avg_valid_loss)

            global_valid_acc = correct / total
            global_valid_acc_list.append(global_valid_acc)
            global_valid_loss_list.append(val_loss)
            writer.add_scalar("valid_loss", val_loss, cur_epoch)
            writer.add_scalar("valid_accuracy", global_valid_acc, cur_epoch)

            print("Epoch [{}/{}], avg valid acc: {}, global valid acc: {}, avg valid loss: {}".format(cur_epoch + 1,
                                                                                                      epoch,
                                                                                                      avg_valid_acc,
                                                                                                      global_valid_acc,
                                                                                                      avg_valid_loss))

        current_epoch_valid_acc = global_valid_acc_list[-1]
        if current_epoch_valid_acc > best_val_acc:
            print("Validation Acc increased ({:.6f} -> {:.6f}). Saving model ... ".format(best_val_acc,
                                                                                          current_epoch_valid_acc))

            torch.save(model.state_dict(), save_path)
            best_val_acc = current_epoch_valid_acc
            # best_val_loss = single_valid_loss

        scheduler.step()
        print("epoch ", epoch, " lr ", scheduler.get_last_lr())

    writer.close()

    if not os.path.exists(metrics_path):
        os.mkdir(metrics_path)
    avg_train_acc_df = pd.DataFrame(avg_train_acc_list)
    avg_train_acc_df.to_csv(os.path.join(metrics_path, "avg_train_acc.csv"), index=False, header=False)

    avg_train_loss_df = pd.DataFrame(avg_train_loss_list)
    avg_train_loss_df.to_csv(os.path.join(metrics_path, "avg_train_loss.csv"), index=False, header=False)

    global_train_acc_df = pd.DataFrame(global_train_acc_list)
    global_train_acc_df.to_csv(os.path.join(metrics_path, "global_train_acc.csv"), index=False, header=False)

    global_train_loss_df = pd.DataFrame(global_train_loss_list)
    global_train_loss_df.to_csv(os.path.join(metrics_path, "global_train_loss.csv"), index=False, header=False)

    # valid 评估数据
    avg_valid_acc_df = pd.DataFrame(avg_valid_acc_list)
    avg_valid_acc_df.to_csv(os.path.join(metrics_path, "avg_valid_acc.csv"), index=False, header=False)

    avg_valid_loss_df = pd.DataFrame(avg_valid_loss_list)
    avg_valid_loss_df.to_csv(os.path.join(metrics_path, "avg_valid_loss.csv"), index=False, header=False)

    global_valid_acc_df = pd.DataFrame(global_valid_acc_list)
    global_valid_acc_df.to_csv(os.path.join(metrics_path, "global_valid_acc.csv"), index=False, header=False)

    global_valid_loss_df = pd.DataFrame(global_valid_loss_list)
    global_valid_loss_df.to_csv(os.path.join(metrics_path, "global_valid_loss.csv"), index=False, header=False)


def evaluate(model, test_loader, device):
    model.to(device)
    print("               Evaluating phase")


if __name__ == '__main__':
    print(torch.__version__)
    print('DEVICE = {}'.format(device))

    train_loader, valid_loader, test_loader = get_loader()

    print('get data success ... ')
    if not os.path.exists('./saved_dict'):
        os.mkdir('./saved_dict')
    model = SecurityCommitModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train(model, criterion, optimizer, scheduler, train_loader, valid_loader, device)

    # evaluate(model, test_loader, device)
