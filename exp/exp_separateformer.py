from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import separateformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

from draw_picture import draw_picture

import warnings

warnings.filterwarnings('ignore')


class Exp_Separateformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Separateformer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'separateformer': separateformer
        }
        assert self.args.model == 'separateformer'
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.mix,
            self.args.separate_factor,
            self.args.step,
            self.device
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据
    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'AQ': Dataset_Custom
        }
        Data = data_dict[self.args.data]  # Dataset_ETT_hour
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            # batch_size 32 freq h
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:  # train
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        # 从csv文件中读取数据，并按照batch_size=32进行分组，每个为[32,96,512]
        # data_set :8521
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        # train 8521
        # val 2857
        # test 2857
        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,  # 8521
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(vali_loader):
            true = batch_y
            pred, _ = self._process_one_batch(vali_data, batch_x, batch_x_mark, batch_y, batch_y_mark)
            # 只选取预测部分
            pred = pred[:, self.args.label_len:, :]
            true = true[:, self.args.label_len:, :]
            if i % 20 == 0 or i == len(vali_loader) - 1:
                draw_picture(true, pred, 'vali {}th true-red pred-blue'.format(i), 'vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss2 = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_x_mark, batch_y, batch_y_mark)

                if i % 20 == 0 or i == len(train_loader) - 1:
                    # draw_picture(true, batch_x, 'train {}th true-red batch_x-blue'.format(i))
                    draw_picture(true, pred, 'train {}th true-red pred-blue'.format(i), 'train')
                #用于backward
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                # 只选取预测部分
                pred = pred[:, self.args.label_len:, :]
                true = true[:, self.args.label_len:, :]
                #用于输出计算误差
                loss2 = criterion(pred, true)
                train_loss2.append(loss2.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss2.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss2 = np.average(train_loss2)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss2, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch_x, batch_x_mark, batch_y, batch_y_mark)
            # 只选取预测部分
            pred = pred[:, self.args.label_len:, :]
            true = true[:, self.args.label_len:, :]
            if i % 20 == 0 or i == len(test_loader) - 1:
                # draw_picture(true, batch_x, 'train {}th true-red batch_x-blue'.format(i))
                draw_picture(true, pred, 'test {}th true-red pred-blue'.format(i), 'test')

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_data, batch_x, batch_x_mark, batch_y, batch_y_mark)
            # 只选取预测部分
            pred = pred[:, self.args.label_len:, :]
            true = true[:, self.args.label_len:, :]
            if i % 20 == 0 or i == len(pred_loader) - 1:
                # draw_picture(true, batch_x, 'train {}th true-red batch_x-blue'.format(i))
                draw_picture(true, pred, 'pred {}th true-red pred-blue'.format(i), 'pred')
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_x_mark, batch_y, batch_y_mark):
        """
        输入原数据以及相应的时间戳，输出经过模型后的输出和原数据
        :param dataset_object:
        :param batch_x:
        :param batch_x_mark:
        :return:
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        batch_y = batch_y.to(self.device)
        # outputs为经过预测后的输出 batch_y为原序列
        # outputs[batch_size, sequence_len, enc_in], true[batch_size, sequence_len, enc_in]
        return outputs, batch_y
