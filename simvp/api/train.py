# Copyright (c) CAIRI AI Lab. All rights reserved

import time
import logging
import pickle
import json
import torch
import numpy as np
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table
import ipdb 
from tqdm import tqdm

from simvp.core import metric, Recorder
from simvp.methods import method_maps
from simvp.utils import (set_seed, print_log, output_namespace, check_dir,
                         get_dataset)

from torch.utils.tensorboard import SummaryWriter

import imageio

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


class NodDistExperiment(object):
    """ Experiment with non-dist PyTorch training and evaluation """

    def __init__(self, args):
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.args.method = self.args.method.lower()

        # ipdb.set_trace()
        self._preparation()
        print_log(output_namespace(self.args))

        T, C, H, W = self.args.in_shape
        if self.args.method == 'simvp':
            _tmp_input = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            flops = FlopCountAnalysis(self.method.model, _tmp_input)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        # print_log(self.method.model)
        # print_log(flop_count_table(flops))

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:', device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.image_path = osp.join(self.args.res_dir, 'images')
        check_dir(self.image_path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.path, 'train_{}.log'.format(timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
        
        
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

        # load weights
        if self.args.load:
            self._load()
        

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.vali_loader, self.test_loader = get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self):
        print('Loading weights from {} in epoch {}'.format(self.args.weight_path, self.args.pretrain_epoch))
        # self.method.model.load_state_dict(torch.load(self.args.weight_path + '.pth'))
        # fw = open(self.args.weight_path + '.pkl', 'rb')
        self.method.model.load_state_dict(torch.load(osp.join(self.args.weight_path, str(self.args.pretrain_epoch) + '.pth')))
        fw = open(osp.join(self.args.weight_path, str(self.args.pretrain_epoch) + '.pkl'), 'rb')
        
        
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(verbose=True)
        summary_writer = SummaryWriter(self.args.res_dir + 'tensorboard')
        num_updates = 0
        # constants for other methods:
        eta = 1.0  # PredRNN
        for epoch in tqdm(range(self.args.pretrain_epoch+1, self.args.pretrain_epoch+1+self.config['epoch'])):
            loss_mean = 0.0

            # self.test()

            if self.args.method in ['simvp', 'crevnet', 'phydnet']:
                num_updates, loss_mean = self.method.train_one_epoch(
                    self.train_loader, epoch, num_updates, loss_mean)
            else:
                raise ValueError(f'Invalid method name {self.args.method}')

            summary_writer.add_scalar('train_loss', loss_mean, epoch)

            # 保存结果
            self.method.test_once(self.vali_loader, epoch, self.args.res_dir)
            
            if epoch % self.args.save_step == 0:
                self._save(str(epoch))
            # self._save(str(epoch))



    def vali(self, vali_loader):
        preds, trues, val_loss = self.method.vali_one_epoch(self.vali_loader)

        mae, mse = metric(
            preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, return_ssim_psnr=False)
        print_log('val\t mse:{}, mae:{}'.format(mse, mae))
        if has_nni:
            nni.report_intermediate_result(mse)

        return val_loss

    def test(self):
        inputs, trues, preds = self.method.test_one_epoch(self.test_loader, self.args.res_dir)
        # print("output: ", inputs.shape, trues.shape, preds.shape)
        print("test end")

        print(inputs.shape, trues.shape, preds.shape)
        mae, mse, ssim, psnr, sharp = metric(
            preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, return_ssim_psnr=True)
        metrics = np.array([mae, mse,ssim, psnr, sharp])
        print_log('mse:{}, mae:{}, ssim:{}, psnr:{}, sharp: {}'.format(mse, mae, ssim, psnr, sharp))
        
