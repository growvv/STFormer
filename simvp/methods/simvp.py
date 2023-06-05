import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter

from simvp.models import SimVP_Model
from .base_method import Base_method
import imageio

from simvp.utils import check_dir
import os
import ipdb

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()

        for batch_x, batch_y in tqdm(train_loader):
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)

            loss = self.criterion(pred_y, batch_y)
            loss.backward()
            self.model_optim.step()
            self.scheduler.step()
            
            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))

            # train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, loss_mean

    def vali_one_epoch(self, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self._predict(batch_x)
            loss = self.criterion(pred_y, batch_y)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()
                                                  ), [pred_y, batch_y], [preds_lst, trues_lst]))

            if i * batch_x.shape[0] > 1000:
                break
    
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
        
        total_loss = np.average(total_loss)

        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        return preds, trues, total_loss

    def test_one_epoch(self, test_loader, save_dir, **kwargs):
        # 输出模型的大小
        print('model size: {:.2f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        return [], [], []
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)


        save_dir = os.path.join(save_dir, 'npy')
        check_dir(save_dir)
        
        for idx, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            # ipdb.set_trace()
            with torch.no_grad():
                pred_y = self._predict(batch_x.to(self.device))  # [1, 10, 1, 256, 256]

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            

            np.save(os.path.join(save_dir, f'inputs_{idx}.npy'), batch_x.detach().cpu().numpy()) # inputs_lst: [1, 1, 10, 1, 256, 256]
            np.save(os.path.join(save_dir, f'trues_{idx}.npy'), batch_y.detach().cpu().numpy())
            np.save(os.path.join(save_dir, f'preds_{idx}.npy'), pred_y.detach().cpu().numpy())
            

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        # 
        return inputs, trues, preds
    

    def test_once(self, test_loader, epoch, save_dir):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in tqdm(test_loader):
            with torch.no_grad():
                pred_y = self._predict(batch_x.to(self.device))

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            
            break

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])


        # import ipdb
        # ipdb.set_trace()
        save_gifs = save_dir + '/gifs/'
        check_dir(save_gifs)

        # 把一系列图片保存为gif
        def save_gif(imgs, path):
            imgs = imgs.transpose(0, 2, 3, 1)
            imgs = imgs * 70
            imgs = imgs.astype(np.uint8)
            imageio.mimsave(path, imgs)

        for i in range(min(batch_x.shape[0], 4)):
            # 把inputs[i], trues[i], preds[i]拼起来放到同一排
            pics = np.concatenate([inputs[i], trues[i], preds[i]], axis=-1)
            save_gif(pics, save_gifs + f'{epoch}_{i}.gif')

    # def save_model(self, save_dir, epoch):
    #     save_dir = os.path.join(save_dir, 'checkpoints')
    #     check_dir(save_dir)
    #     save_model_path = os.path.join(save_dir, f'{epoch}.pth')
        
    #     torch.save(self.model.state_dict(), save_model_path)