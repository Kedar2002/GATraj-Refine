'''
Author: Mengmeng Liu
Date: 2022/09/24
'''
from utils import *
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
class Processor():
    def __init__(self, args):
        self.args = args
        Dataloader = DataLoader_bytrajec2
        self.lr=self.args.learning_rate
        self.dataloader_gt = Dataloader(args,is_gt=True)
        model = import_class(args.model)
        self.net = model(args)
        if self.args.phase == "train":
            print("self.args.phase",self.args.phase)
            self.net.train()
        else:
            self.net.eval()
        self.init_lr = self.args.learning_rate
        self.step_ratio = self.args.step_ratio
        self.lr_step=self.args.lr_step
        if self.args.using_cuda:
            self.net=self.net.cuda()
        else:
            self.net=self.net.cpu()
        # Construct Adam after moving parameters; load_state_dict maps moments
        # onto their parameter devices while retaining host-side step counters.
        self.set_optimizer()
        self.epoch = 1
        self.load_model()
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')



    def save_model(self,epoch):
        model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
        numpy_state = np.random.get_state()
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'seed': getattr(self.args, 'seed', 1),
            'experiment_config': vars(self.args).copy(),
            'base_epoch': getattr(self.net, 'base_epoch', None),
            'rng_state': {
                'python': random.getstate(),
                'numpy': (
                    numpy_state[0],
                    torch.from_numpy(numpy_state[1].astype(np.int64)),
                    int(numpy_state[2]),
                    int(numpy_state[3]),
                    float(numpy_state[4]),
                ),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }, model_path + '.tmp')
        os.replace(model_path + '.tmp', model_path)


    def load_model(self):
        if self.args.load_model > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            if not os.path.isfile(self.args.model_save_path):
                raise FileNotFoundError('Requested checkpoint does not exist: {}'.format(self.args.model_save_path))
            print('Loading checkpoint')
            checkpoint = torch.load(self.args.model_save_path, map_location='cpu', weights_only=False)
            saved_config = checkpoint.get('experiment_config', {})
            required_keys = [
                'stage', 'base_sha256', 'validation_set',
                'hidden_size', 'x_encoder_layers', 'x_encoder_head', 'final_mode',
                'obs_length', 'pred_length', 'input_offset', 'input_mix',
                'input_position', 'mlp_decoder', 'SR', 'pass_time', 'ifGaussian',
                'test_set', 'seed', 'neighbor_thred', 'batch_size', 'val_fraction',
                'dataset', 'seq_length', 'min_obs', 'input_size', 'output_size', 'ifvalid',
            ]
            if self.args.stage == 'refiner':
                required_keys += ['endpoint_refine', 'shape_refine', 'shape_components', 'refiner_hidden',
                                  'max_endpoint_correction', 'max_shape_correction']
                if self.args.phase == 'train':
                    required_keys += ['refiner', 'endpoint_loss_weight', 'correction_penalty']
            if self.args.phase == 'train':
                required_keys += ['num_epochs', 'learning_rate', 'eta_min', 'randomRotate', 'clip']
            if not saved_config and self.args.phase == 'train':
                raise ValueError('Legacy checkpoint lacks experiment metadata; use a fresh run for a paired comparison.')
            mismatches = [key for key in required_keys if key in saved_config
                          and saved_config[key] != getattr(self.args, key)]
            if mismatches:
                raise ValueError('Checkpoint experiment differs in: ' + ', '.join(mismatches))
            model_epoch = int(checkpoint['epoch'])
            if model_epoch != self.args.load_model:
                raise ValueError('Checkpoint epoch differs from requested epoch')
            self.epoch = model_epoch + 1
            self.net.load_state_dict(checkpoint['state_dict'])
            if hasattr(self.net, 'base_epoch'):
                self.net.base_epoch = checkpoint.get('base_epoch')
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                # Legacy checkpoints contain the optimizer's saved learning
                # rate but no scheduler state. Restore the scheduler position
                # and preserve that rate for the next scheduler step.
                legacy_state = self.scheduler.state_dict()
                legacy_state['last_epoch'] = model_epoch + 1
                legacy_state['_step_count'] = model_epoch + 2
                legacy_state['_last_lr'] = [group['lr'] for group in self.optimizer.param_groups]
                self.scheduler.load_state_dict(legacy_state)
                print('Checkpoint lacks scheduler state; continuing from its saved learning rate.')
            if 'rng_state' in checkpoint:
                rng_state = checkpoint['rng_state']
                random.setstate(rng_state['python'])
                numpy_state = rng_state['numpy']
                np.random.set_state((
                    numpy_state[0],
                    numpy_state[1].cpu().numpy().astype(np.uint32, copy=False),
                    numpy_state[2],
                    numpy_state[3],
                    numpy_state[4],
                ))
                torch.set_rng_state(rng_state['torch'].cpu())
                if torch.cuda.is_available() and rng_state['cuda']:
                    # A Kaggle checkpoint can contain two GPU states while a
                    # resumed Colab runtime exposes only one GPU.
                    states = rng_state['cuda']
                    torch.cuda.set_rng_state_all(states[:torch.cuda.device_count()])
                    saved_gpu = saved_config.get('gpu', 0)
                    if self.args.using_cuda and saved_gpu < len(states):
                        torch.cuda.set_rng_state(states[saved_gpu], device=self.args.gpu)
            print('Loaded checkpoint at epoch', model_epoch)


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam((p for p in self.net.parameters() if p.requires_grad),lr=self.lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,\
        T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('Testing begin')
        test_error, test_final_error, _= self.test_epoch(self.args.load_model)
        print('Set: {}, epoch: {:.5f},test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set,self.args.load_model,test_error,test_final_error))
        with open(os.path.join(self.args.model_dir, 'metrics_{}.json'.format(self.args.load_model)), 'w') as output:
            json.dump({'test_set': self.args.test_set, 'seed': self.args.seed,
                       'epoch': self.args.load_model, 'minADE': test_error,
                       'minFDE': test_final_error, 'num_modes': self.args.final_mode,
                       'parameters': sum(p.numel() for p in self.net.parameters()),
                       'experiment': vars(self.args)}, output, indent=2)

    def playtrain(self):
        print('Training begin')
        test_error, test_final_error,first_erro_test,val_final_error,val_error,val_erro_first=0,0,0,0,0,0
        run_end_epoch = self.args.num_epochs
        epochs_per_run = getattr(self.args, 'epochs_per_run', 0)
        if epochs_per_run > 0:
            run_end_epoch = min(run_end_epoch, self.epoch + epochs_per_run - 1)
        for epoch in range(self.epoch, run_end_epoch + 1):
            print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_epoch(epoch)
            val_error, val_final_error, val_erro_first = self.val_epoch(epoch)
            self.scheduler.step()
            # if epoch == self.args.num_epochs:
            self.save_model(epoch)
            if epoch == self.args.num_epochs:
                test_error, test_final_error, first_erro_test = self.test_epoch(epoch)
            #log files
            self.log_file_curve.write(str(epoch) + ',' + str(train_loss) + ',' + str(
                val_error) + ',' + str(val_final_error) + ','+str(val_erro_first)+ ','\
                +str(test_error) + ',' + str(test_final_error) + ','+str(first_erro_test)+ '\n')

            self.log_file_curve.close()
            self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')
            #console log
            print('----epoch {} \n train_loss={:.5f}, valid_error={:.3f}, valid_final={:.3f}, valid_first={:.3f}\n\
                test_error={:.3f},test_final={:.3f},test_first={:.3f}\n'\
            .format(epoch, train_loss,val_error, val_final_error,val_erro_first,test_error,test_final_error,first_erro_test))
            model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + str(epoch) + '.tar'



    def train_epoch(self,epoch):
        """   batch_abs: the (orientated) batch
              batch_norm: the batch shifted by substracted the last position. ??? What is the impact of zeros
              shift_value: the last observed position
              seq_list: [seq_length, num_peds], mask for position with actual values at each frame for each ped
              nei_list: [seq_length, num_peds, num_peds], mask for neigbors at each frame
              nei_num: [seq_length, num_peds], neighbors at each frame for each ped
              batch_pednum: list, number of peds in each batch"""
        self.net.train()
        loss_epoch=0
        for batch in range(self.dataloader_gt.trainbatchnums):
            start = time.time()
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch,epoch)#batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            self.net.zero_grad()
            GATraj_loss, full_pre_tra = self.net.forward(inputs_fw, epoch, iftest=False)
            if GATraj_loss == 0:
                continue
            loss_epoch += GATraj_loss.item()
            GATraj_loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in self.net.parameters() if p.requires_grad), self.args.clip)
            self.optimizer.step()
            end= time.time()
            if batch%self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.\
                format(batch,self.dataloader_gt.trainbatchnums, epoch,GATraj_loss.item(), end - start))
        train_loss_epoch = loss_epoch / self.dataloader_gt.trainbatchnums


        return train_loss_epoch

    @torch.no_grad()
    def val_epoch(self,epoch):
        self.net.eval()
        error_epoch,final_error_epoch, first_erro_epoch = 0,0,0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list= [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5,1e-5,1e-5

        for batch in range(self.dataloader_gt.valbatchnums):
            if batch%100 == 0:
                print('testing batch',batch,self.dataloader_gt.valbatchnums)
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch,epoch) #batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt=tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            GATraj_loss,full_pre_tra= self.net.forward(inputs_fw, epoch, iftest=True)
            if GATraj_loss == 0:
                continue
            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch

    @torch.no_grad()
    def test_epoch(self,epoch):
        self.net.eval()
        error_epoch, final_error_epoch, first_erro_epoch = 0, 0, 0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5, 1e-5, 1e-5


        for batch in range(self.dataloader_gt.testbatchnums):
            if batch%100 == 0:
                print('testing batch',batch,self.dataloader_gt.testbatchnums)
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch,epoch)
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
            GATraj_loss,full_pre_tra = self.net.forward(inputs_fw, epoch, iftest=True)
            if GATraj_loss == 0:
                continue

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)
            first_erro_epoch += min(first_erro_epoch_list)
            final_error_epoch += min(final_error_epoch_list)
            error_epoch += min(error_epoch_list)
            error_cnt_epoch += error_cnt
            final_error_cnt_epoch += final_error_cnt
            first_erro_cnt_epoch += first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch
