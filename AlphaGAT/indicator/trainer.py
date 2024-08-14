import time

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from indicator.batchloss import Batch_Loss
from indicator.config import Config
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from indicator.model.IMM import IMM
from indicator.model.MyTimeMixer import TimeMix
from indicator.model.TCN import TCN
from indicator.model.Transformer import TF
from indicator.model.iTransformer import ITF
from indicator.rubishCode.StockModern import MODATN

from indicator.utils import get_loaders, get_logger


class  Trainer():
    def __init__(self, config:Config):
        self.learning_rate =config.learning_rate
        self.max_epochs = config.epochs
        self.device = config.device

        self.model_dict = {
            'IMM': IMM,
            'ITF': ITF,
            'TCN': TCN,
            'TF': TF,
            'MODATN':  MODATN,
            'TimeMix': TimeMix
        }

        if config.model in self.model_dict:
            cls = globals()[config.model]
            self.model = cls(config).to(self.device).float()
            # self.model = self.model_dict[config.model](config).float()


        else:
            raise NotImplementedError ("No model found")


        #self.model = IMM(config).to(self.device)


        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate)
        self.scheduler = Scheduler(self.optimizer, milestones=[20,50,100], gamma=0.1)

        self.clip_grad_norm = config.clip_grad_norm
        self.batch_loss = Batch_Loss(config)
        self.tensorboard = SummaryWriter(f"{config.cwd}/tensorboard")

        self.train_dataloader,  self.valid_dataloader, self.test_dataloader = get_loaders(config=config)

        self.logger = get_logger(config.log_file)

        self.epochs = 1
        self.best_val = -np.inf

        '''model save parameter'''
        self.if_over_write = config.if_over_write
        self.save_gap = config.save_gap
        self.save_counter = 0
        self.cwd = config.cwd
        self._logging_parameter(config)







    def run(self):

        epoch_train_loss = []
        epoch_train_loss_IC = []
        epoch_train_loss_CV = []
        epoch_valid_loss = []
        epoch_valid_loss_IC= []
        epoch_valid_loss_CV = []
        epoch_test_loss = []
        epoch_test_loss_IC = []
        epoch_test_loss_CV = []
        epoch_index = []
        for epoch in range(self.epochs, self.max_epochs+1):
            self.logger.info('=================================================================')
            start_time = time.time()

            # LR Decay

            ###Training
            train_loss, valid_loss, test_loss = self._train_one_epoch()
            self.scheduler.step()
            epoch_train_loss.append(train_loss[0])
            epoch_train_loss_IC.append(train_loss[1])
            epoch_train_loss_CV.append(train_loss[2])

            epoch_valid_loss.append(valid_loss[0])
            epoch_valid_loss_IC.append(valid_loss[1])
            epoch_valid_loss_CV.append(valid_loss[2])

            epoch_test_loss.append(test_loss[0])
            epoch_test_loss_IC.append(test_loss[1])
            epoch_test_loss_CV.append(test_loss[2])

            epoch_index.append(epoch)
            end_time = time.time()
            elapsed_time = start_time - end_time
            self.logger.info("Epoch {:3d}/{:3d}: Time: {:.2f}]".format(
                epoch, self.max_epochs, elapsed_time / 60))

            ####save model
            prev_max_val = self.best_val
            self.best_val = max(self.best_val, test_loss[1])

            if_save = test_loss[1] > prev_max_val
            self.save_counter += 1
            model_path = None
            if if_save:
                if self.if_over_write:
                    model_path = f"{self.cwd}/model.pt"
                else:
                    model_path = f"{self.cwd}/model__{self.epochs:04}_{self.best_val:.4f}.pt"
            elif self.save_counter >= self.save_gap:
                self.save_counter =0
                # if self.if_over_write:
                #     model_path = f"{self.cwd}/model.pt"
                # else:
                model_path = f"{self.cwd}/model__{self.epochs:04}_{self.best_val:.4f}.pt"

            if model_path:
                torch.save(self.model, model_path)


        self.logger.info("PLOT THE RESULT")
        fig_path =f"{self.cwd}/loss_result.png"
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_index, epoch_train_loss, label="train")
        plt.plot(epoch_index, epoch_valid_loss, label="valid")
        plt.plot(epoch_index, epoch_test_loss, label="test")
        plt.legend()
        plt.savefig(fig_path)
        plt.clf()

        fig_path =f"{self.cwd}/lossIC_result.png"
        plt.xlabel("Epoch")
        plt.ylabel("LossIC")
        plt.plot(epoch_index, epoch_train_loss_IC, label="train")
        plt.plot(epoch_index, epoch_valid_loss_IC, label="valid")
        plt.plot(epoch_index, epoch_test_loss_IC, label="test")
        plt.legend()
        plt.savefig(fig_path)
        plt.clf()

        fig_path =f"{self.cwd}/lossCV_result.png"
        plt.xlabel("Epoch")
        plt.ylabel("LossCV")
        plt.plot(epoch_index, epoch_train_loss_CV, label="train")
        plt.plot(epoch_index, epoch_valid_loss_CV, label="valid")
        plt.plot(epoch_index, epoch_test_loss_CV, label="test")
        plt.legend()
        plt.savefig(fig_path)
        plt.clf()



        plt.close('all')
        self.logger.info("Train End!!!!!!")
        torch.save(self.model,f"{self.cwd}/model_end.pt")




    def _logging_parameter(self, config:Config):
        self.logger.info('Start Training, the hyperparameters are shown below:')
        self.logger.info("*************************************************")
        self.logger.info('--save_dir:   {}'.format(config.cwd))
        self.logger.info('--batch_size:   {}'.format(config.batch_size))
        self.logger.info('--day_windows:   {}'.format(config.day_windows))
        self.logger.info('--epochs:   {}'.format(config.epochs))
        self.logger.info('--learning_rate:   {}'.format(config.learning_rate))
        self.logger.info('--gpu_id:   {}'.format(config.device))
        self.logger.info('--drop:   {}'.format( config.drop))
        self.logger.info('--alphas:   {}'.format(config.alphas))
        self.logger.info('--model:   {}'.format(config.model))
        self.logger.info('--use_cov:   {}'.format(config.use_cov))
        self.logger.info('--factor_cov:   {}'.format(config.factor_cov))
        self.logger.info('--use_attention:   {}'.format(config.use_attention))



    def _train_one_epoch(self):
        self.model.train()
        train_epoch_loss = []
        train_epoch_loss_IC = []
        train_epoch_loss_CV = []
        for idx, (data, targets) in enumerate(tqdm(self.train_dataloader)):
            # data = data.to(self.device)
            # targets = targets.to(self.device)
            preds = self.model(data)

            ####loss.shape == [batch, num_indicator]
            loss, IC_loss, CV_loss = self.batch_loss(preds,targets)
            assert torch.isfinite(loss).all(), "Loss is NaN or infinite!"
            # loss = torch.abs(loss).mean()
            ###minize
            loss = -loss
            self.optimizer_update(loss)
            train_epoch_loss.append(-loss.cpu().detach().numpy())
            train_epoch_loss_IC.append(IC_loss.cpu().detach().numpy())
            train_epoch_loss_CV.append(CV_loss.cpu().detach().numpy())

        mean_train_loss = np.stack(train_epoch_loss).mean()
        mean_train_loss_IC = np.stack(train_epoch_loss_IC).mean()
        mean_train_loss_CV = np.stack(train_epoch_loss_CV).mean()


        with torch.no_grad():
            self.model.eval()
            val_epoch_loss = []
            val_epoch_loss_IC = []
            val_epoch_loss_CV = []
            test_epoch_loss = []
            test_epoch_loss_IC = []
            test_epoch_loss_CV = []


            for idx, (data, targets) in enumerate(tqdm(self.valid_dataloader)):
                # data = data.to(self.device)
                # targets = targets.to(self.device)
                preds = self.model(data)

                ####loss.shape == [batch, num_indicator]
                loss, IC_loss, CV_loss  = self.batch_loss(preds, targets)

                val_epoch_loss.append(loss.cpu().detach().numpy())
                val_epoch_loss_IC.append(IC_loss.cpu().detach().numpy())
                val_epoch_loss_CV.append(CV_loss.cpu().detach().numpy())
            mean_val_loss = np.stack(val_epoch_loss).mean()
            mean_val_loss_IC = np.stack(val_epoch_loss_IC).mean()
            mean_val_loss_CV = np.stack(val_epoch_loss_CV).mean()


            for idx, (data, targets) in enumerate(tqdm(self.test_dataloader)):
                # data = data.to(self.device)
                # targets = targets.to(self.device)
                preds = self.model(data)

                ####loss.shape == [batch, num_indicator]
                loss, IC_loss, CV_loss = self.batch_loss(preds, targets)

                test_epoch_loss.append(loss.cpu().detach().numpy())
                test_epoch_loss_IC.append(IC_loss.cpu().detach().numpy())
                test_epoch_loss_CV.append(CV_loss.cpu().detach().numpy())
            mean_test_loss = np.stack(test_epoch_loss).mean()
            mean_test_loss_IC = np.stack(test_epoch_loss_IC).mean()
            mean_test_loss_CV = np.stack(test_epoch_loss_CV).mean()
        self.logger.info('Epoch {:3d}: Train Loss: {:.4f},  IC_loss: {:.4f}, CV_Loss : {:.6f}\n'
                         ' Val Loss: {:.4f},  IC_loss: {:.4f}, CV_Loss : {:.6f}\n '
                         'Test Loss: {:.4f},   IC_loss: {:.4f}, CV_Loss : {:.6f}'
                         .format(self.epochs, mean_train_loss, mean_train_loss_IC, mean_train_loss_CV,
                                 mean_val_loss, mean_val_loss_IC, mean_val_loss_CV,
                                 mean_test_loss, mean_test_loss_IC, mean_test_loss_CV))

        self.tensorboard.add_scalar("train", mean_train_loss)
        self.tensorboard.add_scalar("valid", mean_val_loss)
        self.tensorboard.add_scalar("test", mean_test_loss)
        self.epochs += 1

        return [mean_train_loss, mean_train_loss_IC, mean_train_loss_CV], [mean_val_loss, mean_val_loss_IC, mean_val_loss_CV],\
               [mean_test_loss, mean_test_loss_IC, mean_test_loss_CV]

    def optimizer_update(self,  objective: Tensor):
        self.optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=self.optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        self.optimizer.step()






