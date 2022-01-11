# import os
import numpy as np
from tqdm import tqdm

# import sys
# sys.path.append("./rloss/pytorch/pytorch-deeplab_v3_plus")

# from rloss_deeplab.mypath import Path
# from dataloaders.custom_transforms import denormalizeimage
from abcd.rloss_deeplab.modeling.deeplab import *
# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
from abcd.rloss_deeplab.utils.lr_scheduler import LR_Scheduler
# from utils.saver import Saver
# from utils.summaries import TensorboardSummary
from abcd.rloss_deeplab.utils.metrics import Evaluator
# from dataloaders.utils import decode_seg_map_sequence

# from DenseCRFLoss import DenseCRFLoss

from torchvision.utils import make_grid
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
import torch
from argparse import Namespace
from utils import *

class BaseModel(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        if isinstance(hparams, dict):
            print('Converting hparam dict to namespace!')
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.evaluator = Evaluator(self.nclass)
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.hparams, **kwargs)

        # Clear start epoch if fine-tuning
        if self.hparams.ft:
            self.hparams.start_epoch = 0

        self.save_hyperparameters()

    def log(self, name, value):
      self.logger.experiment.log({name : value})

    def forward(self, x):
        pass

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.hparams.momentum, 
                                                weight_decay=self.hparams.weight_decay, 
                                                nesterov=self.hparams.nesterov)
        self.scheduler = LR_Scheduler(self.hparams.lr_scheduler, self.hparams.lr,
                                            self.hparams.epochs, self.num_img_tr)
        return self.optimizer #[self.optimizer], [self.scheduler]

    def get_loss_val(self, batch, batch_idx):
        image, target = batch['image'], batch['label']
        # global_step = batch_idx + self.num_img_tr * self.val_counter
        # croppings = (target!=254).float()
        target[target==254]=255

        output = self.forward(image)
        celoss = self.criterion(output, target)

        # flat_output = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
        #                                             dataset=self.hparams.dataset)
        # self.logger.experiment.log({'val/Image-output-target':[wandb.Image(image[0]), wandb.Image(flat_output[0]), wandb.Image(target[0])] }, commit=False)
        # img_overlay = 0.3*image[:3].clone().cpu().data + 0.7*flat_output
        # self.logger.experiment.log({'val/Overlay':wandb.Image(img_overlay)}, commit=False)
        mask = torch.max(output[:1],1)[1].detach()
        self.val_img_logs += [wb_mask(image[0].cpu().numpy().transpose([1,2,0]), mask[0].cpu().numpy(), target[0].cpu().numpy())]

        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        target = target.cpu().numpy()
        self.evaluator.add_batch(target, pred)
        result = {
          'ce_loss': celoss
        }
        return result

    def validation_summary(self, outputs):
      test_loss = 0.0
      masks = self.val_img_logs
      self.val_img_logs = []
      print(len(masks))
      for output in outputs:
        test_loss += output['ce_loss']

      # Fast test during the training
      Acc = self.evaluator.Pixel_Accuracy()
      Acc_class = self.evaluator.Pixel_Accuracy_Class()
      mIoU = self.evaluator.Mean_Intersection_over_Union()
      FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
      if len(masks)>10:
        self.logger.experiment.log({'val/Examples':masks[:50]}, commit=False)
      self.logger.experiment.log({'val/mIoU': mIoU}, commit=False)
      self.logger.experiment.log({'val/Acc': Acc}, commit=False)
      self.logger.experiment.log({'val/Acc_class': Acc_class}, commit=False)
      self.logger.experiment.log({'val/fwIoU': FWIoU}, commit=False)
      self.logger.experiment.log({'val/loss_epoch': test_loss.item()})
      print('Validation:')
      print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
      print('Loss: %.3f' % test_loss)
      self.evaluator.reset()

    def validation_epoch_end(self, validation_step_outputs):
      self.validation_summary(validation_step_outputs)

    def get_loss(self, batch, batch_idx):
        i = batch_idx
        epoch = self.current_epoch
        image, target = batch['image'], batch['label']
        target[target==254]=255
        self.scheduler(self.optimizer, i, epoch, self.best_pred)
        self.optimizer.zero_grad()
        output = self.forward(image)
        celoss = self.criterion(output, target)
        return celoss
        
    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.get_loss_val(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.get_loss_val(batch, batch_idx)

    def validation(self, val_loader, epoch=0):
        self.eval()
        self.evaluator.reset()
        tbar = tqdm(val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target==254]=255
            if image.is_cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.forward(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.log('val/total_loss_epoch', test_loss)
        self.log('val/mIoU', mIoU)
        self.log('val/Acc', Acc)
        self.log('val/Acc_class', Acc_class)
        self.log('val/fwIoU', FWIoU)
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

if __name__ == "__main__":
    base_model = BaseModel()
    import pdb;pdb.set_trace()