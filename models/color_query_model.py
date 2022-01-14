# import os
import numpy as np
from tqdm import tqdm

from utils import *

import sys
sys.path.append("./rloss/pytorch/pytorch-deeplab_v3_plus")
# from mypath import Path
# from dataloaders.custom_transforms import denormalizeimage
from modeling.deeplab import *
# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
# from utils.saver import Saver
# from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
# from dataloaders.utils import decode_seg_map_sequence

# from DenseCRFLoss import DenseCRFLoss

from torchvision.utils import make_grid
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
import torch
from argparse import Namespace


class BaseModel(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.evaluator = Evaluator(self.hparams.nclass)
        self.best_pred = 0.0
        self.val_img_logs = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        if self.hparams.ft:
            self.hparams.start_epoch = 0
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.hparams.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.hparams.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.hparams.momentum, 
                                                weight_decay=self.hparams.weight_decay, 
                                                nesterov=self.hparams.nesterov)
        self.scheduler = LR_Scheduler(self.hparams.lr_scheduler, self.hparams.lr,
                                            self.hparams.epochs, self.hparams.num_img_tr)
        return self.optimizer #[self.optimizer], [self.scheduler]

    def get_loss_val(self, batch, batch_idx):
        image, target = batch['image'], batch['label']
        target[target==254]=255
        output = self.forward(image)
        celoss = self.criterion(output, target.long())
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
      for output in outputs:
        test_loss += output['ce_loss']

      # Fast test during the training
      Acc = self.evaluator.Pixel_Accuracy()
      Acc_class = self.evaluator.Pixel_Accuracy_Class()
      mIoU = self.evaluator.Mean_Intersection_over_Union()
      FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
      if len(masks)>10:
        m = self.hparams.get('visualize_num_examples') or 10
        self.logger.experiment.log({'val/Examples':masks[:m]}, commit=False)
      self.logger.experiment.log({'val/mIoU': mIoU}, commit=False)
      self.logger.experiment.log({'val/Acc': Acc}, commit=False)
      self.logger.experiment.log({'val/Acc_class': Acc_class}, commit=False)
      self.logger.experiment.log({'val/fwIoU': FWIoU}, commit=False)
      self.logger.experiment.log({'val/loss_epoch': test_loss.item()})
      # print('Validation:')
      # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
      # print('Loss: %.3f' % test_loss)
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
        celoss = self.criterion(output, target.long())
        return celoss
        
    # def log(self, name, value):
    #   wandb.log({name: value})

    def training_step(self, batch, batch_idx):
        ce_loss = self.get_loss(batch, batch_idx)
        self.log('train/ce', ce_loss.item())
        return ce_loss
    
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


import wandb
def to_img(img_tensor, **kwargs):
  return wandb.Image((img_tensor.cpu().detach().numpy() * 255).astype(np.uint8), **kwargs)
log_num = 0

# TODO: Make this a layer in a sequence of decoding layers
class ColorDecoder(nn.Module):
  def __init__(self, num_classes=21, feature_dim=256):
    super().__init__()
    self.num_classes = num_classes
    self.feature_dim = feature_dim
    self.softmax = nn.Softmax(dim=1)
    self.coarse_cls = nn.Conv2d(feature_dim, num_classes, kernel_size=1, stride=1)
    
  def log(self, d, commit=False):
    if log_num % 10 == 0:
      wandb.log(d, commit=commit)

  def forward(self, feature_map, x, low=None):
    image = x
    x = x.permute(0,2,3,1) # bs, 3, h, w -> bs, h, w, 3
    if low is not None:
      low = F.interpolate(low, size=image.size()[2:], mode='bilinear', align_corners=True)
      x = torch.concat([x, low], dim=-1)
    
    x_dim = x.size(-1)
    logits_map = self.coarse_cls(feature_map) # c
    coarse_segments = self.softmax(logits_map) # bs, h, w, num_classes, dim(x) = bs, h, w, 3
    coarse_segments = F.interpolate(coarse_segments, size=image.size()[2:], mode='bilinear', align_corners=True) # s, dim(s) = bs, h, w, num_classes
    # sanity check
    mask = torch.max(coarse_segments[:1],1)[1].detach()
    mask = wandb.Image(image[0].cpu().numpy().transpose([1,2,0]), masks={
      "prediction" : {"mask_data" : mask[0].cpu().numpy(), "class_labels" : labels()}})
    self.log({'debug/coarse_segments': mask}, commit=False)

    image_segments_masked =  [  x * coarse_segments[::,i].unsqueeze(-1).expand(-1,-1,-1,x_dim) for i in range(self.num_classes) ] # num_classes x (bs, h, w, 3)
    # sanity check
    self.log({'debug/image_segments_masked': to_img(image_segments_masked[0][0])}, commit=False)

    q = [ torch.mean(s, dim=(1,2)) for s in image_segments_masked ] # mean color of the segment, num_classes x (bs, 3)
    # sanity check
    self.log({'debug/query': [to_img(q[i][0].unsqueeze(0).unsqueeze(0).expand(50, 50,-1), caption=segmentation_classes[i]) for i in range(self.num_classes)]}, commit=False)

    attn_maps = [ torch.sum(x * q[i].unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), x.size(2), -1), dim=-1) for i in range(self.num_classes) ] # num_classes x  (bs, h, w)
    # sanity check
    self.log({'debug/attn_maps': [to_img(attn_maps[i][0], caption=segmentation_classes[i]) for i in range(self.num_classes)] }, commit=False)

    segments_by_color = torch.cat([a.unsqueeze(1) for a in attn_maps],  dim=1) # bs, num_classes, h, w
    # sanity check
    mask = torch.max(segments_by_color[:1],1)[1].detach()
    mask = wandb.Image(image[0].cpu().numpy().transpose([1,2,0]), masks={
      "prediction" : {"mask_data" : mask[0].cpu().numpy(), "class_labels" : labels()}})
    self.log({'debug/finer_segments': mask}, commit=False)

    global log_num
    log_num += 1
    return segments_by_color


class SimpleColorDecoder(nn.Module):
  def __init__(self, num_classes=21, feature_dim=256):
    super().__init__()
    self.num_classes = num_classes
    self.feature_dim = feature_dim
    self.softmax = nn.Softmax(dim=1)
    self.coarse_cls = nn.Conv2d(feature_dim, num_classes, kernel_size=1, stride=1)

  def forward(self, feature_map, x):
    logits_map = self.coarse_cls(feature_map) # c
    coarse_segments = self.softmax(logits_map) # bs, h, w, num_classes, dim(x) = bs, h, w, 3
    coarse_segments = F.interpolate(coarse_segments, size=x.size()[2:], mode='bilinear', align_corners=True) # s, dim(s) = bs, h, w, num_classes
    x = x.permute(0,2,3,1) # bs, 3, h, w -> bs, h, w, 3
    image_segments_masked =  [  x * coarse_segments[::,i].unsqueeze(-1).expand(-1,-1,-1,3) for i in range(self.num_classes) ] # num_classes x (bs, h, w, 3)
    q = [ torch.mean(s, dim=(1,2)) for s in image_segments_masked ] # mean color of the segment, num_classes x (bs, 3)
    attn_maps = [ torch.sum(x * q[i].unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), x.size(2), -1), dim=-1) for i in range(self.num_classes) ] # num_classes x  (bs, h, w)
    segments_by_color = torch.cat([a.unsqueeze(1) for a in attn_maps],  dim=1) # bs, num_classes, h, w
    return segments_by_color


class ColorModel(BaseModel):
    def __init__(self, hparams, encoder=None):
        super().__init__(hparams)
        if encoder is None:
          model = DeepLab(num_classes=self.hparams.nclass,
                            backbone=self.hparams.backbone,
                            output_stride=self.hparams.out_stride,
                            sync_bn=self.hparams.sync_bn,
                            freeze_bn=self.hparams.freeze_bn)
          encoder = model.backbone
        self.encoder = encoder
        self.decoder = ColorDecoder(num_classes=self.hparams.nclass, feature_dim=320) # resnet feature map dim
        
    def forward(self, x):
      feature_map, _ = self.encoder(x)
      return self.decoder(feature_map, x)

    def configure_optimizers(self):
        train_params = [{'params': self.get_1x_lr_params(), 'lr': self.hparams.lr},
                        {'params': self.get_10x_lr_params(), 'lr': self.hparams.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.hparams.momentum, 
                                                weight_decay=self.hparams.weight_decay, 
                                                nesterov=self.hparams.nesterov)
        self.scheduler = LR_Scheduler(self.hparams.lr_scheduler, self.hparams.lr,
                                            self.hparams.epochs, self.hparams.num_img_tr)
        return self.optimizer

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

