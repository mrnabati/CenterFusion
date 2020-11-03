from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from test import prefetch_test
import json

def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  lr = opt.lr

  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  if opt.val_intervals < opt.num_epochs or opt.eval:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.val_split), batch_size=1, shuffle=False, 
              num_workers=1, pin_memory=True)

    if opt.eval:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir, n_plots=opt.eval_n_plots, 
                                  render_curves=opt.eval_render_curves)
      return

  print('Setting up train data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, opt.train_split), batch_size=opt.batch_size, 
        shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=True
  )

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'

    # log learning rate
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
      logger.scalar_summary('LR', lr, epoch)
      break
    
    # train one epoch
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    
    # log train results
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    
    # evaluate
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        
        # evaluate val set using dataset-specific evaluator
        if opt.run_dataset_eval:
          out_dir = val_loader.dataset.run_eval(preds, opt.save_dir, 
                                                n_plots=opt.eval_n_plots, 
                                                render_curves=opt.eval_render_curves)
          
          # log dataset-specific evaluation metrics
          with open('{}/metrics_summary.json'.format(out_dir), 'r') as f:
            metrics = json.load(f)
          logger.scalar_summary('AP/overall', metrics['mean_ap']*100.0, epoch)
          for k,v in metrics['mean_dist_aps'].items():
            logger.scalar_summary('AP/{}'.format(k), v*100.0, epoch)
          for k,v in metrics['tp_errors'].items():
            logger.scalar_summary('Scores/{}'.format(k), v, epoch)
          logger.scalar_summary('Scores/NDS', metrics['nd_score'], epoch)
      
      # log eval results
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    
    # save this checkpoint
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    
    # update learning rate
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
