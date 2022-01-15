#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager , SetDataManager
# from data.maxdisLoss import maxDistLoss
import configs

import wrn_mixup_model
import res_mixup_model
# import wrn_mixup_model_p
from io_utils import parse_args, get_resume_file ,get_assigned_file
from os import path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
image_size = 84

from res_mixup_model import resnet10
from make_loss.label_smoothing_CE import CrossEntropyLabelSmooth
from make_loss.com_loss import com_Loss
from make_loss.pri_loss import pri_Loss
from RSNet import RSNetwork

def train_s2m2(base_loader, base_loader_test, model, start_epoch, stop_epoch, params, tmp):

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    smce = CrossEntropyLabelSmooth(num_classes=params.num_classes)
    Loss_com = com_Loss()
    Loss_pri = pri_Loss()

    if params.model == 'RSN_wrn28' :
        rotate_classifier = nn.Sequential(nn.Linear(640,4))
    elif params.model == 'ResNet18':
        rotate_classifier = nn.Sequential(nn.Linear(512,4))

    rotate_classifier.cuda()

    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])

    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ], lr=1e-3, amsgrad=True)

    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        train_smceLoss = 0
        train_comLoss = 0
        train_priLoss = 0
        d_loss = 0
        rotate_loss = 0
        correct = 0
        total = 0
        torch.cuda.empty_cache()
        
        for batch_idx, (inputs, targets) in enumerate(base_loader):
            
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            lam = np.random.beta(params.alpha, params.alpha)
            # f , outputs , target_a , target_b, mix_cls_scores, mixed_label, sim_feat, diff_feat, min_pairs = model(inputs, targets, mixup_hidden= True , mixup_alpha = params.alpha , lam = lam)
            features, outputs , target_a , target_b, mix_cls_scores, mixed_label, sim_feat, diff_feat, min_pairs = \
                model(inputs, targets, mixup_hidden= True , mixup_alpha = params.alpha , lam = lam)
            #loss_dists = dist_loss(f)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            # # smce
            loss_smce = smce(mix_cls_scores, mixed_label)
            train_smceLoss += loss_smce.data.item()
            # com loss
            beta = 1.0
            loss_com = beta * Loss_com(sim_feat, min_pairs)
            train_comLoss += loss_com.data.item()
            # pri loss
            gamma = 1.0
            loss_pri = gamma * Loss_pri(diff_feat, min_pairs)
            train_priLoss += loss_pri.data.item()
            # total loss
            total_loss = loss + loss_smce + com_Loss + pri_Loss
            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            
            _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            total += f.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
            
            bs = inputs.size(0)
            inputs_ = []
            targets_ = []
            a_ = []
            indices = np.arange(bs)
            np.random.shuffle(indices)
            
            split_size = int(bs/4)
            for j in indices[0:split_size]:
                x90 = inputs[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                inputs_ += [inputs[j], x90, x180, x270]
                targets_ += [targets[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            inputs = Variable(torch.stack(inputs_,0))
            targets = Variable(torch.stack(targets_,0))
            a_ = Variable(torch.stack(a_,0))

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                a_ = a_.cuda()

            
            rf , outputs = model(inputs)
            rotate_outputs = rotate_classifier(rf)
            rloss =  criterion(rotate_outputs,a_)
            closs = criterion(outputs, targets)
            loss = (rloss+closs)/2.0
            
            rotate_loss += rloss.data.item()
            
            loss.backward()
            
            optimizer.step()
            
            
            if batch_idx%10 ==0 :
                print('{0}/{1}'.format(batch_idx, len(base_loader)),
                             'Loss: %.3f | Acc: %.3f%% | RotLoss: %.3f | smceLoss: %.3f | comLoss: %.3f | priLoss: %.3f'
                             % (train_loss/(batch_idx+1), 100.*correct/total, rotate_loss/(batch_idx+1), train_smceLoss/(batch_idx+1),
                                train_comLoss/(batch_idx+1), train_priLoss/(batch_idx+1) ) )
     

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        # if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
        #     outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
        #     torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
         

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f , outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                             % (test_loss/(batch_idx+1), 100.*correct/total ))
            # if (test_loss < min_loss and epoch > 200) or epoch%50==0:
            #     if test_loss < min_loss:
            #         min_loss = test_loss
            #     outfile = os.path.join(params.checkpoint_dir,
            #                            '{:d}_{:.3f}.tar'.format(epoch, (test_loss / (batch_idx + 1))))
            #     torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
       
    return model


if __name__ == '__main__':
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
    base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    base_datamgr_test    = SimpleDataManager(image_size, batch_size = params.test_batch_size)
    base_loader_test     = base_datamgr_test.get_data_loader( base_file , aug = False )


    if params.model == 'WideResNet28_10':
        model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes)
    elif params.model == 'ResNet18':
        model = res_mixup_model.resnet18(num_classes=params.num_classes)
    elif params.model == 'RSN_wrn28':
        model = RSNetwork(num_classes=params.num_classes)
            
    
    if params.method =='S2M2_R':
        if use_gpu:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
            model.cuda()

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir )
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']

            model.load_state_dict(state)
    
        model = train_s2m2(base_loader, base_loader_test,  model, start_epoch, start_epoch+stop_epoch, params, {})


   



  
