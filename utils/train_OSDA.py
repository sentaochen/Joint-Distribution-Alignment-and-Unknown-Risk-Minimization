import torch
import torch.nn as nn

import time
import math
import torch.nn.functional as F
from utils.lr_schedule import inv_lr_scheduler, multi_step_lr_scheduler
from utils.eval import test, predict, test_OSDA
from utils import globalvar as gl

def finetune_for_OSDA(args, model, optimizer, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        num_step = 0
        tar_data_iter = iter(dataloaders['tar_pretrain'])
        for src_input, src_label in dataloaders['src_pretrain']:
            src_input, src_label = src_input.to(DEVICE), src_label.to(DEVICE)

            if num_step > 0 and num_step % len(dataloaders['tar_pretrain']) == 0:
                tar_data_iter = iter(dataloaders['tar_pretrain'])
            num_step = num_step + 1
            tar_input, _ = next(tar_data_iter)
            tar_input = tar_input.to(DEVICE)


            source_output, _, source_softmax, target_softmax, _ = model(src_input, tar_input)
            cls_loss = criterion(source_output, src_label)

            # use entropy values to classify target unknown data, and cal target unknown loss
            target_entropy = -torch.sum(target_softmax * torch.log(target_softmax), dim=1)
            source_entropy_max =  torch.max(-torch.sum(source_softmax * torch.log(source_softmax), dim=1))
            tar_unk = torch.zeros(len(tar_input)).to(DEVICE,torch.int64)
            weights = torch.zeros(len(tar_input)).to(DEVICE,torch.int64)
            tar_unk[target_entropy > source_entropy_max] = args.source_classes_num
            weights[target_entropy > source_entropy_max] = 1
            tar_unknown_loss = -torch.mean(torch.sum(torch.log(target_softmax) * F.one_hot(tar_unk, args.source_classes_num+ 1), dim=1)*weights)
            lambd = 2 / (1 + math.exp(-10 * epoch / args.epochs)) - 1
            loss = cls_loss + 0.1 * lambd * tar_unknown_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc_tar, all_target_pseudo_labels, all_target_label = test(dataloaders['tar_test'], model)
        acc_os, acc_os_star, acc_unknown, acc_hos = test_OSDA(all_target_label, all_target_pseudo_labels, args.source_classes_num)
        print("epoch:{:01d}, acc_os:{:06f}, acc_os_star:{:06f}, acc_unknown:{:06f}, acc_hos:{:06f}".format(epoch, acc_os, acc_os_star, acc_unknown, acc_hos))
    torch.save(model.state_dict(), '{}/best(PT)_{}_{}_{}.pth'.format(check_path, args.net, args.source, args.target))
    print('final model save!')
    time_pass = time.time() - start_time
    print('PreTraining complete in {}h {}m {:.0f}s\n'.format(time_pass//3600, time_pass%3600//60, time_pass%60))




def train_for_OSDA(args, model, optimizer, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    acc_tar, all_target_pseudo_labels, all_target_label = test(dataloaders['tar_test'], model)
    acc_os, acc_os_star, acc_unknown, acc_hos = test_OSDA(all_target_label, all_target_pseudo_labels, args.source_classes_num)
    print("Initial model: acc_os:{:06f}, acc_os_star:{:06f}, acc_unknown:{:06f}, acc_hos:{:06f}".format(acc_os, acc_os_star, acc_unknown, acc_hos))
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    if args.early:
        best_acc = 0
        counter = 0
    src_data_l = iter(dataloaders['src_train_l'])
    tar_data_ul = iter(dataloaders['tar_train_ul'])
    
    start_time = time.time()
    avg_div_loss, avg_cls_loss, avg_unk_loss = 0, 0, 0
    for step in range(1, args.steps + 1):
        model.train()
        current_lr = inv_lr_scheduler(optimizer, step, args.lr_mult, init_lr=args.lr)
        src_input, src_label = next(src_data_l)
        tar_input, _ = next(tar_data_ul)
        optimizer.zero_grad()
        src_input, src_label = src_input.to(DEVICE), src_label.to(DEVICE)
        tar_input = tar_input.to(DEVICE)
        source_output, _, _, target_softmax, div_loss = model(src_input, tar_input, src_label, train=True)
        cls_loss = criterion(source_output, src_label)
        
        tar_unk = torch.zeros(len(target_softmax)).to(DEVICE,torch.int64)+args.source_classes_num
        with torch.no_grad():
            p_unk_softmax = target_softmax[:,-1]
        p_unk_softmax1 = p_unk_softmax.clone().detach()
        # tar_unknown_loss = -torch.mean(torch.sum(torch.log(target_softmax) * F.one_hot(tar_unk, args.source_classes_num+ 1), dim=1)*p_unk_softmax1) #use all target data to cal target unknown loss
        target_prob, target_l = torch.max(target_softmax, 1)
        tar_unknown_loss = -torch.mean(torch.sum(torch.log(target_softmax[target_l==args.source_classes_num]) * F.one_hot(tar_unk[target_l==args.source_classes_num], args.source_classes_num+ 1), dim=1)*p_unk_softmax1[target_l==args.source_classes_num])
        lambd = 2 / (1 + math.exp(-10 * step / args.steps)) - 1
        loss = cls_loss + 0.1 * lambd * tar_unknown_loss + 0.1 * lambd * div_loss
        loss.backward()
        optimizer.step()
        avg_div_loss += div_loss
        avg_cls_loss += cls_loss
        avg_unk_loss += tar_unknown_loss
        if step > 0 and step % args.log_interval == 0:
            print('Learning rate: {:.8f}'.format(current_lr))
            print('Step: [{}/{}]: lambd:{}, div_loss:{:.4f}, cls_loss:{:.4f}, tar_unknown_loss:{:.4f}'.format(step, args.steps, lambd, avg_div_loss, avg_cls_loss, avg_unk_loss))
            avg_div_loss, avg_cls_loss, avg_unk_loss = 0, 0, 0
        if step > 0 and step % args.save_interval == 0:
            print('{} step train time: {:.1f}s'.format(args.save_interval, time.time()-start_time))
            test_time = time.time()
            acc_tar, all_target_pseudo_labels, all_target_label = test(dataloaders['tar_test'], model)
            acc_os, acc_os_star, acc_unknown, acc_hos = test_OSDA(all_target_label, all_target_pseudo_labels, args.source_classes_num)
            print("Step: [{}/{}]: acc_os:{:06f}, acc_os_star:{:06f}, acc_unknown:{:06f}, acc_hos:{:06f}".format(step, args.steps, acc_os, acc_os_star, acc_unknown, acc_hos))
        
            if step % args.update_interval == 0 :
                pseudo_labels = predict(model, dataloaders['tar_test'])
                dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)
                dataloaders['src_train_l'].batch_sampler.label_sampler.set_batch_num(args.update_interval)
                src_data_l = iter(dataloaders['src_train_l'])
                tar_data_ul = iter(dataloaders['tar_train_ul'])
            if args.early:
                if acc_hos > best_acc:
                    best_acc = acc_hos
                    counter = 0
                    if args.save_check : 
                        torch.save(model.state_dict(), '{}/best_OSDA_{}_to_{}.pth'.format(check_path, args.source, args.target))
                else:
                    counter += 1
                    if counter > args.patience:
                        print('early stop! training_step:{}'.format(step))
                        break
            seconds = time.time() - start_time
            print('{} step cost time: {}h {}m {:.0f}s\n'.format(step, seconds//3600, seconds%3600//60, seconds%60))
    time_pass = time.time() - start_time
    print('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training_step:{}, acc_hos:{}'.format(step, best_acc))

