import os
import pickle
import numpy as np
import shutil
import tqdm

from configurations.configuration import Configuration
from models.afd_sl import AFDSL
from trainers.supervised_trainer import SLTrainer
from data_util.dataset import GridDataset
from tools.calculate_a_p_r_f1 import calculate_evaluation
from trainers.supervised_trainer import Criterion

from torch import cuda
import torch
from torch import nn

from tensorboardX import SummaryWriter

import sys


def evaluation(TP, FP, TN, FN, N):
    accuracy = (TP + TN) / N
    precision = TP / (TP + FP) if TP + FP != 0 else 0.
    recall = TP / (TP + FN) if TP + FN != 0 else 0.
    f1 = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN != 0 else 0.
    return accuracy, precision, recall, f1


def fault_diagnosis_train(config,
                          save_dir='experiment_3/train_data_g',
                          filename='train_data_g_filenames.txt'):
    """

    :param filename:
    :param save_dir:
    :type config: Configuration
    :param config:
    :return:
    """

    if cuda.is_available():
        if config.gpu_id >= 0:
            compute_device = torch.device('cuda:%d' % config.gpu_id)
        else:
            compute_device = torch.device('cuda')
    else:
        compute_device = torch.device('cpu')
    # print('coumpte device: %s ' % compute_device)

    # build model
    afd_sl = AFDSL(emb_size=config.emb_size, T=config.T,
                   compute_device=compute_device)

    criterion = Criterion(beta=config.beta)

    sl_trainer = SLTrainer(afd_sl=afd_sl,
                           criterion=criterion,
                           learning_rate=config.lr,
                           clip_norm=config.clip_norm,
                           compute_device=compute_device,
                           ckpt_dir=config.ckpt_dir)
    sl_trainer.to(compute_device)

    # load dataset
    grid_dataset = GridDataset(data_dir=config.data_dir,
                               save_dir=save_dir,
                               filename=filename)

    # process dir
    if config.retrain:
        # checkpoint
        if os.path.exists(os.path.join(config.ckpt_dir, 'afd_sl.pth')):
            os.remove(os.path.join(config.ckpt_dir, 'afd_sl.pth'))
        # train step
        if os.path.exists(os.path.join(config.ckpt_dir, 'train_step.dat')):
            os.remove(os.path.join(config.ckpt_dir, 'train_step.dat'))
        # log
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)

    # load model
    sl_trainer.load()

    # load ckpt train_step
    train_step = 0
    epoch = 0
    if os.path.exists(os.path.join(config.ckpt_dir, 'train_step.dat')):
        train_step = pickle.load(open(os.path.join(config.ckpt_dir, 'train_step.dat'), 'rb'))
    if os.path.exists(os.path.join(config.ckpt_dir, 'epoch.dat')):
        train_step = pickle.load(open(os.path.join(config.ckpt_dir, 'epoch.dat'), 'rb'))

    # log
    log_writer = SummaryWriter(config.log_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    total_loss = 0.
    batch_count = 0
    while True:
        if epoch >= config.num_epoch:
            break

        # sample examples
        batch_examples, batch_filenames, ones_travel = grid_dataset.sample(config.batch_size)
        batch_loss = sl_trainer.train_step(batch_examples)
        # print log
        sys.stdout.write('\r< %dth epoch %dth step>: < batch_loss %.5f >' %
                         (epoch + 1, train_step + 1, batch_loss.item()))
        # logging
        log_writer.add_scalar('train/batch_loss', batch_loss.item(), train_step)

        # calculate total loss
        total_loss += batch_loss.item()
        batch_count += 1
        if ones_travel:
            # print('\n|| train step %d total loss: %.5f ||' % (train_step + 1, total_loss / batch_count))
            # logging
            log_writer.add_scalar('train/total_loss', total_loss / batch_count, epoch)
            total_loss = 0.
            batch_count = 0
            epoch += 1
            sl_trainer.save()

        if (train_step + 1) % config.save_every == 0:
            sl_trainer.save()
            pickle.dump(train_step, open(os.path.join(config.ckpt_dir, 'train_step.dat'), 'wb'))
            pickle.dump(epoch, open(os.path.join(config.ckpt_dir, 'epoch.dat'), 'wb'))

        train_step += 1

    return train_step


def fault_diagnosis_infer(config,
                          save_dir='experiment_3/test_data_g',
                          filename='test_data_g_filenames.txt'):
    """

    :param filename:
    :param save_dir:
    :type config: Configuration
    :param config:
    :return:
    """
    if cuda.is_available():
        if config.gpu_id >= 0:
            compute_device = torch.device('cuda:%d' % config.gpu_id)
        else:
            compute_device = torch.device('cuda')
    else:
        compute_device = torch.device('cpu')
    # print('coumpte device: %s ' % compute_device)

    # build model
    afd_sl = AFDSL(emb_size=config.emb_size, T=config.T,
                   compute_device=compute_device)

    ckpt_filename = os.path.join(config.ckpt_dir, 'afd_sl.pth')
    # print(ckpt_filename)
    if os.path.exists(ckpt_filename):
        state_dict = torch.load(ckpt_filename)
        if 'model' in state_dict:
            afd_sl.load_state_dict(state_dict['model'])
        else:
            afd_sl.load_state_dict(state_dict)

    # load dataset
    grid_dataset = GridDataset(data_dir=config.data_dir,
                               save_dir=save_dir,
                               filename=filename,
                               shuffle=False)

    all_tp = 0
    all_fp = 0
    all_tn = 0
    all_fn = 0
    all_n = 0
    for idx in tqdm.tqdm(range(len(grid_dataset)), total=len(grid_dataset)):
        example, filename = grid_dataset[idx]  # type: (Example, str)

        # # debug
        # print('debug')
        # example = pickle.load(open('dataset/data/118_4_10_10_58.example', 'rb'))
        # filename = 'dataset/data/118_4_10_10_58.example'

        predict = afd_sl(example)  # Tensor (num_dev,)
        # predict = torch.argmax(predict, dim=1)  # type: torch.Tensor # Tensor (num_dev,)

        # print('=============================================================')
        # print('%d th example: "%s"' % (idx + 1, filename))

        # 计算评价指标
        predict = predict.cpu().detach().numpy()  # numpy (num_dev,)
        predict = (predict > config.threshold).astype(np.int)  # numpy (num_dev,)
        label = example.device_state  # numpy (num_dev,)
        tp, fp, tn, fn, n = calculate_evaluation(predict, label)
        accuracy, precision, recall, f1 = evaluation(tp, fp, tn, fn, n)

        # print('<TP %d> <FP %d> <TN %d> <FN %d>' % (tp, fp, tn, fn))
        # print('<A %.5f> <P %.5f> <R %.5f> <F1 %.5f>' % (accuracy, precision, recall, f1))

        # 统计整个数据集
        all_tp += tp
        all_fp += fp
        all_tn += tn
        all_fn += fn
        all_n += n

    all_accuracy, all_precision, all_recall, all_f1 = evaluation(all_tp, all_fp, all_tn, all_fn, all_n)
    print('\n*************************************************************')
    print('<TP %d> <FP %d> <TN %d> <FN %d>' % (all_tp, all_fp, all_tn, all_fn))
    print('<A %.5f> <P %.5f> <R %.5f> <F1 %.5f>' % (all_accuracy, all_precision, all_recall, all_f1))
    print('*************************************************************')

    return all_accuracy, all_precision, all_recall, all_f1


if __name__ == '__main__':
    from configurations.configuration import config
    from data_util.dataset import Example
    from data_util.dataset import Topology

    config.data_dir = '/data/lf/graduate_projects/dataset'
    fault_diagnosis_train(config)
