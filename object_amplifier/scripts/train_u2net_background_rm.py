import os
import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import glob
import os

from loguru import logger

from .data_loader import (
    RescaleT, RandomCrop, ToTensorLab, SalObjDataset
)
from object_amplifier.model import U2NET, U2NETP
from object_amplifier import DATA_PATH, TRAIN_IMAGES, TRAIN_LABELS, MODELS_PATH
from object_amplifier.settings import U2NetSettings



# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """
    Calculates the binary cross-entropy loss for multiple prediction branches and fuses them together.

    :param d0: Prediction tensor from the first branch.
    :param d1: Prediction tensor from the second branch.
    :param d2: Prediction tensor from the third branch.
    :param d3: Prediction tensor from the fourth branch.
    :param d4: Prediction tensor from the fifth branch.
    :param d5: Prediction tensor from the sixth branch.
    :param d6: Prediction tensor from the seventh branch.
    :param labels_v: Ground truth labels for the input samples.

    :return: A tuple containing the individual binary cross-entropy losses for each branch
             (loss0, loss1, loss2, loss3, loss4, loss5, loss6) and the fused loss (loss).
    """
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    logger.info("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

    return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss



def train_u2net():
    """Train U2NET"""
     
    ####################################################
    ######     DEFINE SETTINGS & DIRECTORIES     #######
    ####################################################
    logger.info("Define directories and model settings")
    settings = U2NetSettings()
    model_name = settings['model_name']
    tra_image_dir, tra_label_dir = str(TRAIN_IMAGES) + "/", str(TRAIN_LABELS) + "/"
    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = str(MODELS_PATH) + "/"

    # Define model configuration
    epoch_num = settings['epoch_num']
    batch_size_train = settings['batch_size_train']
    train_num = settings['train_num']

    tra_img_name_list = glob.glob(f"{tra_image_dir}*{image_ext}")

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(str(tra_label_dir) + imidx + label_ext)

    logger.info("---")
    logger.info("train images: ", len(tra_img_name_list))
    logger.info("train labels: ", len(tra_lbl_name_list))
    logger.info("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    logger.info("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    ####################################################
    ######                TRAINING               #######
    ####################################################
    logger.info("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # logger.info statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            logger.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


if __name__ == "__main__":
    train_u2net()
