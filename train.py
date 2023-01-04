import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from model import Unet
from data_factory import get_data
from config import cfg


def main():
    print('testing')
    """
    This script trains the model.
    Train (and val) loaders are used for training (and saving checkpoints).
    Trained models and training logs (train and vall loss curves) are saved to the disk.
    """
    # get the model
    model = Unet(in_channels=cfg.data.input_channels,
                 out_channels=2,
                 feature_reduction=4,
                 norm_type=cfg.model.norm_type)
    model.to('cuda:0')

    optim = torch.optim.Adam(model.parameters(),
                             lr=cfg.train.learning_rate,
                             weight_decay=cfg.train.l2_reg)

    # get dataloaders
    train_loader, val_loader, _ = get_data(cfg)

    # set up training

    # lr schedular
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=cfg.train.lr_decay_every, gamma=cfg.train.lr_decay)

    # loss function
    weight = torch.tensor([0.05, 1.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)

    # training logs
    train_loss_log = []
    val_loss_log = []

    best_val_loss = 999.0

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(
            'output directory ', out_dir,
            ' already exists. Make sure you are not overwriting previously trained model...'
        )

    print('configurations: ', cfg)
    print('starting training')

    # Train
    for epoch in range(cfg.train.num_epochs):
        # begin training
        loss_train = 0
        model.train()
        for i, data in enumerate(train_loader):
            optim.zero_grad()

            shaded = data[0].cuda()
            dem = data[1].cuda().unsqueeze(1)
            naip_image = data[2].cuda()
            labels = data[3].long().cuda()
            dem_dxy = data[5].cuda()
            dem_dxy_pre = data[6].cuda().unsqueeze(1)

            if i == 0:
                print_now = True
            else:
                print_now = False

            predictions = model(shaded, dem, naip_image, dem_dxy, dem_dxy_pre)

            loss = criterion(predictions, labels)

            loss.backward()

            loss_train += loss.item()

            optim.step()

            # printing
            if (i + 1) % 20 == 0:
                print('[Ep ', epoch + 1, '] train loss: ', loss_train / (i + 1))

        # end of training for this epoch
        loss_train /= len(train_loader)

        # begin validation
        loss_val = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                optim.zero_grad()  # clear gradients
                shaded = data[0].cuda()
                dem = data[1].cuda().unsqueeze(1)
                naip_image = data[2].cuda()
                labels = data[3].long().cuda()
                dem_dxy = data[5].cuda()
                dem_dxy_pre = data[6].cuda().unsqueeze(1)

                predictions = model(shaded, dem, naip_image, dem_dxy,
                                    dem_dxy_pre)

                loss = criterion(predictions, labels)

                loss_val += loss.item()
        # end of validation
        loss_val /= len(val_loader)

        # End of epoch
        scheduler.step()

        train_loss_log.append(loss_train)
        val_loss_log.append(loss_val)

        print('End of epoch ', epoch + 1, ' , Train loss: ', loss_train,
              ', val loss: ', loss_val)

        # save best model checkpoint
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            fname = 'model_dict.pth'
            torch.save(model.state_dict(), os.path.join(out_dir, fname))
            print('=========== model saved at epoch: ', epoch + 1,
                  ' =================')

    # save model checkpoint at the end
    fname = 'model_dict_end.pth'
    torch.save(model.state_dict(), os.path.join(out_dir, fname))
    print('model saved at the end of training: ')

    # save loss curves
    plt.figure()
    plt.plot(train_loss_log)
    plt.plot(val_loss_log)
    plt.legend(['train loss', 'test loss'])
    fname = os.path.join(out_dir, 'loss.png')
    plt.savefig(fname)

    # Saving train and val loss logs
    log_name = os.path.join(out_dir, "training_logs.txt")
    with open(log_name, 'w') as result_file:
        result_file.write('Validation loss ')
        result_file.write(str(val_loss_log))
        result_file.write('\nTraining loss  ')
        result_file.write(str(train_loss_log))


if __name__ == '__main__':
    main()
