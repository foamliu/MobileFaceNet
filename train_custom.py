"""
This script will first load the pre-trained model which was trained on faces_emore and tested on LFW

Then, will train the model on our custom dataset

"""
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq, pretrained_model_path
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
from lfw_eval import lfw_test
from mobilefacenet import MobileFaceNet, ArcMarginModel
from utils import parse_args, save_checkpoint, AverageMeter, accuracy, get_logger, clip_gradient

import pdb

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = float('-inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = MobileFaceNet()
        pretrained_model = torch.load(pretrained_model_path)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        metric_fc = ArcMarginModel(args)
        optimizer = torch.optim.SGD([{'params': model.conv1.parameters()},
                                     {'params': model.dw_conv.parameters()},
                                     {'params': model.features.parameters()},
                                     {'params': model.conv2.parameters()},
                                     {'params': model.gdconv.parameters()},
                                     {'params': model.conv3.parameters(), 'weight_decay': 4e-4},
                                     {'params': model.bn.parameters()},
                                     {'params': metric_fc.parameters()}],
                                    lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay, nesterov=True)

        model = nn.DataParallel(model)
        metric_fc = nn.DataParallel(metric_fc)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']


    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)

        lr = optimizer.param_groups[0]['lr']
        print('\nLearning rate={}\n'.format(lr))

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)
        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        lfw_acc, threshold = lfw_test(model)
        writer.add_scalar('model/lfw_acc', lfw_acc, epoch)
        writer.add_scalar('model/threshold', threshold, epoch)

        # Check if there was an improvement
        is_best = lfw_acc > best_acc
        best_acc = max(lfw_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_acc, is_best)
        scheduler.step(epoch)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        print(feature.shape)
        output = metric_fc(feature, label)  # class_id_out => [N, 10575]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
