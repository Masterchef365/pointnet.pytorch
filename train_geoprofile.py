from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
from wacknet.model import WackNet
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from geoprofile import Geoprofile
import h5py as h5

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

writer = SummaryWriter()

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

dataset = h5.File(opt.dataset, "r")

#top_color = dataset['top_color']
#right_color = dataset['right_color']
#bottom_color = dataset['bottom_color']
#left_color = dataset['left_color']
#
#ds_iter = zip(zip(pclds, top_color, right_color, bottom_color, left_color), labels)
#for x, l in ds_iter:
#    pcld, top, right, bottom, left = x
#    print(pcld.shape, top.shape, right.shape, bottom.shape, left.shape, l.shape)

num_classes = 45

#testdataloader = torch.utils.data.DataLoader(
#    dataset['test'],
#    batch_size=opt.batchSize,
#    num_workers=int(opt.workers))

print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

narrow_width = 200
wide_width = 500
n_rows = 200
classifier = WackNet(narrow_width, wide_width, n_rows, k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = 0

weights = [0.0 if i == 0 else 1.0 for i in range(num_classes)]
weights = torch.tensor(weights).cuda()

sample_idx = 0
for epoch in range(opt.nepoch):
    scheduler.step()
    labels = divide_chunks(dataset['pcld_labels'], opt.batchSize)
    pclds = divide_chunks(dataset['pcld'], opt.batchSize)
    top_imgs = divide_chunks(dataset['top_color'], opt.batchSize)
    right_imgs = divide_chunks(dataset['right_color'], opt.batchSize)
    bottom_imgs = divide_chunks(dataset['bottom_color'], opt.batchSize)
    left_imgs = divide_chunks(dataset['left_color'], opt.batchSize)
    dataloader = zip(zip(pclds, top_imgs, right_imgs, bottom_imgs, left_imgs), labels)
    print(f"Epoch {epoch+1}")
    for i, data in enumerate(dataloader, 0):
        (points, top, right, bottom, left), target = data
        points = torch.from_numpy(points).transpose(1, 2).cuda()
        top = torch.from_numpy(top).transpose(1, 3).cuda().float() / 255.
        right = torch.from_numpy(right).transpose(1, 3).cuda().float() / 255.
        bottom = torch.from_numpy(bottom).transpose(1, 3).cuda().float() / 255.
        left = torch.from_numpy(left).transpose(1, 3).cuda().float() / 255.
        target = torch.from_numpy(target).cuda().long()

        optimizer.zero_grad()

        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points, top, right, bottom, left)

        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]

        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target, weight=weights)
        #loss = F.nll_loss(pred, target)

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1]

        # Ignore emtpy labels when testing accuracy
        nonempty = pred_choice != 0
        n_nonempty = torch.count_nonzero(nonempty)

        correct = (pred_choice.eq(target.data) & nonempty).cpu().sum()

        accuracy = correct.item()/n_nonempty
        writer.add_scalar('Loss/train', loss.item(), sample_idx)
        writer.add_scalar('Accuracy/train', accuracy, sample_idx)

        if n_nonempty != 0:
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), accuracy))
        else:
            print("Rejected; nonempty == 0")
        
        sample_idx += 1

        #if i % 10 == 0:
        #    j, data = next(enumerate(testdataloader, 0))
        #    points, target = data
        #    points, target = points.cuda(), target.cuda()
        #    classifier = classifier.eval()
        #    pred, _, _ = classifier(points)
        #    pred = pred.view(-1, num_classes)
        #    target = target.view(-1, 1)[:, 0]
        #    loss = F.nll_loss(pred, target)
        #    pred_choice = pred.data.max(1)[1]
        #    correct = pred_choice.eq(target.data).cpu().sum()
        #    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

    if epoch % 10 == 0:
        print("Saving...")
        torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))

### benchmark mIOU
#shape_ious = []
#for i,data in tqdm(enumerate(testdataloader, 0)):
#    points, target = data
#    points = points.transpose(2, 1)
#    points, target = points.cuda(), target.cuda()
#    classifier = classifier.eval()
#    pred, _, _ = classifier(points)
#    pred_choice = pred.data.max(2)[1]
#
#    pred_np = pred_choice.cpu().data.numpy()
#    target_np = target.cpu().data.numpy() - 1
#
#    for shape_idx in range(target_np.shape[0]):
#        parts = range(num_classes)#np.unique(target_np[shape_idx])
#        part_ious = []
#        for part in parts:
#            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#            if U == 0:
#                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#            else:
#                iou = I / float(U)
#            part_ious.append(iou)
#        shape_ious.append(np.mean(part_ious))
#
#print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))