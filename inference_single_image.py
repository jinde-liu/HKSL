from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tr
from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap
from utils.metrics import Evaluator
import matplotlib.pyplot as plt
import torch.nn as nn
from args import Args_occ5000
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(
    "/home/kidd/kidd1/pytorch-deeplab-xception/run/occ5000/deeplab_v3+_noflip/50-0.7067296461287129-model_best.pth.tar")
checkpoint2 = torch.load(
    '/home/kidd/kidd1/HKSL/run/occ5000/deeplab_v3+_noflip_kinematic/50-0.7107144180176619-model_best.pth.tar'
)
args = Args_occ5000()
args2 = Args_occ5000()
args2.use_kinematic = True
model = DeepLab(
    args=args,
    num_classes=13,
    backbone='resnet',
    output_stride=16,
    sync_bn=True,
    freeze_bn=False)
model2 = DeepLab(
    args=args2,
    num_classes=13,
    backbone='resnet',
    output_stride=16,
    sync_bn=True,
    freeze_bn=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)
torch.set_grad_enabled(False)

model2.load_state_dict(checkpoint2['state_dict'])
model2.eval()
model2.to(device)
torch.set_grad_enabled(False)

def transform(image):
    return tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)


# Read eval image and gt
dataset_base = '/home/kidd/kidd1/Occ5000'
im_list = []
ann_list = []
with open(dataset_base + '/list/val_all2500.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        im_path = line[0:line.find('.png') + 4]
        ann_path = line[line.find('.png') + 5:]
        assert os.path.isfile(dataset_base + im_path)
        assert os.path.isfile(dataset_base + ann_path)
        im_list.append(dataset_base + im_path)
        ann_list.append(dataset_base + ann_path)
assert (len(im_list) == len(ann_list)), 'number not same in im and ann!'
print('Number of images in {}:{:d}'.format('val', len(im_list)))
for i in range(len(im_list)):
    if i % 24 == 0:
        print('processed %d images'%i)
    image = Image.open(im_list[i])
    gt_im = Image.open(ann_list[i])
    gt = np.array(gt_im)
    gt_rgb = decode_segmap(gt, dataset="occ5000")

    # Inference and set the visual color map
    inputs = transform(image).to(device)
    output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
    output2 = model2(inputs.unsqueeze(0)).squeeze().cpu().numpy()
    pred = np.argmax(output, axis=0)
    pred_rgb = decode_segmap(pred, dataset="occ5000")
    pred2 = np.argmax(output2, axis=0)
    pred_rgb2 = decode_segmap(pred2, dataset="occ5000")

    fig = plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(gt_rgb)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(pred_rgb)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(pred_rgb2)
    plt.axis('off')
    plt.savefig('/home/kidd/kidd1/HKSL/run/occ5000/deeplab_v3+_noflip_kinematic/results_images/' + str(i) + '.png')
    #plt.show()
    plt.close(fig)

    # eval = Evaluator(13)
    # eval.reset()
    # eval.add_batch(gt, pred)
    # miou = eval.Mean_Intersection_over_Union()
    # print(miou)
    # class_miou = eval.Class_Intersection_over_Union()
    # print(class_miou)

