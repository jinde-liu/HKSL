from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class CIHPSegmentation(Dataset):
    """Dataloader for occ5000 dataset
    """
    NUM_CLASSES = 20

    def __init__(self, args, base_dir=Path.db_root_dir('cihp'), split='train'):
        """Set image id, image path and annotation

        base_dir: path to VOC dataset directory
        split: train or val

        self.images -- split images' path
        self.annotations -- split annotations' path
        """
        # Image and annotation dir
        super().__init__()
        if split == 'train' or split == 'val':
            self.split = split
        else:
            raise Exception('split can only be "train" or "val"')
        self._base_dir = base_dir

        self.args = args
        # Image set list dir
        _splits_dir = os.path.join(self._base_dir, 'list')

        # self.im_ids = []
        self.images = []
        self.annotations = []
        if self.split == 'train':
            im_path = os.path.join(self._base_dir, 'Train', 'Images')
            ann_path = os.path.join(self._base_dir, 'Train', 'Category_ids')
        else:
            im_path = os.path.join(self._base_dir, 'Validation', 'Images')
            ann_path = os.path.join(self._base_dir, 'Validation', 'Category_ids')
        # Get image and ann path
        with open(os.path.join(os.path.join(_splits_dir, self.split + '_id.txt')), "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                im_name = line + '.jpg'
                ann_name = line + '.png'
                assert os.path.isfile(os.path.join(im_path, im_name))
                assert os.path.isfile(os.path.join(ann_path, ann_name))
                self.images.append(os.path.join(im_path, im_name))
                self.annotations.append(os.path.join(ann_path, ann_name))

        assert (len(self.images) == len(self.annotations))
        print('Number of images in {}:{:d}'.format(split, len(self.images)))

    def __len__(self):
        """Returen dataset length
        """
        return len(self.images)

    def __str__(self):
        return 'CIHP(split=' + str(self.split) + ')'

    def __getitem__(self, index):
        """Get one item from dataset
        index: item's idx
        """
        _img, _ann = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _ann}
        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        """Load img and ann
        index: idx for loading image
        """
        image = Image.open(self.images[index]).convert('RGB')
        annotation = Image.open(self.annotations[index])
        return image, annotation

    def transform_tr(self, sample):
        """Transformations for images
        sample: {image:img, annotation:ann}

        Note: the mean and std is from imagenet
        """
        if self.args.no_flip:
            composed_transforms = transforms.Compose([
                tr.RandomRotate(degree=self.args.rotate_degree),
                tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size,
                                   scale_ratio=self.args.scale_ratio, fill=0),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
            return composed_transforms(sample)
        else:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size,
                                   scale_ratio=self.args.scale_ratio, fill=0),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
            return composed_transforms(sample)

    def transform_val(self, sample):
        """Transformations for images
        sample: {image:img, annotation:ann}

        Note: the mean and std is from imagenet
        """
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=0),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


#################################################
#   test code for cihp.py
#################################################
if __name__ == '__main__':
    # show random batch_size images and annotaions
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import args

    args = args.Args_cihp()
    cihp_train = CIHPSegmentation(args, split='train')
    data_loader = DataLoader(cihp_train, batch_size=args.batch_size, shuffle=False)

    for i, sample in enumerate(data_loader):
        for j in range(sample['image'].size()[0]):
            img = sample['image'].numpy()[j]
            gt = sample['label'].numpy()[j]
            seg_map = decode_segmap(gt, dataset='cihp')
            # Pytroch tensor chanel comes first then h and w
            img_tmp = np.transpose(img, axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img_tmp)
            plt.subplot(1, 2, 2)
            plt.imshow(seg_map)

        if i == 0:
            break
    plt.show()


