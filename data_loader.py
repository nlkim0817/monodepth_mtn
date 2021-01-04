import os
from PIL import Image
from glob import glob
from torch.utils.data import Dataset


class KaistLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):

        scenes = ['Campus','Residential','Suburb','Urban'];
        if mode == 'train':
           rgb_left_dir  = os.path.join(root_dir, 'training/{}/LEFT');
           rgb_right_dir = os.path.join(root_dir, 'training/{}/RIGHT');
           thm_left_dir  = os.path.join(root_dir, 'training/{}/THERMAL');
        else:
           rgb_left_dir  = os.path.join(root_dir, 'testing/{}/LEFT');
           rgb_right_dir = os.path.join(root_dir, 'testing/{}/RIGHT');
           thm_left_dir  = os.path.join(root_dir, 'testing/{}/THERMAL');

        rgb_left_paths, rgb_right_paths, thm_left_paths = [],[],[];
        for scene in scenes:
            rgb_left_path = glob( os.path.join( rgb_left_dir.format(scene), '*.jpg') );
            rgb_left_path = sorted( rgb_left_path );
            rgb_left_paths += rgb_left_path;

            rgb_right_path = glob( os.path.join( rgb_right_dir.format(scene), '*.jpg') );
            rgb_right_path = sorted( rgb_right_path );
            rgb_right_paths += rgb_right_path;

            thm_left_path = glob( os.path.join( thm_left_dir.format(scene), '*.jpg') );
            thm_left_path = sorted( thm_left_path );
            thm_left_paths += thm_left_path;

        self.thm_left_paths  = thm_left_paths;
        self.rgb_left_paths  = rgb_left_paths;
        self.rgb_right_paths = rgb_right_paths;
        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.thm_left_paths)

    def __getitem__(self, idx):
        thm_left_image = Image.open(self.thm_left_paths[idx]).convert("RGB");
        rgb_left_image = Image.open(self.rgb_left_paths[idx]);
        if self.mode == 'train' or self.mode == 'val':
            rgb_right_image = Image.open(self.rgb_right_paths[idx])
            sample = {'left_image': rgb_left_image, 'left_thm_image': thm_left_image,
                      'right_image': rgb_right_image}
            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
               left_image = self.transform(thm_left_image)
            return left_image


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image
