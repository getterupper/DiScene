import os
import mmcv
import numpy as np
import torch
import pickle
import os.path as osp
from tqdm import tqdm
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
import math, cv2
from mmcv.image.io import imread
from mmcv.parallel import DataContainer as DC

from loaders.pipelines.transform_2d import Resize, NormalizeImage, PrepareForNet
from loaders.pipelines.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D
from loaders.ssc_metrics import SSCMetrics
from models.utils import sparse2dense

from .metric3d_utils import transform_test_data_scalecano

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

@DATASETS.register_module()
class OccScanNetDataset(Dataset):
    def __init__(
        self,
        data_path, 
        num_frames=1,
        offset=0,
        grid_size_occ=[60, 60, 36],
        coarse_ratio=2,
        empty_idx=0,
        phase='train',
        final_dim=[480, 640], 
        resize_lim=[1.0, 1.0],
        num_pts=21600,
        pretrained_depth_model='DepthAnythingV2',
        data_tg='base'
        ):
        super(OccScanNetDataset, self).__init__()

        self.occscannet_root = data_path
        self.phase = phase
        
        self.num_frames = num_frames
        self.offset = offset
        self.grid_size_occ = grid_size_occ
        self.grid_size_occ_coarse = (np.array(grid_size_occ) // coarse_ratio).astype(np.uint32)
        self.coarse_ratio = coarse_ratio
        self.empty_idx = empty_idx
        self.phase = phase
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.pretrained_depth_model = pretrained_depth_model

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        if data_tg == 'base':
            subscenes_list = f'{self.occscannet_root}/{self.phase}.txt'
        elif data_tg == 'mini' or data_tg == 'small' or data_tg == 'tiny' or data_tg == 'attn':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_{data_tg}.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
        
        self.num_pts = num_pts
        # pipeline
        if self.phase == 'train':
            transforms_pipeline = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=True),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms_pipeline = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=False),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms_pipeline = transforms_pipeline

        # set group flag for the samplers
        self._set_group_flag()
    
    def __len__(self):
        return len(self.used_subscenes)
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # idx = 0
        if self.phase != 'train':
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example
    
    def pipeline(self, input_dict):
        imgs, metas, occs = input_dict['imgs'], input_dict['meta'], input_dict['occs']

        # deal with img augmentation
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        for t in self.transforms_pipeline:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
        FN, C, H, W = imgs.shape
        # imgs = imgs.reshape(FN, C, H, W)
        metas['img_shape'] = imgs_dict['img_shape']
        if imgs_dict.get('img_aug_matrix'):
            img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
            metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)

        img_metas = DC(metas, cpu_only=True)

        result = {
            'img': imgs,
            'img_metas': img_metas,
            'voxel_semantics': occs.squeeze(0)
        }
        return result

    def get_data_info(self, index):
        name = self.used_subscenes[index]
        with open(name, 'rb') as f:
            data = pickle.load(f)
        
        name_without_ext = os.path.splitext(name)[0]
        this_name = name_without_ext.split('gathered_data/')[-1]
        
        meta = {}
        meta['name'] = this_name # 'scene0000_00/00000'
        meta['scene_size'] = self.scene_size
        cam_pose = data['cam_pose']
        meta['cam2world'] = cam_pose
        world2cam = np.linalg.inv(cam_pose)
        meta['world2cam'] = world2cam
        
        rgb_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.jpg'
        depth_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.png'
        depth_gt_np = Image.open(depth_path).convert('I;16')
        depth_gt_np = np.array(depth_gt_np) / 1000.0

        meta['depth_path'] = depth_path

        if self.pretrained_depth_model == 'DepthAnythingV2':
            transform = Compose([
                Resize(
                    width=480,
                    height=480,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            img_depthbranch = cv2.imread(rgb_path)
            img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
            img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
            sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
            img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
            depth_gt = torch.from_numpy(sample['depth']).unsqueeze(0)
            meta['depth_gt'] = depth_gt
            depth_valid_mask = (torch.isnan(depth_gt) == 0)
            depth_gt[depth_valid_mask == 0] = 0
            meta['img_depthbranch'] = img_depthbranch
            meta['depth_gt_valid'] = depth_gt
        elif self.pretrained_depth_model == 'DepthAnythingV1':
            img_depthbranch = Image.open(rgb_path).convert("RGB")
            img_depthbranch = img_depthbranch.resize((640, 480))
            depth_gt_np = cv2.resize(depth_gt_np, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth_gt = torch.from_numpy(depth_gt_np).unsqueeze(0)
            meta['depth_gt'] = depth_gt
            depth_valid_mask = (torch.isnan(depth_gt) == 0)
            depth_gt[depth_valid_mask == 0] = 0
            meta['img_depthbranch'] = img_depthbranch
            meta['depth_gt_valid'] = depth_gt
        elif self.pretrained_depth_model == 'Metric3Dv2-Small':
            depth_gt_np = cv2.resize(depth_gt_np, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth_gt = torch.from_numpy(depth_gt_np).unsqueeze(0)
            meta['depth_gt'] = depth_gt
            depth_valid_mask = (torch.isnan(depth_gt) == 0)
            depth_gt[depth_valid_mask == 0] = 0
            meta['depth_gt_valid'] = depth_gt

            data_basic=dict(
                canonical_space = dict(
                    img_size=(540, 960),
                    focal_length=1000.0,
                ),
                depth_range=(0, 1),
                depth_normalize=(0.1, 200),
                crop_size = (616, 1064),
            )
            normalize_scale = data_basic['depth_range'][1]

            rgb_origin = cv2.imread(rgb_path)[:, :, ::-1].copy()
            cam_intrin = data['intrinsic']
            fx, fy, cx, cy = cam_intrin[0, 0], cam_intrin[1, 1], cam_intrin[0, 2], cam_intrin[1, 2]
            intrinsic = [fx, fy, cx, cy]
            rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, data_basic)
            rgb_input = rgb_input[None, ...]
            meta['img_depthbranch'] = rgb_input
            meta['pad'] = pad
            meta['normalize_scale'] = normalize_scale
            meta['label_scale_factor'] = label_scale_factor
        elif self.pretrained_depth_model == 'Metric3Dv2-Giant':
            depth_gt_np = cv2.resize(depth_gt_np, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth_gt = torch.from_numpy(depth_gt_np).unsqueeze(0)
            meta['depth_gt'] = depth_gt
            depth_valid_mask = (torch.isnan(depth_gt) == 0)
            depth_gt[depth_valid_mask == 0] = 0
            meta['depth_gt_valid'] = depth_gt

            data_basic=dict(
                canonical_space = dict(
                    img_size=(540, 960),
                    focal_length=1000.0,
                ),
                depth_range=(0, 1),
                depth_normalize=(0.1, 200),
                crop_size = (616, 1064),
            )
            normalize_scale = data_basic['depth_range'][1]

            rgb_origin = cv2.imread(rgb_path)[:, :, ::-1].copy()
            cam_intrin = data['intrinsic']
            fx, fy, cx, cy = cam_intrin[0, 0], cam_intrin[1, 1], cam_intrin[0, 2], cam_intrin[1, 2]
            intrinsic = [fx, fy, cx, cy]
            rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, data_basic)
            rgb_input = rgb_input[None, ...]
            meta['img_depthbranch'] = rgb_input
            meta['pad'] = pad
            meta['normalize_scale'] = normalize_scale
            meta['label_scale_factor'] = label_scale_factor
        elif self.pretrained_depth_model == 'GT' or self.pretrained_depth_model == 'None':
            transform = Compose([
                Resize(
                    width=480,
                    height=480,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            img_depthbranch = cv2.imread(rgb_path)
            img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
            img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
            sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
            img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
            depth_gt = torch.from_numpy(sample['depth']).unsqueeze(0)
            meta['depth_gt'] = depth_gt
            depth_valid_mask = (torch.isnan(depth_gt) == 0)
            depth_gt[depth_valid_mask == 0] = 0
            meta['img_depthbranch'] = img_depthbranch
            meta['depth_gt_valid'] = depth_gt
        else:
            raise NotImplementedError

        pix_x = torch.arange(0, 640, dtype=torch.int)
        pix_y = torch.arange(0, 480, dtype=torch.int)
        pix_xx = pix_x[:, None].expand(640, 480)
        pix_yy = pix_y[None, :].expand(640, 480)
        pix_xx_yy = torch.stack([pix_xx, pix_yy], dim=-1) # actual space
        meta['coords_2d'] = pix_xx_yy
        
        meta['rgb_path'] = rgb_path
        N_img = []
        this_img = imread(rgb_path, 'unchanged').astype(np.float32)
        this_H, this_W, _ = this_img.shape
        new_H, new_W = 480, 640
        # resize
        new_img = cv2.resize(this_img, (new_W, new_H))
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        N_img.append(new_img)
        img = np.stack(N_img, 0) # [1, 968, 1296, 3]
        this_H, this_W= new_H, new_W
        img = [img] # [1, 1, 968, 1296, 3]
        
        cam_intrin = data['intrinsic']
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor
        
        meta['cam_k'] = cam_intrin[:3, :3]
        viewpad = np.eye(4)
        viewpad[:meta['cam_k'].shape[0], :meta['cam_k'].shape[1]] = meta['cam_k']
        meta['cam2img'] = viewpad
        world2img = (viewpad @ world2cam)
        meta['world2img'] = world2img

        vox_origin = data["voxel_origin"]
        meta['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
        target = data["target_1_4"]
        target = np.transpose(target, (1, 0, 2))  # 60, 60, 36
        target[target == 0] = 12  # unknown
        target[target == 255] = 0  # empty
        occ = target  # (60, 60, 36)
        occ = [occ]  # [1, 60, 60, 36]
        
        meta['label'] = occ
        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)

        input_dict = {
            'imgs': imgs,
            'meta': meta,
            'occs': occs
        }
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, is_save=False, **eval_kwargs):
        results_dict = {}
        results_dict.update(
            self.eval_miou(occ_results, runner=runner, show_dir=show_dir, is_save=is_save, **eval_kwargs))
        return results_dict

    def eval_miou(self, occ_results, runner=None, show_dir=None, is_save=False, **eval_kwargs):
        print('\nStarting Evaluation...')
        metric = SSCMetrics(n_classes=12)

        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            result_dict = occ_results[i]
            info = self.get_data_info(i)
            occ_labels = info['occs'].squeeze(0)

            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                dense_shape=occ_labels.shape,
                empty_value=12)
            
            if is_save:
                name = info['meta']['name']
                save_dir = os.path.join('./save_results', name)
                os.makedirs(save_dir, exist_ok=True)

                np.save(os.path.join(save_dir, 'pred.npy'), occ_pred)
                # np.save(os.path.join(save_dir, 'gt.npy'), occ_labels)
            
            # ignore unknown
            occ_pred[occ_pred == 0] = 255
            occ_pred[occ_pred == 12] = 0
            occ_labels[occ_labels == 0] = 255
            occ_labels[occ_labels == 12] = 0
            
            metric.add_batch(occ_pred, occ_labels)

        stats = metric.get_stats()
            
        info_sem_cls = stats["iou_ssc"]
        info_sem = stats["iou_ssc_mean"]
        info_geo = stats["iou"]

        print(f'Current val iou of sem_cls is {[round(info_sem_cls[cls_] * 100, 2) for cls_ in range(len(info_sem_cls))]}')
        print(f'Current val iou of sem is {round(info_sem * 100, 2)}')
        print(f'Current val iou of geo is {round(info_geo * 100, 2)}')
        
        return {
            'IoU': info_geo,
            'mIoU': info_sem
        }