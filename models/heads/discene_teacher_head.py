import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from ..bbox.utils import decode_points
import numpy as np
import random
from mmcv import Config
import os

import sys
# NOTE: modify below to /your/path/to/DiScene
sys.path.append('/path/to/DiScene')
sys.path.append('/path/to/DiScene/depth_anything/metric_depth')
sys.path.append('/path/to/DiScene/Depth-Anything-V2/metric_depth')
sys.path.append('/path/to/DiScene/Metric3D')
from ..bbox.utils import decode_bbox, decode_points, encode_points

@HEADS.register_module()
class DiSceneHead_Teacher(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 ignore_label=0,
                 empty_label=12,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 pretrained_depth_model='DepthAnythingV2',
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # depth branch
        self.pretrained_depth_model = pretrained_depth_model
        if pretrained_depth_model == 'DepthAnythingV1':
            # NOTE: modify below to /your/path/to/DiScene
            overrite = {"pretrained_resource": "local::/path/to/DiScene/checkpoints/depth_anything_metric_depth_indoor.pt"}
            from depth_anything.metric_depth.zoedepth.models.builder import build_model as build_depthany_model
            from depth_anything.metric_depth.zoedepth.utils.config import get_config as get_depthany_config
            conf = get_depthany_config("zoedepth", "infer", "nyu", **overrite)
            # conf['img_size'] = [480, 640]
            from pprint import pprint
            # pprint(conf)
            model = build_depthany_model(conf)
            self.depth_model = model.to('cuda')
            for p in self.depth_model.parameters():
                p.requires_grad = False
        elif pretrained_depth_model == 'DepthAnythingV2':
            from depth_anything_v2.dpt import DepthAnythingV2
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            self.depth_model = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
            # NOTE: modify below to /your/path/to/DiScene
            checkpoint = torch.load('/path/to/DiScene/checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_key = k[len('module.'):] 
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.depth_model.load_state_dict(new_state_dict)
            for p in self.depth_model.parameters():
                p.requires_grad = False
        elif self.pretrained_depth_model == 'Metric3Dv2-Small':
            from mono.utils.mldb import load_data_info, reset_ckpt_path
            from mono.model.monodepth_model import get_configured_monodepth_model
            # NOTE: modify below to /your/path/to/DiScene
            cfg = Config.fromfile('/path/to/DiScene/Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py')
            cfg.load_from = '/path/to/.cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth'
            cfg.batch_size = 1
            # load data info
            data_info = {}
            load_data_info('data_info', data_info=data_info)
            cfg.mldb_info = data_info
            # update check point info
            reset_ckpt_path(cfg.model, data_info)
            depth_model = get_configured_monodepth_model(cfg, ).cuda()
            # load ckpt
            load_path = cfg.load_from
            if os.path.isfile(load_path):
                checkpoint = torch.load(load_path, map_location="cpu")
                ckpt_state_dict  = checkpoint['model_state_dict']
                depth_model.load_state_dict(ckpt_state_dict, strict=False)
            self.depth_model = depth_model
            for p in self.depth_model.parameters():
                p.requires_grad = False
        elif self.pretrained_depth_model == 'Metric3Dv2-Giant':
            from mono.utils.mldb import load_data_info, reset_ckpt_path
            from mono.model.monodepth_model import get_configured_monodepth_model
            # NOTE: modify below to /your/path/to/DiScene
            cfg = Config.fromfile('/path/to/DiScene/Metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py')
            cfg.load_from = '/path/to/.cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth'
            cfg.batch_size = 1
            # load data info
            data_info = {}
            load_data_info('data_info', data_info=data_info)
            cfg.mldb_info = data_info
            # update check point info
            reset_ckpt_path(cfg.model, data_info)
            depth_model = get_configured_monodepth_model(cfg, ).cuda()
            # load ckpt
            load_path = cfg.load_from
            if os.path.isfile(load_path):
                checkpoint = torch.load(load_path, map_location="cpu")
                ckpt_state_dict  = checkpoint['model_state_dict']
                depth_model.load_state_dict(ckpt_state_dict, strict=False)
            self.depth_model = depth_model
            for p in self.depth_model.parameters():
                p.requires_grad = False
        elif self.pretrained_depth_model == 'GT' or self.pretrained_depth_model == 'None':
            self.depth_model = None
        else:
            raise NotImplementedError

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)
    
    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas, is_train=False):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        
        if self.depth_model is not None:
            self.depth_model.eval()
        
        depth_preds = []
        assert B == len(img_metas)
        for i in range(B):
            if self.pretrained_depth_model == 'DepthAnythingV1':
                image_ = img_metas[i]['img_depthbranch']
                depth_pred = self.depth_model.infer_pil(image_, output_type="tensor", with_flip_aug=False)
                depth_pred = depth_pred.to(mlvl_feats[0].device)
            elif self.pretrained_depth_model == 'DepthAnythingV2':
                image_ = img_metas[i]['img_depthbranch'].to(mlvl_feats[0].device)
                depth_pred = self.depth_model.infer_image(image_, 480, 640, 480)
            elif self.pretrained_depth_model == 'Metric3Dv2-Small':
                image_ = img_metas[i]['img_depthbranch'].to(mlvl_feats[0].device)
                depth_pred, _, _ = self.depth_model.inference({'input': image_})
                depth_pred = depth_pred.squeeze()

                pad = img_metas[i]['pad']
                normalize_scale = img_metas[i]['normalize_scale']
                label_scale_factor = img_metas[i]['label_scale_factor']
                img_shape = img_metas[i]['img_shape'][0]
                depth_pred = depth_pred[pad[0] : depth_pred.shape[0] - pad[1], pad[2] : depth_pred.shape[1] - pad[3]]
                depth_pred = torch.nn.functional.interpolate(depth_pred[None, None, :, :], [img_shape[0], img_shape[1]], mode='bilinear').squeeze() # to original size
                depth_pred = depth_pred * normalize_scale / label_scale_factor
                depth_pred = (depth_pred > 0) * (depth_pred < 300) * depth_pred
            elif self.pretrained_depth_model == 'Metric3Dv2-Giant':
                image_ = img_metas[i]['img_depthbranch'].to(mlvl_feats[0].device)
                depth_pred, _, _ = self.depth_model.inference({'input': image_})
                depth_pred = depth_pred.squeeze()

                pad = img_metas[i]['pad']
                normalize_scale = img_metas[i]['normalize_scale']
                label_scale_factor = img_metas[i]['label_scale_factor']
                img_shape = img_metas[i]['img_shape'][0]
                depth_pred = depth_pred[pad[0] : depth_pred.shape[0] - pad[1], pad[2] : depth_pred.shape[1] - pad[3]]
                depth_pred = torch.nn.functional.interpolate(depth_pred[None, None, :, :], [img_shape[0], img_shape[1]], mode='bilinear').squeeze() # to original size
                depth_pred = depth_pred * normalize_scale / label_scale_factor
                depth_pred = (depth_pred > 0) * (depth_pred < 300) * depth_pred
            elif self.pretrained_depth_model == 'GT':
                depth_pred = img_metas[i]['depth_gt'].to(mlvl_feats[0].device).squeeze(0)  # using GT
            else:
                raise NotImplementedError

            depth_preds.append(depth_pred)

        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts, query_feats = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            depth_preds,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    all_query_feats=query_feats,
                    depth_preds=depth_preds)
    
    def forward_anchors(self, mlvl_feats, img_metas, depth_preds, anchor_points):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        query_feat = anchor_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts, query_feats = self.transformer(
            anchor_points,
            query_feat,
            mlvl_feats,
            depth_preds,
            img_metas=img_metas,
        )

        return dict(init_points=anchor_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    all_query_feats=query_feats)

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_freq_cfg = self.train_cfg.get('cls_freq')
        cls_freq_cfg = np.array(cls_freq_cfg)
        cls_freq_offset = self.train_cfg.get('cls_freq_offset')

        cls_weights = refine_pts.new_tensor(1 / np.log(cls_freq_cfg) - cls_freq_offset) * 400.
        cls_weights[self.ignore_label] = 0.
        label_weights = cls_weights * \
            self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr')
        empty_weights = self.train_cfg.get('empty_weights')

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr)
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes')
        rare_weights = self.train_cfg.get('rare_weights')
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx)
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)
        
        # TODO: re-weight for loss_pts between pred_pts and pred_paired_pts
        pred_pts_weights = refine_pts.new_ones(pred_paired_pts.shape[0])
        dist = torch.norm(refine_pts - pred_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr)
        pred_pts_weights[mask] = empty_weights

        for cls_idx in rare_classes:
            mask = (refine_pts_labels == cls_idx)
            pred_pts_weights[mask] = pred_pts_weights[mask].clamp(min=rare_weights)

        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights, 
                gt_pts_weights, pred_pts_weights)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0) # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights, pred_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list, 
             gt_masks_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts_weights = torch.cat(pred_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  weight=pred_pts_weights[..., None],
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts, img_metas=None):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict
    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_cls_scores = pred_dicts['all_cls_scores']
        all_refine_pts = pred_dicts['all_refine_pts']
        cls_scores = all_cls_scores[-1].sigmoid()
        refine_pts = all_refine_pts[-1]

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr')  # TODO: wtf is this?
        score_thr = self.test_cfg.get('score_thr')

        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (cls_scores > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]

            pts = torch.cat([refine_pts, cls_scores], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1], 
                                        self.voxel_num[2], self.num_classes))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

            labels = scores.argmax(dim=-1)
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = torch.logical_and(voxel_semantics[i] != self.empty_label, voxel_semantics[i] != self.ignore_label)
            cam_mask = torch.ones_like(voxel_semantics[i])
            curr_coors = coors[mask]
            curr_masks = cam_mask[mask]
            curr_labels = voxel_semantics[i][mask]

            gt_points.append(curr_coors)
            gt_masks.append(curr_masks) # camera mask (not used) and not empty
            gt_labels.append(curr_labels)
        
        return gt_points, gt_masks, gt_labels
