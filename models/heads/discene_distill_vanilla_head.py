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
from torch.utils.data import WeightedRandomSampler
import random
from mmcv import Config
import os

from ..bbox.utils import decode_bbox, decode_points, encode_points
from .matcher import HungarianMatcher

@HEADS.register_module()
class DiSceneHead_Distill_Vanilla(BaseModule):
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
                 loss_guide_pts_weight=3.0,
                 loss_guide_feats_weight=2.0,
                 loss_prior_pts_weight=3.0,
                 loss_prior_feats_weight=2.0,
                 loss_anchor_pts_weight=3.0,
                 loss_anchor_feats_weight=2.0,
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

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self.loss_guide_pts_weight   = loss_guide_pts_weight * 0.2
        self.loss_guide_feats_weight = loss_guide_feats_weight * 0.2

        self.loss_prior_pts_weight   = loss_prior_pts_weight * 0.2
        self.loss_prior_feats_weight = loss_prior_feats_weight * 0.2

        self.loss_anchor_pts_weight   = loss_anchor_pts_weight * 0.5
        self.loss_anchor_feats_weight = loss_anchor_feats_weight * 0.5

        self.matcher = HungarianMatcher(cost_pts=1.)

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)
    
    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas, is_train=False):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts, query_feats = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    all_query_feats=query_feats)
    
    def get_anchors(self, voxel_semantics=None):
        voxel_semantics = voxel_semantics.long()
        B, W, H, Z = voxel_semantics.shape
        cls_freq_cfg = self.train_cfg.get('cls_freq')
        cls_freq_cfg = np.array(cls_freq_cfg)
        cls_freq_offset = self.train_cfg.get('cls_freq_offset')
        cls_weights = torch.Tensor(1 / np.log(cls_freq_cfg) - cls_freq_offset).to(voxel_semantics.device)
        cls_weights *= 800.

        x = torch.arange(0, W, dtype=torch.float32, device=voxel_semantics.device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=voxel_semantics.device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=voxel_semantics.device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        anchor_points, anchor_labels = [], []
        for i in range(B):
            curr_mask = torch.logical_and(voxel_semantics[i] != self.empty_label, voxel_semantics[i] != self.ignore_label)
            curr_coors = coors[curr_mask]
            curr_labels = voxel_semantics[i][curr_mask]

            curr_weights = torch.zeros_like(curr_labels).to(voxel_semantics.device).float()
            for cls in range(self.num_classes):
                curr_weights[curr_labels == cls] = cls_weights[cls]
            sampler = WeightedRandomSampler(curr_weights, num_samples=self.num_query, replacement=False)
            curr_anchor_coors = curr_coors[list(sampler)]
            curr_anchor_labels = curr_labels[list(sampler)]

            curr_anchor_points = encode_points(curr_anchor_coors, self.pc_range)
            anchor_points.append(curr_anchor_points)
            anchor_labels.append(curr_anchor_labels)
        
        anchor_points = torch.stack(anchor_points)[:, :, None, :]
        anchor_labels = torch.stack(anchor_labels)
        return anchor_points, anchor_labels
    
    def forward_anchors(self, mlvl_feats, img_metas, anchor_points):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        query_feat = anchor_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts, query_feats = self.transformer(
            anchor_points,
            query_feat,
            mlvl_feats,
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
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def loss_single_match(self,
                    cls_scores,
                    refine_pts,
                    query_feats,
                    teacher_cls_scores,
                    teacher_refine_pts,
                    teacher_query_feats):
        B, Q, R = cls_scores.shape[:-1]
        
        query_ctrs = refine_pts.mean(dim=2)
        query_ctrs = decode_points(query_ctrs, self.pc_range)

        teacher_query_ctrs = teacher_refine_pts.mean(dim=2)
        teacher_query_ctrs = decode_points(teacher_query_ctrs, self.pc_range)

        outputs = dict(pred_points=query_ctrs)
        targets = [dict(points=teacher_query_ctrs[i]) for i in range(B)]

        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)

        src_points = outputs['pred_points'][idx]
        tgt_points = torch.cat([t['points'][ind] for t, (_, ind) in zip(targets, indices)], dim=0)
        loss_guide_pts = F.l1_loss(src_points, tgt_points, reduction='none').sum() / (B * Q)
        loss_guide_pts = loss_guide_pts * self.loss_guide_pts_weight

        src_feats = query_feats.flatten(0, 1)
        teacher_feats = [teacher_query_feats[i] for i in range(B)]
        target_feats = torch.cat([teacher_feat[ind] for teacher_feat, (_, ind) in zip(teacher_feats, indices)], dim=0)
        f_s = F.normalize(src_feats, p=2, dim=1)
        f_t = F.normalize(target_feats, p=2, dim=1)
        loss_guide_feats = nn.MSELoss(reduction='none')(f_s, f_t).sum() / (B * Q)
        loss_guide_feats = loss_guide_feats * self.loss_guide_feats_weight

        return loss_guide_pts, loss_guide_feats

    def loss_single_match_prior(self,
                    cls_scores,
                    refine_pts,
                    query_feats,
                    teacher_cls_scores,
                    teacher_refine_pts,
                    teacher_query_feats):
        B, Q, R = cls_scores.shape[:-1]

        src_ctrs = refine_pts.mean(dim=2).reshape(-1, 3)
        src_ctrs = decode_points(src_ctrs, self.pc_range)
        tgt_ctrs = teacher_refine_pts.mean(dim=2).reshape(-1, 3)
        tgt_ctrs = decode_points(tgt_ctrs, self.pc_range)
        loss_prior_pts = F.l1_loss(src_ctrs, tgt_ctrs, reduction='none').sum() / (B * Q)
        loss_prior_pts = loss_prior_pts * self.loss_prior_pts_weight

        src_feats = query_feats.flatten(0, 1)
        tgt_feats = teacher_query_feats.flatten(0, 1)
        f_s = F.normalize(src_feats, p=2, dim=1)
        f_t = F.normalize(tgt_feats, p=2, dim=1)
        loss_prior_feats = nn.MSELoss(reduction='none')(f_s, f_t).sum() / (B * Q)
        loss_prior_feats = loss_prior_feats * self.loss_prior_feats_weight

        return loss_prior_pts, loss_prior_feats

    def loss_single_match_anchor(self,
                    cls_scores,
                    refine_pts,
                    query_feats,
                    teacher_cls_scores,
                    teacher_refine_pts,
                    teacher_query_feats):
        B, Q, R = cls_scores.shape[:-1]

        src_ctrs = refine_pts.mean(dim=2).reshape(-1, 3)
        src_ctrs = decode_points(src_ctrs, self.pc_range)
        tgt_ctrs = teacher_refine_pts.mean(dim=2).reshape(-1, 3)
        tgt_ctrs = decode_points(tgt_ctrs, self.pc_range)
        loss_anchor_pts = F.l1_loss(src_ctrs, tgt_ctrs, reduction='none').sum() / (B * Q)
        loss_anchor_pts = loss_anchor_pts * self.loss_anchor_pts_weight

        src_feats = query_feats.flatten(0, 1)
        tgt_feats = teacher_query_feats.flatten(0, 1)
        f_s = F.normalize(src_feats, p=2, dim=1)
        f_t = F.normalize(tgt_feats, p=2, dim=1)
        loss_anchor_feats = nn.MSELoss(reduction='none')(f_s, f_t).sum() / (B * Q)
        loss_anchor_feats = loss_anchor_feats * self.loss_anchor_feats_weight

        return loss_anchor_pts, loss_anchor_feats

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, 
             voxel_semantics, 
             mask_camera, 
             preds_dicts, 
             teacher_preds_dicts=None, 
             prior_dicts=None, 
             teacher_prior_dicts=None, 
             anchor_dicts=None,
             teacher_anchor_dicts=None,
             img_feats=None, 
             teacher_img_feats=None, 
             img_metas=None):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']
        all_query_feats = preds_dicts['all_query_feats']

        teacher_all_cls_scores = teacher_preds_dicts['all_cls_scores']
        teacher_all_refine_pts = teacher_preds_dicts['all_refine_pts']
        teacher_all_query_feats = teacher_preds_dicts['all_query_feats']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list)
        
        loss_guide_feats_2d = init_points.new_tensor(0)
        for lvl in range(len(img_feats)):
            # b_, n_, c_, h_, w_ = img_feats[lvl].shape
            src_feat = img_feats[lvl].permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.embed_dims)
            tgt_feat = teacher_img_feats[lvl].permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.embed_dims)
            loss_guide_feats_2d += nn.MSELoss(reduction='sum')(F.normalize(src_feat, p=2, dim=1), F.normalize(tgt_feat, p=2, dim=1)) / src_feat.shape[0]
        
        losses_guide_pts, losses_guide_feats = multi_apply(
            self.loss_single_match, all_cls_scores, all_refine_pts, all_query_feats,
            teacher_all_cls_scores, teacher_all_refine_pts, teacher_all_query_feats)
        
        student_prior_all_cls_scores = prior_dicts['all_cls_scores']
        student_prior_all_refine_pts = prior_dicts['all_refine_pts']
        student_prior_all_query_feats = prior_dicts['all_query_feats']

        teacher_prior_all_cls_scores = teacher_prior_dicts['all_cls_scores']
        teacher_prior_all_refine_pts = teacher_prior_dicts['all_refine_pts']
        teacher_prior_all_query_feats = teacher_prior_dicts['all_query_feats']
        
        losses_prior_pts, losses_prior_feats = multi_apply(
            self.loss_single_match_prior, student_prior_all_cls_scores, student_prior_all_refine_pts, student_prior_all_query_feats,
            teacher_prior_all_cls_scores, teacher_prior_all_refine_pts, teacher_prior_all_query_feats)

        student_anchor_all_cls_scores = anchor_dicts['all_cls_scores']
        student_anchor_all_refine_pts = anchor_dicts['all_refine_pts']
        student_anchor_all_query_feats = anchor_dicts['all_query_feats']

        teacher_anchor_all_cls_scores = teacher_anchor_dicts['all_cls_scores']
        teacher_anchor_all_refine_pts = teacher_anchor_dicts['all_refine_pts']
        teacher_anchor_all_query_feats = teacher_anchor_dicts['all_query_feats']
        
        losses_anchor_pts, losses_anchor_feats = multi_apply(
            self.loss_single_match_anchor, student_anchor_all_cls_scores, student_anchor_all_refine_pts, student_anchor_all_query_feats,
            teacher_anchor_all_cls_scores, teacher_anchor_all_refine_pts, teacher_anchor_all_query_feats)

        loss_dict = dict()
        loss_dict['loss_guide_img_feats'] = loss_guide_feats_2d
        
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

        loss_dict['loss_guide_pts'] = losses_guide_pts[-1]
        loss_dict['loss_guide_feats'] = losses_guide_feats[-1]

        loss_dict['loss_prior_pts'] = losses_prior_pts[-1]
        loss_dict['loss_prior_feats'] = losses_prior_feats[-1]

        loss_dict['loss_anchor_pts'] = losses_anchor_pts[-1]
        loss_dict['loss_anchor_feats'] = losses_anchor_feats[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_guide_pts_i, loss_guide_feats_i, loss_prior_pts_i, loss_prior_feats_i, loss_anchor_pts_i, loss_anchor_feats_i in \
                zip(losses_cls[:-1], losses_pts[:-1], losses_guide_pts[:-1], losses_guide_feats[:-1], losses_prior_pts[:-1], losses_prior_feats[:-1], losses_anchor_pts[:-1], losses_anchor_feats[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            
            loss_dict[f'd{num_dec_layer}.loss_guide_pts'] = loss_guide_pts_i
            loss_dict[f'd{num_dec_layer}.loss_guide_feats'] = loss_guide_feats_i

            loss_dict[f'd{num_dec_layer}.loss_prior_pts'] = loss_prior_pts_i
            loss_dict[f'd{num_dec_layer}.loss_prior_feats'] = loss_prior_feats_i

            loss_dict[f'd{num_dec_layer}.loss_anchor_pts'] = loss_anchor_pts_i
            loss_dict[f'd{num_dec_layer}.loss_anchor_feats'] = loss_anchor_feats_i

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
