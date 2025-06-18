import time
import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

@DETECTORS.register_module()
class DiScene_Distill(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=True,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 teacher_model=None,
                 teacher_weight=None,
                 pretrained=None):
        super().__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
                         pts_fusion_layer, img_backbone, pts_backbone, img_neck,
                         pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
                         train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask

        self.memory = {}
        self.queue = queue.Queue()

        self.teacher_cfg = teacher_model
        self.teacher_weight = teacher_weight
        self.teacher_model = build_model(self.teacher_cfg)
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def load_teacher(self):
        checkpoint = torch.load(self.teacher_weight, map_location='cuda')['state_dict']
        self.teacher_model.load_state_dict(checkpoint, strict=True)
        # load_checkpoint(teacher_model, teacher_weight, map_location='cuda', strict=True)
        self.teacher_model.to('cuda')
        self.teacher_model.eval()

        # Teacher-Guided Initialization
        new_transformer_state_dict = dict()
        for k, v in checkpoint.items():
            elif k.startswith('pts_bbox_head.transformer.'):
                new_k = k[len('pts_bbox_head.transformer.'):]
                new_transformer_state_dict[new_k] = v
        self.pts_bbox_head.transformer.load_state_dict(new_transformer_state_dict, strict=True)

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      voxel_semantics=None,
                      mask_camera=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_feat(img, img_metas)
        src_feats = [img_feats[lvl].clone() for lvl in range(len(img_feats))]
        prior_src_feats = [img_feats[lvl].clone() for lvl in range(len(img_feats))]
        anchor_src_feats = [img_feats[lvl].clone() for lvl in range(len(img_feats))]
        outs = self.pts_bbox_head(img_feats, img_metas, is_train=True)
        depth_preds = outs['depth_preds']

        teacher_img_feats = self.teacher_model.extract_feat(img, img_metas)
        tgt_feats = [teacher_img_feats[lvl].clone() for lvl in range(len(teacher_img_feats))]
        anchor_tgt_feats = [teacher_img_feats[lvl].clone() for lvl in range(len(teacher_img_feats))]
        teacher_outs = self.teacher_model.pts_bbox_head(teacher_img_feats, img_metas, is_train=True)
        teacher_prior_points = teacher_outs['init_points']
        teacher_depth_preds = teacher_outs['depth_preds']

        # loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, src_feats, tgt_feats, img_metas]

        # Prior-Level Knowledge Distillation
        prior_outs = self.pts_bbox_head.forward_anchors(prior_src_feats, img_metas, depth_preds, teacher_prior_points)
        # loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, prior_outs, teacher_outs, src_feats, tgt_feats, img_metas]

        # Anchor-Level Knowledge Distillation
        anchor_points, anchor_labels = self.pts_bbox_head.get_anchors(voxel_semantics=voxel_semantics)
        anchor_outs = self.pts_bbox_head.forward_anchors(anchor_src_feats, img_metas, depth_preds, anchor_points)
        teacher_anchor_outs = self.teacher_model.pts_bbox_head.forward_anchors(anchor_tgt_feats, img_metas, teacher_depth_preds, anchor_points)
        # loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, anchor_outs, teacher_anchor_outs, src_feats, tgt_feats, img_metas]

        # Query-Level Knowledge Distillation
        # loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, src_feats, tgt_feats, img_metas]

        # Total
        # loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, src_feats, tgt_feats, img_metas]
        loss_inputs = [voxel_semantics, mask_camera, outs, teacher_outs, prior_outs, teacher_outs, anchor_outs, teacher_anchor_outs, src_feats, tgt_feats, img_metas]

        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        return self.simple_test(img_metas, img, **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, img_metas)
        return self.pts_bbox_head.get_occ(outs, img_metas[0], rescale=rescale)
    
    def simple_test(self, img_metas, img=None, rescale=False, voxel_semantics=None):
        # world_size = get_dist_info()[1]
        return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        img_feats = self.extract_feat(img, img_metas)
        return self.simple_test_pts(img_feats, img_metas, rescale=rescale)
