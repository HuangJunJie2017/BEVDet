# Copyright (c) Phigent Robotics. All rights reserved.

import os
import torch
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .. import builder
from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self,img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x = self.img_view_transformer([x] + img[1:])
        x = self.bev_encoder(x)
        return [x]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
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
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get('combine_type','output')
        if combine_type=='output':
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type=='feature':
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class BEVDetSequential(BEVDet):
    def __init__(self, aligned=False, distill=None, pre_process=None,
                 pre_process_neck=None, detach=True, test_adj_ids=None, **kwargs):
        super(BEVDetSequential, self).__init__(**kwargs)
        self.aligned = aligned
        self.distill = distill is not None
        if self.distill:
            self.distill_net = builder.build_neck(distill)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        self.test_adj_ids = test_adj_ids
    
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, rot, tran, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H,
                                 W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            if self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)
                if self.pre_process_neck:
                    bev_feat = self.pre_process_neck_net(bev_feat)
                else:
                    bev_feat = bev_feat[0]
            bev_feat_list.append(bev_feat)
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES(BEVDetSequential):
    def __init__(self, before=False, interpolation_mode='bilinear',**kwargs):
        super(BEVDetSequentialES, self).__init__(**kwargs)
        self.before=before
        self.interpolation_mode=interpolation_mode

    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape
        _,v,_ =trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current frame to adjacent frame
        l02c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l02c[:,:,:3,:3] = rots[0]
        l02c[:,:,:3,3] = trans[0]
        l02c[:,:,3,3] =1

        l12c = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        l12c[:,:,:3,:3] = rots[1]
        l12c[:,:,:3,3] = trans[1]
        l12c[:,:,3,3] =1
        # l0tol1 = l12c.matmul(torch.inverse(l02c))[:,0,:,:].view(n,1,1,4,4)
        l0tol1 = l02c.matmul(torch.inverse(l12c))[:,0,:,:].view(n,1,1,4,4)

        l0tol1 = l0tol1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l0tol1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output


    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, _ , _, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


class BEVDepth_Base():
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)


    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def get_depth_loss(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (~(depth_gt == 0)).reshape(B, N, 1, H, W).expand(B, N,
                                                                       self.img_view_transformer.D,
                                                                       H, W)
        depth_gt = torch.clip(torch.floor(depth_gt) -
                              self.img_view_transformer.grid_config['dbound'][
                                  0],
                              0, self.img_view_transformer.D).to(torch.long)
        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32)
        depth = depth.sigmoid().view(B, N, self.img_view_transformer.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit,
                                            weight=loss_weight)
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth


@DETECTORS.register_module()
class BEVDepth(BEVDepth_Base, BEVDet):
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:])
        x = self.bev_encoder(x)
        return [x], depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepth4D(BEVDepth_Base, BEVDetSequentialES):
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x], depth_digit_list[0]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
