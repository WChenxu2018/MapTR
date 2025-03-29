import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS, BACKBONES
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
from mmdet3d.models.detectors.base import Base3DDetector
import math
from typing import Optional
import torch
from torch import nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models import BACKBONES, build_loss
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention

POS_INF = 64500


class VectorGnn(nn.Module):
    def __init__(self, dim):
        super(VectorGnn, self).__init__()
        self.dim = dim

        # 使用 Linear 层代替 Conv1d 层
        self.query_fc = nn.Linear(dim, dim // 2)
        self.key_fc = nn.Linear(dim, dim // 2)
        self.value_fc = nn.Linear(dim, dim)

    def forward(self, x, mask):
        # 输入 x 的形状: [B, P, C]
        B, P, C = x.shape

        # 计算 query, key 和 value
        query = self.query_fc(x)  # [B, P, D/2]
        key = self.key_fc(x)      # [B, P, D/2]
        value = self.value_fc(x)  # [B, P, D]

        D = query.size(-1)  # D 是 query 的最后一维大小，即 dim / 2

        # reshape 掩码张量，扩展到与 attention 矩阵相同的维度
        mask = mask.view(B, P, 1)

        # 计算注意力分数
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(D)  # [B, P, P]
        
        # 应用掩码，将无效位置设置为负无穷大
        attention = attention + (mask - 1) * POS_INF
        
        # 计算 softmax 归一化的注意力权重
        attention = F.softmax(attention, dim=-1)  # [B, P, P]

        # 使用注意力权重矩阵与 value 相乘得到最终输出
        output = torch.matmul(attention, value)  # [B, P, D]

        # 使用掩码更新输出张量，以确保无效位置的值被正确屏蔽
        output *= mask

        return output
    
@ATTENTION.register_module()
class StaticAttentionNet(BaseModule):
    def __init__(self, map_gnn_dim, gnn_encoder_layer_num) -> None:
        super().__init__()
        self.gnn_dim = map_gnn_dim
        self.gnn_encoders = nn.ModuleList()
        for l in range(gnn_encoder_layer_num):
            self.gnn_encoders.append(VectorGnn(self.gnn_dim))

    def forward(self, subgraph_output, mask_polyline):
        gnn_f = subgraph_output
        for l, gnn in enumerate(self.gnn_encoders):
            gnn_f = gnn(gnn_f, mask_polyline)
        return gnn_f

    def forward_dummy(self, subgraph_output, mask_polyline):
        gnn_f = subgraph_output
        for l, gnn in enumerate(self.gnn_encoders):
            gnn_f = gnn.forward_dummy(gnn_f, mask_polyline)
        return gnn_f

@BACKBONES.register_module()
class VectornetExtractor(BaseModule):
    def __init__(
        self,
        static_input_dims: int = 5,
        embedding_dims: int = 256,
        ff_dims: int = 512,
        map_num: int = 2,
        output_dims: int = 256,
        static_type_num: int = 3,
        node_num: int = 20,
        vector_num: int = 10,
        init_cfg: Optional[dict] = None,
        freeze_weights: bool = False,
        freeze_bn_stat: bool = False,
        attention_net = None
    ):
        """Head initialization.

        Args:
            save_dir (str, optional): The path the save the results.
                This is usually used during inferring the model.
                Defaults to None.
            **kwargs: Other parameters. Please refer to the `BaseModule`.
        """
        super().__init__(
            init_cfg=init_cfg,
        )
        self.static_input_dims = static_input_dims
        self.embedding_dims = embedding_dims
        self.ff_dims = ff_dims
        self.output_dims = output_dims
        self.static_type_num = static_type_num
        self.node_num = node_num
        self.map_num = map_num
        self.attention_net = build_attention(attention_net)
        self.static_embedding_layer1 = nn.Sequential(
            nn.Linear(self.static_input_dims-2, self.ff_dims),
            nn.ReLU(),
            nn.Linear(self.ff_dims, self.embedding_dims),
        )
        self.type_embedding_layer = nn.Embedding(self.static_type_num, self.embedding_dims//2)
        
        self.map_embedding = nn.Embedding(self.map_num, self.embedding_dims//2)
        
        self.static_embedding_layer2 = nn.Sequential(
            nn.Linear(2 * self.embedding_dims, self.ff_dims),
            nn.ReLU(),
            nn.Linear(self.ff_dims, self.embedding_dims),         
        )
        self.map_subgraph_encode1 = nn.Sequential(
            nn.Conv2d(self.embedding_dims, self.output_dims//2, 1), 
            nn.BatchNorm2d(self.output_dims//2), 
            nn.ReLU(inplace=True)
        )
        self.map_subgraph_encode2 = nn.Sequential(
            nn.Conv2d(self.output_dims, self.output_dims//2, 1), 
            nn.BatchNorm2d(self.output_dims//2), 
            nn.ReLU(inplace=True)
        )
        self.map_subgraph_encode3 = nn.Sequential(
            nn.Conv2d(self.output_dims, self.output_dims//2, 1), 
            nn.BatchNorm2d(self.output_dims//2), 
            nn.ReLU(inplace=True)
        )
        self.vector_num = vector_num
        
        # self.map2_embedding = nn.Embedding(self.vector_num, self.embedding_dims)

    def process_input(self,  vectornet_feature_static: torch.Tensor):
        """
            Args:
            - vectornet_feature_static (B, P, N, 5), (4, 80, 10, 5)
            Outputs:
            - map_features (B, P, N, self.embedding_dims)
        """        
        static_feature = self.static_embedding_layer1(vectornet_feature_static[...,:4])
        static_type_embeddings = self.type_embedding_layer(vectornet_feature_static[...,-2].long())
        map_source_type_embeddings = self.map_embedding(vectornet_feature_static[...,-1].long())
        static_feature = torch.cat([static_feature, static_type_embeddings, map_source_type_embeddings], dim=-1)
        static_feature = self.static_embedding_layer2(static_feature) # B, 80, 8, embedding_dims
        return static_feature

    def maxpooling_concat(self, mid_input: torch.Tensor, mask: torch.Tensor, concat=True):
        # mid_input = mid_input * (torch.ones_like(mask).to(mask.device) + torch.full([int(d) for d in mask.shape], -1.).to(mask.device) * mask)
        mid_input = mid_input * (1 - mask)
        # outputs = F.max_pool2d(mid_input, kernel_size=(1,self.node_num))
        outputs = F.max_pool2d(mid_input, kernel_size=(1,self.node_num), stride=(1, 1))
        if concat:
            outputs = outputs.repeat(1,1,1,self.node_num)
            outputs = torch.cat([mid_input, outputs], dim = 1)
        return outputs

    def map_subgraph_network(self, subgraph_input: torch.Tensor, subgraph_input_mask: torch.Tensor):
        '''
            to extract features in polylines between nodes
        '''
        B, P, N, D = subgraph_input.shape
        # to speed up, we use conv net to achieve mlp, the permute() function is to enable this
        subgraph_input = subgraph_input.view(B,P,N,-1).permute(0,3,1,2).contiguous() # (B, D, P, N)
        # subgraph_mask = mask.view(B,1,1,P,N).repeat(1,A,1,1,1).view(B*A,1,P,N)
        subgraph_mask = subgraph_input_mask.view(B,1,P,N).repeat(1, self.output_dims//2, 1, 1) #（B,D,P,N)
        subgraph_mid_output = self.map_subgraph_encode1(subgraph_input)
        subgraph_output = self.maxpooling_concat(subgraph_mid_output, subgraph_mask)

        subgraph_mid_output = self.map_subgraph_encode2(subgraph_output)
        subgraph_output = self.maxpooling_concat(subgraph_mid_output, subgraph_mask)

        subgraph_mid_output = self.map_subgraph_encode3(subgraph_output)
        subgraph_output = self.maxpooling_concat(subgraph_mid_output, subgraph_mask)

        subgraph_output = self.maxpooling_concat(subgraph_output, subgraph_mask.repeat(1, 2, 1, 1), concat=False)  # B, D, P, 1

        subgraph_output = subgraph_output.view(B,-1,P).transpose(2,1).contiguous() # B, P, D
        if self.node_num > 1:
            mask_static_polyline = F.max_pool2d(
                subgraph_input_mask, kernel_size=(1, subgraph_input_mask.shape[-1])
            )
        else:
            mask_static_polyline = mask_static_polyline
        # mask_map = subgraph_input_mask[..., :1].view(B, P) # 为啥只取第一个vector的mask
        subgraph_output = subgraph_output * mask_static_polyline
        return subgraph_output, mask_static_polyline

    def forward(
        self,
        vectornet_feature_static: torch.Tensor,
        vectornet_mask_static: torch.Tensor,
    ):        
        """
            Args:
            - vectornet_feature_static (B, P, N, 5)
            - vectornet_mask_static (B, P, N)
            Outputs:
            - map_feature (B, 1, 1, P, D)
            - mask_map (B, 1, 1, P)
        """
        B, P, N = vectornet_mask_static.shape
        half_P = P // 2
        vectornet_feature_static_map1 = vectornet_feature_static[:, :half_P, :self.node_num] 
        vectornet_mask_static_map1 = vectornet_mask_static[:, :half_P, :self.node_num]
        map1_feature = self.process_input(vectornet_feature_static_map1)
        subgraph1_output, mask1_static_polyline = self.map_subgraph_network(map1_feature, vectornet_mask_static_map1)

        vectornet_feature_static_map2 = vectornet_feature_static[:, half_P:, :self.node_num]
        vectornet_mask_static_map2 = vectornet_mask_static[:, half_P:, :self.node_num]
        map2_feature = self.process_input(vectornet_feature_static_map2)
        subgraph2_output, mask2_static_polyline = self.map_subgraph_network(map2_feature, vectornet_mask_static_map2)
        
        subgraph_output = torch.cat((subgraph1_output, subgraph2_output), dim = 1)
        mask_static_polyline = torch.cat((mask1_static_polyline, mask2_static_polyline), dim = 1)

        enc_feat_static = self.attention_net(
            subgraph_output, mask_static_polyline
        )
        # enc_feat_static = enc_feat_static.squeeze(-1).permute(0, 2, 1)
        # mask_static_polyline = mask_static_polyline.squeeze(-1).permute(0, 2, 1)

        
        return enc_feat_static, mask_static_polyline


@DETECTORS.register_module()
class MapFusionTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 vector_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 ):

        super(MapFusionTR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.vector_backbone = builder.build_backbone(vector_backbone)
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          map_feature,
                          map_mask,
                          gt_bboxes_3d,
                          gt_labels_3d):


        outs = self.pts_bbox_head(map_feature, map_mask)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # print(losses.item())
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

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
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      model_input_map1 = None,
                      model_input_map1_mask = None,
                      model_input_map2 = None,
                      model_input_map2_mask = None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      meta = None
                      ):
        input_map = torch.cat((model_input_map1, model_input_map2), dim = 1)
        input_map_mask = torch.cat((model_input_map1_mask, model_input_map2_mask), dim = 1)

        map_feature, map_mask = self.vector_backbone(input_map, input_map_mask)

        losses = dict()
        losses_pts = self.forward_pts_train(map_feature, map_mask, gt_bboxes_3d,
                                            gt_labels_3d)

        losses.update(losses_pts)
        for k in ['loss_cls', 'loss_bbox', 'loss_iou', 'loss_pts', 'loss_dir']:
            loss = losses_pts[k]
            print(f"Loss: {k}, {loss}")
        return losses

    def forward_test(self, 
                     model_input_map1 = None,
                      model_input_map1_mask = None,
                      model_input_map2 = None,
                      model_input_map2_mask = None,
                      meta = None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **kwargs):
        input_map = torch.cat((model_input_map1, model_input_map2), dim = 1)
        input_map_mask = torch.cat((model_input_map1_mask, model_input_map2_mask), dim = 1)
        map_feature, map_mask = self.vector_backbone(input_map, input_map_mask)
        outs = self.pts_bbox_head(map_feature, map_mask)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # print("outs ", outs)
        ### hack         bbox_list_output = [dict() for i in range(len(meta))]
        bbox_list_output = [dict()]
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas = None, rescale=True)
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        for result_dict, pts_bbox in zip(bbox_list_output, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list_output


    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict




@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
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
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 ):

        super(MapTR,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue) #img: torch.Size([4, 6, 3, 480, 800])
        
        return img_feats #torch.Size([4, 6, 256, 15, 25])


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

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
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, None, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
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
                      ):
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
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None
        
        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None,points=None,  **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        # import pdb;pdb.set_trace()
        return outs['bev_embed'], bbox_results
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list


@DETECTORS.register_module()
class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16. 
    We provide this version to resolve this issue.
    """
    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
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
                      prev_bev=None,
                      ):
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
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev=prev_bev)
        losses.update(losses_pts)
        return losses


    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data['img']
        img_metas = data['img_metas']
        img_feats = self.extract_feat(img=img,  img_metas=img_metas)
        prev_bev = data.get('prev_bev', None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
