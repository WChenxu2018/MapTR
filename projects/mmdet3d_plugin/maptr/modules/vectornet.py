from typing import Optional
import torch
from torch import nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models import BACKBONES, build_loss

# @BACKBONES.register_module()
class VectornetExtractor(BaseModule):
    def __init__(
        self,
        static_input_dims: int = 5,
        embedding_dims: int = 256,
        ff_dims: int = 512,
        output_dims: int = 256,
        static_type_num: int = 3,
        node_num: int = 8,
        init_cfg: Optional[dict] = None,
        freeze_weights: bool = False,
        freeze_bn_stat: bool = False,
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
            freeze_weights=freeze_weights,
            freeze_bn_stat=freeze_bn_stat,
        )
        self.static_input_dims = static_input_dims
        self.embedding_dims = embedding_dims
        self.ff_dims = ff_dims
        self.output_dims = output_dims
        self.static_type_num = static_type_num
        self.node_num = node_num

        self.static_embedding_layer1 = nn.Sequential(
            nn.Linear(self.static_input_dims-1, self.ff_dims),
            nn.ReLU(),
            nn.Linear(self.ff_dims, self.embedding_dims//2),
        )
        self.type_embedding_layer = nn.Embedding(self.static_type_num, self.embedding_dims//2)
        self.static_embedding_layer2 = nn.Sequential(
            nn.Linear(self.embedding_dims, self.ff_dims),
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

    def process_input(self,  vectornet_feature_static: torch.Tensor):
        """
            Args:
            - vectornet_feature_static (B, P, N, 5), (4, 80, 10, 5)
            Outputs:
            - map_features (B, P, N, self.embedding_dims)
        """        
        static_feature = self.static_embedding_layer1(vectornet_feature_static[...,:4])
        static_type_embeddings = self.type_embedding_layer(vectornet_feature_static[...,-1].long())
        static_feature = torch.cat([static_feature, static_type_embeddings], dim=-1)
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
        mask_map = subgraph_input_mask[..., :1].view(B, P) # 为啥只取第一个vector的mask
        return subgraph_output, mask_map

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
        vectornet_feature_static = vectornet_feature_static[:, :, :self.node_num] # B, 80, 8, 5
        vectornet_mask_static = vectornet_mask_static[:, :, :self.node_num] # B, 80, 8   0代表存在

        map_feature = self.process_input(vectornet_feature_static)
        ego_map_feature, mask_map = self.map_subgraph_network(map_feature, vectornet_mask_static)
        
        return ego_map_feature[:, 1:], mask_map[:, 1:]