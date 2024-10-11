import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from modules import *

HIDDEN_SIZE_SPACIAL = 256
KEYPOINTS_NUM = 17
BOUNDING_BOX_POINT_NUM = 4
ATTN_EMB_SIZE = 128
NUM_PEDESTRIAN = 10
SEMENTIC_CLASS_NUM = 10


class PedestrianCrossingModel(nn.Module):
    def __init__(self):
        super(PedestrianCrossingModel, self).__init__()

        # Bouding Box embedding
        self.bb_emb = nn.Linear(BOUNDING_BOX_POINT_NUM, 128)
        # Pbb 和 Pbp (Bounding Box 和 Body Pose)
        self.shift_body_pose = ShiftGRUModule(
            input_size=2 * KEYPOINTS_NUM, hidden_size=128
        )
        # Bounding Box和Body Pose的GRU
        self.gru_bb_bp = nn.GRU(2 * 128, hidden_size=128, batch_first=True)
        # 对Vehicle Speed进行嵌入
        self.vehicle_speed_emb = nn.Linear(1, 128)
        # 对Bounding Box、Body Pose、Vehicle Speed的长短期依赖关系进行摘要
        self.gru_bb_bp_vs = nn.GRU(2 * 128, ATTN_EMB_SIZE, batch_first=True)
        # Attention mechanism of spatial kinematic
        self.attention_spatial_kinematic = AdditiveAttention(ATTN_EMB_SIZE)

        # Local motion extractor
        self.local_motion_module = LocalMotionExtractor()

        # Local content extractor
        self.local_content_module = LocalContentExtractor()

        # gru of local content and local motion
        self.gru_lm_lc = nn.GRU(2 * 128, ATTN_EMB_SIZE)

        # Attention mechanism of local content and local motion
        self.attention_lm_lc = AdditiveAttention(ATTN_EMB_SIZE)

        # Semantic Context 和 Categorical Depth
        self.semantic_context_extractor = SementicContextExtractor(SEMENTIC_CLASS_NUM)
        self.categorical_depth_extractor = SementicContextExtractor()
        # Semantic Context 和 Categorical Depth的GRU
        self.gru_sc_cd = nn.GRU(1024, 512)

        # Attention mechanism of spatial context
        self.attention_sc_cd = AdditiveAttention(512)
        self.fc_sc_cd = nn.Linear(512, 128)

        # Attention of all，使用hugging face上预训练过的encoder
        model_name = "prajjwal1/bert-small"
        self.attn_all = AutoModel.from_pretrained(model_name)

        # 分类层
        self.classifier = nn.Linear(128, NUM_PEDESTRIAN)

    def forward(
        self,
        pbb_input,
        pbp_input,
        vehicle_speed,
        motion_input,
        content_input,
        semantic_context_input,
        depth_input,
    ):
        # Pbb 和 Pbp 处理
        pbb_out = self.bb_emb(pbb_input)  # 输出 shape: (batch_size, seq_len, 128)
        pbp_out = self.shift_body_pose(pbp_input)  # 输出 shape: (batch, seq_len, 128)
        bb_bp_out, hn_bb_pp = self.gru_bb_bp(
            torch.cat((pbb_out, pbp_out), -1)
        )  # (batch_size, seq_len, 128)
        speed_emb = self.vehicle_speed_emb(vehicle_speed)  # (batch_size, seq_len, 128)
        kinematic_fusion, hn_bb_bp_vs = self.gru_bb_bp_vs(
            torch.cat((F.relu(bb_bp_out), speed_emb), -1)
        )  # (batch_size, seq_len, 128)
        context_kinematic_fusion, _ = self.attention_spatial_kinematic(
            kinematic_fusion, hn_bb_bp_vs
        )  # (batch_size, 128)

        # Local Motion 和 Local Content 处理
        motion_out = self.local_motion_module(
            motion_input
        )  # 输出 shape: (batch, seq_len, 128)
        content_out = self.local_content_module(
            content_input
        )  # 输出 shape: (batch, seq_len, 128)

        local_content_motion = torch.cat(
            (content_out, motion_out), -1
        )  # (batch_size, seq_len, 2*128)
        lc_lm_fusion, hn_lc_lm = self.gru_lm_lc(
            local_content_motion
        )  # (batch_size, seq_len, 128), (batch_size, 128)
        context_lc_lm, _ = self.attention_lm_lc(
            lc_lm_fusion, hn_lc_lm
        )  # (batch_size, hidden_size)

        # 语义上下文和深度特征处理
        semantic_context_out = F.relu(
            self.semantic_context_extractor(semantic_context_input)
        )  # (batch_size, seq_len, 512)
        depth_out = F.relu(
            self.categorical_depth_extractor(depth_input)
        )  # (batch_size, seq_len, 512)
        sc_cd_fusion, hn_sc_cd = self.gru_sc_cd(
            torch.cat((semantic_context_out, depth_out, -1))
        )
        context_sc_cd, _ = self.attention_sc_cd(
            sc_cd_fusion, hn_sc_cd
        )  # (batch_size, 512)
        context_sc_cd = self.fc_sc_cd(context_sc_cd)

        cls_emb = nn.Parameter(torch.randn(context_sc_cd.shape)).unsqueeze(1)
        sentence = torch.stack(
            (cls_emb, context_kinematic_fusion, context_lc_lm, context_sc_cd), dim=1
        )
        input_emb = torch.cat((cls_emb, sentence), dim=1)
        cls_out = self.attn_all(input_emb)["last_hidden_state"][:, 0, :]
        final_output = self.classifier(cls_out)

        return torch.sigmoid(final_output)  # 【驾驶员减速意愿预测】


# 模型实例
model = PedestrianCrossingModel()
