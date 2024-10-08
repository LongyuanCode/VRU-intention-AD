import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

HIDDEN_SIZE_SPACIAL = 256
KEYPOINTS_NUM = 17
BOUNDING_BOX_POINT_NUM = 4
ATTN_EMB_SIZE = 128
NUM_PEDESTRIAN = 10


class ShiftGRUModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ShiftGRUModule, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

    def forward(self, x):
        # 假设 Shift 操作在数据预处理阶段已经完成
        output, _ = self.gru(x)
        return output


class CNNGRUModule(nn.Module):
    def __init__(self, input_channels, conv_output_size, gru_hidden_size):
        super(CNNGRUModule, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_output_size,
            kernel_size=3,
            padding=1,
        )
        self.gru = nn.GRU(
            input_size=conv_output_size, hidden_size=gru_hidden_size, batch_first=True
        )

    def forward(self, x):
        cnn_out = F.relu(self.cnn(x))
        # 将卷积输出 reshaped 以适应 GRU 输入维度
        cnn_out = cnn_out.view(
            cnn_out.size(0), cnn_out.size(1), -1
        )  # 展平成适合 GRU 的形状
        gru_out, _ = self.gru(cnn_out)
        return gru_out


class LocalContentExtractor(nn.Module):
    def __init__(self):
        super(LocalContentExtractor, self).__init__()

        # Conv3D layer: 只在空间维度上进行卷积
        self.conv3d = nn.Conv3d(
            in_channels=3,
            out_channels=512,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        # 3D Max Pooling layer: 只在空间维度上进行融合
        self.maxpool3d = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.fc = nn.Linear(512 * 56 * 56, 512)
        # GRU layer (m x 512 -> m x 128)
        self.gru = nn.GRU(input_size=512, hidden_size=128, batch_first=True)

    def forward(self, x):
        # Input x: (batch_size, m, 224, 224, 3)
        batch_size, m, h, w, channels = x.shape
        # 对维度进行切换: (batch_size, 3, m, h, w)
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 3, m, 224, 224)
        # 对空间维度h、w进行卷积计算
        x = self.conv3d(x)  # (batch_size, 512, m, h', w')
        # 对空间维度h、w进行融合
        x = self.maxpool3d(x)  # (batch_size, 512, m, 56, 56)
        # 展平
        batch_size, c, m, h_prime, w_prime = x.shape
        x = x.view(
            batch_size, m, c * h_prime * w_prime
        )  # (batch_size, m, 512 * 56 * 56)
        x = self.fc(x)  # (batch_size, m, 512)
        output, _ = self.gru(x)  # (batch_size, m, 128)

        return output


class LocalMotionExtractor(nn.Module):
    def __init__(self):
        super(LocalMotionExtractor, self).__int__()
        # Conv2d layer: 对光流估计的结果(H, W, 2)进行空间上的卷积计算
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(3, 3))
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
        self.fc = nn.LazyLinear(256)

    def forward(self, x):
        # Input x: (batch_size, m, h, w, 2)
        batch_size, m, h, w, channels = x.shape
        x = x.view(batch_size * m, h, w, channels)
        x = x.permute(0, 3, 1, 2)  # (batch_size*m, channels, h, w)
        # 光流估计算法输出的两个通道分别表示水平和垂直方向上像素位移分量
        x = self.conv2d(x)  # (batch_size*m, 256, h', w')
        batch_size_m, cnn_out_channels, h_prime, w_prime = x.shape
        x = x.view(batch_size, m, cnn_out_channels, h_prime, w_prime)
        x = x.view(batch_size, m, cnn_out_channels * h_prime * w_prime)
        x = self.fc(x)  # (batch_size, m, 256)
        output, _ = self.gru(x)  # (bathc_size, m, 128)

        return output


class SementicContextExtractor(nn.Module):
    def __init__(self):
        super(SementicContextExtractor, self).__init__()

        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.maxpool3d = nn.MaxPool3d(
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
        )
        self.fc = nn.Linear(128 * 128, 512)

    def forward(self, x):
        # Input x: (batch_size, m, h=512, w=512)
        x = x.unsqueeze(1)  # (batch_size, c=1, m, h, w)
        x = self.conv3d(x)  # (batch_size, c=1, m, h'=h, w'=w)
        x = self.maxpool3d(x)  # (batch_size, c=1, m, 128, 128)
        x = x.squeeze(1)  # (batch_size, m, 128, 128)
        batch_size, m, h, w = x.shape
        x = x.view(batch_size, m, h * w)  # (batch_size, m, 128*128)
        output = self.fc(x)  # (batch_size, m, 512)

        return output


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        # 定义线性变换 W_h, W_s, v
        self.W_h = nn.Linear(hidden_size, hidden_size)  # 作用在 GRU 输出 h_t 上
        self.W_s = nn.Linear(hidden_size, hidden_size)  # 作用在最终状态 s 上
        self.v = nn.Linear(hidden_size, 1)  # 计算得分

    def forward(self, h, s):
        """
        h: (batch_size, seq_len, hidden_size) - GRU 所有时间步的输出
        s: (batch_size, hidden_size) - GRU 最后一个时间步的隐藏状态
        """
        # 将 s 扩展到 (batch_size, seq_len, hidden_size)
        s_expanded = s.unsqueeze(1).expand_as(
            h
        )  # s_expanded: (batch_size, seq_len, hidden_size)

        # 计算 e_t: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, 1)
        e_t = self.v(
            torch.tanh(self.W_h(h) + self.W_s(s_expanded))
        )  # e_t: (batch_size, seq_len, 1)

        # 注意力权重 alpha_t: (batch_size, seq_len, 1)
        alpha_t = F.softmax(e_t, dim=1)  # 对 seq_len 维度进行 softmax

        # 计算上下文向量 c: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        context = torch.sum(alpha_t * h, dim=1)  # 加权求和，消去 seq_len 维度

        return context, alpha_t  # 返回上下文向量和注意力权重


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
        self.semantic_context_extractor = SementicContextExtractor()
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

        return torch.sigmoid(final_output)  # 行人横穿意图的预测 (0.0 - 1.0)


# 模型实例
model = PedestrianCrossingModel()
