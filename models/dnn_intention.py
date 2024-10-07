import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE_SPACIAL = 256
KEYPOINTS_NUM = 17
BOUNDING_BOX_POINT_NUM = 4
ATTN_EMB_SIZE = 128

class ShiftGRUModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ShiftGRUModule, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x):
        # 假设 Shift 操作在数据预处理阶段已经完成
        output, _ = self.gru(x)
        return output

class CNNGRUModule(nn.Module):
    def __init__(self, input_channels, conv_output_size, gru_hidden_size):
        super(CNNGRUModule, self).__init__()
        self.cnn = nn.Conv2d(in_channels=input_channels, out_channels=conv_output_size, kernel_size=3, padding=1)
        self.gru = nn.GRU(input_size=conv_output_size, hidden_size=gru_hidden_size, batch_first=True)
    
    def forward(self, x):
        cnn_out = F.relu(self.cnn(x))
        # 将卷积输出 reshaped 以适应 GRU 输入维度
        cnn_out = cnn_out.view(cnn_out.size(0), cnn_out.size(1), -1)  # 展平成适合 GRU 的形状
        gru_out, _ = self.gru(cnn_out)
        return gru_out
    
class LocalContentExtractor(nn.Module):
    def __init__(self):
        super(LocalContentExtractor, self).__init__()
        
        # Conv3D layer: 只在空间维度上进行卷积
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
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
        x = x.view(batch_size, m, c * h_prime * w_prime)  # (batch_size, m, 512 * 56 * 56)
        x = self.fc(x)  # (batch_size, m, 512)
        output, _ = self.gru(x)  # (batch_size, m, 128)
        
        return output
    
class LocalMotionExtractor(nn.Module):
    def __init__(self):
        super(LocalMotionExtractor, self).__int__()
        # Conv2d layer: 对光流估计的结果(H, W, 2)进行空间上的卷积计算
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(3,3))
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
        self.fc = nn.LazyLinear(256)
    
    def forward(self, x):
        # Input x: (batch_size, m, h, w, 2)
        batch_size, m, h, w, channels = x.shape
        x = x.view(batch_size*m, h, w, channels)
        x = x.permute(0, 3, 1, 2)   # (batch_size*m, channels, h, w)
        # 光流估计算法输出的两个通道分别表示水平和垂直方向上像素位移分量
        x = self.conv2d(x)  # (batch_size*m, 256, h', w')
        batch_size_m, cnn_out_channels, h_prime, w_prime = x.shape
        x = x.view(batch_size, m, cnn_out_channels, h_prime, w_prime)
        x = x.view(batch_size, m, cnn_out_channels * h_prime * w_prime)
        x = self.fc(x)  # (batch_size, m, 256)
        output, _ = self.gru(x) # (bathc_size, m, 128)

        return output
        


class AttentionModule(nn.Module):
    def __init__(self, hidden_size_spacial, attn_emb_size):
        super(AttentionModule, self).__init__()
        
        # Weight matrix Wp for computing the score
        self.W_p = nn.Linear(hidden_size_spacial, hidden_size_spacial, bias=False)
        
        # Weight matrix Wc for combining the context vector
        self.W_c = nn.Linear(hidden_size_spacial * 2, attn_emb_size)
        
    def forward(self, h_m, h_s):
        """
        h_m: (batch_size, hidden_size_spacial), final hidden state of GRU
        h_s: (batch_size, m, hidden_size_spacial), hidden states for all time steps
        """
        # Step 1: Calculate attention scores (alpha_t)
        # score(h_m, h_s) = h_m^T W_p h_{s^t}, where h_s is the preceding hidden state
        h_m = h_m.unsqueeze(1)  # Expand to (batch_size, 1, hidden_size_spacial)
        score = torch.bmm(h_m, self.W_p(h_s).transpose(1, 2))  # (batch_size, 1, m)
        
        # Step 2: Calculate attention weights alpha_t using softmax
        alpha_t = F.softmax(score, dim=-1)  # (batch_size, 1, m)

        # Step 3: Calculate context vector h_c as weighted sum of h_s using alpha_t
        h_c = torch.bmm(alpha_t, h_s)  # (batch_size, 1, hidden_size_spacial)
        h_c = h_c.squeeze(1)  # (batch_size, hidden_size_spacial)

        # Step 4: Combine h_c and h_m into a new representation using W_c
        combined = torch.cat((h_c, h_m.squeeze(1)), dim=-1)  # (batch_size, hidden_size_spacial * 2)
        A = torch.tanh(self.W_c(combined))  # (batch_size, attn_emb_size)

        return A, alpha_t

class PedestrianCrossingModel(nn.Module):
    def __init__(self):
        super(PedestrianCrossingModel, self).__init__()

        # Bouding Box embedding
        self.bb_emb = nn.Linear(BOUNDING_BOX_POINT_NUM, HIDDEN_SIZE_SPACIAL)
        # Pbb 和 Pbp (Bounding Box 和 Body Pose)
        self.shift_body_pose = ShiftGRUModule(input_size=2*KEYPOINTS_NUM, hidden_size=HIDDEN_SIZE_SPACIAL)
        # Bounding Box和Body Pose的GRU
        self.gru_bb_bp = nn.GRU(2*HIDDEN_SIZE_SPACIAL, hidden_size=HIDDEN_SIZE_SPACIAL, batch_first=True)
        # 对Vehicle Speed进行嵌入
        self.vehicle_speed_emb = nn.Linear(1, HIDDEN_SIZE_SPACIAL)
        # 对Bounding Box、Body Pose、Vehicle Speed的长短期依赖关系进行摘要
        self.gru_bb_bp_vs = nn.GRU(2*HIDDEN_SIZE_SPACIAL, ATTN_EMB_SIZE)

        # Plm (Local Motion)
        self.local_motion_module = CNNGRUModule(input_channels=3, conv_output_size=256, gru_hidden_size=128)
        
        # Plc (Local Content)
        self.local_content_module = CNNGRUModule(input_channels=3, conv_output_size=256, gru_hidden_size=128)
        
        # Attention Mechanism to fuse Local Motion and Local Content
        self.attention_module = AttentionModule(hidden_size=128)
        
        # Semantic Context 和 Categorical Depth
        self.semantic_context_conv3d = nn.Conv3d(in_channels=3, out_channels=512, kernel_size=3, padding=1)
        self.categorical_depth_conv3d = nn.Conv3d(in_channels=3, out_channels=512, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc_context_depth = nn.Linear(512 * 512, 512)

        # 最终的 GRU 和 Attention 模块
        self.final_gru = nn.GRU(input_size=512 + 256 + 256 + 128, hidden_size=128, batch_first=True)
        self.fc_output = nn.Linear(128, 1)
    
    def forward(self, pbb_input, pbp_input, motion_input, content_input, semantic_context_input, depth_input):
        # Pbb 和 Pbp 处理
        pbb_out = self.shift_bounding_box(pbb_input)  # 输出 shape: (batch, seq_len, 256)
        pbp_out = self.shift_body_pose(pbp_input)     # 输出 shape: (batch, seq_len, 256)
        
        # Local Motion 和 Local Content 处理
        motion_out = self.local_motion_module(motion_input)  # 输出 shape: (batch, seq_len, 128)
        content_out = self.local_content_module(content_input)  # 输出 shape: (batch, seq_len, 128)
        
        # 注意力机制结合 Local Motion 和 Local Content
        fused_local = self.attention_module(motion_out, content_out)  # 输出 shape: (batch, 128)
        
        # 语义上下文和深度特征处理
        semantic_context_out = self.flatten(F.relu(self.semantic_context_conv3d(semantic_context_input)))
        depth_out = self.flatten(F.relu(self.categorical_depth_conv3d(depth_input)))
        
        # 合并特征
        context_depth_out = F.relu(self.fc_context_depth(torch.cat((semantic_context_out, depth_out), dim=-1)))
        
        # 最终特征合并并经过 GRU
        combined_features = torch.cat((pbb_out[:, -1, :], pbp_out[:, -1, :], fused_local, context_depth_out), dim=-1)
        gru_out, _ = self.final_gru(combined_features.unsqueeze(1))  # 增加一个时间维度以适应 GRU
        final_output = self.fc_output(gru_out[:, -1, :])  # 取最后时间步的输出
        
        return torch.sigmoid(final_output)  # 行人横穿意图的预测 (0.0 - 1.0)

# 模型实例
model = PedestrianCrossingModel()
