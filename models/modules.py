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
    def __init__(self, input_channel_num=1):
        super(SementicContextExtractor, self).__init__()

        self.conv3d = nn.Conv3d(
            in_channels=input_channel_num,
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
        # x = x.unsqueeze(1)  # (batch_size, c, m, h, w)
        x = self.conv3d(x)  # (batch_size, c=1, m, h'= h, w'= w)
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
        self.dropout = nn.Dropout()

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
        context = torch.sum(self.dropout(alpha_t) * h, dim=1)  # 加权求和，消去 seq_len 维度

        return context, alpha_t  # 返回上下文向量和注意力权重
