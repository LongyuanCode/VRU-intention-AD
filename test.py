import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
hidden_size = 5

context_kinematic_fusion = torch.randn(batch_size, hidden_size)
context_lc_lm = torch.randn(batch_size, hidden_size)
context_sc_cd = torch.randn(batch_size, hidden_size)

# 2. 定义一个 [CLS] embedding，形状为 (batch_size, hidden_size)
cls_emb = nn.Parameter(torch.randn(batch_size, hidden_size))

# 3. 拼接这些张量
# 将 3 个张量拼接成一个 (batch_size, 3, hidden_size) 的张量
context_combined = torch.stack([context_kinematic_fusion, context_lc_lm, context_sc_cd], dim=1)
print(context_combined.shape)
    
""" if __name__ == '__main__':
    extractor = SementicContextExtractor().cuda()
    input = torch.randn(5, 2, 512, 512).cuda()
    output = extractor(input)
    print(output.shape) """
