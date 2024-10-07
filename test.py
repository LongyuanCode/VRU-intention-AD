import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(128*128, 512)
    
    def forward(self, x):
        # Input x: (batch_size, m, h=512, w=512)
        x = x.unsqueeze(1)  # (batch_size, c=1, m, h, w)
        x = self.conv3d(x)  # (batch_size, c=1, m, h'=h, w'=w)
        x = self.maxpool3d(x)   # (batch_size, c=1, m, 128, 128)
        x = x.squeeze(1)    # (batch_size, m, 128, 128)
        batch_size, m, h, w = x.shape
        x = x.view(batch_size, m, h*w)  # (batch_size, m, 128*128)
        output = self.fc(x)  # (batch_size, m, 512)

        return output
    
if __name__ == '__main__':
    extractor = SementicContextExtractor().cuda()
    input = torch.randn(5, 2, 512, 512).cuda()
    output = extractor(input)
    print(output.shape)
