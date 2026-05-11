import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        z1: view1 的 projector output, shape [N, D]
        z2: view2 的 projector output, shape [N, D]
        """

        batch_size = z1.size(0)

        # 1. normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 2. 合併成 2N 筆資料
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # 3. 算所有 pair 的 cosine similarity
        similarity_matrix = torch.matmul(z, z.T)  # [2N, 2N]

        # 4. 除以 temperature
        similarity_matrix = similarity_matrix / self.temperature

        # 5. 自己跟自己不能算，所以把 diagonal 填成很小
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # 6. 建立正確答案 label
        # 第 i 張的 positive pair 是 i + N
        # 第 i+N 張的 positive pair 是 i
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # 7. CrossEntropyLoss
        loss = self.criterion(similarity_matrix, labels)

        return loss


if __name__ == "__main__":
    loss_fn = NTXentLoss(temperature=0.5)

    # 假裝 batch size = 4, projector output dim = 128
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)

    loss = loss_fn(z1, z2)

    print("z1 shape:", z1.shape)
    print("z2 shape:", z2.shape)
    print("loss:", loss.item())