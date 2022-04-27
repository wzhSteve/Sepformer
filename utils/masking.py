import torch
# triangular 三角形
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            #保留对角线以上的部分，不包括对角线
            #如
            #[0,1,1,1]
            #[0,0,1,1]
            #[0,0,0,1]
            #[0,0,0,0]
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property #装饰器
    def mask(self):
        return self._mask

class ProbMask():
    """
    B:batch_szie
    H:head_number
    L:length such as L_V L_Q L_Z
    scores: probsparse number
    用于掩盖不重要的信息
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # .triu(1) 保留对角线以上的部分，不包括对角线
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask