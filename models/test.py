import torch
def _prob_QK( Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
    # Q [B, H, L, D]
    # sample_k = U_part = c*ln(L_k)
    # E: embedding
    B, H, L_K, E = K.shape
    _, _, L_Q, _ = Q.shape

    # calculate the sampled Q_K
    # L_K, E ->L_Q, L_K, E 采用广播机制，相当于复制了L_Q个L_K, E
    K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
    # index_sample张量为   取值为[0,L_K) 维度为(L_Q, sample_k)
    index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
    # 此时
    # K_sample: B, H, L_Q, sample_k, E     从L_K中随机选取sample_k个
    K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
    # 计算 QK^T
    # Q: B, H, L_Q, E   K_sameple: B, H, L_Q, sample_k, E
    # Q_K_sample: B, H, L_Q, sample_k
    Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

    # find the Top_k query with sparisty measurement
    # max-mean measurement
    # torch.max(input, dim) max(-1)[0]为最后一维的最大值 返回值含有value和index [0]表示value 为一个list
    # M:B, H, L_Q
    M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
    # 选取n_top个值
    # topk() 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标
    #M_top: B, H, n_top
    M_top = M.topk(n_top, sorted=False)[1]

    # use the reduced Q to calculate Q_K
    # Q_reduce: B, H, n_top, D
    Q_reduce = Q[torch.arange(B)[:, None, None],
               torch.arange(H)[None, :, None],
               M_top, :]  # factor*ln(L_q)
    Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

    return Q_K, M_top


B = 4;H = 2; L_Q = 5; L_K = 4;D = 6
Q = torch.tensor(torch.rand(B, H, L_Q, D))
K = torch.tensor(torch.rand(B, H, L_K, D))
sample_k = 2;n_top = 3
Q_K, M_TOP = _prob_QK(Q, K, sample_k, n_top)
pass