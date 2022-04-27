import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import separate_encoder, separate_encoder_layer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, AttentionLayer
from models.embed import DataEmbedding


class separateformer(nn.Module):
    def __init__(self,enc_in, dec_in,c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8,
                 dropout=0.05, attn='full', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, mix=True, separate_factor=2, step=4,
                 device=torch.device('cuda:0')):
        super(separateformer,self).__init__()
        self.seq_len = seq_len
        self.pred_len = out_len #预测序列长度
        self.label_len = label_len
        self.attn = attn #attn模块选取
        self.output_attention = output_attention
        self.separate_factor = separate_factor
        self.dropout = dropout
        self.step = step
        #self.activation = F.gelu if activation == 'gelu' else F.relu
        self.activation = F.elu
        self.d_model = d_model
        self.c_out = c_out

        #encoding ETT中enc_in dec_in都为7 d_model为512，即把七个多元变量通过线性映射到512维上
        self.enc_embedding = DataEmbedding(enc_in,d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in,d_model,embed,freq,dropout)
        #true encoder 2layer
        self.encoder = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        self.encoder2 = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        #pred encoder  1layer
        self.encoder_pred = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        #pred decoder  1layer
        self.decoder = Decoder(self.seq_len, self.label_len, self.pred_len, self.step, self.separate_factor, n_heads, mix,
                               self.dropout, self.d_model, self.c_out, self.activation)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None):
        """
        :param x_enc: encoder的输入[batch_size, sequence_len, c_in=7]
        :param x_mark_enc: 输入的时间戳[batche_size, sequence_len, 4]
        :param enc_self_mask:
        :return:
        """

        # # dec_out_mean[batch_size, 1, d_model]
        dec_out_mean = torch.mean(x_dec[:,:self.label_len, :], dim=1).view(x_dec.shape[0], 1, x_dec.shape[2])
        # # temp[batch_size, pred_len, d_model]
        temp = dec_out_mean.repeat(1, self.pred_len, 1)
        # temp = torch.ones([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(x_dec.device)
        # # dec_out中占位符为已知部分的均值或0 作为预测部分的encoder的输入
        x_dec = x_dec[:, :self.label_len, :]
        x_dec = torch.cat([x_dec, temp], dim=1)
        # first embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, layer_output_true = self.encoder(enc_out,attn_mask=enc_self_mask)
        #pred encoder
        dec_out, layer_output_pred = self.encoder_pred(dec_out, attn_mask=enc_self_mask)
        #decoder
        output = self.decoder(enc_out, dec_out, layer_output_true ,layer_output_pred)
        return output # [B, L, D]




