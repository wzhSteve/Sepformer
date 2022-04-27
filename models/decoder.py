import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import FullAttention, AttentionLayer
class separate_decoder_layer(nn.Module):
    """
    功能：
    """
    def __init__(self,attention, d_model, dropout=0.1, activation="gelu", separate_factor=2, step=4):
        """
        :param attention: 注意力机制
        :param d_model:
        :param dff:
        :param dropout:
        :param activation:
        :param separate_factor: 为每层的输入与输出比
        :param step: 每层输入长度
        """
        super(separate_decoder_layer,self).__init__()
        self.step = step
        self.attention = attention #attentionlayer
        self.linear1 = nn.Linear(d_model, d_model,bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #降维的linear
        self.linear2 = nn.Linear(step, step*separate_factor,bias=True)
        self.norm3 = nn.LayerNorm(step*separate_factor)
        self.dropout = nn.Dropout(dropout)

        #self.activation = F.relu if activation == "relu" else F.gelu
        self.activation = F.elu
    def forward(self,x,attn_mask=None):
        """
        :param x: 输入B L D
        :param attn_mask: mask
        :return:y=attn*value为attention模块的输出， attn就是softmax(qk^T)
        """
        new_x, attn = self.attention(x,x,x,attn_mask = attn_mask)
        y = x + self.dropout(new_x)

        #降维 y:[batch_size, step, d_model]->[batch_size, step//factor, d_model]
        y = self.dropout(self.activation(self.linear2(y.transpose(-1, -2))))
        y = y.transpose(-1, -2)

        return y, attn
class DecoderLayer(nn.Module):
    def __init__(self, L_in, L_out, dropout=0.1, activation="gelu"):
        """
        :param L_in: 输入len
        :param L_out: 输出len
        :param dropout:
        :param activation:

        :param x: ffn输入
        :return y: ffn输出
        """
        super(DecoderLayer, self).__init__()
        self.linear = nn.Linear(L_in, L_out,bias=True)
        self.norm = nn.LayerNorm(L_out)
        self.dropout = nn.Dropout(dropout)
        #self.activation = F.relu if activation == "relu" else F.gelu
        self.activation = F.elu
    def forward(self, x):
        y = self.linear(x)
        y = self.dropout(self.activation(y))
        y = self.norm(y)

        return y

class Decoder(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, step, separate_factor, n_heads, mix, dropout=0.1, d_model=512, c_out=7,
                 activation='gelu'):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len + label_len

        self.dropout = dropout
        self.d_model = d_model
        self.c_out = c_out

        self.separate_factor = [3, 2]  # 第一层，二层，三层
        self.step = [24, 8, 4]
        # 层数
        self.layer_len = 2
        #self.activation = F.gelu if activation == 'gelu' else F.relu
        self.activation = F.elu

        self.true_encoder_list1 = nn.ModuleList([]) #用于转换true encoder和pred encoder之间的维度
        self.true_encoder_list2 = nn.ModuleList([])  # 用于转换true encoder和pred encoder之间的维度

        self.conv_module = nn.ModuleList([]) #用于true和pred两个encoder之间层间输出进行合并 输入cat(true, pred) 输出降维
        self.attention_layer_list = nn.ModuleList([])
        self.attention_layer_list2 = nn.ModuleList([])
        self.module = nn.ModuleList([])  # 用于decoder逆向进行维度变换（小->大）
        self.module2 = nn.ModuleList([])
        count = 0
        sequence_len = seq_len
        while(count < self.layer_len):
            old_pred = self.pred_len
            sequence_len = sequence_len//self.separate_factor[count]
            self.pred_len = self.pred_len//self.separate_factor[count]

            #向module中加入decoder layer
            self.attention_layer_list.append(AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                               d_model, n_heads, mix=mix))
            self.module.append(separate_decoder_layer(self.attention_layer_list[count], d_model, dropout=dropout,
                                       activation=activation, step=self.step[self.layer_len-count], separate_factor=self.separate_factor[self.layer_len-1-count]))
            self.attention_layer_list2.append(
                AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),
                               d_model, n_heads, mix=mix))
            self.module2.append(separate_decoder_layer(self.attention_layer_list2[count], d_model, dropout=dropout,
                                                      activation=activation, step=self.step[self.layer_len - count],
                                                      separate_factor=self.separate_factor[self.layer_len - 1 - count]))
            self.conv_module.append(DecoderLayer(2*self.pred_len, self.pred_len, self.dropout, self.activation))
            self.true_encoder_list1.append(DecoderLayer(sequence_len, self.pred_len, self.dropout, self.activation))
            #self.true_encoder_list2.append(DecoderLayer(sequence_len, self.pred_len, self.dropout, self.activation))
            count = count + 1
        # #true merging
        # self.linear_merg_true = nn.Linear(2*sequence_len, sequence_len)
        # self.norm_merg_true = nn.LayerNorm(sequence_len)

        #true encoder to z_1
        self.linear_z1 = nn.Linear(sequence_len, self.pred_len)
        self.norm_z1 = nn.LayerNorm(self.pred_len)
        #true pred encoder to z
        self.linear_enc_true_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.norm_enc_true_pred = nn.LayerNorm(self.pred_len)
        #z to decoder
        self.linear_zout = nn.Linear(self.pred_len, self.pred_len)
        self.norm_zout = nn.LayerNorm(self.pred_len)



        self.linear_out = nn.Linear(d_model, c_out, bias=True)
        self.linear_out2 = nn.Linear(c_out, c_out, bias=True)
        self.norm_out = nn.LayerNorm(c_out)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, enc_out_true, enc_out_pred, layer_output_true, layer_output_pred):
        """
        :param enc_out_true: [batch_size, sequence_len, d_model] list里面两个
        :param enc_out_pred: [batch_size, pred_len, d_model]
        :param layer_output_true: a list 里面两个
        :param layer_output_pred:
        :return:
        """
        #true merg
        # enc_out_true = self.norm_merg_true(self.linear_merg_true(enc_out_true.transpose(-1, -2))).transpose(-1, -2)
        #z
        enc_out_true = self.norm_z1(self.dropout_out(self.activation(self.linear_z1(enc_out_true.transpose(-1, -2))))) #B D L
        output = self.norm_enc_true_pred(self.dropout_out(self.linear_enc_true_pred(torch.cat([enc_out_true ,enc_out_pred.transpose(-1, -2)], dim=2)))) #B D L
        layer_len = self.layer_len
        while(layer_len > 0):
            #下标
            layer_len = layer_len - 1
            #layer_output_temp[batch_size, d_model, L]
            #true encoder维度变换到pred encoder
            layer_output_temp = self.true_encoder_list1[layer_len](layer_output_true[layer_len].transpose(-1, -2))#B D L
            #拼接true encoder pred encoder，下一步输入decoder中
            layer_output_temp = self.conv_module[layer_len](torch.cat([layer_output_temp, layer_output_pred[layer_len].transpose(-1, -2)], dim=-1)) #B D L
            #output为全局特征，layer_output_temp为局部特征，对其进行维度变换，与encoder相反
            output = output.transpose(-1, -2)#B L D
            layer_output_temp = layer_output_temp.transpose(-1, -2)#B L D
            cnt = layer_output_temp.shape[1] // self.step[layer_len+1]  # 该层块的个数cnt
            #用于存放本层的输出
            next_output = torch.tensor([]).to(output.device)
            for i in range(cnt):
                ii = i * self.step[layer_len+1]
                # fft 分频 mean为低频 div为高频
                output_temp = output[:, ii:ii + self.step[layer_len+1], :]
                layer_output_temp_temp = layer_output_temp[:, ii:ii + self.step[layer_len+1], :]
                # 将通过attention后的全局特征和局部特征相加
                output_temp, _ = self.module[self.layer_len - 1 -layer_len](output_temp)
                output_div_temp, _ = self.module2[self.layer_len-1-layer_len](layer_output_temp_temp)#B L D
                output_temp = output_temp + output_div_temp
                next_output = torch.cat([next_output,output_temp], dim=1)#B L D
            output = next_output.transpose(-1, -2)#B D L
        output = output.transpose(-1, -2)#B L D
        output = self.linear_out2(self.linear_out(output))
        return output #output[batch_size, pred_len + label_len, c_out]
