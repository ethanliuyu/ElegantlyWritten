import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from models.quantize import LFQ

from dataset import handSvgDataset
from models.seqDec import TransformerDec , TransformerRef
from models.seqEnc import SVGEmbedding, TransformerEnc



def denumericalize(cmd, n=128):
    cmd = cmd / n * 64
    return cmd


def _sample_categorical(*args_logits):
    temperature = 0.0001
    if len(args_logits) == 1:
        arg_logits, = args_logits
        return Categorical(logits=arg_logits / temperature).sample()
    return (*(Categorical(logits=arg_logits / temperature).sample() for arg_logits in args_logits),)


class Seq2Seq(nn.Module):
    def __init__(self, batch_size, device, sql_len):
        super().__init__()

        self.device = device

        self.selectNum = 3




        self.SVGEmbedding = SVGEmbedding()  # 绘图参数嵌入

        self.InputSeqEnc = TransformerEnc()  # 输入绘图参数的编码器

        self.StyleSeqEnc = TransformerEnc()  # 输入绘图参数的编码器

        self.seqdec = TransformerDec()

        self.RefDec=TransformerRef()



        self.RefCharNum = 504

        #self.RefFC = nn.Linear(512, self.selectNum * self.RefCharNum)  # 用来判断那几个字与输入字符最相近

        self.RefFC =nn.Sequential(
            # nn.Linear(512,1024,bias = False),
            # nn.ReLU(),
            nn.Linear(512,self.RefCharNum,bias = False)
        )

        self.clssNum = 11

        self.clssFC = nn.Linear(512, self.clssNum)  # 用来约束字符类别



        codebook_dim=512

        self.CharIndFC = nn.Linear(codebook_dim, self.RefCharNum)

        self.style_Aggr = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1, batch_first=True)

        self.style_Aggr_norm = nn.LayerNorm(512)


        self.vq = LFQ(
            codebook_size=4096,
            dim=512,
        )

        # 填充0  起始标志 1 结束标志 2   M 3   L 4

        cmd_weights = torch.tensor([1., 1.1, 1.1, 1.2, 0.8]).to(self.device)
        # 创建交叉熵损失函数

        self.cmdCE = nn.CrossEntropyLoss(reduction='sum', weight=cmd_weights, ignore_index=0)

        args_weights = torch.ones(128).to(self.device)
        # args_weights[0] = 1.1  # 填充 0
        # args_weights[1] = 1.1  # 起始标志 1
        # args_weights[2] = 1.1  # 结束标志 2

        self.argsCE = nn.CrossEntropyLoss(reduction='sum', weight=args_weights, ignore_index=0)

        self.argsMSE = nn.MSELoss()

        self.clssCE = nn.CrossEntropyLoss(label_smoothing=0.05)
        # 损失函数定义
        self.RefLoss = nn.CrossEntropyLoss(label_smoothing=0.05, reduction='sum')

        #约束该字符是哪个字符
        self.charIndLoss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='sum')

        self.lr = 0.0004


        self.optimizer_seq = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
            eps=1e-8,
            amsgrad=True
        )

        self.apply(self._init_weights)



    # 初始化权重
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            # 对权重进行均匀初始化
            torch.nn.init.xavier_uniform_(m.weight)
            # 如果有偏置 则对偏置使用0初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是层归一化，则将偏置和权重初始化为0 和 1
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_input(self, trg_command, trg_args,in_command,in_args, input_command, input_args, style_command, style_args,
                  classIndex, RefIdx):  # 设置数据
        self.trg_command = trg_command
        self.trg_args = trg_args
        self.input_command = input_command
        self.input_args = input_args
        self.style_command = style_command
        self.style_args = style_args
        self.classIndex = classIndex
        self.RefIdx = RefIdx
        self.in_command=in_command
        self.in_args=in_args

    def forward(self):

        # 因为添加了类别嵌入，因此为了保留类别嵌入 所以添加一个mask  -3为被填充的数据，被填充的数据设置为 False
        input_mask = (self.input_command == -3)

        # 进行嵌入
        inputSeq = self.SVGEmbedding(self.input_command, self.input_args)

        # 将维度变为 [batch*5, 200, 2] 将5个参考字符合并在 batch 维度上
        style_args = rearrange(self.style_args, 'b h w c -> (b h) w c')

        style_command = rearrange(self.style_command, 'b h c -> (b h) c')

        styleSeq = self.SVGEmbedding(style_command, style_args)

        style_mask = (style_command == -3)  # 创建一个mask对于未使用的部分进行掩码

        input_latent = self.InputSeqEnc(inputSeq, input_mask) #输入编码器

        inputSeq = input_latent #去除CLS标记

        style_latent = self.StyleSeqEnc(styleSeq, style_mask) #风格编码器

        styleSeq = style_latent #[:,1:,:] #去除CLS标记

        style_cls = style_latent[:, :1, :]  # 取出CLS

        style_cls = rearrange(style_cls, '(b k) n d -> b (k n) d', b=inputSeq.size(0))

        # 取出第一个class_token 用来做类别分析
        #input_cls = input_latent[:, :1, :] #去输入的 cls


        in_input = self.SVGEmbedding(self.in_command, self.in_args)


        #####################################
        #创建mask
        tgt_mask = torch.triu(torch.ones(in_input.size(1), in_input.size(1)), diagonal=1)

        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

        num_heads = 8

        tgt_mask = tgt_mask.unsqueeze(0).expand(num_heads * in_input.size(0), -1, -1).to(self.device)

        #####################################

        #style_ref,_=self.style_ref_Aggr(self.style_tensor,styleSeq,styleSeq)

        #style_ref = rearrange(style_ref, '(b h) w c -> b (w h) c', b=inputSeq.size(0))

        styleSeq = rearrange(styleSeq, '(b h) w c -> b (w h) c', b=inputSeq.size(0))

        styleSeq, indices, vq_loss = self.vq(styleSeq)

        #styleSeq = self.style_Aggr_norm(styleSeq)

        style_aggr, self.style_aggr_weights = self.style_Aggr(inputSeq, styleSeq, styleSeq)

        style_aggr = inputSeq + self.style_Aggr_norm(style_aggr)

        command_logits, args_logits = self.seqdec(x=in_input, ref=style_aggr, tgt_mask=tgt_mask)

        style_class_logits = self.clssFC(style_cls)


        b = self.RefIdx.size(0)  # 获取 batch size

        # 创建值为 -2 的张量，维度为 [b, 1]
        prepend = torch.full((b, 1), -2, dtype=self.RefIdx.dtype, device=self.RefIdx.device)

        # 创建值为 -1 的张量，维度为 [b, 1]
        append = torch.full((b, 1), -1, dtype=self.RefIdx.dtype, device=self.RefIdx.device)

        self.RefIdx = torch.cat((prepend, self.RefIdx, append), dim=1)


        tgt_mask = torch.triu(torch.ones(self.RefIdx.size(1), self.RefIdx.size(1)), diagonal=1)

        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

        num_heads = 8

        tgt_mask = tgt_mask.unsqueeze(0).expand(num_heads * self.RefIdx.size(0), -1, -1).to(self.device)

        ref_logits=self.RefDec(x=self.RefIdx,ref=input_latent,tgt_mask=tgt_mask)


        #ref_latent,_=self.ref_Aggr(self.noise_tensor,input_latent,input_latent)

        #combined_tensor = torch.cat((ref_latent, style_ref), dim=1)

        #ref_logits= self.RefFC(ref_latent)

        return command_logits, args_logits, ref_logits, style_class_logits, vq_loss

    def optimize_parameters(self):

        command_logits, args_logits, ref_logits, style_class_logits, vq_loss = self.forward()
        self.out_command_logits = command_logits
        self.out_args_logits = args_logits
        return self.backward_seq2seq(command_logits, args_logits, ref_logits, style_class_logits, vq_loss)

    def backward_seq2seq(self, command_logits, args_logits, ref_logits, style_class_logits, vq_loss):

        class_lable = self.classIndex.unsqueeze(1).repeat(1, self.selectNum)  # 重塑标签为 [b, k]


        class_loss = self.clssCE(style_class_logits.reshape(-1, self.clssNum), class_lable.reshape(-1))


        N, S, _ = ref_logits.shape

        #ref_logits = ref_logits.reshape(N, S, self.selectNum, self.RefCharNum)

        # 计算相关字符选择损失


        #self.RefIdx1=torch.cat((self.RefIdx, self.RefIdx), dim=1)



        ref_loss = self.RefLoss(ref_logits[:,:-1,:].reshape(-1, self.RefCharNum+2), self.RefIdx[:,1:].reshape(-1)+2)

        self.out_idx = _sample_categorical(ref_logits)


        n_commands = 5

        args_dim = 128


        loss_cmd = self.cmdCE(command_logits.reshape(-1, n_commands), self.trg_command.reshape(-1) + 3)

        loss_args = self.argsCE(args_logits.reshape(-1, args_dim), self.trg_args.reshape(-1) + 3)  # shift due


        line_mask = (self.trg_command == 0).float() + (self.trg_command == 2).float() + (self.trg_command == 3).float()

        # 弹出最后一个，以保证维度相同
        line_mask = line_mask[:, :-1]

        # 计算实际长度
        trg_seqlen = torch.sum(line_mask == 1, dim=1)

        # 在最后一个维度上插入新维度
        line_mask = line_mask.unsqueeze(2)

        # 复制新维度，使其形状为 [2, 199, 6]
        line_mask = line_mask.repeat(1, 1, 6)

        # trg_args 第一个元素为[0,0] 作为起始标志

        # 直线插值损失

        trg_args = denumericalize(self.trg_args)

        trg_p0 = trg_args[:, :-1, :]

        trg_p1 = trg_args[:, 1:, :]

        t = 0.25
        trg_pts_line = trg_p0 + t * (trg_p1 - trg_p0)

        for t in [0.5, 0.75]:
            coord_t = trg_p0 + t * (trg_p1 - trg_p0)
            trg_pts_line = torch.cat((trg_pts_line, coord_t), -1)
        trg_pts_line = trg_pts_line * line_mask

        args_prob2 = F.softmax(args_logits / 0.1, -1)

        c = torch.argmax(args_prob2, -1).unsqueeze(-1).float() - args_prob2.detach()

        p_argmax = args_prob2 + c

        p_argmax = torch.mean(p_argmax, -1) - 1  # 还原原始

        p_argmax = denumericalize(p_argmax)  # 对坐标进行缩放，避免产生较大损失无法优化

        out_p0 = p_argmax[:, :-1, :]  # 原始数据在开头添加了[0 0]标记，为了能够并行计算，去除最后一个元素

        out_p1 = p_argmax[:, 1:, :]  #

        t = 0.25
        out_pts_line = out_p0 + t * (out_p1 - out_p0)
        for t in [0.5, 0.75]:
            coord_t = out_p0 + t * (out_p1 - out_p0)
            out_pts_line = torch.cat((out_pts_line, coord_t), -1)

        out_pts_line = out_pts_line * line_mask

        aux_pts_loss = torch.pow((out_pts_line - trg_pts_line), 2) * line_mask

        loss_aux = torch.mean(aux_pts_loss, dim=-1, keepdim=False)

        loss_aux = torch.mean(torch.sum(loss_aux / trg_seqlen.unsqueeze(-1), -1))

        #MSE_loss = self.argsMSE(_sample_categorical(args_logits), (self.trg_args + 3).float())

        # seq_loss = ref_loss + loss_cmd + loss_args + loss_aux * 0.01 + class_loss+vq_loss
        seq_loss = loss_cmd + loss_args + vq_loss + ref_loss  + loss_aux

        # 清空梯度
        self.optimizer_seq.zero_grad()
        # 反向传播
        seq_loss.backward()
        # 跟新参数
        self.optimizer_seq.step()

        return ref_loss, loss_cmd, loss_args, loss_aux, class_loss, vq_loss

    def sample(self):

        tensors = {
            'trg_command': self.trg_command,
            'trg_args': self.trg_args,
            'out_command': _sample_categorical(self.out_command_logits),
            'out_args': _sample_categorical(self.out_args_logits),
            'out_idx': self.out_idx,
            'input_idx': self.RefIdx,
            "style_aggr_weights":self.style_aggr_weights,
            "style_command" : self.style_command,
            "style_args" : self.style_args
        }

        # 保存字典到文件
        torch.save(tensors, './out_sample.pt')


if __name__ == '__main__':

    crPath = "./CharacterData/cr_mapping.json"
    refPath = "./CharacterData/RefCharList.json"
    lmdbPath = "./DataPreparation/LMDB/lmdb"
    dataset = handSvgDataset(crPath, refPath, lmdbPath)

    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(batch_size, device, sql_len=dataset.sqeLen)

    for i, (
            trg_command, trg_args,in_command,in_args, input_command, input_args, style_command, style_args, classIndex,
            RefIdx) in enumerate(
        dataloader):

        model.set_input(trg_command, trg_args,in_command,in_args, input_command, input_args, style_command, style_args, classIndex, RefIdx)
        model.optimize_parameters()

        break
