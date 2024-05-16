import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import  numpy as  np
from dataset import handSvgDataset
from models.model import Seq2Seq
import  random

#torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

setup_seed(3407)

crPath= "./CharacterData/cr_mapping.json"
refPath = "./CharacterData/RefCharList.json"
lmdbPath = "./DataPreparation/LMDB/lmdb"
dataset = handSvgDataset(crPath,refPath, lmdbPath)

batch_size=2

# PyTorch 在执行反向传播计算时执行额外的检查，以帮助找出可能导致计算图中出现错误的地方。
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

Model = Seq2Seq(batch_size,device,sql_len=dataset.sqeLen).to(device)  # 将模型移动到 GPU 上

# 训练
num_epochs = 400
start_epoch = 0

save_path = "./experiment/checkpoint/mymodel.pth"  # 要保存的文件路径，会在当前目录下创建一个 models 子目录

if not os.path.exists("./experiment/checkpoint"):  # 判断路径是否存在
    os.makedirs("./experiment/checkpoint")  # 使用os模块创建路径

if os.path.exists(save_path):
    print("load model")
    # 加载模型和优化器状态
    checkpoint = torch.load(save_path)
    Model.load_state_dict(checkpoint['model_state_dict'])
    Model.optimizer_seq.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

# 训练模式
Model.train()

# 定义进度条
pbar = tqdm(total=len(dataloader), desc='Training', unit='batch')

for epoch in range(start_epoch, num_epochs):

    for i, batch in enumerate(dataloader):

        try:
                # 向前传播

            # 将批次中的所有张量移动到指定设备
            batch = [t.to(device) for t in batch]

            trg_command, trg_args,in_command,in_args, input_command, input_args, style_command, style_args, classIndex, RefIdx = batch

            Model.set_input(trg_command, trg_args,in_command,in_args, input_command, input_args, style_command, style_args, classIndex,
                            RefIdx)

            ref_loss, loss_cmd, loss_args, loss_aux, class_loss,vq_loss = Model.optimize_parameters()

            pbar.n = i

            pbar.set_description(
            "Epoch {} ref_loss: {:.4f} loss_cmd: {:.4f} loss_args: {:.4f} loss_aux: {:.4f} class_loss: {:.4f} vq_loss: {:.4f}"
            .format(epoch + 1,
                    ref_loss.item(),
                    loss_cmd.item(),
                    loss_args.item(),
                    loss_aux.item(),
                    class_loss.item(),
                    vq_loss.item()
                    ))

            if i % 100 == 0:

              Model.sample()

              torch.save({
                    'epoch': epoch,
                    'model_state_dict': Model.state_dict(),
                    'optimizer_state_dict': Model.optimizer_seq.state_dict(),
                }, save_path)


        except Exception as e:

            pbar.n = i
            # 发生异常时的处理，可以根据实际情况选择记录日志、打印错误信息等
            print(f"An error occurred: {e}")
            # 跳过当前次迭代，继续下一次
            continue


    # 重置进度条
    pbar.reset()

# 关闭进度条
pbar.close()



