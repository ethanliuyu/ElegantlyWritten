import json
import os
import pickle

import lmdb
import numpy as np
from torch.utils.data import Dataset, DataLoader

#content_reference_json = "../cr_mapping.json"
import random



class handSvgDataset(Dataset):
    def __init__(self, crPath,refPath, lmdbPath):

        self.sqeLen = 180
        print("load content cr_mapping json")
        with open(crPath, 'r') as f:
            self.cr_mapping = json.load(f)
        print("load content RefCharList json")
        with open(refPath, 'r') as f:
            self.RefCharList = json.load(f)



        print("load lmdb")

        self.clssNum=0

        self.RefCharNum=len(self.RefCharList)

        # lmdbPaht="../getFont/lmdbdata/lmdb"
        self.env = self.load_lmdb(lmdbPath)

        self.className, self.class_to_idx, self.allFile = self._make_dataset()

        self.Ref_to_idx = {char_name: i for i, char_name in enumerate(self.RefCharList)}
        print("Ref_to_idx", len(self.Ref_to_idx))

    def load_lmdb(self, lmdb_path):
        """
        load_lmdb
        """
        lmdb_path = os.path.join(lmdb_path)
        env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        return env

    def _make_dataset(self):

        with self.env.begin() as txn:
            key = "className"
            value_bytes = txn.get(key.encode('utf-8'))

            if value_bytes is not None:
                className = pickle.loads(value_bytes)

        self.clssNum = len(className)

        print('number of fonts: ', self.clssNum)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(className)}

        print(class_to_idx)

        allFile = []
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    key_str = key.decode('utf-8')
                    if key_str != "className":
                        allFile.append(key_str)

        print("sample num", len(allFile))
        return className, class_to_idx, allFile

    def get_sample_pair(self, className, charName):

        unicode_code_point = int(charName, 16)
        unicode_character = chr(unicode_code_point)
        style_unis = self.cr_mapping[unicode_character]  # 获取与该字 包含共同部件的内容

        style_command = []
        style_args = []

        # style_unis = ['娘', '年', '念', '捻', '撵']  # 临时测试
        RefIdx = []

        # className = "018.pot"  # 临时测试

        with self.env.begin() as txn:
            for tempChar in style_unis:

                tempName = hex(ord(tempChar))[2:].upper()
                key = '{}_{}'.format(className, tempName)
                value_bytes = txn.get(key.encode('utf-8'))

                # 如果没有这个字 就随机选择一个
                if value_bytes is None:
                    while True:
                        print("random select style char")
                        # 生成一个随机整数，范围在 0 到 len(RefCharList)-1 之间
                        random_index = random.randint(0, len(RefCharList) - 1)
                        tempChar = RefCharList[random_index]
                        tempName = hex(ord(tempChar))[2:].upper()
                        key = '{}_{}'.format(className, tempName)
                        value_bytes = txn.get(key.encode('utf-8'))
                        # 找到符合条件的就退出
                        if value_bytes is not None:
                            break

                path = pickle.loads(value_bytes)
                temp_command, temp_args = self.bulid_numpy_data(path, addNnoise=False, addToken=False,addCls=True)
                # 将数组追加到
                style_command.append(temp_command)
                # 使用 np.stack 进行垂直堆叠
                style_args.append(temp_args)
                # 创建选择参考风格的id
                RefIdx.append(self.Ref_to_idx[tempChar])

            style_command = np.stack(style_command)
            style_args = np.stack(style_args)
        return style_command, style_args, np.array(RefIdx)

    def bulid_numpy_data(self, path, addNnoise=False, addToken=False, felax=False,addCls=False):

        # 在path中读取命令类型
        command = path["command"]
        # 将M转为1 将 L 转为2
        command = [0 if x == 'M' else 1 for x in command]

        # 读取坐标值
        args = path["para"]
        # 坐标值转为numpy
        args = np.array(args)


        if felax:
            # 缩短和延长笔画
            args = self.felax_line(path["command"], args)

        if addNnoise:
            # 添加噪声
            noise_array = np.random.randint(low=-5, high=5, size=args.shape)
            args = args + noise_array

        invalid_values = np.any((args >= 120) | (args <= 0))
        if invalid_values:
            # print("数组 trg_args 中存在无效值(大于等于 224 或小于 0)")

            # print(args)
            # 将所有大于 224 的值设置为 220
            args[args >= 120] = 120

            # 将所有小于 1 的值设置为 1
            args[args <= 0] = 1

        if addToken:
            # 在开头添加<SOS>
            command = [-2] + command
            # 在末尾添加<EOS>
            command = command + [-1]

            # 添加 <SOS> 在开头
            sos_token = np.array([[-2, -2]])  # 假设 <SOS> 的编码是 [-2, -2]
            args = np.insert(args, 0, sos_token, axis=0)
            # 添加 <EOS> 在末尾
            eos_token = np.array([[-1, -1]])  # 假设 <EOS> 的编码是 [-1, -1]

            args = np.insert(args, len(args), eos_token, axis=0)

        command = np.array(command)

        command = np.pad(command, (0, max(0, self.sqeLen - len(command))), constant_values=-3)

        # 计算需要填充的数量
        padding_length = max(0, self.sqeLen - len(args))
        # 使用 [-1, -1] 进行填充
        padding_array = np.array([[-3, -3]])
        # 在数组的末尾添加填充
        args = np.pad(args, ((0, padding_length), (0, 0)), 'constant', constant_values=padding_array)

        #添加类别嵌入时，在头部添加类别嵌入长度为 self.sqeLen+1
        if addCls:


            command=np.insert(command, 0, -2)



            cls_token = np.array([[-2, -2]])  # 假设 <CLS> 的编码是 [-2, -2]

            args = np.insert(args, 0, cls_token, axis=0)

            if len(args) > self.sqeLen+1:
                raise ValueError("The length of 'args' exceeds the maximum")

            return command, args

        else:
            if len(args) > self.sqeLen:
                raise ValueError("The length of 'args' exceeds the maximum")

            return command, args

    def felax_line(self, command, para):
        def bezier_curve(P0, P1, t):
            return np.round((1 - t) * P0 + t * P1)  # 取整四舍五入

        # t_values = [0.7,0.75,0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]

        t_values = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
        # para = np.array(para)

        for i in range(len(command)):
            if command[i] == "M":
                P0 = para[i]
            else:
                P1 = para[i]
                # 随机选择一个 t 值
                t = random.choice(t_values)
                P0 = bezier_curve(P0, P1, t)

                para[i] = P0

        return para

    def __len__(self):

        return len(self.allFile)

    def __getitem__(self, index):
        try:
            key = self.allFile[index]

            with self.env.begin() as txn:
                value_bytes = txn.get(key.encode('utf-8'))

                if value_bytes is not None:
                    path = pickle.loads(value_bytes)

            split_parts = key.split("_")
            charName = split_parts[1]  # 获取当前是那个字符
            className = split_parts[0]  # 获取该字符是那个字体类别
            classIndex = self.class_to_idx[className]

            # print(charName,className,classIndex,path)

            # 目标数据 需要添加起始标记
            trg_command, trg_args = self.bulid_numpy_data(path, addNnoise=False, addToken=True)

            # 输入数据 不需要添加起始标记等
            input_command, input_args = self.bulid_numpy_data(path, addNnoise=True, addToken=False, felax=True,addCls=True)

            # 参考风格数据
            style_command, style_args, RefIdx = self.get_sample_pair(className, charName)



            return trg_command, trg_args, input_command, input_args, style_command, style_args, classIndex, RefIdx

        except Exception as e:
            print(f"An error occurred while processing data item {index}: {e}")
            # 处理错误，例如返回一个默认的数据项或None
            # return default_data_item
            # 或者
            new_idx = random.randint(0, len(self.allFile) - 1)
            return self.__getitem__(new_idx)

if __name__ == '__main__':
    refPath = "./cr_mapping.json"
    lmdbPath = './lmdbdata/lmdb/'
    dataset = handSvgDataset(refPath, lmdbPath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        # Do something with the data
        # print(path)
        # print(target)

        break

