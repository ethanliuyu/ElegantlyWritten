import os
import numpy as np
import struct
import pdb





import json

charListPath="./charlist.json"

with open(charListPath, 'r') as f:
    charList = json.load(f)

def glyph_to_svg(svgPath):
    # 构造 SVG 文件内容字符串
    svg_content = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    svg_content += f'<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224" viewBox="0 0 224 224">\n'
    svg_content += f'<g transform="matrix(1 0 0 1 0 0)">\n'
    svg_content += f'<path d="{svgPath}" stroke = "black" fill="none"  stroke-width="3" />\n'
    svg_content += '</g>\n</svg>\n'
    return svg_content


def read_from_pot_dir(pot_dir,outPath):
    def one_file(f):
        while True:
            # 文件头，交代了该sample所占的字节数以及label以及笔画数
            header = np.fromfile(f, dtype='uint8', count=8)
            if not header.size:
                break

            sample_size = header[0] + (header[1] << 8)

            tagcode = header[2] + (header[3] << 8) + (header[4] << 16) + (header[5] << 24)

            stroke_num = header[6] + (header[7] << 8)

            # 以下是参考官方POTView的C++源码View部分的Python解析代码
            traj = []

            for i in range(stroke_num):  # 一共有几个笔画，就循环几次
                while True:
                    header = np.fromfile(f, dtype='int16', count=2)
                    x, y = header[0], header[1]
                    if x == -1 and y == 0:  # -1 0 代表当前笔画的结束
                        traj.append([100000, 100000])  # 插入一个很大的数用来标记笔画的结束
                        break
                    else:
                        traj.append([x, y])

            # 最后还一个标志文件结尾的(-1, -1)
            header = np.fromfile(f, dtype='int16', count=2)

            pts = np.array(traj)

            # 将数组中的所有数乘以0.9并取整
            pts = np.round(pts * 0.6).astype(int)

            # 找到第一列的最小值和最大值
            xmin = np.min(pts[:, 0])
            pts[:, 0] -= xmin

            sorted_col1 = np.unique(np.sort(pts[:, 0]))
            xmax = sorted_col1[-2]
            xmin = sorted_col1[0]

            # 找到第二列的最小值和最大值
            ymin = np.min(pts[:, 1])
            pts[:, 1] -= ymin

            sorted_col1 = np.unique(np.sort(pts[:, 1]))
            ymax = sorted_col1[-2]
            ymin = sorted_col1[0]

            ## 计算平移量，将小图像放在大图像中心
            maxImage = 224
            translate_x = abs((maxImage - (xmax - xmin)) // 2)

            translate_y = abs((maxImage - (ymax - ymin)) // 2)

            pts[:, 0] += translate_x
            pts[:, 1] += translate_y

            paths = ""

            path = "M {} {} \n".format(pts[0][0], pts[0][1])
            paths += path
            stroke_start_tag = False

            for i in range(1, len(pts)):

                if pts[i][0] > 1000 and pts[i][1] > 1000:
                    stroke_start_tag = True
                    continue
                if stroke_start_tag:
                    path = "M {} {} \n".format(pts[i][0], pts[i][1])
                    stroke_start_tag = False
                else:
                    if pts[i][0]<0 or pts[i][1]<0:
                        pdb.post_mortem(pts)  # 进入调试模式
                    path = "L {} {} \n".format(pts[i][0], pts[i][1])

                paths += path

            temp=paths
            yield temp, tagcode

    a = 1
    for file_name in os.listdir(pot_dir):
        if file_name.endswith('.pot'):
            save_path = os.path.join(outPath, file_name)
            print(save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(pot_dir, file_name)
            with open(file_path, 'rb') as f:
                for paths, tagcode in one_file(f):

                    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312', errors='replace')
                    if tagcode_unicode in charList:

                        print(tagcode_unicode)

                        hex_code = format(ord(tagcode_unicode), 'x')
                        file_name = f"\\u{hex_code}"

                        #unicode_hex = ' '.join([f'\\u{ord(char):04x}' for char in tagcode_unicode])

                        #unicode_hex = ' '.join([hex(ord(char)) for char in tagcode_unicode])

                        #unicode_hex = ''.join(unicode_hex.split(b'\x00'.decode()))  # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题

                        svg_data = glyph_to_svg(paths)
                        output_path =os.path.join(save_path, f"{file_name}.svg")
                        with open(output_path, "w") as f:
                            f.write(svg_data)
                        a+=1
                        print(a,file_name)

                        # if a%10==0:
                        #     break


if __name__=="__main__":
    datePath="./POTData/"

    outPath="./SVGData/"

    read_from_pot_dir(datePath,outPath)
