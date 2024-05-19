import torchimport osdef glyph_to_svg(svgPath1, svgPath2):    # 构造 SVG 文件内容字符串    svg_content = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'    svg_content += f'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">\n'    svg_content += f'<g transform="matrix(1 0 0 1 0 0)">\n'    svg_content += f'<path d="{svgPath1}" stroke = "black" fill="none" stroke-width="3" />\n'    svg_content += f'<path d="{svgPath2}" stroke = "red" fill="none" stroke-width="3" />\n'    svg_content += '</g>\n</svg>\n'    return svg_content# 加载字典loaded_tensors = torch.load('out_sample.pt', map_location=torch.device('cpu'))trg_command = loaded_tensors["trg_command"]trg_args = loaded_tensors["trg_args"]out_command = loaded_tensors["out_command"]out_args = loaded_tensors["out_args"]out_idxs=loaded_tensors["out_idx"]input_idxs=loaded_tensors["input_idx"]num=10command = trg_command[num].numpy()outcmd= out_command[num].numpy()-3outargs = out_args[num].numpy()-3trgargs = trg_args[num].numpy()out_idx=out_idxs[num].numpy()input_idx=input_idxs[num].numpy()print(out_idx)print(input_idx)out_path = ""tra_path = ""for i in range(len(command)):    if command[i] == -2:        continue    elif command[i] == 0:        out_path += " ".join([" M", str(outargs[i][0]), str(outargs[i][1])])        tra_path += " ".join([" M", str(trgargs[i][0]), str(trgargs[i][1])])    elif command[i] == 1:        out_path += " ".join([" L", str(outargs[i][0]), str(outargs[i][1])])        tra_path += " ".join([" L", str(trgargs[i][0]), str(trgargs[i][1])])svg_data=glyph_to_svg(tra_path, out_path)save_path="./"file_name="test"output_path =os.path.join(save_path, f"{file_name}.svg")with open(output_path, "w") as f:    f.write(svg_data)