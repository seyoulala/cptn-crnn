#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019/6/10 下午2:26 
# @Author : Ethan
# @Site :  
# @File : uti.py
# @Software: PyCharm


import numpy as np
import os
import sys
from PIL import Image
import  PIL.ImageOps
import matplotlib.image as  mping
import matplotlib.pyplot as plt
from models.my_crnn import CRNN
import torch
import cv2

from decimal import Decimal
from decimal import getcontext

# image_path ='/media/ethan/0207380E139801C9/t10k-images'
# output_result_path = "/media/ethan/0207380E139801C9/work_dir/ocr/crnn_chinese_characters_rec/to_lmdb/traifapiao.txt"
#
# with open(output_result_path,'w') as file:
# 	images_list = os.listdir(image_path)
# 	for i in images_list:
# 		label = i.split('_')[0]
# 		file.write(i+' '+label)
# 		file.write("\n")
# image_list = os.listdir(image_path)[0]
# image= Image.open('test_images/number3.png')
# print(image.size[0],image.size[1])
# plt.imshow(image)
# plt.show()
# txt_file = open('to_lmdb/111.txt','a+')
#
# with open(output_result_path,'r') as file:
# 	max_length = 10
# 	for line in file.readlines():
# 		image = line.split(' ')[0]
# 		label = line.split(' ')[1].replace('\n','')
# 		if len(label)<max_length:
# 			label = str(label)
# 			label+= (max_length-len(label))*'0'
# 			txt_file.write(image+' '+label)
# 			txt_file.write('\n')
# 		else:
# 			label = str(label)
# 			label = label[:max_length]
# 			txt_file.write(image+' '+label)
# 	txt_file.close()




# image = Image.open(path)
# X =np.zeros((46,300,3))
# X[:]=255
# w,h = image.size
# X[:,:w]= image
# X = X/255.0
# plt.imshow(X)
# plt.show()
# path = 'to_lmdb/t10k-images'
# ### 随机拼接手写数字
# name_list = os.listdir(path)
# number_file = open('to_lmdb/number_write.txt','w')
# #
# #
# t = 0
# while t<15000:
# 	ims = []
# 	number_list =[]
# 	select_list = np.random.choice(name_list, size=10)
# 	for line in select_list:
# 		label = line.split('_')[0]
# 		number_list.append(label)
# 	for item in select_list:
# 		img = Image.open(os.path.join(path, item))
# 		ims.append(img)
# 	w, h = ims[0].size
# 	result = Image.new(ims[0].mode, (w * 10, h))
# 	for i, im in enumerate(ims):
# 		result.paste(im, box=(i * w, 0))
# 	target_path = os.path.join('to_lmdb/number', '%d.jpg' % t)
# 	number_file.write('%d.jpg'%t+' '+ ''.join(number_list))
# 	number_file.write('\n')
# 	result = PIL.ImageOps.invert(result)
# 	result.save(target_path)
# 	t += 1


# p拼接


# #padding图片到相同的宽度
# def matrixtoimage(data):
# 	data =data*255
# 	new_image = Image.fromarray(data.astype(np.uint8)).convert('L')
# 	return  new_image
#
# max_w = 0
# for item in os.listdir(path):
# 	image=Image.open(os.path.join(path,item))
# 	w,h = image.size
# 	#找到图片中的最大宽度
# 	if w>max_w:
# 		max_w=w
# i=0
# for item in os.listdir(path):
# 	image = Image.open(os.path.join(path,item))
# 	w,h = image.size
# 	X = np.zeros((h,max_w,3))
# 	X[:,:w] = image
# 	X/=255.0
# 	new_img = matrixtoimage(X)
# 	target_image_path = os.path.join('to_lmdb/train_images', '%d.jpg' %i )
# 	if i==54:
# 		i+=2
# 	else:
# 		i+=1
# 	new_img.save(target_image_path)

#
# ## padding图片
# path = 'to_lmdb/fapiao.txt'
# file =  open('to_lmdb/fapiao1.txt','w')
#
# max_length = 0
# with open(path,'r') as f:
# 	for line in f.readlines():
# 		image = line.split(' ')[0]
# 		label = line.split(' ')[1].replace('\n','')
# 		if len(label)>max_length:
# 			max_length = len(label)
#
# with open(path,'r') as f:
# 	for line in f.readlines():
# 		image = line.split(' ')[0]
# 		label = line.split(' ')[1].replace('\n','')
# 		if len(label)<max_length:
# 			label = str(label)
# 			label+= (max_length-len(label))*'0'
# 			file.write(image+' '+label)
# 			file.write('\n')
# 		else:
# 			label = str(label)
# 			label = label[:max_length]
# 			file.write(image+' '+label)
# 			file.write('\n')
# 	file.close()


#


#
# def numbertostr(str_num):
# 	num_to_ch_dic = {0: '零', '.': '点', 1: '壹', 2: '贰', 3: '叁', 4: '肆', 5: '伍', 6: '陆', 7: '柒', 8: '捌', 9: '玖', 10: 'z'}
# 	li_dw = ['萬', '亿', '萬']
# 	character_num = ['壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
# 	li_dot = ['分', '角']
# 	li_mod = list("拾佰仟")
# 	# str_num = "1000000001000001"
# 	final_str = ''
#
# 	detail_dw = list("分角圆拾佰仟")
# 	for i in li_dw:
# 		detail_dw.append(i)
# 		for j in li_mod:
# 			detail_dw.append(j)
# 	detail_dw.reverse()
# 	# print(detail_dw)
# 	max_len = len(detail_dw)
#
# 	li_num = list(str(int(Decimal(str_num) * 100)).rjust(max_len, '0'))
# 	# print(li_num)
# 	li_dw.append('圆')
# 	index_tmp = 0
# 	while index_tmp < max_len:
# 		if detail_dw[index_tmp] in li_dw:
# 			if li_num[index_tmp] == '0':
# 				li_num[index_tmp] = '10'
# 		li_num[index_tmp] = num_to_ch_dic[int(li_num[index_tmp])]
# 		index_tmp = index_tmp + 1
# 	# print(li_num)
# 	li_num_str = ''.join(li_num)
# 	# print(li_num_str)
# 	li_num_str = li_num_str.replace('零零零z', '---z')
# 	li_num_str = li_num_str.replace('零零z', '--z')
# 	li_num_str = li_num_str.replace('z零零', 'z--')
# 	li_num_str = li_num_str.replace('z零', 'z-')
# 	li_num_str = li_num_str.replace('零z', '-z')
# 	# print(li_num_str)
# 	li_num = list(li_num_str)
# 	index_tmp = 0
#
# 	start_sign = 0
# 	while index_tmp < max_len:
# 		if start_sign == 0:
# 			if li_num[index_tmp] in character_num:
# 				start_sign = 1
# 				final_str = li_num[index_tmp] + detail_dw[index_tmp]
# 		elif start_sign == 1:
# 			if li_num[index_tmp] == 'z':
# 				final_str = final_str + detail_dw[index_tmp] + '零'
# 			elif li_num[index_tmp] == '-':
# 				final_str = final_str + ''
# 			elif li_num[index_tmp] == '零':
# 				final_str = final_str + '零'
# 			else:
# 				final_str = final_str + li_num[index_tmp] + detail_dw[index_tmp]
# 		index_tmp = index_tmp + 1
# 	if start_sign == 0:
# 		final_str = '零圆零分整'
# 	else:
# 		# print(final_str)
# 		final_str = final_str.replace('零零', '零')
# 		final_str = final_str.replace('亿萬', '亿零')
# 		final_str = final_str + '整'
# 		final_str = final_str.replace('圆零整', '圆整')
# 		final_str = final_str.replace('角零整', '角整')
# 		final_str = final_str.replace('零零', '零')
#
# 	return  final_str

# path ='to_lmdb/number/9.jpg'
# img = Image.open(path).convert('L')
# invert_img = PIL.ImageOps.invert(img)
#
# # plt.imshow(invert_img,cmap=plt.get_cmap('gray'))
# # plt.show()
# invert_img.save('ttt.jpg')


##整合发票数据
# bignumber='零壹贰叁肆伍陆柒捌玖拾佰仟万亿元角分整圆'
#
# path ='to_lmdb/alltext'
# all_txt = open('to_lmdb/bignumber.txt','w')
# name_list =sorted(os.listdir(path))
# 	# file = open(os.path.join(path,item),'r')
# i=0
# for item in name_list:
# 	with open(os.path.join(path,item),'r',encoding='utf-8') as file:
# 		for line in file.readlines():
# 			print(line)
# 			if line is not None:
# 				for char in line:
# 					if char in bignumber:
# 						all_txt.write(line)
# 						i+=1
# 						break
# 					else:
# 						continue
# print(i)

#已有图像

import torch
from torch.autograd import Variable

from graphviz import Digraph

def make_dot(var, params=None):
    """
    画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    蓝色节点表示有梯度计算的变量Variables;
    橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left',
                              fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # 多输出场景 multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    return dot

