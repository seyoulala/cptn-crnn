#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019/6/8 上午10:51 
# @Author : Ethan
# @Site :  
# @File : crnn_finetune.py 
# @Software: PyCharm


from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import models.my_crnn as mycrnn
import os
import utils
import dataset
import models.crnn as crnn
import re
import params



##初始化参数
def init_args():
	args = argparse.ArgumentParser()
	args.add_argument('--trainroot', help='path to dataset', default='./to_lmdb/train/')
	args.add_argument('--valroot', help='path to dataset', default='./to_lmdb/train/')
	args.add_argument('--cuda', action='store_true', help='enables cuda', default=False)

	return args.parse_args()


# custom weights initialization called on crnn
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


def val(net, dataset, criterion, max_iter=100):
	print('Start val')
	# for p in net.parameters():
	# 	p.requires_grad = False
	#固定住bp层以及dropout层参数
	net.eval()
	with torch.no_grad():

		data_loader = torch.utils.data.DataLoader(
			dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
		val_iter = iter(data_loader)
		i = 0
		n_correct = 0
		loss_avg = utils.averager()

		max_iter = min(max_iter, len(data_loader))
		for i in range(max_iter):
			data = val_iter.next()
			i += 1
			cpu_images, cpu_texts = data
			batch_size = cpu_images.size(0)
			utils.loadData(image, cpu_images)
			# 返回word索引以及整条text长度
			t, l = converter.encode(cpu_texts)
			utils.loadData(text, t)
			utils.loadData(length, l)
			preds = net(image)
			# preds = crnn(image)
			# preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
			preds_size = torch.IntTensor([preds.size(0)] * batch_size)

			cost = criterion(preds, text, preds_size, length) / batch_size
			loss_avg.add(cost)
			_, preds = preds.max(2)
			preds = preds.transpose(1, 0).contiguous().view(-1)
			# word索引解码为文字,预测结果
			sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
			#原始序列标签
			list_1 = []
			for i in cpu_texts:
				list_1.append(i.decode('utf-8', 'strict'))
			for pred, target in zip(sim_preds, list_1):
				if pred == target:
					n_correct += 1
		#
		# raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
		# for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
		# 	print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

		print(n_correct)
		print(max_iter * params.batchSize)
		accuracy = n_correct / float(max_iter * params.batchSize)
		print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(crnn, criterion, optimizer, train_iter):
	data = train_iter.next()
	cpu_images, cpu_texts = data
	batch_size = cpu_images.size(0)
	utils.loadData(image, cpu_images)
	#解码文本长度，以及长度
	t, l = converter.encode(cpu_texts)
	#复制文本
	utils.loadData(text, t)
	utils.loadData(length, l)
	preds = crnn(image)
	# preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
	preds_size = torch.IntTensor([preds.size(0)] * batch_size)
	#CTC损失输入为预测结果,label，预测结果长度，实际长度
	cost = criterion(preds, text, preds_size, length) / batch_size
	# crnn.zero_grad()
	optimizer.zero_grad()
	cost.backward()
	optimizer.step()
	return cost


def training(crnn, train_loader, criterion, optimizer):
	for total_steps in range(params.niter):
		train_iter = iter(train_loader)
		i = 0
		print("total number", len(train_loader))
		while i < len(train_loader):
			# for p in crnn.parameters():
			# 	p.requires_grad = True
			#训练阶段
			crnn.train()
			cost = trainBatch(crnn, criterion, optimizer, train_iter)
			loss_avg.add(cost)
			i += 1
			if i % params.displayInterval == 0:
				print('[%d/%d][%d/%d] Loss: %f' %
				      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
				loss_avg.reset()
			if i % params.valInterval == 0:
				val(crnn, test_dataset, criterion)
		# 每两个epoch就保存一次模型
		if (total_steps + 1) % params.saveInterval == 0:
			torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, total_steps, i))


if __name__ == '__main__':
	args = init_args()
	manualSeed = random.randint(1, 10000)  # fix seed
	random.seed(manualSeed)
	np.random.seed(manualSeed)
	torch.manual_seed(manualSeed)
	cudnn.benchmark = True


	# store model path
	if not os.path.exists('./expr'):
		os.mkdir('./expr')

	# read train set
	#创建自由读取的数据集
	train_dataset = dataset.lmdbDataset(root=args.trainroot)
	assert train_dataset
	#是否随机采样
	if not params.random_sample:
		sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
	else:
		sampler = None

	# images will be resize to 32*160
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=params.batchSize,
		shuffle=True, sampler=sampler,
		num_workers=int(params.workers),
		collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))

	# read test set
	# images will be resize to 32*160,w为160,h为32
	#再测试时,将图像
	test_dataset = dataset.lmdbDataset(
		root=args.valroot, transform=dataset.resizeNormalize((280, 32)))
	# test_dataset = dataset.lmdbDataset(root=args.valroot,transform=dataset.resizescale())
	#类别个数
	nclass = len(params.alphabet) + 1
	nc = 1
	#讲字符进行转换
	converter = utils.strLabelConverter(params.alphabet)
	criterion = CTCLoss()
	# criterion = torch.nn.CTCLoss()
	# cnn and rnn
	image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
	text = torch.IntTensor(params.batchSize * 5)
	length = torch.IntTensor(params.batchSize)


	# crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
	crnn = crnn.CRNN(6736,hidden_unit=256)

	crnn_model_path = 'trained_models/netCRNN_4_48000.pth'
	#导入预训练模型权重
	print("loading pretrained model from %s" % crnn_model_path)
	crnn.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
	# 获取预训练的参数
	pretrained_dict = crnn.state_dict()

	# mycrnn = mycrnn.CRNN(params.imgH, nc, nclass, params.nh)
	mycrnn = mycrnn.CRNN(class_num=nclass,hidden_unit=256)
	mycrnn_dict = mycrnn.state_dict()

	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mycrnn_dict}
	# 使用预训练模型来更新参数
	mycrnn_dict.update(pretrained_dict)
	mycrnn.load_state_dict(mycrnn_dict)

	if args.cuda:
		crnn.cuda()
		image = image.cuda()
		criterion = criterion.cuda()

	# mycrnn.apply(weights_init)
	# if params.crnn != '':
	# 	print('loading pretrained model from %s' % params.crnn)
	# 	crnn.load_state_dict(torch.load(params.crnn))
	#
	# image = Variable(image)
	# text = Variable(text)
	# length = Variable(length)

	# loss averager
	loss_avg = utils.averager()

	# setup optimizer
	if params.adam:
		optimizer = optim.Adam(filter(lambda p: p.requires_grad,mycrnn.parameters()), lr=params.lr,
		                       betas=(params.beta1, 0.999))
	elif params.adadelta:
		optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,mycrnn.parameters()), lr=params.lr)
	else:
		optimizer = optim.RMSprop(filter(lambda p: p.requires_grad,mycrnn.parameters()), lr=params.lr)

	# print(mycrnn)
	training(mycrnn, train_loader, criterion, optimizer)
#



