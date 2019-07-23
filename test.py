import numpy as np
import sys, os
import time
import cn2an
sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import  models.my_crnn as mycrnn
import alphabets
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--images_path', type=str, default='test_images/test1.png', help=
parser.add_argument('--images_path', type=str, default='data/res', help='the path to your images')

opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
# crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
# crnn_model_path ='trained_models/netCRNN_4_48000.pth'
crnn_model_path = 'expr/crnn_Rec_done_1199_27.pth'
alphabet = str1
nclass = len(alphabet)+1


# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    #字符的编码和解码
    converter = utils.strLabelConverter(alphabet)
    #讲图片转化为灰度图
    image = cropped_image.convert('L')
    # scale = image.size[1]*1.0/32

    ## 训练的时候将高度变为32,宽缩放相同比例
    w = int(image.size[0] / (280/160))
    #讲图片resize成宽为32
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    print(image.size())
    # image = Variable(image)


    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    ## 在transpose,view之前需要进行一次contiguous
    preds = preds.transpose(1, 0).contiguous().view(-1)

    # preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds_size = torch.IntTensor([preds.size(0)])
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    #修改图片中的错别字
    if '元' in sim_pred:
        sim_pred = sim_pred.replace('元','圆')
    # print(sim_pred)

    return sim_pred


if __name__ == '__main__':

	# crnn network
    # model = crnn.CRNN(32, 1, nclass, 256)
    # model = crnn.CRNN(nclass,hidden_unit=256)
    model= mycrnn.CRNN(nclass,hidden_unit=256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path,map_location='cpu'))
    
    # started = time.time()
    # ## read an image
    # image = Image.open(opt.images_path)
    #
    # crnn_recognition(image, model)
    # finished = time.time()
    # print('elapsed time: {0}'.format(finished-started))
    txt_file = open('ocr_image_test.txt','w')

    #用于预测所有的结果
    files = sorted(os.listdir(opt.images_path))
    for file in files:
        full_path = os.path.join(opt.images_path,file)
        print("====================")
        print("ocr image is %s" %full_path)
        started = time.time()
        image = Image.open(full_path)

        result = crnn_recognition(image,model)
        #将结果变为阿拉伯数字
        # check_vaild = ['元','圆','整']
        # for item in check_vaild:
        #     if item in result:
        #         result = result.replace(item,'')
        # try:
        #     number = cn2an.cn2an(result,'strict')
        # except ValueError as e:
        #     print('中文格式不符合')
        #     number = 'notvaild'
        print('results: {0}'.format(result))

        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))
        txt_file.writelines(file+' '+result)
        txt_file.write('\n')
    txt_file.close()



    