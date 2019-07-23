import alphabets

random_sample = True
keep_ratio = True #是否保持缩放比
adam = False
adadelta = False
saveInterval = 100
valInterval = 20
n_test_disp = 10
displayInterval = 5
experiment = './expr'
alphabet = alphabets.alphabet
crnn = '' #是否导入预训练模型
beta1 =0.5
lr = 0.0001
niter = 2000 #迭代次数
nh = 256 #隐藏层数量
imgW = 280 #宽度
imgH = 32 #图片的高度
batchSize = 16 #batch大小
workers = 2 #处理数据的线程
