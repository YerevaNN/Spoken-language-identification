""" Usage: python get_output_layers.py test|val
"""
import sys
import caffe
import numpy as np

caffe.set_mode_gpu()

deploy = '../prototxt/deploy.augm_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024rd0.3-1024rd0.3.prototxt'
model = 'augm_dropout0.3_on_augm84K-lr0.01_30K_iter_75000'
model_path = '../models/' + model + '.caffemodel'

"""
####################### networks with no augmentation ##########################
net = caffe.Classifier(deploy, model_path)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 1, 256, 858)

folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/png/'
cnt = 12320
file = open('../valEqual.csv', 'r')
prob_file = open('probs/val/' + model + '.csv', 'w')

for iter in range(cnt):
    name = file.readline().split(',')[0]
    net.blobs['data'].data[...] = transformer.preprocess('data', 
            caffe.io.load_image(folder + name + '.png', color=False))
    probs = net.forward()['loss'][0]
    probs = [str(x) for x in probs]
    prob_file.write(','.join(probs) + '\n')
    
    if (iter % 100 == 0):
        print "processed %d images" % (iter + 1)
"""

######################### networks with augmentation ###########################
assert sys.argv[1] in ('test', 'val')
dataset = sys.argv[1]
augm_cnt = 20
cnt = 12320

if (dataset == 'val'):
    folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/pngaugm/'
    file = open('../valEqual.csv', 'r')
else:
    folder = '../test/pngaugm/'
    file = open('../testingData.csv', 'r')

# sum - mean of augm_cnt versions of speech
# log - mean of logs of augm_cnt versions of speech
# dense - last dense layer, 1024 outputs
prob_file_sum = open('probs/' + dataset + '/' + model + '.sum' + str(augm_cnt) + '.csv', 'w')
prob_file_log = open('probs/' + dataset + '/' + model + '.log' + str(augm_cnt) + '.csv', 'w')
dense_file = open('probs/' + dataset + '/'+ model + '.dense' + str(augm_cnt) + '.csv', 'w')

net = caffe.Classifier(deploy, model_path)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

net.blobs['data'].reshape(augm_cnt, 1, 256, 768)
for iter in range(cnt):
    if (dataset == 'val'):
        name = file.readline().split(',')[0]
    else:
        name = file.readline().strip()[:-4]
    X = np.zeros((augm_cnt, 1, 256, 768), dtype=np.float32)
    for index in range(augm_cnt):
        augm_path = folder + name + '.' + str(index) + '.png'
        X[index] = transformer.preprocess('data', caffe.io.load_image(augm_path, color=False))

    net.blobs['data'].data[...] = X
    out = net.forward()['loss']
    probs_sum = out.mean(axis=0)
    probs_log = np.log(out + 1e-7).mean(axis=0)
    dense = net.blobs['ip2new'].data
    
    probs_sum = [str(x) for x in probs_sum]
    prob_file_sum.write(','.join(probs_sum) + '\n')
    
    probs_log = ["%f" % x for x in probs_log]
    prob_file_log.write(','.join(probs_log) + '\n')
    
    for index in range(augm_cnt):
        tmp = [str(x) for x in dense[index]]
        dense_file.write(','.join(tmp) + '\n')
    
    if (iter % 10 == 0):
        print "processed %d images" % (iter + 1)
