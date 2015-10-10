import sys
import caffe
import numpy as np

caffe.set_mode_gpu()

# info about classes
file = open('trainingData.csv')
data = file.readlines()[1:]
langs = set()
for line in data:
    filepath, language = line.split(',')
    language = language.strip()
    langs.add(language)
langs = sorted(langs)


# network parameters:
deploy_name = 'main_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024rd0.5-1024rd0.5_DLR'
network_name = deploy_name + '_150K-momentum'
iterations = '51000'

net = caffe.Classifier(model_file='prototxt/deploy.' + deploy_name + '.prototxt',
                       pretrained_file='models/' + network_name + '_iter_' + iterations + '.caffemodel')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
net.blobs['data'].reshape(1, 1, 256, 858)

predict_set = sys.argv[1]

if (predict_set == "test"):
    folder = 'test/png/'
    f = open('testingData.csv')
    cnt = 12320
    print_file = open('predictions/test_' + network_name + '_iter_' + iterations + '.csv', 'w')
elif (predict_set == "val"):
    folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/pngaugm/' ## stegh dreci augm
    f = open('valDataNew.csv')
    cnt = 16176
    print_file = open('predictions/validation_' + network_name + '_iter_' + iterations + '.csv', 'w')
else: # train
    folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/png/'
    f = open('trainingDataNew.csv')
    cnt = 10000
    print_file = open('predictions/train_' + network_name + '_iter_' + iterations + '.csv', 'w')
    
preds = []
labels = []
topcoder_score = 0
processed = 0

for iter in range(cnt):
    st = f.readline()
    if (predict_set == "val" or predict_set == "train"):
        (name, label) = st.split(',')
        label = int(label)
    else:
        name = st.strip()[:-4]
    processed += 1
    
    net.blobs['data'].data[...] = transformer.preprocess('data', 
        caffe.io.load_image(folder + name + '.png', color=False))
    
    out = net.forward()['loss'][0]

    pred = sorted([(x, it) for it, x in enumerate(out)], reverse=True)
    
    if (predict_set == "val" or predict_set == "train"):
        if (pred[0][1] == label):
            topcoder_score = topcoder_score + 1000
        elif (pred[1][1] == label):
            topcoder_score = topcoder_score + 400
        elif (pred[2][1] == label): 
            topcoder_score = topcoder_score + 160
    
    for i in range(3):
        lang_id = pred[i][1]
        lang = langs[lang_id]
        print_file.write(name + '.mp3,' + lang + ',' + str(i + 1) + '\n')

    if (iter % 100 == 0):
        print >> sys.stderr, "processed %d / %d images" % (iter, cnt)
        print >> sys.stderr, "score: ", topcoder_score
        print >> sys.stderr, "expected score:", topcoder_score / processed * 35200

print >> sys.stderr, "Final score: ", topcoder_score, " / ", cnt, "000"
print >> sys.stderr, "expected score:", topcoder_score / processed * 35200
