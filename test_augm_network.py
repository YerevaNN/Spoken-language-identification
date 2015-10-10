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
deploy_name = 'augm_32r-2-64r-2-64r-2-128r-2-128r-2-256r-2-1024rd0.3-1024rd0.3'
network_name = 'augm_dropout0.3_on_augm84K-lr0.01_30K'
iterations = '90000'
aveSamples = 20 # average over this many samples

net = caffe.Classifier(model_file='prototxt/deploy.' + deploy_name + '.prototxt',
                       pretrained_file='models/' + network_name + '_iter_' + iterations + '.caffemodel')

net.blobs['data'].reshape(1, 1, 256, 768)
predict_set = sys.argv[1]

if (predict_set == "test"):
    folder = 'test/png/'
    f = open('testingData.csv')
    cnt = 12320
    print_file = open('predictions/test_' + network_name + '_iter_' + iterations + '_' + str(aveSamples) + '.csv', 'w')
elif (predict_set == "val"):
    folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/pngaugm/'
    f = open('valEqual.csv')
    cnt = 12320
    print_file = open('predictions/validation_' + network_name + '_iter_' + iterations + '_' + str(aveSamples) + '.csv', 'w')
else: # train
    folder = '/home/brainstorm/caffe/Data/mnt/3/language/train/pngaugm/'
    f = open('trainEqual.csv')
    cnt = 10000
    print_file = open('predictions/train_' + network_name + '_iter_' + iterations + '_' + str(aveSamples) + '.csv', 'w')
    
preds = []
labels = []
topcoder_score = 0.0
processed = 0

for iter in range(cnt):
    st = f.readline()
    if (predict_set == "val" or predict_set == "train"):
        (name, label) = st.split(',')
        label = int(label)
    else:
        name = st.strip()[:-4]
    processed += 1
    out = np.zeros((176, ))
    for randomIndex in range(aveSamples):
        image = caffe.io.load_image(folder + name + '.' + str(randomIndex) + '.png', color=False)
        image = np.transpose(image, (2, 0, 1))
        #image = np.concatenate([image, np.zeros((1, 256, 858 - 768), dtype=np.float32)], axis=2)
        net.blobs['data'].data[...] = image
        out += net.forward()['loss'][0]

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
        print >> sys.stderr, network_name + '_iter_' + iterations + '_' + str(aveSamples)
        print >> sys.stderr, "processed %d / %d images (%d samples/mp3)" % (iter, cnt, aveSamples)
        print >> sys.stderr, "score: ", topcoder_score
        print >> sys.stderr, "expected score:", topcoder_score / processed * 35200

print >> sys.stderr, "Final score: ", topcoder_score, " / ", cnt, "000"
print >> sys.stderr, "expected score:", topcoder_score / processed * 35200
