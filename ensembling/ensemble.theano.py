""" Usage: python ensemble.theano.py model1 [another_model]*
    
for GPU mode
    1. export PATH=$PATH:/usr/local/cuda-6.5/bin
    2. THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags='-arch=sm_30' python ensemble.theano.py model1 [another_model]*
"""

import cPickle as pickle
import sys
import caffe
import numpy as np

caffe.set_mode_gpu()

def get_score(probs, label):
    pred = sorted([(x, it) for it, x in enumerate(probs)], reverse=True)
    if (pred[0][1] == label):
        return 1000
    if (pred[1][1] == label):
        return 400
    if (pred[2][1] == label): 
        return 160
    return 0
    
def get_full_score(preds, labels):
    topCoderScore = 0.0
    for i in range(len(labels)):
        topCoderScore += get_score(preds[i], labels[i])  
    
    return topCoderScore / len(labels) * 3520

####################### COLLECTING INFO ABOUT LANGS ############################
file = open('../trainingData.csv')
data = file.readlines()[1:]
langs = set()
for line in data:
    filepath, language = line.split(',')
    language = language.strip()
    langs.add(language)
langs = sorted(langs)
file.close()

n_models = len(sys.argv) - 1
X = np.zeros((12320, n_models * 176), dtype=np.float32)
for iter in range(n_models):
    csvpath = 'probs/val/' + sys.argv[iter + 1]
    csv = open(csvpath, 'r')
    for row_id, line in enumerate(csv.readlines()):
        mas = line.split(',')
        mas = np.array([float(x) for x in mas], dtype=np.float32)
        X[row_id, 176*iter:176*(iter+1)] = mas
    csv.close()
    
Y = []
label_file = open('../valEqual.csv')
for line in label_file.readlines():
    Y.append(int(line.split(',')[1]))
label_file.close()

print "X.shape =", X.shape
print "len(Y) =", len(Y)

for iter in range(n_models):
    print "score of model %d = %f" % (iter+1, get_full_score(X[:, 176*iter:176*(iter+1)], Y))


######################### TRAINING ENSEMBLING MODEL ############################
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as layers

n_train_examples = 10000
X = X.astype(theano.config.floatX)
trainX = X[:n_train_examples]
trainY = Y[:n_train_examples]
valX = X[n_train_examples:]
valY = Y[n_train_examples:]

input_var = T.matrix('X')
target_var = T.ivector('y')

from lasagne.nonlinearities import softmax, sigmoid, rectify
network = lasagne.layers.InputLayer((None, X.shape[1]), input_var)
network = lasagne.layers.DenseLayer(network, 4000, nonlinearity=rectify)
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5), 176, nonlinearity=softmax)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 0 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

params = lasagne.layers.get_all_params(network, trainable=True)
learning_rate = theano.shared(np.float32(0.2))
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
train_fn = theano.function([input_var, target_var], loss, updates=updates)
validation_fn = theano.function([input_var, target_var], loss)

for epoch in range(1000):
    train_loss = train_fn(trainX, trainY)
    val_loss = validation_fn(valX, valY)
    print "Epoch %d: train_loss = %f, val_loss = %f, lr = %f" % (epoch + 1, train_loss, val_loss, learning_rate.get_value())
    if (epoch > 0 and epoch % 200 == 0):
        learning_rate.set_value(np.float32(learning_rate.get_value() * 0.7))
    
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], test_prediction)
all_predictions = predict_fn(valX)

score = 0.0
for probs, label in zip(all_predictions, valY):
    score += get_score(probs, label)
print "Final score on ensembling validaion = %f" % score
print "Expected score = %f" % (score / len(valY) * 3520)


print "\n\n==> creating submission..."
X = np.zeros((12320, n_models * 176), dtype=np.float32)
for iter in range(n_models):
    csvpath = 'probs/test/' + sys.argv[iter + 1]
    csv = open(csvpath, 'r')
    for row_id, line in enumerate(csv.readlines()):
        mas = line.split(',')
        mas = np.array([float(x) for x in mas], dtype=np.float32)
        X[row_id, 176*iter:176*(iter+1)] = mas
    csv.close()

prediction = predict_fn(X)
print "prediction.shape =", prediction.shape
ensembled = open('ensembled.csv', 'w')
for probs in prediction:
    out = [str(x) for x in probs]
    ensembled.write(','.join(out) + '\n')


"""
######################### SAVING MODEL TO BE ABLE TO REPRODUCE #################
print "==> Saving model..."
with open("model.pickle", 'w') as save_file:
	pickle.dump(obj = {'params' : layers.get_all_param_values(network)}, file = save_file, protocol = -1)
"""
