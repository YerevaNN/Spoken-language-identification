import random
import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.nonlinearities import rectify, softmax, sigmoid, tanh

import PIL.Image as Image
from base_network import BaseNetwork

floatX = theano.config.floatX


class Network(BaseNetwork):
    
    def __init__(self, train_list_raw, test_list_raw, png_folder, batch_size, l2, mode, rnn_num_units, batch_norm, **kwargs):
        
        print "==> not used params in network class:", kwargs.keys()
        self.train_list_raw = train_list_raw
        self.test_list_raw = test_list_raw
        self.png_folder = png_folder
        self.batch_size = batch_size
        self.l2 = l2
        self.mode = mode
        self.num_units = rnn_num_units
        self.batch_norm = batch_norm
        
        self.input_var = T.tensor3('input_var')
        self.answer_var = T.ivector('answer_var')
        
        # scale inputs to be in [-1, 1]
        input_var_norm = 2 * self.input_var - 1
        
        print "==> building network"
        example = np.random.uniform(size=(self.batch_size, 858, 128), low=0.0, high=1.0).astype(np.float32) #########
        answer = np.random.randint(low=0, high=176, size=(self.batch_size,)) #########

        # InputLayer       
        network = layers.InputLayer(shape=(None, 858, 128), input_var=input_var_norm)
        print layers.get_output(network).eval({self.input_var:example}).shape

        # GRULayer
        network = layers.GRULayer(incoming=network, num_units=self.num_units)
        print layers.get_output(network).eval({self.input_var:example}).shape
        
        # BatchNormalization Layer
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
            print layers.get_output(network).eval({self.input_var:example}).shape
        
        # GRULayer
        network = layers.GRULayer(incoming=network, num_units=self.num_units, only_return_final=True)
        print layers.get_output(network).eval({self.input_var:example}).shape
        
        # BatchNormalization Layer
        # There are some states, where this layer was disabled
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
            print layers.get_output(network).eval({self.input_var:example}).shape
        
        # Last layer: classification
        network = layers.DenseLayer(incoming=network, num_units=176, nonlinearity=softmax)
        print layers.get_output(network).eval({self.input_var:example}).shape

        self.params = layers.get_all_params(network, trainable=True)
        self.prediction = layers.get_output(network)
        
        self.loss_ce = lasagne.objectives.categorical_crossentropy(self.prediction, self.answer_var).mean()
        if (self.l2 > 0):
            self.loss_l2 = self.l2 * lasagne.regularization.regularize_network_params(network, 
                                                                    lasagne.regularization.l2)
        else:
            self.loss_l2 = 0
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.003)
        
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.answer_var], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.answer_var],
                                       outputs=[self.prediction, self.loss])
    
    
    def say_name(self):
        return "rnn_2layers_5khz.GRU.num_units%d" % self.num_units
    
    
    def read_batch(self, data_raw, batch_index):

        start_index = batch_index * self.batch_size
        end_index = start_index + self.batch_size
        
        data = np.zeros((self.batch_size, 858, 128), dtype=np.float32)
        answers = []
        
        for i in range(start_index, end_index):
            answers.append(int(data_raw[i].split(',')[1]))
            name = data_raw[i].split(',')[0]
            path = self.png_folder + name + ".png"
            im = Image.open(path)
            data[i - start_index, :, :] = np.transpose(np.array(im).astype(np.float32) / 256.0)[:, :128]

        answers = np.array(answers, dtype=np.int32)
        return data, answers
    
    