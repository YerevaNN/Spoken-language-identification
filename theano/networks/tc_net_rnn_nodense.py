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
    
    def __init__(self, train_list_raw, test_list_raw, png_folder, batch_size, dropout, l2, mode, batch_norm, rnn_num_units, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()
        self.train_list_raw = train_list_raw
        self.test_list_raw = test_list_raw
        self.png_folder = png_folder
        self.batch_size = batch_size
        self.dropout = dropout
        self.l2 = l2
        self.mode = mode
        self.batch_norm = batch_norm
        self.num_units = rnn_num_units
        
        self.input_var = T.tensor4('input_var')
        self.answer_var = T.ivector('answer_var')
        
        print "==> building network"
        example = np.random.uniform(size=(self.batch_size, 1, 128, 858), low=0.0, high=1.0).astype(np.float32) #########
        answer = np.random.randint(low=0, high=176, size=(self.batch_size,)) #########
       
        network = layers.InputLayer(shape=(None, 1, 128, 858), input_var=self.input_var)
        print layers.get_output(network).eval({self.input_var:example}).shape
        
        # CONV-RELU-POOL 1
        network = layers.Conv2DLayer(incoming=network, num_filters=16, filter_size=(7, 7), 
                                     stride=1, nonlinearity=rectify)
        print layers.get_output(network).eval({self.input_var:example}).shape
        network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=2, ignore_border=False)
        print layers.get_output(network).eval({self.input_var:example}).shape
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
        
        # CONV-RELU-POOL 2
        network = layers.Conv2DLayer(incoming=network, num_filters=32, filter_size=(5, 5), 
                                     stride=1, nonlinearity=rectify)
        print layers.get_output(network).eval({self.input_var:example}).shape
        network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=2, ignore_border=False)
        print layers.get_output(network).eval({self.input_var:example}).shape
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)

        
        # CONV-RELU-POOL 3
        network = layers.Conv2DLayer(incoming=network, num_filters=32, filter_size=(3, 3), 
                                     stride=1, nonlinearity=rectify)
        print layers.get_output(network).eval({self.input_var:example}).shape
        network = layers.MaxPool2DLayer(incoming=network, pool_size=(3, 3), stride=2, ignore_border=False)
        print layers.get_output(network).eval({self.input_var:example}).shape
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
        
        self.params = layers.get_all_params(network, trainable=True)
        
        output = layers.get_output(network)
        num_channels  = 32 
        filter_W = 104
        filter_H = 13
        # NOTE: these constants are shapes of last pool layer, it can be symbolic 
        # explicit values are better for optimizations
        
        channels = []
        for channel_index in range(num_channels):
            channels.append(output[:, channel_index, :, :].transpose((0, 2, 1)))
        
        rnn_network_outputs = []
        for channel_index in range(num_channels):
            rnn_input_var = channels[channel_index]
            
            # InputLayer       
            network = layers.InputLayer(shape=(None, filter_W, filter_H), input_var=rnn_input_var)

            # GRULayer
            network = layers.GRULayer(incoming=network, num_units=self.num_units, only_return_final=True)
            
            # BatchNormalization Layer
            if (self.batch_norm):
                network = layers.BatchNormLayer(incoming=network)
              
            # add params 
            self.params += layers.get_all_params(network, trainable=True)
            
            rnn_network_outputs.append(layers.get_output(network))
        
        all_output_var = T.concatenate(rnn_network_outputs, axis=1)
        print all_output_var.eval({self.input_var:example}).shape
        
        # InputLayer
        network = layers.InputLayer(shape=(None, self.num_units * num_channels), input_var=all_output_var)
        
        """
        # DENSE 1
        network = layers.DenseLayer(incoming=network, num_units=512, nonlinearity=rectify)
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
        if (self.dropout > 0):
            network = layers.dropout(network, self.dropout)
        print layers.get_output(network).eval({self.input_var:example}).shape
        """
        
        # Last layer: classification
        network = layers.DenseLayer(incoming=network, num_units=176, nonlinearity=softmax)
        print layers.get_output(network).eval({self.input_var:example}).shape
        
    
        self.params += layers.get_all_params(network, trainable=True)
        self.prediction = layers.get_output(network)
    
        #print "==> param shapes", [x.eval().shape for x in self.params]
        
        self.loss_ce = lasagne.objectives.categorical_crossentropy(self.prediction, self.answer_var).mean()
        if (self.l2 > 0):
            self.loss_l2 = self.l2 * lasagne.regularization.apply_penalty(self.params, 
                                                                          lasagne.regularization.l2)
        else:
            self.loss_l2 = 0
        self.loss = self.loss_ce + self.loss_l2
        
        #updates = lasagne.updates.adadelta(self.loss, self.params)
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
        return "tc_net_rnn.3conv.GRU.num_units%d.nodense.5khz" % self.num_units
    
    
    def read_batch(self, data_raw, batch_index):

        start_index = batch_index * self.batch_size
        end_index = start_index + self.batch_size
        
        data = np.zeros((self.batch_size, 1, 128, 858), dtype=np.float32)
        answers = []
        
        for i in range(start_index, end_index):
            answers.append(int(data_raw[i].split(',')[1]))
            name = data_raw[i].split(',')[0]
            path = self.png_folder + name + ".png"
            im = Image.open(path)
            data[i - start_index, 0, :, :] = np.array(im).astype(np.float32)[:128, :] / 256.0

        answers = np.array(answers, dtype=np.int32)
        return data, answers
    