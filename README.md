# Spoken language identification with deep learning

Read more in the following blog posts:

* [About TopCoder contest and our CNN-based solution implemented in Caffe](http://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks/) (October 2015)
* [About combining CNN and RNN using Theano/Lasagne](http://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/) (June 2016)

Theano/Lasagne models are [here](/theano). The basic steps to run them are:

* Download the dataset from [here](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16555&pm=13978) or use your own dataset.
* Create spectrograms for recording using `create_spectrograms.py` or `augment_data.py`. The latter will also augment the data by randomly perturbing the spectrograms and cropping a random interval of length 9s from the recording. 
* Create listfiles for training set and validation set, where each row of the a listfile describes one example and has 2 values seperated by a comma. The first one is the name of the example, the second one is the label (counting starts from 0). A typical listfile will look like [this](https://gist.github.com/Harhro94/aa11fe6b454c614cdedea882fd00f8d7).
* Change the `png_folder` and listfile paths in [`theano/main.py`](/theano/main.py).
* Run `theano/main.py`.