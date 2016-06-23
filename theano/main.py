import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json
import importlib

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

# TODO: add argument to choose training set
parser.add_argument('--network', type=str, default="network_batch", help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--mode', type=str, default="train", help='mode: train/test/test_on_train')
parser.add_argument('--batch_size', type=int, default=32, help='no commment')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--log_every', type=int, default=100, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=50000, help='save state every x iteration')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--no-batch_norm', dest="batch_norm", action='store_false', help='batch normalization')
parser.add_argument('--rnn_num_units', type=int, default=500, help='number of hidden units if the network is RNN')
parser.add_argument('--equal_split', type=bool, default=False, help='use trainEqual.csv and valEqual.csv')
parser.add_argument('--forward_cnt', type=int, default=1, help='if forward pass is nondeterministic, then how many forward passes are averaged')

parser.set_defaults(batch_norm=True)
args = parser.parse_args()
print args

if (args.equal_split):
    train_listfile = open("/mnt/hdd615/Hrayr/Spoken-language-identification/trainEqual.csv", "r")
    test_listfile = open("/mnt/hdd615/Hrayr/Spoken-language-identification/valEqual.csv", "r")
else:
    train_listfile = open("/mnt/hdd615/Hrayr/Spoken-language-identification/trainingDataNew.csv", "r")
    test_listfile = open("/mnt/hdd615/Hrayr/Spoken-language-identification/valDataNew.csv", "r")

train_list_raw = train_listfile.readlines()
test_list_raw = test_listfile.readlines()

print "==> %d training examples" % len(train_list_raw)
print "==> %d validation examples" % len(test_list_raw)

train_listfile.close()
test_listfile.close()

args_dict = dict(args._get_kwargs())
args_dict['train_list_raw'] = train_list_raw
args_dict['test_list_raw'] = test_list_raw
args_dict['png_folder'] = "/mnt/hdd615/Hrayr/Spoken-language-identification/train/png/"
    


print "==> using network %s" % args.network
network_module = importlib.import_module("networks." + args.network)
network = network_module.Network(**args_dict)


network_name = args.prefix + '%s.bs%d%s%s' % (
    network.say_name(),
    args.batch_size, 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "")
    
print "==> network_name:", network_name


start_epoch = 0
if args.load_state != "":
    start_epoch = network.load_state(args.load_state) + 1

def do_epoch(mode, epoch):
    # mode is 'train' or 'test' or 'predict'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()

    batches_per_epoch = network.get_batches_per_epoch(mode)
    all_prediction = []

    for i in range(0, batches_per_epoch):
        step_data = network.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        log = step_data["log"]
        
        avg_loss += current_loss
        if (mode == "predict" or mode == "predict_on_train"):
            all_prediction.append(prediction)
            for pass_id in range(args.forward_cnt-1):
                step_data = network.step(i, mode)
                prediction += step_data["prediction"]
                current_loss += step_data["current_loss"]
            prediction /= args.forward_cnt
            current_loss /= args.forward_cnt
            
        for x in answers:
            y_true.append(x)
        
        for x in prediction.argmax(axis=1):
            y_pred.append(x)
        
        if ((i + 1) % args.log_every == 0):
            cur_time = time.time()
            print ("  %sing: %d.%d / %d \t loss: %3f \t avg_loss: %.5f \t %s \t time: %.2fs" % 
                (mode, epoch, (i + 1) * args.batch_size, batches_per_epoch * args.batch_size, 
                 current_loss, avg_loss / (i + 1), log, cur_time - prev_time))
            prev_time = cur_time
      
    
    #print "confusion matrix:"
    #print metrics.confusion_matrix(y_true, y_pred)
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
    
    if (mode == "predict"):
        all_prediction = np.vstack(all_prediction)
        pred_filename = "predictions/" + ("equal_split." if args.equal_split else "") + \
                         args.load_state[args.load_state.rfind('/')+1:] + ".csv"
        with open(pred_filename, 'w') as pred_csv:
            for x in all_prediction:
                print >> pred_csv, ",".join([("%.6f" % prob) for prob in x])
                    
    return avg_loss / batches_per_epoch


if args.mode == 'train':
    print "==> training"   	
    for epoch in range(start_epoch, args.epochs):
        do_epoch('train', epoch)
        test_loss = do_epoch('test', epoch)
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, test_loss)
        print "==> saving ... %s" % state_name
        network.save_params(state_name, epoch)
        
elif args.mode == 'test':
    do_epoch('predict', 0)
elif args.mode == 'test_on_train':
    do_epoch('predict_on_train', 0)
else:
    raise Exception("unknown mode")