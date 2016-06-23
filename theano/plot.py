import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import os


#parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--plot', type=str, default='plot.png', help='plotfile name with .png')
parser.add_argument('--log', type=str, default='log.txt', help='log file name')
parser.add_argument('--winVal', type=int, default='200', help='window for Val')
parser.add_argument('--winTrain', type=int, default='200', help='window for Train')
parser.add_argument('--no-legend', dest='legend', action='store_false')
parser.add_argument('--no-accuracy', dest='accuracy', action='store_false')
parser.add_argument('--no-loss', dest='loss', action='store_false')
parser.add_argument('--start_epoch', type=float, default=-1.0, help='start plotting from that epoch')
parser.set_defaults(loss=True)
parser.set_defaults(legend=True)
parser.set_defaults(accuracy=True)

args = parser.parse_args()

plotname = args.plot
windowVal = args.winVal
windowTrain = args.winTrain
accuracy = []


def movingAverage(loss, window):
    mas = []
    for i in range(len(loss)):
        j = i - window + 1
        if (j < 0):
            j = 0
        sum = 0.0
        for k in range(window):
            sum += loss[j + k]
        mas.append(sum / window)
    return mas


def plotTrainVal(filename, index, plotLabel):
    valx = []
    valy = []
    trainx = []
    trainy = []
    train_accuracyx = []
    train_accuracyy = []
    val_accuracyx = []
    val_accuracyy = []
    
    with open(filename, 'r') as logfile: 
        for st in logfile.readlines():
            head = st.split('\t')[0].strip()

            if (head[:7] == 'testing' or head[:8] == 'training'):
                iteration_expr = head[head.find(':')+1:]
                divpos = iteration_expr.find('/')
                first = iteration_expr[:divpos]
                iterations_per_epoch = float(iteration_expr[divpos+1:])
                dotpos = first.find('.')
                epoch = float(first[:dotpos])
                iteration = float(first[dotpos+1:])
                x = epoch + iteration / iterations_per_epoch
                
                st_loss = st[st.find("avg_loss"):]
                cur_loss = float(st_loss[st_loss.find(':')+1:st_loss.find('\t')])
                
                if (head[:7] == 'testing'):
                    valx.append(x)
                    valy.append(cur_loss)
                else:
                    trainx.append(x)
                    trainy.append(cur_loss)
            
            if st.strip()[:8] == "accuracy":
                cur_accuracy = float(st[st.find(':')+1:st.find("percent")]) / 100.0
                if (len(train_accuracyx) > len(val_accuracyx)):
                    val_accuracyx.append(valx[-1])
                    val_accuracyy.append(cur_accuracy)
                else:
                    train_accuracyx.append(trainx[-1])
                    train_accuracyy.append(cur_accuracy)

    while(len(valx) > 0 and valx[0] < args.start_epoch):
        valx = valx[1:]
        valy = valy[1:]

    while(len(trainx) > 0 and trainx[0] < args.start_epoch):
        trainx = trainx[1:]
        trainy = trainy[1:]


    #window config
    wndVal = min(windowVal, int(0.8 * len(valy)))
    wndTrain = min(windowTrain, int(0.8 * len(trainy)))
    
    print "Train length: ", len(trainy), " \t\t window: ", wndTrain
    print "Val length: ", len(valy), " \t\t window: ", wndVal
    
    #movAvg and correcting length
    #valy = movingAverage(valy, wndVal)
    #trainy = movingAverage(trainy, wndTrain)
    #valx = valx[:len(valy)]
    #trainx = trainx[:len(trainy)]
    

    #plotting
    greenDiff = 50
    redBlueDiff = 50
    
    if (args.loss):
        plt.plot(trainx, trainy, '#00' + hex(index * greenDiff)[2:] 
                + hex(256 - index * redBlueDiff)[2:],
                label=plotLabel + " train")
        plt.hold(True)

        plt.plot(valx, valy, '#' + hex(256 - index * redBlueDiff)[2:] 
                + hex(index * greenDiff)[2:] + '00',
                label=plotLabel + " validation")
        plt.hold(True)
    
    if (args.accuracy):
        plt.plot(train_accuracyx, train_accuracyy, '#000000',
                label=plotLabel + " train_accuracy")
        plt.hold(True)

        plt.plot(val_accuracyx, val_accuracyy, '#00FF00',
                label=plotLabel + " val_accuracy")
        plt.hold(True)
                
    print "plot index =", index
    for (x, y) in zip(val_accuracyx, val_accuracyy):
        print "\tepoch = %.0f, accuracy = %f" % (x - 1, y)
    print '\tMax: %f // Epoch: %d' % (max(val_accuracyy), val_accuracyx[val_accuracyy.index(max(val_accuracyy))])


plotTrainVal(args.log, 1, args.log)


if (args.legend):
    plt.legend(loc='upper right', fontsize='x-small')
plt.gcf().savefig(plotname)

