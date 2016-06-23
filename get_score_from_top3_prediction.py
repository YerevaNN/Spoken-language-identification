""" USAGE: python get_score_fromcsv.py --prediction= --anwser=
   
    Prediction file may have less lines
    
    Each line of prediction file must contain at least 3 integers: labels of top3
    predictions, then it may have some additional information
"""
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prediction', type=str)
parser.add_argument('--answer', type=str, default='valDataNew.csv')
args = parser.parse_args()
print args


# info about classes
file = open('trainingData.csv')
data = file.readlines()[1:]
langs = set()
for line in data:
    filepath, language = line.split(',')
    language = language.strip()
    langs.add(language)
langs = sorted(langs)


prediction_file = open(args.prediction, 'r')
prediction_lines = prediction_file.readlines()
answer_file = open(args.answer, 'r')
answer_lines = answer_file.readlines()
cnt = len(prediction_lines)
top_coder_score = 0.0
correct = 0

wrong_answers = open('wrong_answers.txt', 'w')

for iter in range(cnt):
    st = answer_lines[iter]
    (name, label) = st.split(',')
    label = int(label)

    pred = prediction_lines[iter].split(',')
    pred = [int(x) for x in pred]

    if (pred[0] == label):
        correct += 1
        top_coder_score = top_coder_score + 1000
    elif (pred[1] == label):
        #correct += 1
        top_coder_score = top_coder_score + 400
    elif (pred[2] == label):
        #correct += 1
        top_coder_score = top_coder_score + 160

    if (pred[0] != label):
        print >> wrong_answers, (answer_lines[iter] + str(pred[3 + pred[0]]) + ',' + str(pred[3 + pred[1]]) + ',' + 
            str(pred[3 + pred[2]]) + ', votes for correct answer: ' + str(pred[3 + label])) 

    if ((iter + 1) % 100 == 0):
        print >> sys.stderr, "processed %d / %d images" % (iter + 1, cnt)
        print >> sys.stderr, "expected score:", top_coder_score / (iter + 1) * 35200

print >> sys.stderr, "Final score: ", top_coder_score, " / ", cnt, "000"
print >> sys.stderr, "expected score:", top_coder_score / cnt * 35200
print >> sys.stderr, "Accuracy: ", 100.0 * correct / cnt