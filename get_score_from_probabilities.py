""" USAGE: python get_score_from_probabilities.py --prediction= --anwser=
    prediction file may have less lines
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

    out = prediction_lines[iter].split(',')
    out = [float(x) for x in out]
    pred = [(x, it) for it, x in enumerate(out)]
    pred = sorted(pred, reverse=True)

    if (pred[0][1] == label):
        correct += 1
        top_coder_score = top_coder_score + 1000
    elif (pred[1][1] == label):
        #correct += 1
        top_coder_score = top_coder_score + 400
    elif (pred[2][1] == label): 
        #correct += 1
        top_coder_score = top_coder_score + 160

    if (pred[0][1] != label):
        print >> wrong_answers, answer_lines[iter] + prediction_lines[iter]
    
    if ((iter + 1) % 100 == 0):
        print >> sys.stderr, "processed %d / %d images" % (iter + 1, cnt)
        print >> sys.stderr, "expected score:", top_coder_score / (iter + 1) * 35200

print >> sys.stderr, "Final score: ", top_coder_score, " / ", cnt, "000"
print >> sys.stderr, "expected score:", top_coder_score / cnt * 35200
print >> sys.stderr, "Accuracy: ", 100.0 * correct / cnt