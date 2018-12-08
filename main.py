#!/usr/bin/env python2.7
import random
import numpy
from sklearn.linear_model import LogisticRegression

# def extract_features():

labelsXArray = ['Drive', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'FirstDown', 'posteam',
                'PlayType', 'PassAttempt', 'RushAttempt', 'HomeTeam']

labelsYArray = ['Yards.Gained', 'Touchdown', 'Safety', 'InterceptionThrown', 'Reception']

labels = {}
labelsX = {}
labelsY = {}

'''
def convert_data(line, type):
    if type == 'X':
        for i in range(len(labelsXArray)):
            if i == 7 && 


'''


def split_lines(input_file, seed, output1, output2):
    random.seed(seed)

    f1 = open(output1, "w")
    f2 = open(output2, "w")

    out1 = []
    out2 = []

    first_line = True

    for line in open(input_file, 'r').readlines():
        if first_line:
            first_line = False
            i = 0
            xi = 0
            yi = 0
            for label in line.split(','):
                if label in labelsXArray:
                    labelsX[label] = xi
                    xi += 1
                if label in labelsYArray:
                    labelsY[label] = yi
                    yi += 1
                labels[i] = label
                i += 1
            continue

        if len(line.split(',')) != 102:
            continue

        if random.random() > 0.5:
            out1.append(line)
        else:
            out2.append(line)

    for i in out1:
        f1.write(i)
    for j in out2:
        f2.write(j)
    f1.close()
    f2.close()


def read_data(filename):
    X = []
    Y = []

    for play in open(filename, 'r').readlines():
        lineX = []
        lineY = []
        i = 0

        for element in play.split(','):

            if labels[i] in labelsX:
                lineX.append(element)
            if labels[i] in labelsY:
                lineY.append(element)

            i += 1

        if lineX[labelsX['PlayType']] != 'Pass' and lineX[labelsX['PlayType']] != 'Run':
            continue
        if lineX[labelsX['PlayType']] == 'Pass':
            lineX[labelsX['PlayType']] = float(-1)
        if lineX[labelsX['PlayType']] == 'Run':
            lineX[labelsX['PlayType']] = float(1)

        if lineX[labelsX['posteam']] == lineX[labelsX['HomeTeam']]:
            lineX[labelsX['posteam']] = float(1)
        else:
            lineX[labelsX['posteam']] = float(-1)
        lineX[labelsX['HomeTeam']] = float(1)
        if 'NA' in (lineX or lineY):
            continue

        t = numpy.array(lineX)
        X.append(t.astype(numpy.float))


        yrdstogo = int(lineX[labelsX['ydstogo']])
        attempts = 1
        if float(lineX[labelsX['down']]) < 4:
            attempts = 4 - float(lineX[labelsX['down']])
        gain = float(lineY[labelsY['Yards.Gained']])
        if gain > yrdstogo / attempts:
            Y.append(1)
        else:
            Y.append(0)

    return X, Y



def checkNA(X, Y):
    for v in range(len(X)):
        for value in range(len(X[v])):
            if X[v][value] == 'NA':
                print value
                print X[v][8]
                print v
                return False
    for vector in Y:
        for value in range(len(vector)):
            if vector[value] == 'NA':
                print value
                return False
    return True


if __name__ == '__main__':
    split_lines('dataset.csv', 2, 'train', 'test')
    print 'labelsX'
    print labelsX
    print 'labelsY'
    print labelsY
    Xtrain, Ytrain = read_data('train')
    Xtest, Ytest = read_data('test')
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)
    predictions = model.predict(Xtest)
    print predictions
