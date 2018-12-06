#!/usr/bin/env python2.7
import random

# def extract_features():

labelsXArray = ['Drive', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'FirstDown', 'posteam',
                'DefensiveTeam', 'PlayType', 'PassAttempt', 'PassLocation', 'RushAttempt', 'HomeTeam', 'AwayTeam']

labelsYArray = ['Yards.Gained', 'Touchdown', 'Safety', 'PassOutcome', 'PassLength', 'AirYards', 'YardsAfterCatch',
                'InterceptionThrown', 'Reception']

labels = {}
labelsX = {}
labelsY = {}


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
                    labelsX[xi] = label
                    xi += 1
                if label in labelsYArray:
                    labelsY[yi] = label
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
            if labels[i] in labelsX.values():
                lineX.append(element)
            if labels[i] in labelsY.values():
                lineY.append(element)
            i += 1

        X.append(lineX)
        Y.append(lineY)

    return X, Y


if __name__ == '__main__':
    split_lines('/Users/mohammed/Downloads/dataset.csv', 2, 'train', 'test')
    print labels
    print labelsX
    print labelsY
    X, Y = read_data('train')
    print X[:10]
    print Y[:10]
