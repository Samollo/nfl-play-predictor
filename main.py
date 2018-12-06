#!/usr/bin/env python2.7
import random


#def extract_features():

labelsArray = ['Drive','down','TimeSecs','PlayTimeDiff','yrdline100','ydstogo','FirstDown','posteam','DefensiveTeam','Yards.Gained','Touchdown','Safety','PlayType','PassAttempt','PassOutcome','PassLength','PassLocation','AirYards','YardsAfterCatch','InterceptionThrown''RushAttempt','Reception','HomeTeam','AwayTeam']

labelsWeWant = ['','']

def split_lines(input, seed, output1, output2):
    f1 = open(output1, "w")
    f2 = open(output2, "w")
    random.seed(seed)
    out1 = []
    out2 = []
    labels = {}
    first_line = True
    for line in open(input, 'r').readlines():
        if first_line:
            first_line = False
            i = 0
            for label in line.split(','):
                labels[label] = i
                i += 1
            continue
        array = line.split(',')
        toAppend = array[labels['']] +
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
        array = play.split(',')
        toAppend = array[2:4] + array[7:8] + array[11:12]
        toAppend.append(array[14])
        toAppend.append(array[29])
        toAppend.append(array[32])
        toAppend.append(array[43])
        X.append(toAppend)


    return X, Y


if __name__ == '__main__':
    split_lines('dataset.csv', 2, 'train', 'test')
    read_data('train')