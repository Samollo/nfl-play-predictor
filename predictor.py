import random
import numpy
import math

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# def extract_features():

labelsXArray = ['Drive', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'FirstDown', 'posteam',
                'PlayType', 'PassAttempt', 'RushAttempt', 'HomeTeam']

labelsYArray = ['Yards.Gained', 'Touchdown', 'Safety', 'InterceptionThrown', 'Reception']

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
    corrupted = 0

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
            corrupted += 1
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
    print 'corrupted line '
    print corrupted


def read_data(filename):
    X = []
    Y = []

    post_processed = 0

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
            post_processed += 1
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
            post_processed += 1
            continue

        t = numpy.array(lineX)
        X.append(t.astype(numpy.float))

        yrdstogo = float(lineX[labelsX['ydstogo']])
        attempts = 1
        if float(lineX[labelsX['down']]) < 4:
            attempts = 4 - float(lineX[labelsX['down']])
        gain = float(lineY[labelsY['Yards.Gained']])
        if gain > yrdstogo / attempts:
            Y.append(1)
        else:
            Y.append(0)

    print 'post processed line '
    print post_processed

    return X, Y


def read_data_knn(filename):
    X = []
    Y = []

    post_processed = 0

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
            post_processed += 1
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
            post_processed += 1
            continue

        t = numpy.array(lineX)
        X.append(t.astype(numpy.float))

        yrdstogo = float(lineX[labelsX['ydstogo']])
        attempts = 1
        if float(lineX[labelsX['down']]) < 4:
            attempts = 4 - float(lineX[labelsX['down']])
        gain = float(lineY[labelsY['Yards.Gained']])
        if gain > yrdstogo / attempts:
            Y.append(True)
        else:
            Y.append(False)

    return X, Y


def productivity_eval(train_x, train_y, test_x, test_y):
    model = LogisticRegression()
    model.fit(train_x, train_y)
    isProd = 0.
    totalProd = 0.
    trueProd = 0.

    predictions = model.predict(test_x)

    for idx in range(len(predictions)):
        if predictions[idx] == 0 and test_y[idx] == 0:
            continue
        if predictions[idx] == 0 and test_y[idx] == 1:
            totalProd += 1.
            continue
        if predictions[idx] == 1 and test_y[idx] == 0:
            isProd += 1.
            totalProd += 1.
            continue
        if predictions[idx] == 1 and test_y[idx] == 1:
            trueProd += 1.
            isProd += 1.
            totalProd += 1.

    recall = isProd / totalProd
    precision = trueProd / isProd
    return recall, precision, model.coef_[0]


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


def simple_distance(data1, data2):
    if len(data1) != len(data2):
        return

    value = 0
    for i in range(len(data1)):
        value += math.pow(data1[i] - data2[i], 2)

    return math.sqrt(value)


def distant_with_weight(data1, data2, weight):
    if len(data1) != len(data2):
        return

    value = 0
    for i in range(len(data1)):
        value += math.pow((data1[i] - data2[i]) * weight[i], 2)

    return math.sqrt(value)


def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out


def k_nearest_neighbors(x, points, dist_function, w, k):
    distance = {}
    result = []

    for point in range(len(points)):
        if len(points[point]) != len(x):
            return []
        else:
            distance[point] = dist_function(x, points[point], w)

    for key, value in sorted(distance.iteritems(), key=lambda (ky, vl): (vl, ky))[:k]:
        result.append(key)

    return result


def find_best_k(train_x, train_y, dist_function, w):
    value = {}
    k_value = sampled_range(1, len(train_x) / 10, 20)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(train_x):
        X_train, X_test = numpy.array(train_x)[train_index], numpy.array(train_x)[test_index]
        y_train, y_test = numpy.array(train_y)[train_index], numpy.array(train_y)[test_index]
        for i in range(len(k_value)):
            if k_value[i] in value:
                value[k_value[i]] += eval_produtivity_classifier(numpy.asfarray(X_train), y_train,
                                                                 numpy.asfarray(X_test), y_test,
                                                                 is_productive,
                                                                 dist_function, w, k_value[i])
            else:
                value[k_value[i]] = eval_produtivity_classifier(numpy.asfarray(X_train), y_train,
                                                                numpy.asfarray(X_test), y_test,
                                                                is_productive,
                                                                dist_function, w, k_value[i])
    best_v = 10000
    best_k = 0
    for k, v in value.items():
        if v < best_v:
            best_v = v
            best_k = k
    print 'Taux pour k =', best_k, ':', float(best_v) / float(10)
    return best_k


def is_productive(x, train_x, train_y, dist_function, w, k):
    if len(train_x) != len(train_y):
        return

    true_productive = 0
    distance = k_nearest_neighbors(x, train_x, dist_function, w, k)

    for i in range(len(distance)):
        if train_y[distance[i]]:
            true_productive += 1

    ratio = float(true_productive) / float(len(distance))

    if ratio == 0.5:
        return train_y[distance[0]]
    else:
        return ratio > 0.5


def eval_produtivity_classifier(train_x, train_y, test_x, test_y, classifier, dist_function, w, k):
    false_detection = 0.0

    for point in range(len(test_x)):
        v = classifier(test_x[point], train_x, train_y, dist_function, w, k)
        if v != test_y[point]:
            false_detection += 1.0
    return false_detection / float(len(test_x))


def eval_produtivity(train_x, train_y, test_x, test_y, classifier, dist_function, k):
    isProd = 0.0
    totalProd = 0.0
    trueProd = 0.0

    for point in range(len(test_x)):
        v = classifier(test_x[point], train_x, train_y, dist_function, w, k)
        if not v and not test_y[point]:
            continue
        if not v and test_y[point]:
            totalProd += 1.0
            continue
        if v and not test_y[point]:
            isProd += 1.
            totalProd += 1.
            continue
        if v and test_y[point]:
            trueProd += 1.
            isProd += 1.
            totalProd += 1.

    recall = isProd / totalProd
    precision = trueProd / isProd

    return recall, precision


if __name__ == '__main__':
    split_lines('dataset.csv', 2, 'train', 'test')
    Xtrain, Ytrain = read_data('train')
    Xtest, Ytest = read_data('test')

    r, p, w = productivity_eval(Xtrain, Ytrain, Xtest, Ytest)
    print r, p

    Xtrain, Ytrain = read_data_knn('train')
    Xtest, Ytest = read_data_knn('test')

    j = 1000

    k = find_best_k(Xtrain[:j], Ytrain[:j], distant_with_weight, w)
    print eval_produtivity(Xtrain[:j], Ytrain[:j], Xtest[:100], Ytest[:100], is_productive,
                           distant_with_weight, k)
