import cv2


def norm(image):
    return [point for row in
            [list(map(int, row / 255)) for row in list(cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1])]
            for point in row]


def analysis(tested_values, actual_values):
    tn = 0
    fn = 0
    fp = 0
    tp = 0
    for pair in zip(tested_values, actual_values):
        if pair[0] == 0 and pair[1] == 0:
            tn += 1
        elif pair[0] == 0 and pair[1] == 1:
            fn += 1
        elif pair[0] == 1 and pair[1] == 0:
            fp += 1
        else:
            tp += 1
    return float(tn), float(fn), float(fp), float(tp)


def evaluation(processed, actual):
    tn, fn, fp, tp = analysis(norm(processed), norm(actual))
    accuracy = (tp + tn) / (tn + fn + fp + tp) # trafność
    sensitivity = tp / (tp + fn)  # czułość
    specificity = tn / (fp + tn)  # swoistość
    print('trafność: {}'.format(accuracy), 'czułość: {}'.format(sensitivity), 'swoistość: {}'.format(specificity))
    # TODO miary dla danych niezrównoważonych
