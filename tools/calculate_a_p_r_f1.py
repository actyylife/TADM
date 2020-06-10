import numpy as np


def calculate_evaluation(predict, label):
    """

    :type label: np.ndarray
    :type predict: np.ndarray
    :param predict: numpy (num_dev,)
    :param label: numpy (num_dev,)
    :return:
    """
    predict = predict.astype(np.int)
    label = label.astype(np.int)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for predict_value, label_value in zip(predict, label):
        if predict_value == 1 and label_value == 1:
            TP += 1
        if predict_value == 1 and label_value == 0:
            FP += 1
        if predict_value == 0 and label_value == 0:
            TN += 1
        if predict_value == 0 and label_value == 1:
            FN += 1

    # # # debug
    # print('<TP %d> <FP %d> <TN %d> <FN %d>' % (TP, FP, TN, FN))

    return TP, FP, TN, FN, len(label)
