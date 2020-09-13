from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import classification_report


def classification2(reference_list, prediciton_list):
    # micro_accuracy = accuracy_score(reference_list, prediciton_list)
    micro_precision = precision_score(reference_list, prediciton_list)
    micro_recall = recall_score(reference_list, prediciton_list)
    micro_f1 = f1_score(reference_list, prediciton_list)

    return micro_precision, micro_recall, micro_f1

def classificationN(reference_list, prediciton_list):
    # micro_accuracy = accuracy_score(reference_list, prediciton_list)
    micro_precision = precision_score(reference_list, prediciton_list, average="micro")
    micro_recall = recall_score(reference_list, prediciton_list, average="micro")
    micro_f1 = f1_score(reference_list, prediciton_list, average="micro")

    macro_precision = precision_score(reference_list, prediciton_list, average="macro")
    macro_recall = recall_score(reference_list, prediciton_list, average="macro")
    macro_f1 = f1_score(reference_list, prediciton_list, average="macro")

    weighted_precision = precision_score(reference_list, prediciton_list, average="weighted")
    weighted_recall = recall_score(reference_list, prediciton_list, average="weighted")
    weighted_f1 = f1_score(reference_list, prediciton_list, average="weighted")

    return (micro_precision, micro_recall, micro_f1), (macro_precision, macro_recall, macro_f1), (weighted_precision, weighted_recall, weighted_f1)

def classificationM(reference_list, prediciton_list):
    print(classification_report(reference_list, prediciton_list))

    # micro_accuracy = accuracy_score(reference_list, prediciton_list)
    micro_precision = precision_score(reference_list, prediciton_list, average="micro")
    micro_recall = recall_score(reference_list, prediciton_list, average="micro")
    micro_f1 = f1_score(reference_list, prediciton_list, average="micro")

    macro_precision = precision_score(reference_list, prediciton_list, average="macro")
    macro_recall = recall_score(reference_list, prediciton_list, average="macro")
    macro_f1 = f1_score(reference_list, prediciton_list, average="macro")

    weighted_precision = precision_score(reference_list, prediciton_list, average="weighted")
    weighted_recall = recall_score(reference_list, prediciton_list, average="weighted")
    weighted_f1 = f1_score(reference_list, prediciton_list, average="weighted")

    return (micro_precision, micro_recall, micro_f1), (macro_precision, macro_recall, macro_f1), (weighted_precision, weighted_recall, weighted_f1)






def main():
    reference_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    prediciton_list = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    print(classification2(reference_list, prediciton_list))  # (0.5, 0.5, 0.6, 0.5454545454545454)
    print('-'*100)

    reference_list = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    prediciton_list = [1, 2, 2, 2, 3, 1, 2, 3, 3, 3]
    print(classificationN(reference_list, prediciton_list))
    print('-'*100)

    reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
    print(classificationM(reference_list, prediciton_list))


if __name__ == '__main__':
    main()

