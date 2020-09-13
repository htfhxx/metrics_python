
def evaluate_2(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    # tp = ((y_true==1) & (y_pred==1)).sum()
    # fp = ((y_true==0) & (y_pred==1)).sum()
    # fn = ((y_true==1) & (y_pred==0)).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1



def evaluate_N(y_true, y_pred, N, average=None):
    tp_list,fp_list, fn_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
    for i in range(1, N+1):
        y_true_tmp = [1 if j==i else 0 for j in y_true]
        y_pred_tmp = [1 if j==i else 0 for j in y_pred]
        # tp, fp, fn = count_tp_fp_fn(y_true_tmp, y_pred_tmp)
        tp = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 1 and b == 0)
        tp_list[i-1]=tp
        fp_list[i-1]=fp
        fn_list[i-1]=fn

    if average == 'micro':
        tp = sum(tp_list)
        fp = sum(fp_list)
        fn = sum(fn_list)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    elif average == 'macro':
        precision_list, recall_list, f1_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
        for i in range(1, N+1):
            precision_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fp_list[i-1] )
            recall_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fn_list[i-1] )
            if (precision_list[i-1] + recall_list[i-1]) == 0:
                f1_list[i-1] = 0.0
            else:
                f1_list[i-1] = 2 * (precision_list[i-1] * recall_list[i-1]) / (precision_list[i-1] + recall_list[i-1])
        return sum(precision_list) / N, sum(recall_list) / N, sum(f1_list) / N
    elif average == 'weighted':
        precision_list, recall_list, f1_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
        num_list = [0 for i in range(N)]
        for i in range(1, N+1):
            precision_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fp_list[i-1] )
            recall_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fn_list[i-1] )
            if (precision_list[i-1] + recall_list[i-1]) == 0:
                f1_list[i-1] = 0.0
            else:
                f1_list[i-1] = 2 * (precision_list[i-1] * recall_list[i-1]) / (precision_list[i-1] + recall_list[i-1])
            num_list[i-1] = sum(1 for a in y_true if a == i)
        assert sum(num_list) == len(y_true) == len(y_pred)
        percent_list = [a/len(y_true) for a in num_list]
        func = lambda x, y: x * y
        return sum(map(func, precision_list, percent_list)), sum(map(func, recall_list, percent_list)), sum(map(func, f1_list, percent_list))
    else:
        print('wrong average !')
        exit()



def evaluate_Multi(y_true, y_pred, N, average=None):
    # reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    # prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
    tp_list,fp_list, fn_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
    for i in range(1, N+1):
        y_true_tmp = [1 if j[i-1]==1 else 0 for j in y_true]
        y_pred_tmp = [1 if j[i-1]==1 else 0 for j in y_pred]
        # print("y_true_tmp: ",y_true_tmp)
        # print("y_pred_tmp: ",y_pred_tmp)
        tp = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true_tmp, y_pred_tmp) if a == 1 and b == 0)
        tp_list[i-1]=tp
        fp_list[i-1]=fp
        fn_list[i-1]=fn
    if average == 'micro':
        tp = sum(tp_list)
        fp = sum(fp_list)
        fn = sum(fn_list)
        if tp ==0:
            return 0.0, 0.0, 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall)== 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    elif average == 'macro':
        precision_list, recall_list, f1_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
        for i in range(1, N+1):
            precision_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fp_list[i-1] )
            recall_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fn_list[i-1] )
            if (precision_list[i-1] + recall_list[i-1]) == 0:
                f1_list[i-1] = 0.0
            else:
                f1_list[i-1] = 2 * (precision_list[i-1] * recall_list[i-1]) / (precision_list[i-1] + recall_list[i-1])
        return sum(precision_list) / N, sum(recall_list) / N, sum(f1_list) / N
    elif average == 'weighted':
        precision_list, recall_list, f1_list = [0 for i in range(N)],[0 for i in range(N)],[0 for i in range(N)]
        num_list = [0 for i in range(N)]
        for i in range(1, N+1):
            precision_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fp_list[i-1] )
            recall_list[i-1] = tp_list[i-1] / ( tp_list[i-1] + fn_list[i-1] )
            if (precision_list[i-1] + recall_list[i-1]) == 0:
                f1_list[i-1] = 0.0
            else:
                f1_list[i-1] = 2 * (precision_list[i-1] * recall_list[i-1]) / (precision_list[i-1] + recall_list[i-1])
            # print('y_true: ',y_true)
            num_list[i-1] = sum(1 for a in y_true if a[i-1] ==1)

        # assert sum(num_list) == len(y_true) == len(y_pred)
        # print('num_list: ', num_list)
        percent_list = [a/sum(num_list) for a in num_list]
        func = lambda x, y: x * y
        return sum(map(func, precision_list, percent_list)), sum(map(func, recall_list, percent_list)), sum(map(func, f1_list, percent_list))

    else:
        print('wrong average !')
        exit()


def main():
    reference_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    prediciton_list = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    print(evaluate_2(reference_list, prediciton_list))
    print('-'*100)

    reference_list = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    prediciton_list = [1, 2, 2, 2, 3, 1, 2, 3, 3, 3]
    print(evaluate_N(reference_list, prediciton_list, 3,average='micro'))
    print(evaluate_N(reference_list, prediciton_list, 3,average='macro'))
    print(evaluate_N(reference_list, prediciton_list, 3,average='weighted'))
    print('-'*100)

    reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
    print(evaluate_Multi(reference_list, prediciton_list, 3, average='micro'))
    print(evaluate_Multi(reference_list, prediciton_list, 3, average='macro'))
    print(evaluate_Multi(reference_list, prediciton_list, 3, average='weighted'))


if __name__ == '__main__':
    main()

