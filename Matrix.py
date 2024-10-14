import pickle
import numpy as np

def get_score(y_true, y_probs):
    pred_one_num = np.sum(y_probs) # TP + FP
    target_one_num = np.sum(y_true) # TP + FN

    TP = np.sum(y_probs*y_true) # TP
    FP = pred_one_num - TP
    FN = target_one_num - TP
    TN = y_true.shape[0] - pred_one_num - FN

    Acc = (TP+TN)/(TP+TN+FP+FN)
    if TP + FP == 0: 
        Precision = -1
    else:
        Precision = TP/(TP+FP)
    if TP + FN == 0:
        Recall = -1
    else:
        Recall = TP / (TP + FN)
    if TN + FP == 0:
        Specificity = -1
    else:
        Specificity = TN / (TN + FP)
    if Precision is -1 or Recall is -1:
        F1 = -1
    else:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    return Acc, Precision, Recall, Specificity, F1

def get_scores(y_true, y_probs, all_classes):
    if True:
        # print('\nEligible_imgs: ', NoFindingIndex)
        # print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)
        GT_and_probs = {'y_true': y_true, 'y_probs': y_probs}
        with open('GT_and_probs', 'wb') as handle:
            pickle.dump(GT_and_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    '''
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall  = Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    F1 = 2 * (P * R) / (P + R)
    P: Precision
    R: Recall
    '''
    class_scores = {}
    for i, key in enumerate(all_classes):
        Acc, Prec, Reca, Spec, F1 = get_score(y_true[:, i], y_probs[:, i])
        class_scores[i] = [Acc, Prec, Reca, Spec, F1]
        print('class %s: Acc:%f, Prec:%f, Reca:%f, Spec:%f, F1:%f'%(key, Acc, Prec, Reca, Spec, F1))
    
    return class_scores
