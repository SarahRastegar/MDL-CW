from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy
from sklearn.metrics import average_precision_score, recall_score ,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2


def draw(layer_outs_test,layer_outs_train,y_test,y_train,msg):
    order = numpy.dot(layer_outs_test, numpy.log(1+layer_outs_train.T))
            #  numpy.dot(1-layer_outs_test, numpy.log(1-layer_outs_train.T))
    ret = numpy.argsort(-order, axis=1)
    relevance = numpy.zeros(ret.shape)
    for i in range(0, y_test.shape[0]):
        relevance[i, :] = numpy.argmax(y_train[ret[i, :], :], axis=1) == numpy.argmax(y_test[i, :])
    cumrel = numpy.cumsum(relevance, axis=1)
    a = numpy.array(range(1, y_train.shape[0]+1))
    precision = cumrel / a[None, :]
    a = cumrel[:, -1]
    recall = cumrel / a[:, None]

    numLevels = 11
    avg_prec = numpy.zeros(numLevels)
    std_recall = numpy.linspace(0, 1, numLevels)
    ax = numpy.linspace(0, 1, 21)
    for i in range(0, numLevels):
        precision[recall < std_recall[i]] = -numpy.inf
        avg_prec[i] = numpy.mean(numpy.max(precision, axis=1))

    plt.clf()
    plt.plot(std_recall, avg_prec, color='turquoise', lw=lw,
             label='Mean average Precision-recall curve')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xticks(ax)
    plt.yticks(ax)
    plt.rc('grid', linestyle="-", color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to {0}'.format(msg))
    plt.legend(loc="lower right")
    plt.show()


def evaluate(y_true,y_pred,n_classes):
    y_pred = label_binarize(y_pred,range(n_classes))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])


    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                     average="micro")

    # Plot Precision-Recall curve
    # plt.clf()
    # plt.plot(recall[0], precision[0], lw=lw, color='navy',
    #      label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(float(MAP)/float(n_classes)))
    # plt.legend(loc="lower left")
    # plt.show()

    # Plot Precision-Recall curve for each class
    # plt.clf()
    # plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
    #          label='micro-average Precision-recall curve (area = {0:0.2f})'
    #          ''.format(average_precision["micro"]))
    # # for i, color in zip(range(n_classes), colors):
    # #     plt.plot(recall[i], precision[i], color=color, lw=lw,
    # #              label='Precision-recall curve of class {0} (area = {1:0.2f})'
    # #              ''.format(i, average_precision[i]))
    #
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Extension of Precision-Recall curve to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    average_precisions = []
    tp=numpy.zeros(len(y_true))
    fp=numpy.zeros(len(y_true))
    tn = numpy.zeros(len(y_true))
    fn = numpy.zeros(len(y_true))
    npos=0#numpy.zeros(len(y_true))
    for index in range(n_classes):

        #row_indices_sorted = numpy.argsort(-y_pred[:, index])

        y_true_cls = y_true[:, index]
        y_pred_cls = y_pred[:, index]
        #cm = confusion_matrix(y_true_cls>0, y_pred_cls>0)

        tp = ((y_true_cls+y_pred_cls)==2)+tp
        fp = (y_true_cls<y_pred_cls)+fp
        fn = (y_true_cls>y_pred_cls)+fn
        tn = ((y_true_cls+y_pred_cls)==0)+tn
        npos = numpy.sum(y_true_cls)+npos




    fp = numpy.cumsum(fp)
    tp = numpy.cumsum(tp)
    fn = numpy.cumsum(fn)


    #rec = tp * 1.0 / npos

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    rec = tp * 1.0 / numpy.maximum((tp+fn), numpy.finfo(numpy.float64).eps)
    prec = tp * 1.0/numpy.maximum((tp + fp), numpy.finfo(numpy.float64).eps)
    #prec = numpy.cumsum(prec)
    #rec = numpy.cumsum(rec)

    mrec = numpy.concatenate(([0.], rec, [1.]))
    mpre = numpy.concatenate(([0.], prec, [0.]))
    stdrec=numpy.linspace(0,1,11)
    stdrec[-1]=stdrec[-1]-numpy.finfo(numpy.float64).eps
    stdpre=numpy.zeros(11)
    # compute the precision envelope
    j=0
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])
    for i in range(0, mpre.size - 1, 1):
        if j<11 and stdrec[j]<mrec[i]:
            stdpre[j]=mpre[i]
            j=j+1
    if rec[-1] < stdrec[-1]:
        stdpre[-1]= mpre[-1]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = numpy.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    average_precisions.append(numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


    plt.clf()
    plt.plot(stdpre,stdrec, color='turquoise', lw=lw,
             label='Mean average Precision-recall curve (area = {0:0.2f})'
             ''.format(average_precision["micro"]))
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(recall[i], precision[i], color=color, lw=lw,
    #              label='Precision-recall curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()
