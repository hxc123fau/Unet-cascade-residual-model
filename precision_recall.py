import numpy as np
from matplotlib import pyplot as plt


class calculation_precison_recall():

    def calculation(self, output, mask):
        # prediction=np.load(path_output)
        # label=np.load(path_label)
        prediction = output
        label = mask
        prediction = np.reshape(prediction, (1, -1))
        label = np.reshape(label, (1, -1))
        # print('shape',prediction.shape,label.shape)

        # precision_score = ((label == 1.) * (prediction == 1.)).sum() / float((prediction == 1.).sum())
        # recall_score = ((label == 1.) * (prediction == 1.)).sum() / (label == 1.).sum()
        # print('recall1',((label == 1.) * (prediction == 1.)).sum() )
        # print('recall2',(label == 1.).sum())

        precision_score = ((label ==0.) * (prediction ==0.)).sum() / (prediction ==0).sum()
        recall_score = ((label ==0.) * (prediction ==0.)).sum() / (label ==0.).sum()
        # print('recall1',((label ==0.) * (prediction ==0.)).sum() )
        # print('recall2',(label ==0.).sum())

        numerator = 2 * precision_score * recall_score
        denominator = precision_score + recall_score
        f1_score = numerator / denominator

        return precision_score, recall_score, f1_score
