import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    predictions = np.load("/home/felix/line_ws/src/line_tools/python/line_net/predictions_train.npy")
    gts = np.load("/home/felix/line_ws/src/line_tools/python/line_net/ground_truths_train.npy")

    predictions = np.squeeze(predictions)
    print(predictions.shape)
    print(gts.shape)

    pred_arg = np.argmax(predictions, axis=1)
    gt_arg = np.argmax(gts, axis=1)
    corrects = np.where(pred_arg == gt_arg)
    correct_values = pred_arg[corrects]
    print("Percentage of correct predictions: {}".format(corrects[0].shape[0]))
    print(correct_values)
    print(np.unique(correct_values))

    print("Number of correct preds if only k = 3:")
    print(np.where(gt_arg == 3)[0].shape[0])

    plt.hist(correct_values, bins=range(31))
    plt.show()

    for i in range(1):
        if np.argmax(predictions[i, :]) == np.argmax(gts[i, :]):
            fig, axs = plt.subplots(2)
            axs[0].bar(range(31), predictions[i, :])
            axs[1].bar(range(31), gts[i, :])
            plt.show()

