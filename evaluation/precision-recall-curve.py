# Import necessary libraries
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle


# Binarize the output
# This process transforms the multiclass labels into a binary format that's easier for the model to process.
y_true_bin = label_binarize(test_gen.classes, classes=[0, 1, 2])
n_classes = y_true_bin.shape[1]

# Predict the probabilities for the test set
y_score = model.predict(test_gen)

# Here, we initialize dictionaries to hold precision, recall and average_precision values for each class
precision = dict()
recall = dict()
average_precision = dict()

# Now, for each class, we calculate precision and recall values and store them in the dictionaries
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

# Next, we calculate the micro-averaged precision and recall values over all classes, giving us a single score that summarizes the overall performance of our model.
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_true_bin, y_score, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

# Now we plot the precision-recall curves. The first plot is a micro-averaged curve, which gives an overall summary of model performance.

# Setup figure size
plt.figure(figsize=(10, 7))
# Plot the micro-averaged Precision-Recall curve
plt.plot(recall["micro"], precision["micro"], label='micro-average Precision-recall curve (area = {0:0.2f})'.format(average_precision["micro"]))

# We then plot a precision-recall curve for each class separately. This allows us to see how well the model performs for each individual class.
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

# Finally, we set the labels and title for our plot and show it.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()
