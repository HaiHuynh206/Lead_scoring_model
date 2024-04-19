import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, classification_report, auc
import matplotlib.pyplot as plt
import seaborn as sns

def draw_cm(actual, predicted):
    """
    Draws a confusion matrix for the actual vs predicted labels.

    Parameters:
    - actual: Array-like of actual target values.
    - predicted: Array-like of predicted target values.
    """
    cm = confusion_matrix(actual, predicted, labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='vlag',
                xticklabels=["Not Converted", "Converted"],
                yticklabels=["Not Converted", "Converted"])
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

def draw_roc(actual, probs):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - actual: Array-like of actual target values.
    - probs: Array-like of predicted probabilities for the positive class.
    """
    fpr, tpr, thresholds = roc_curve(actual, probs, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#477ca8', lw=2, linestyle='-', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#cb3335', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Fits the model using the training data, makes predictions, and evaluates the model's performance.

    Parameters:
    - model: The machine learning model to be evaluated.
    - x_train: Training feature dataset.
    - y_train: Training target dataset.
    - x_test: Test feature dataset.
    - y_test: Test target dataset.

    Returns:
    - Prints out the accuracy score, error rate, classification report, confusion matrix, and Gini index for both training and test datasets.
    """
    model = model.fit(x_train, y_train)
    predict_train_y = model.predict(x_train)
    predict_test_y = model.predict(x_test)

    # Various performance metrics
    print("**Accuracy Score**")
    train_accuracy = accuracy_score(y_train, predict_train_y)
    test_accuracy = accuracy_score(y_test, predict_test_y)
    print("Train Accuracy is: %s" % (train_accuracy))
    print("\nTest Accuracy is: %s" % (test_accuracy))
    print("---------------------------------------------------------")

    print("\n**Accuracy Error**")
    train_error = (1 - train_accuracy)
    test_error = (1 - test_accuracy)
    print("Train Error: %s" % (train_error))
    print("\nTest Error: %s" % (test_error))
    print("---------------------------------------------------------")

    print("\n**Classification Report**")
    train_cf_report = pd.DataFrame(classification_report(y_train, predict_train_y, output_dict=True))
    test_cf_report = pd.DataFrame(classification_report(y_test, predict_test_y, output_dict=True))
    print("Train Classification Report:")
    print(train_cf_report)
    print("\nTest Classification Report:")
    print(test_cf_report)
    print("---------------------------------------------------------")

    print("\n**Confusion Matrix**")
    train_conf = confusion_matrix(y_train, predict_train_y)
    test_conf = confusion_matrix(y_test, predict_test_y)
    print("Train Confusion Matrix Report:")
    print((train_conf))
    print("\nTest Confusion Matrix Report:")
    print((test_conf))
    print("---------------------------------------------------------")

    gini_train = 2 * roc_auc_score(y_train, predict_train_y) - 1
    gini_test = 2 * roc_auc_score(y_test, predict_test_y) - 1
    print("\n**Gini Index**")
    print("Train Gini Index: %s" % (gini_train))
    print("\nTest Gini Index: %s" % (gini_test))