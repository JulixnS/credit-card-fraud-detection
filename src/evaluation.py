import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, name, X_test, y_test):

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: ' + name + ' \n(Top Left: True Neg | Top Right: False Pos\nBottom Left: False Neg | Bottom Right: True Pos)')
    plt.savefig("confusion_matrixes/" + name + "_matrix.png")