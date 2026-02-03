from src.data_processing import load_data, split_data
from src.model import train_model, save, load
from src.evaluation import plot_confusion_matrix
from sklearn.metrics import classification_report
import os

models = ["random_forest_v1", "knn_v1"]

def main():
    df = load_data("data/creditcard.csv")   #loads only the important features of the data

    X_train, y_train, X_test, y_test = split_data(df)

    for name in models:

        if os.path.exists('models/' + name + '.pkl'):
            model = load('models/' + name + '.pkl')
        else:
            model = train_model(X_train, y_train)
            save(model, 'models/' + name + '.pkl')

        predictions = model.predict(X_test)
        print(name + " classification report: \n")
        print(classification_report(y_test, predictions)) 
        plot_confusion_matrix(model, name, X_test, y_test)

    

if __name__ == "__main__":
    main()