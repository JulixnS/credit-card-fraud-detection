from src.data_processing import load_data, split_data
from src.model import train_model, save, load
from src.evaluation import plot_confusion_matrix
from sklearn.metrics import classification_report
import os


model_path = 'models/random_forest_v1.pkl'

def main():
    df = load_data("data/creditcard.csv")   #loads only the important features of the data

    X_train, y_train, X_test, y_test = split_data(df)

    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = train_model(X_train, y_train)
        save(model, model_path)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions)) 
    plot_confusion_matrix(model, X_test, y_test)

    

if __name__ == "__main__":
    main()