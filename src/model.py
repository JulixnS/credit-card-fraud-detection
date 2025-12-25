from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 42)
    model.fit(X_train, y_train)

    return model

def save(model, filepath):
    joblib.dump(model, filepath)
    print("Model saved to: " + filepath)

def load(filepath):
    return joblib.load(filepath)