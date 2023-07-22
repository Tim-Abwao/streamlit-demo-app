import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=1
)

if __name__ == "__main__":
    model = RandomForestClassifier(random_state=2)
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):2%}")
    joblib.dump(model, "model.gz")  # save model to file
