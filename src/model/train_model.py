from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def basic_model_run(X, Y, splitsize=0.3):

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=splitsize
    )  # 70% training and 30% test
    # Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred_accuracy = print(
        "Accuracy:", metrics.accuracy_score(y_test, y_pred)
    )

    return clf, y_pred, y_pred_accuracy
