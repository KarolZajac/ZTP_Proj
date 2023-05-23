import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.preprocessing import LabelEncoder

from dask_ml.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import log_loss
import schedule
import time
import sys


def get_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train.to_dask_array(lengths=True)).compute()
    y_test_pred = model.predict(X_test.to_dask_array(lengths=True)).compute()

    # Test set metrics
    model_test_accuracy = accuracy_score(y_test.to_dask_array(lengths=True), y_test_pred)
    model_test_rmse = mean_squared_error(y_test.to_dask_array(lengths=True), y_test_pred, squared=False)
    model_test_mae = mean_absolute_error(y_test.to_dask_array(lengths=True), y_test_pred)
    model_test_auc = roc_auc_score(y_test.to_dask_array(lengths=True), y_test_pred)
    model_test_loss = log_loss(y_test.to_dask_array(lengths=True), y_test_pred)

    # Training set metrics
    model_train_accuracy = accuracy_score(y_train.to_dask_array(lengths=True), y_train_pred)
    model_train_rmse = mean_squared_error(y_train.to_dask_array(lengths=True), y_train_pred, squared=False)
    model_train_mae = mean_absolute_error(y_train.to_dask_array(lengths=True), y_train_pred)
    model_train_auc = roc_auc_score(y_train.to_dask_array(lengths=True), y_train_pred)
    model_train_loss = log_loss(y_train.to_dask_array(lengths=True), y_train_pred)

    return [model_train_accuracy, model_train_rmse, model_train_mae, model_train_auc, model_train_loss,
            model_test_accuracy, model_test_rmse, model_test_mae, model_test_auc, model_test_loss]


def preprocess(data):
    data = data.dropna()
    data["y"] = data["y"].replace({"yes": 1, "no": 0})

    le = LabelEncoder()
    data['job'] = le.fit_transform(data['job'])
    data['marital'] = le.fit_transform(data['marital'])
    data['education'] = le.fit_transform(data['education'])
    data['housing'] = le.fit_transform(data['housing'])
    data['loan'] = le.fit_transform(data['loan'])
    data['contact'] = le.fit_transform(data['contact'])
    data['month'] = le.fit_transform(data['month'])
    data['poutcome'] = le.fit_transform(data['poutcome'])
    data['default'] = le.fit_transform(data['default'])

    return data


def train(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train.to_dask_array(lengths=True), y_train.to_dask_array(lengths=True))
    return model


def job():
    print("job:")
    data = dd.read_csv("bank.csv", header=0, delimiter=';')
    data = preprocess(data)

    X = data.drop('y', axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = train(X_train, y_train)

    predictions = model.predict(X_test.to_dask_array(lengths=True)).compute()
    metrics = get_metrics(model, X_train, X_test, y_train, y_test)

    with open("./logs", 'a+') as fd:
        fd.write('Model performance for Training set\n')
        fd.write("- Accuracy: {:.4f}\n".format(metrics[0]))
        fd.write('- Root Mean Squared Error: {:4f}\n'.format(metrics[1]))
        fd.write('- Mean Absolute Error: {:4f}\n'.format(metrics[2]))
        fd.write('- AUC: {:4f}\n'.format(metrics[3]))
        fd.write('- Loss: {:4f}\n'.format(metrics[4]))

        fd.write('----------------------------------\n')

        fd.write('Model performance for Test set\n')
        fd.write('- Accuracy: {:.4f}\n'.format(metrics[5]))
        fd.write('- Root Mean Squared Error: {:.4f}\n'.format(metrics[6]))
        fd.write('- Mean Absolute Error: {:.4f}\n'.format(metrics[7]))
        fd.write('- AUC: {:.4f}\n'.format(metrics[8]))
        fd.write('- Loss: {:4f}\n'.format(metrics[9]))


if __name__ == '__main__':

    schedule.every(1).minutes.do(job)

    while 1:
        schedule.run_pending()
        time.sleep(1)