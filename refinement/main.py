from numpy import array
import numpy as np
import time
import xgboost as xgb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from joblib import dump, load
from fastFM import als
from fastFM import sgd
from scipy.sparse import csr_matrix


# XgBoost
class MyXGBoostRegressor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                      max_depth=5, alpha=10, n_estimators=100)

    def load_data(self, file_path="kmeans-time-total.csv"):
        self.data = np.genfromtxt(file_path, delimiter=',', skip_header=True)
        X = self.data[:, 0:5]
        y = self.data[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=12)
        print("Data loaded successfully!")

    def train(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        end_time = time.perf_counter()
        print(f"Model trained! Training time: {(end_time - start_time) * 1000} ms")

    def predict(self):
        start_time = time.perf_counter()
        predictions = self.model.predict(self.X_test)
        end_time = time.perf_counter()
        print(f"Predicting time: {(end_time - start_time) * 1000} ms")

        ''' for over and under estimation '''
        diff_overestimate = 0
        diff_underestimate = 0
        for i, v in enumerate(predictions):
            # pred = sum(v)
            # real = sum(self.y_test[i])
            pred = v
            real = self.y_test[i]
            if pred > real:
                diff_overestimate += (pred - real)
            else:
                diff_underestimate += (real - pred)
        print("diff_overestimate: ", diff_overestimate)
        print("diff_underestimate: ", diff_underestimate)

        # self.y_test = np.array([sum(i) for i in self.y_test])
        # predictions = np.array([sum(i) for i in predictions])

        ''' for error in each iteration '''
        # self.y_test = np.array(self.y_test).T
        # predictions = np.array(predictions).T
        # print("======== print error ========")
        # mse_list = []
        # for i in range(10):
        #     mse = mean_squared_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mse_list.append(mse)
        # print("mse: ", mse_list)
        # for i in
        #
        # mae_list = []
        # for i in range(10):
        #     mae = mean_absolute_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mae_list.append(mae)
        # print("mae: ", mae_list)
        #
        # wmape_list = []
        # for i in range(10):
        #     wmape = np.sum(np.abs(self.y_test[i] - predictions[i])) / np.sum(np.abs(self.y_test[i]))
        #     wmape_list.append(wmape)
        # print("WMAPE: ", wmape_list)
        #
        # smape_list = []
        # for i in range(10):
        #     smape = 100 / len(self.y_test[i]) * np.sum(
        #         2 * np.abs(predictions[i] - self.y_test[i]) / (np.abs(self.y_test[i]) + np.abs(predictions[i])))
        #     smape_list.append(smape)
        # print("SMAPE:", smape_list)

        # mae = mean_absolute_error(self.y_test, predictions)
        # print("MAE: ", mae)
        #
        # wmape = np.sum(np.abs(self.y_test - predictions)) / np.sum(np.abs(self.y_test))
        # print("WMAPE: ", wmape)
        #
        # smape = 100 / len(self.y_test) * np.sum(
        #     2 * np.abs(predictions - self.y_test) / (np.abs(self.y_test) + np.abs(predictions)))
        # print("SMAPE:", smape)


# DistNet(MLP)
class MyDistNet:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    # load row dataset from csv file
    def load_data(self, file_path="kmeans-time-total.csv"):
        self.data = np.genfromtxt(file_path, delimiter=',')
        X = self.data[:, 0:5]
        y = self.data[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=12)
        print("load data successfully!")

    def train(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=5, activation='relu'))
        self.model.add(Dense(64, activation='relu'))  # 隐藏层
        self.model.add(Dense(1))  # 输出层，预测一个连续的值
        self.model.compile(optimizer='adam', loss='mse')

        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train, epochs=1000, verbose=2)
        end_time = time.perf_counter()
        print(f"Model trained! Training time: {(end_time - start_time) * 1000} ms")

        self.model.save('my_mlp_model.h5')

    def predict(self):
        self.model = load_model('my_mlp_model.h5')
        # loss = model.evaluate(self.X_test, self.y_test, verbose=1)
        # print("测试集上的损失：", loss)

        start_time = time.perf_counter()
        predictions = self.model.predict(self.X_test)
        end_time = time.perf_counter()
        print(f"Predicting time: {(end_time - start_time) * 1000} ms")

        ''' for over and under estimation '''
        diff_overestimate = 0
        diff_underestimate = 0
        for i, v in enumerate(predictions):
            # pred = sum(v)
            # real = sum(self.y_test[i])
            pred = v
            real = self.y_test[i]
            if pred > real:
                diff_overestimate += (pred - real)
            else:
                diff_underestimate += (real - pred)
        print("diff_overestimate: ", diff_overestimate)
        print("diff_underestimate: ", diff_underestimate)

        ''' for error in each iteration '''
        # self.y_test = np.array(self.y_test).T
        # predictions = np.array(predictions).T
        # # print(self.y_test.shape)
        # # print(predictions.shape)
        # print("======== print error ========")
        # mse_list = []
        # for i in range(10):
        #     mse = mean_squared_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mse_list.append(mse)
        # print("mse: ", mse_list)
        #
        # mae_list = []
        # for i in range(10):
        #     mae = mean_absolute_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mae_list.append(mae)
        # print("mae: ", mae_list)
        #
        # wmape_list = []
        # for i in range(10):
        #     wmape = np.sum(np.abs(self.y_test[i] - predictions[i])) / np.sum(np.abs(self.y_test[i]))
        #     wmape_list.append(wmape)
        # print("WMAPE: ", wmape_list)
        #
        # smape_list = []
        # for i in range(10):
        #     smape = 100 / len(self.y_test[i]) * np.sum(
        #         2 * np.abs(predictions[i] - self.y_test[i]) / (np.abs(self.y_test[i]) + np.abs(predictions[i])))
        #     smape_list.append(smape)
        # print("SMAPE:", smape_list)

        # self.y_test = np.array([sum(i) for i in self.y_test])
        # predictions = np.array([sum(i) for i in predictions])
        # print("======== print error ========")
        # mse = mean_squared_error(self.y_test, predictions)
        # print("MSE: ", mse)
        #
        # mae = mean_absolute_error(self.y_test, predictions)
        # print("MAE: ", mae)
        #
        # wmape = np.sum(np.abs(self.y_test - predictions)) / np.sum(np.abs(self.y_test))
        # print("WMAPE: ", wmape)
        #
        # smape = 100 / len(self.y_test) * np.sum(
        #     2 * np.abs(predictions - self.y_test) / (np.abs(self.y_test) + np.abs(predictions)))
        # print("SMAPE:", smape)


# 线性回归 (AutoML)
class MyLinearRegressor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = LinearRegression()

    def load_data(self, file_path="kmeans-time-total.csv"):
        self.data = np.genfromtxt(file_path, delimiter=',', skip_header=True)
        X = self.data[:, 0:5]
        y = self.data[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=12)
        print("Data loaded successfully!")

    def train(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        end_time = time.perf_counter()
        print(f"Model trained! Training time: {(end_time - start_time) * 1000} ms")

    def predict(self):
        start_time = time.perf_counter()
        predictions = self.model.predict(self.X_test)
        end_time = time.perf_counter()
        print(f"Predicting time: {(end_time - start_time) * 1000} ms")

        ''' for over and under estimation '''
        diff_overestimate = 0
        diff_underestimate = 0
        for i, v in enumerate(predictions):
            # pred = sum(v)
            # real = sum(self.y_test[i])
            pred = v
            real = self.y_test[i]
            if pred > real:
                diff_overestimate += (pred - real)
            else:
                diff_underestimate += (real - pred)
        print("diff_overestimate: ", diff_overestimate)
        print("diff_underestimate: ", diff_underestimate)

        ''' for error in each iteration '''
        #
        # self.y_test = np.array(self.y_test).T
        # predictions = np.array(predictions).T
        # # print(self.y_test.shape)
        # # print(predictions.shape)
        # print("======== print error ========")
        # mse_list = []
        # for i in range(10):
        #     mse = mean_squared_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mse_list.append(mse)
        # print("mse: ", mse_list)
        #
        # mae_list = []
        # for i in range(10):
        #     mae = mean_absolute_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mae_list.append(mae)
        # print("mae: ", mae_list)
        #
        # wmape_list = []
        # for i in range(10):
        #     wmape = np.sum(np.abs(self.y_test[i] - predictions[i])) / np.sum(np.abs(self.y_test[i]))
        #     wmape_list.append(wmape)
        # print("WMAPE: ", wmape_list)
        #
        # smape_list = []
        # for i in range(10):
        #     smape = 100 / len(self.y_test[i]) * np.sum(
        #         2 * np.abs(predictions[i] - self.y_test[i]) / (np.abs(self.y_test[i]) + np.abs(predictions[i])))
        #     smape_list.append(smape)
        # print("SMAPE:", smape_list)

        # self.y_test = np.array([sum(i) for i in self.y_test])
        # predictions = np.array([sum(i) for i in predictions])
        # print("======== print error ========")
        # mse = mean_squared_error(self.y_test, predictions)
        # print("MSE: ", mse)
        #
        # mae = mean_absolute_error(self.y_test, predictions)
        # print("MAE: ", mae)
        #
        # wmape = np.sum(np.abs(self.y_test - predictions)) / np.sum(np.abs(self.y_test))
        # print("WMAPE: ", wmape)
        #
        # smape = 100 / len(self.y_test) * np.sum(
        #     2 * np.abs(predictions - self.y_test) / (np.abs(self.y_test) + np.abs(predictions)))
        # print("SMAPE:", smape)


# 因子分解机 MF
class MyFM:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # self.model = als.FMRegression(n_iter=2000, init_stdev=0.1, rank=4, l2_reg_w=0.1, l2_reg_V=0.5)

        self.model = sgd.FMRegression(
            n_iter=1,  # 设置为1因为我们会手动控制循环
            init_stdev=0.1,
            rank=4,
            step_size=0.1  # 学习率
        )

    # Load row dataset from csv file
    def load_data(self, file_path="kmeans-time.csv"):
        self.data = np.genfromtxt(file_path, delimiter=',')
        X = self.data[:, 0:5]
        y = self.data[:, 5]
        X_sparse = csr_matrix(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_sparse, y, test_size=0.1,
                                                                                random_state=40)
        print("Data loaded successfully!")

    # def train(self):
    #     self.model.fit(self.X_train, self.y_train)
    #     print("Model trained and ready for predictions!")

    def train(self, n_epochs=2000):
        for epoch in range(n_epochs):
            # 使用 partial_fit 更新模型
            self.model.partial_fit(self.X_train, self.y_train)
            predictions = self.model.predict(self.X_train)
            mse = mean_squared_error(self.y_train, predictions)
            print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {mse}")

    def predict(self):
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        print("测试集上的均方误差 MSE：", mse)
        print("Predictions:")
        print(predictions)
        print("Actual values:")
        print(self.y_test)


# 非线性回归
class MyNonLinearRegression:
    def __init__(self, degree=4):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = LinearRegression()
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree)

    def load_data(self, file_path="kmeans-time-series.csv"):
        self.data = np.genfromtxt(file_path, delimiter=',')
        X = self.data[:, 0:5]
        # print(X.shape)
        y = self.data[:, 5:34]

        # poly mse = 143
        X_poly = self.poly.fit_transform(X)
        # print(X_poly.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_poly, y, test_size=0.1,
                                                                                random_state=12)

        # X_transformed = np.log1p(X)   # log mse = 869
        # X_transformed = np.sqrt(X)    # sqrt mse = 675
        # X_transformed = 1.0 / (X + 1e-8)    # 1/x mse = 1267
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_transformed, y, test_size=0.05, random_state=12)

        print("load dataset successfully！")

    def train(self):
        start_time = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        end_time = time.perf_counter()
        print(f"Model trained! Training time: {(end_time - start_time) * 1000} ms")

        joblib.dump(self.model, 'nlr.pkl')

    def predict(self):
        self.model = joblib.load('nlr.pkl')
        start_time = time.perf_counter()
        predictions = self.model.predict(self.X_test)
        end_time = time.perf_counter()
        print(f"Predicting time: {(end_time - start_time) * 1000} ms")

        ''' for over and under estimation '''
        diff_overestimate = 0
        diff_underestimate = 0
        for i, v in enumerate(predictions):
            pred = sum(v)
            real = sum(self.y_test[i])
            if pred > real:
                diff_overestimate += (pred - real)
            else:
                diff_underestimate += (real - pred)
        print("diff_overestimate: ", diff_overestimate)
        print("diff_underestimate: ", diff_underestimate)

        ''' for error in each iteration '''

        # return predictions

        # self.y_test = np.array(self.y_test).T
        # predictions = np.array(predictions).T
        # # print(self.y_test.shape)
        # # print(predictions.shape)
        # print("======== print error ========")
        # mse_list = []
        # for i in range(10):
        #     mse = mean_squared_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mse_list.append(mse)
        # print("mse: ", mse_list)
        #
        # mae_list = []
        # for i in range(10):
        #     mae = mean_absolute_error(self.y_test[i], predictions[i])
        #     # print(f"{i}-th iteration's MSE: ", mse)
        #     mae_list.append(mae)
        # print("mae: ", mae_list)
        #
        # wmape_list = []
        # for i in range(10):
        #     wmape = np.sum(np.abs(self.y_test[i] - predictions[i])) / np.sum(np.abs(self.y_test[i]))
        #     wmape_list.append(wmape)
        # print("WMAPE: ", wmape_list)
        #
        # smape_list = []
        # for i in range(10):
        #     smape = 100 / len(self.y_test[i]) * np.sum(
        #         2 * np.abs(predictions[i] - self.y_test[i]) / (np.abs(self.y_test[i]) + np.abs(predictions[i])))
        #     smape_list.append(smape)
        # print("SMAPE:", smape_list)

        # self.y_test = np.array([sum(i) for i in self.y_test])
        # predictions = np.array([sum(i) for i in predictions])
        # print("======== print error ========")
        # mse = mean_squared_error(self.y_test, predictions)
        # print("MSE: ", mse)
        #
        # mae = mean_absolute_error(self.y_test, predictions)
        # print("MAE: ", mae)
        #
        # wmape = np.sum(np.abs(self.y_test - predictions)) / np.sum(np.abs(self.y_test))
        # print("WMAPE: ", wmape)
        #
        # smape = 100 / len(self.y_test) * np.sum(
        #     2 * np.abs(predictions - self.y_test) / (np.abs(self.y_test) + np.abs(predictions)))
        # print("SMAPE:", smape)


# def error_in_iterations():


if __name__ == '__main__':
    # model = MyNonLinearRegression(degree=4)
    # model.load_data("kmeans-time-series.csv")
    # model.train()
    # model.predict()

    # xgb_regressor = MyXGBoostRegressor()
    # xgb_regressor.load_data()
    # xgb_regressor.train()
    # xgb_regressor.predict()

    mlr = MyLinearRegressor()
    mlr.load_data()
    mlr.train()
    mlr.predict()

    # fm = MyFM()
    # fm.load_data()
    # fm.train()
    # fm.predict()

    # mlp = MyDistNet()
    # mlp.load_data()
    # mlp.train()
    # mlp.predict()

    # print(mlp.X_train.shape)
    # print(mlp.y_train.shape)
    # print(mlp.X_test.shape)
    # print(mlp.y_test.shape)
