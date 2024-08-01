import matplotlib
import torch
import matplotlib.pyplot as plt
import gpytorch
from matplotlib import font_manager
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main import MyNonLinearRegression
import numpy as np

matplotlib.use('Agg')
plt.style.use("bmh")
plt.rcParams["image.cmap"] = "Blues"


class SimpleSincKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    # this is the kernel function
    # def forward(self, x1, x2, **params):
    #     # apply lengthscale
    #     x1_ = x1.div(self.lengthscale)
    #     x2_ = x2.div(self.lengthscale)
    #     # calculate the distance between inputs
    #     diff = self.covar_dist(x1_, x2_, **params)
    #     # prevent divide by 0 errors
    #     diff.where(diff == 0, torch.as_tensor(1e-20))
    #     # return sinc(diff) = sin(diff) / diff
    #     return torch.sin(diff).div(diff)

    def g(self, i):
        i = i.float()
        exp_term = torch.exp(i) - 1
        linear_term = i
        return torch.where(i < 0, exp_term, linear_term)

    def forward(self, x1, x2, diag=False, **params):
        # x2 = x2[20:]
        # 扩展 x1 和 x2 以便计算所有对之间的差异
        diff = (x1.unsqueeze(1) - x2.unsqueeze(0))#.squeeze(-1)

        # 计算距离，假设 self.covar_dist 已经返回一个与 diff 形状匹配的张量
        # 这里假设 distance 与 diff 形状一致，如果不是这样，您需要调整 self.covar_dist 方法或其调用方式
        distance = self.covar_dist(x1, x2)
        # if distance.shape != diff.shape:
            # raise ValueError("The shape of distance must match diff. Check covar_dist implementation.")

        # 创建全零张量作为输出的一部分
        # zero_tensor = torch.zeros_like(diff)
        zero_tensor = torch.ones_like(diff)
        return zero_tensor.squeeze(-1)

        # # 计算指数部分
        # exp_component = torch.exp(-(torch.log(distance + 1)) ** 2)
        #
        # # 使用 torch.where 根据条件选择输出
        # condition = (diff <= 0) & (distance >= -1)
        #
        # result = torch.where(condition, zero_tensor, zero_tensor)
        #
        # # 返回计算的核矩阵
        # return result

    # def forward(self, x1, x2, diag=False, **params):
    #     # x1 = x1.div(self.lengthscale)
    #     # x2 = x2.div(self.lengthscale)
    #     diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    #
    #     # 应用 forrester_1d 函数的逻辑，如果 x1 - x2 <= -1，则输出为0
    #
    #     diff = self.covar_dist(x1, x2)
    #     zero_tensor = torch.zeros_like(diff)
    #
    #     # x1_ = x1.div(self.lengthscale)
    #     # x2_ = x2.div(self.lengthscale)
    #     # diff = x1_.unsqueeze(1) - x2_.unsqueeze(0)
    #     exp_component = torch.exp(-(torch.log(diff + 1)) ** 2)
    #
    #     # 使用 torch.where 根据条件来选择输出
    #     # diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    #     result = torch.where((x1.unsqueeze(1) - x2.unsqueeze(0) <= 0) & (diff >= -1), zero_tensor, exp_component)
    #
    #     # 返回计算的核矩阵
    #     return result.squeeze(-1)

    # def forward(self, x1, x2, diag=False, **params):
    #     x1 = x1.div(self.lengthscale)
    #     x2 = x2.div(self.lengthscale)
    #     diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    #
    #     # 应用 forrester_1d 函数的逻辑，如果 x1 - x2 <= -1，则输出为0
    #     zero_tensor = torch.zeros_like(diff)
    #
    #     # x1_ = x1.div(self.lengthscale)
    #     # x2_ = x2.div(self.lengthscale)
    #     # diff = x1_.unsqueeze(1) - x2_.unsqueeze(0)
    #     exp_component = torch.exp(-(torch.log(diff + 1)) ** 2)
    #
    #     # 使用 torch.where 根据条件来选择输出
    #     # diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    #     result = torch.where(diff <= -1, zero_tensor, exp_component)
    #
    #     # 返回计算的核矩阵
    #     return result.squeeze(-1)


class BaseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(1)
        # self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = SimpleSincKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GpExecution:
    def __init__(self):
        self.xs = torch.linspace(0, 28, 141).unsqueeze(1)
        # self.xs = torch.linspace(0, 35, 180).unsqueeze(1)
        self.index = None
        self.GPmodel = None
        self.likelihood = None

        # train mnlr
        self.mnlr = MyNonLinearRegression(degree=4)
        self.mnlr.load_data()
        self.value_predict = np.array(self.mnlr.predict())
        self.value_real = np.array(self.mnlr.y_test)

        # set (x,y) for GP
        # self.train_x = torch.tensor([self.xs[i] for i in range(1, 145, 5)][:self.index])
        # self.train_y = torch.tensor([self.value_real[i] / self.value_predict[i] for i in range(29)][:self.index])
        self.train_x = None
        self.train_y = None
        self.row_value_real = None
        self.row_value_perdict = None

    def set_train_xy(self, row, index):
        self.index = index
        self.row_value_real = self.value_real[row]
        self.row_value_perdict = self.value_predict[row]
        # print(self.row_value_real)
        # print(self.row_value_perdict)

        self.train_x = torch.tensor([self.xs[i] for i in range(0, 141, 5)][:index], dtype=torch.float32)
        # self.train_x = torch.tensor([self.xs[i] for i in range(1, 180, 5)][:index], dtype=torch.float32)
        self.train_y = torch.tensor([self.row_value_real[i] / self.row_value_perdict[i] for i in range(29)][:index],
                                    dtype=torch.float32)
        # print(self.train_x.shape)
        # print(self.train_y.shape)

    def build_gp_model(self):
        lengthscale = 100000
        noise = 1e-4

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.GPmodel = BaseGPModel(self.train_x, self.train_y, self.likelihood)

        # fix the hyperparameters
        self.GPmodel.covar_module.lengthscale = lengthscale
        self.GPmodel.likelihood.noise = noise
        self.GPmodel.mean_module.constant = 1

        self.GPmodel.eval()
        self.likelihood.eval()

    def draw_gp_map(self, destination="image/2.4.5.pdf"):
        with torch.no_grad():
            predictive_distribution = self.likelihood(self.GPmodel(self.xs))
            predictive_mean = predictive_distribution.mean
            predictive_lower, predictive_upper = predictive_distribution.confidence_region()

            torch.manual_seed(0)
            # samples = predictive_distribution.sample(torch.Size([5]))

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('none')
        ax.patch.set_alpha(0.0)
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # plt.plot(xs, ys, label="objective", c="28")
        plt.scatter(self.train_x, self.train_y, marker="x", c="k", label="actual ratio")
        plt.plot(self.xs, predictive_mean, label="mean of predicted ratio")
        plt.fill_between(
            self.xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% Confidence Interval"
        )
        # plt.plot(self.xs, samples[0, :], alpha=0.5, label="samples")
        # for i in range(1, samples.shape[0]):
        #     plt.plot(self.xs, samples[i, :], alpha=0.5)
        font_properties = font_manager.FontProperties(size=16)
        plt.legend(loc='upper left', fontsize=15, frameon=True, facecolor='none', edgecolor='none', prop=font_properties)
        plt.savefig(destination)

    def get_gp_error(self):
        with torch.no_grad():
            # Create a tensor from a list of tensors selected by indices
            input_x = torch.tensor([self.xs[i] for i in range(0, 141, 5)])
            given_x_distribution = self.likelihood(self.GPmodel(input_x))
            given_x_mean = given_x_distribution.mean

        rate = [mean.item() for mean in given_x_mean]
        # print(rate)
        real_time = self.row_value_real
        predict_time = [self.row_value_perdict[i] * rate[i] for i in range(29)]
        return predict_time[self.index]
        # return [real_time[self.index + 1], predict_time[self.index + 1], self.row_value_perdict[self.index + 1]]
        # print(real_time)
        # print(predict_time)
        # print([mean_squared_error([self.row_value_perdict[i]], [self.row_value_real[i]])
        #        for i in range(len(self.row_value_perdict))])    # mse in the first phase
        #
        # mse = mean_squared_error([real_time[self.index + 1]], [predict_time[self.index + 1]])
        # print("index: ", self.index)
        # print("mse:", mse)
        # return mse


def error_calculator():
    iter = 10
    adjust_map = np.zeros((201, iter))

    gp = GpExecution()
    for row in range(201):
        for index in range(iter):
            gp.set_train_xy(row, index)
            gp.build_gp_model()
            # gp.draw_gp_map()
            predict_time = gp.get_gp_error()
            adjust_map[row][index] = predict_time

    real_time = np.array(gp.value_real).T
    adjust_map = np.array(adjust_map).T
    # print(self.y_test.shape)
    # print(predictions.shape)
    print("======== print error ========")
    mse_list = []
    for i in range(iter):
        mse = mean_squared_error(real_time[i], adjust_map[i])
        # print(f"{i}-th iteration's MSE: ", mse)
        mse_list.append(mse)
    print("mse: ", mse_list)

    mae_list = []
    for i in range(iter):
        mae = mean_absolute_error(real_time[i], adjust_map[i])
        # print(f"{i}-th iteration's MSE: ", mse)
        mae_list.append(mae)
    print("mae: ", mae_list)

    wmape_list = []
    for i in range(iter):
        wmape = np.sum(np.abs(real_time[i] - adjust_map[i])) / np.sum(np.abs(real_time[i]))
        wmape_list.append(wmape)
    print("WMAPE: ", wmape_list)

    smape_list = []
    for i in range(iter):
        smape = 100 / len(real_time[i]) * np.sum(
            2 * np.abs(adjust_map[i] - real_time[i]) / (np.abs(real_time[i]) + np.abs(adjust_map[i])))
        smape_list.append(smape)
    print("SMAPE:", smape_list)


def f(n, k, m):
    a = 28 * (n + k)
    b = m - 3 * n + 32 - 2 * k
    print("m: ", m)
    print("b: ", b)
    return a / b


def get_m(n, f, k):
    a = 2*n + 28*(n+k)/f + 2*k -32
    return a


if __name__ == '__main__':
    error_calculator()

    # n = 1000000
    # k = 10000
    # m = 30 * 1024 * 1024 / 4
    # print(f(n, k, m))

    # print(get_m(n, 30, k)*6/(1024*1024))

    # gp = GpExecution()
    # gp.set_train_xy(10, 4)
    # gp.build_gp_model()
    # # gp.draw_gp_map("image/4th iteration.pdf")
    # predict_time = gp.get_gp_error()
    # print(predict_time)




# # 2.4.1 Setting up the training data
# xs = torch.linspace(0, 28, 145).unsqueeze(1)
#
# # 2.4.2 Implementing a Gaussian process class
# val_y_predict = [11.34161282, 7.59117915, 7.22696008, 7.08553581, 7.06018923, 6.83196382, 6.83607625, 6.77301407, 6.70213726, 6.70759988, 6.68269758, 6.64271965, 6.70442761, 6.64453557, 6.56208854, 6.56029171, 6.50492022, 6.54041511, 6.61606085, 6.59841863, 6.6685142,  6.54902091, 6.55297523, 6.57711033, 6.58894266, 6.51970038, 6.57985635, 6.5090014, 6.48946426]
# val_y_real = [11.7584895, 8.5163212, 7.5147134, 7.5304439, 7.2535096, 7.4749154, 7.2836293, 7.6450257, 7.7235665, 7.3232264, 6.8895855, 7.2078927, 6.7524595, 7.0271194, 6.8584423, 7.1738951, 6.9948291, 6.8284746, 6.8832117, 7.1058704, 7.0752989, 7.0684063, 6.7949126, 6.9941014, 6.9152496, 6.7949531, 6.9038458, 6.7462257, 6.6000524]


# 2.4.3 Making predictions with a Gaussian process
# declare the GP
# lengthscale = 1
# noise = 1e-4
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = BaseGPModel(None, None, likelihood)
# model.covar_module.lengthscale = lengthscale    # fix the hyperparameters
# model.likelihood.noise = noise
#
# model.mean_module.constant = 1
#
# model.eval()
# likelihood.eval()


# 2.4.4 Visualizing predictions by a Gaussian process
# with torch.no_grad():
#     predictive_distribution = likelihood(model(xs))
#     predictive_mean = predictive_distribution.mean
#     predictive_lower, predictive_upper = predictive_distribution.confidence_region()
#
#     torch.manual_seed(0)
#     samples = predictive_distribution.sample(torch.Size([5]))  # sample number
# plt.figure(figsize=(8, 6))
# plt.plot(xs, predictive_mean.detach(), label="mean")
# plt.fill_between(
#     xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
# )
# plt.plot(xs, samples[0, :], alpha=0.5, label="samples")
#
# for i in range(1, samples.shape[0]):
#     plt.plot(xs, samples[i, :], alpha=0.5)
# plt.ylim(-2, 4)
# plt.legend(fontsize=15)
# plt.savefig("image/2.4.4.png")


# set real sample
# declare the GP
# lengthscale = 15     # smoothness degree
# noise = 1e-4
#
# ###########
# index = 27
# train_x = torch.tensor([xs[i] for i in range(1, 145, 5)][:index])
# train_y = torch.tensor([val_y_real[i]/val_y_predict[i] for i in range(29)][:index])
# ###########
#
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = BaseGPModel(train_x, train_y, likelihood)
#
# # fix the hyperparameters
# model.covar_module.lengthscale = lengthscale
# model.likelihood.noise = noise
# model.mean_module.constant = 1
#
# model.eval()
# likelihood.eval()
#
# # draw
# with torch.no_grad():
#     predictive_distribution = likelihood(model(xs))
#     predictive_mean = predictive_distribution.mean
#     predictive_lower, predictive_upper = predictive_distribution.confidence_region()
#
#     torch.manual_seed(0)
#     samples = predictive_distribution.sample(torch.Size([1]))
#
# plt.figure(figsize=(8, 6))
# # plt.plot(xs, ys, label="objective", c="r")
# plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
# plt.plot(xs, predictive_mean, label="mean")
# plt.fill_between(
#     xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95\% CI"
# )
# plt.plot(xs, samples[0, :], alpha=0.5, label="samples")
# for i in range(1, samples.shape[0]):
#     plt.plot(xs, samples[i, :], alpha=0.5)
# plt.legend(fontsize=15)
# plt.savefig("image/2.4.5.png")
#
#
# with torch.no_grad():
#     # Create a tensor from a list of tensors selected by indices
#     input_x = torch.tensor([xs[i] for i in range(1, 145, 5)])
#     given_x_distribution = likelihood(model(input_x))
#     given_x_mean = given_x_distribution.mean
#
# # Print all predicted means
# # print("Predicted means at given x:")
# rate = [mean.item() for mean in given_x_mean]
# # print(rate)
# real_time = val_y_real
# predict_time = [val_y_predict[i] * rate[i] for i in range(29)]
# print(real_time)
# print(predict_time)
# print([mean_squared_error([val_y_predict[i]], [val_y_real[i]]) for i in range(len(val_y_predict))])
#
# mse = mean_squared_error([real_time[index+1]], [predict_time[index+1]])
# print("index: ", index)
# print("mse:", mse)
# 1.show the process of iteration

# 2.calculate all mse in each iteration

# 3.process dataset in batch


# def g(i):
#     i = i.float()
#     exp_term = -1 + (1e-8)
#     linear_term = torch.log(i+1)
#     return torch.where(i <= -1, exp_term, linear_term)
#
#
# def forrester_1d(x):
#     y = torch.where(x <= -1, torch.zeros_like(x), torch.exp(-(torch.log(x + 1)) ** 2))
#     return y.squeeze(-1)
#
#
# def visualize_gp_belief(model, likelihood, num_samples=5):
#     with torch.no_grad():
#         predictive_distribution = likelihood(model(xs))
#         predictive_mean = predictive_distribution.mean
#         predictive_upper, predictive_lower = predictive_distribution.confidence_region()
#
#     plt.figure(figsize=(8, 6))
#
#     plt.plot(xs, ys, label="objective", c="r")
#     plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
#
#     plt.plot(xs, predictive_mean, label="mean")
#     plt.fill_between(
#         xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
#     )
#
#     torch.manual_seed(0)
#     for i in range(num_samples):
#         plt.plot(xs, predictive_distribution.sample(), alpha=0.5, linewidth=2)
#
#     plt.legend(fontsize=15)
#     plt.savefig("image/6.png")
#
#
# xs = torch.linspace(-3, 3, 101).unsqueeze(1)
# ys = forrester_1d(xs)
#
# torch.manual_seed(0)
# train_x = torch.rand(size=(3, 1)) * 6 - 3
# train_y = forrester_1d(train_x)
#
#
# class ScaleGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# # declare the GP
# lengthscale = 1  # 0.3, 1, 3
# outputscale = 3  # 0.3, 1, 3
# noise = 1e-4
#
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ScaleGPModel(train_x, train_y, likelihood)
#
# # fix the hyperparameters
# model.covar_module.base_kernel.lengthscale = lengthscale
# model.covar_module.outputscale = outputscale
# model.likelihood.noise = noise
#
# model.eval()
# likelihood.eval()
#
# visualize_gp_belief(model, likelihood)

