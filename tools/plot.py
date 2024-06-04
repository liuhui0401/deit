import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 生成 logistic 分布的样本
    u_logistic = torch.rand(100000)
    t_logistic = torch.rand(100000)

    t0 = 0.9
    t1 = 1.1
    # 生成正态分布的样本
    u_normal = torch.normal(mean=0., std=1., size=(100000,))
    t_normal = 1 / (1 + torch.exp(-u_normal)) * (t1 - t0) + t0
    t_normal = torch.minimum(t_normal, torch.ones_like(t_normal))

    # 绘制 logistic 分布的直方图
    plt.hist(t_logistic.numpy(), bins=30, alpha=0.5, label='Logistic Distribution')

    # 绘制正态分布的直方图
    plt.hist(t_normal.numpy(), bins=30, alpha=0.5, label='Normal Distribution')

    plt.xlabel('t Value')
    plt.ylabel('Frequency')
    plt.title('Sampled Data')
    plt.legend()
    plt.grid(True)
    plt.show()
