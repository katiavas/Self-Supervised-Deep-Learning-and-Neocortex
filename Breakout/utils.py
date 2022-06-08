import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    sd = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        # running_avg[i] = scores[i]
    plt.plot(x, running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)


def plot_learning_curve_with_shaded_error(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    std = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        std[i] = np.std(scores[max(0, i-10):(i+1)])
        print(std[i])
        # running_avg[i] = scores[i]
    print(std)
    # print(running_avg - std)
    plt.plot(x, running_avg)
    plt.fill_between(x, running_avg - std, running_avg + std,
                     color='blue', alpha=0.2)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)

'''
def plot_intrinsic_reward(x, intrinsic_reward, figure_file):
    intrinsic = np.zeros(len(intrinsic_reward))
    for i in range(len(intrinsic)):
        # intrinsic[i] = np.mean(intrinsic_reward[max(0, i-100):(i+1)])
        intrinsic[i] = intrinsic_reward[i]
    plt.plot(x, intrinsic)
    plt.xlabel("Number of episodes")
    plt.ylabel("Intrinsic reward")
    plt.title('Plot of intrinsic reward when ICM is on')
    plt.savefig(figure_file)

def plot_learning_curve_with_shaded_error(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    std = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        std[i] = np.std(scores[max(0, i-100):(i+1)])
        # running_avg[i] = scores[i]
    # print(std)
    # print(running_avg - std)
    plt.plot(x, running_avg)
    plt.fill_between(x, running_avg - std, running_avg + std,
                     color='blue', alpha=0.2)
    plt.xlabel("Number of episodes")
    plt.ylabel("Extrinsic reward")
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)'''