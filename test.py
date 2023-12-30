import numpy as np
import matplotlib.pyplot as plt

def moving_averages(array, window_size=10):
    i = 0
    mv_avg = []

    while i < (len(array) - window_size + 1):
        window_average = round(np.sum(array[i:i+window_size]) / window_size, 2)
        mv_avg.append(window_average)
        i += 1

    return mv_avg

fedavg_acc = np.load('output/fedavg_K_S10_acc.npy')
gl_layer_acc = np.load('output/gl_layer_K_S10_acc.npy')
gl_model_acc = np.load('output/gl_model_K_S10_acc.npy')
ll_layer_acc = np.load('output/ll_layer_K_S10_acc.npy')
ll_model_acc = np.load('output/ll_model_K_S10_acc.npy')

fedavg_final_acc = np.load('output/fedavg_K_S10_final_acc.npy')
gl_layer_final_acc = np.load('output/gl_layer_K_S10_final_acc.npy')
gl_model_final_acc = np.load('output/gl_model_K_S10_final_acc.npy')
ll_layer_final_acc = np.load('output/ll_layer_K_S10_final_acc.npy')
ll_model_final_acc = np.load('output/ll_model_K_S10_final_acc.npy')

window_size = 10
fedavg_acc = moving_averages(fedavg_acc, window_size)
gl_layer_acc = moving_averages(gl_layer_acc, window_size)
gl_model_acc = moving_averages(gl_model_acc, window_size)
ll_layer_acc = moving_averages(ll_layer_acc, window_size)
ll_model_acc = moving_averages(ll_model_acc, window_size)

print(fedavg_final_acc)
print(gl_layer_final_acc)
print(gl_model_final_acc)
print(ll_layer_final_acc)
print(ll_model_final_acc)

max_epoch = len(fedavg_acc)
plt.figure()
plt.plot(range(len(fedavg_acc[0:max_epoch])), fedavg_acc[0:max_epoch], label='fedavg', color='r', linestyle='--')
plt.plot(range(len(gl_layer_acc[0:max_epoch])), gl_layer_acc[0:max_epoch], label='gl layer')
plt.plot(range(len(gl_model_acc[0:max_epoch])), gl_model_acc[0:max_epoch], label='gl model')
plt.plot(range(len(ll_layer_acc[0:max_epoch])), ll_layer_acc[0:max_epoch], label='ll layer')
plt.plot(range(len(ll_model_acc[0:max_epoch])), ll_model_acc[0:max_epoch], label='ll model')
plt.legend()
plt.show()

