import os
import numpy as np
import matplotlib.pyplot as plt

cifar10_outputs, cifar10_1_outputs = [], [] 
cifar10_labels, cifar10_1_labels = [], []
for root, dirs, files in os.walk('../model_outputs'):
    for file in files:
        with np.load(root + '/' +file) as data:
            if "cifar10-1" in file:
                cifar10_1_outputs.append(np.argmax(data['preds'], axis=1))
                cifar10_1_labels = data['labels']
            else:
                cifar10_outputs.append(np.argmax(data['preds'], axis=1))
                cifar10_labels = data['labels']

# n_points x n_models
cifar10_outputs = np.transpose(np.array(cifar10_outputs, dtype=np.int64))
cifar10_1_outputs = np.transpose(np.array(cifar10_1_outputs, dtype=np.int64))
num_models = len(cifar10_1_outputs[0])

output_numbers_10_1, output_numbers_10 = np.zeros(10), np.zeros(10)

for image in cifar10_1_outputs:
    output_numbers_10_1[len(list(set(image)))] += 1

for image in cifar10_outputs:
    output_numbers_10[len(list(set(image)))] += 1

plt.bar(np.arange(10), output_numbers_10_1, 0.3, label='CIFAR10.1')
plt.bar(np.arange(10)+0.3, output_numbers_10, 0.3, label='CIFAR10')
plt.xticks([i for i in range(10)], [i for i in range(10)])
plt.title('Number of Unique Incorrect Predictions')
plt.xlabel("Unique Image Class Predictions")
plt.ylabel("Number of Images")
plt.legend(loc='upper right')
plt.savefig('./images/datasets_unique.png')
plt.clf() 

output_numbers_10_1 /= len(cifar10_1_labels)
output_numbers_10 /= len(cifar10_labels)

plt.bar(np.arange(10), output_numbers_10_1, 0.3, label='CIFAR10.1')
plt.bar(np.arange(10)+0.3, output_numbers_10, 0.3, label='CIFAR10')
plt.xticks([i for i in range(10)], [i for i in range(10)])
plt.title('Number of Unique Incorrect Predictions')
plt.xlabel("Unique Image Class Predictions")
plt.ylabel("Number of Images")
plt.legend(loc='upper right')
plt.savefig('./images/datasets_norm_unique.png')
plt.clf() 





# # For each image, count how many times it is misclassified

# print(len(cifar10_1_outputs))
# print(len(cifar10_outputs))
# misclassification_counts = np.zeros(num_models+1, dtype=np.int64)
# incorrect_label_counts = []
# for i in range(10):
#     incorrect_label_counts.append(np.zeros(10, dtype=np.int64))

# for i, label in enumerate(cifar10_1_labels):
#     misclassification_counts[np.sum(cifar10_1_outputs[i] == label)] += 1
#     for model in range(len(cifar10_1_outputs[i])):
#         if cifar10_1_outputs[i][model] != label:
#             incorrect_label_counts[label][cifar10_1_outputs[i][model]] += 1

# plt.stairs(np.flip(misclassification_counts), fill=True)
# ticks = [(2*i + 1)/2 for i in range(len(misclassification_counts))]
# ticklabels = [i for i in range(len(misclassification_counts))]
# plt.xticks(ticks, ticklabels)
# plt.xlabel("Number of Models that incorrectly classify image")
# plt.ylabel("Number of Images")
# plt.title("CIFAR10.1 Misclassifications")
# plt.savefig('./images/all_classes_cifar10_1.png')
# plt.clf()

# misclassification_counts2 = np.zeros(num_models+1, dtype=np.int64)
# incorrect_label_counts2 = []
# for i in range(10):
#     incorrect_label_counts2.append(np.zeros(10, dtype=np.int64))

# for i, label in enumerate(cifar10_labels):
#     misclassification_counts2[np.sum(cifar10_outputs[i] == label)] += 1
#     for model in range(len(cifar10_outputs[i])):
#         if cifar10_outputs[i][model] != label:
#             incorrect_label_counts2[label][cifar10_outputs[i][model]] += 1

# plt.stairs(np.flip(misclassification_counts2), fill=True, color='orange')
# ticks = [(2*i + 1)/2 for i in range(len(misclassification_counts2))]
# ticklabels = [i for i in range(len(misclassification_counts2))]
# plt.xticks(ticks, ticklabels)
# plt.xlabel("Number of Models that incorrectly classify image")
# plt.ylabel("Number of Images")
# plt.title("CIFAR10 Misclassifications")
# plt.savefig('./images/all_classes_cifar10.png')
# plt.clf()

# plt.bar(np.arange(num_models+1), np.flip(misclassification_counts), 0.3, label='CIFAR10.1')
# plt.bar(np.arange(num_models+1)+0.3, np.flip(misclassification_counts2), 0.3, label='CIFAR10')
# plt.ylim((0, 1000))
# plt.legend(loc='upper right')
# plt.xlabel("Number of Models that incorrectly classify image")
# plt.ylabel("Number of Images")
# plt.savefig('./images/all_classes.png')
# plt.clf()

# for label in range(10):
#     plt.bar(np.arange(10), incorrect_label_counts[label], 0.3, label='CIFAR10.1')
#     plt.bar(np.arange(10)+0.3, incorrect_label_counts2[label], 0.3, label='CIFAR10')
#     plt.xticks([i for i in range(10) if i != label], [i for i in range(10) if i != label ])
#     plt.title('Incorrect Predictions for Class {} '.format(label))
#     plt.xlabel("Predicted Class")
#     plt.ylabel("Number of times predicted")
#     plt.legend(loc='upper right')
#     plt.savefig('./images/class_{}.png'.format(label))
#     plt.clf() 