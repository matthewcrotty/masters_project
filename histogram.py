import os
import numpy as np
import matplotlib.pyplot as plt


def parse_data():
    bin1_t = 0.9404
    bin2_t = 0.9622
    bin1_idx, bin2_idx, bin3_idx = [], [], []
    model_acc1, model_acc2 = [], []
    cifar10_outputs, cifar10_1_outputs = [], [] 
    cifar10_labels, cifar10_1_labels = [], []
    cifar10_preds, cifar10_1_preds = [], []

    for root, dirs, files in os.walk('./model_outputs'):
        for file in files:
            with np.load(root + '/' +file) as data:
                if "cifar10-1" in file:
                    cifar10_1_outputs.append(np.argmax(data['preds'], axis=1))
                    cifar10_1_labels = data['labels']
                    model_acc1.append(float(data['acc']))
                    cifar10_1_preds.append(data['preds'])
                else:
                    cifar10_outputs.append(np.argmax(data['preds'], axis=1))
                    cifar10_labels = data['labels']
                    model_acc2.append(float(data['acc']))
                    cifar10_preds.append(data['preds'])
                    if float(data['acc']) < bin1_t:
                        bin1_idx.append(len(model_acc2)-1)
                    elif float(data['acc']) < bin2_t:
                        bin2_idx.append(len(model_acc2)-1)
                    else:
                        bin3_idx.append(len(model_acc2)-1)
    
    return cifar10_outputs, cifar10_1_outputs, cifar10_labels, cifar10_1_labels, bin1_idx, bin2_idx, bin3_idx, cifar10_preds, cifar10_1_preds


def bin_norm_unique_output(cifar10_1_outputs_o, cifar10_outputs_o, cifar10_1_labels, cifar10_labels, bin1_idx, bin2_idx, bin3_idx):
    count = 1
    for bin_idx in [bin1_idx, bin2_idx, bin3_idx]:
        # split into bins
        num_models = len(bin_idx)
        cifar10_1_outputs = [cifar10_1_outputs_o[i] for i in bin_idx]
        cifar10_outputs = [cifar10_outputs_o[i] for i in bin_idx]

        # n_points x n_models
        cifar10_outputs = np.transpose(np.array(cifar10_outputs, dtype=np.int64))
        cifar10_1_outputs = np.transpose(np.array(cifar10_1_outputs, dtype=np.int64))
        

        output_numbers_10_1, output_numbers_10 = np.zeros(10), np.zeros(10)

        for image in cifar10_1_outputs:
            output_numbers_10_1[len(list(set(image)))] += 1

        for image in cifar10_outputs:
            output_numbers_10[len(list(set(image)))] += 1

        output_numbers_10_1 /= len(cifar10_1_labels)
        output_numbers_10 /= len(cifar10_labels)

        plt.bar(np.arange(10), output_numbers_10_1, 0.3, label='CIFAR10.1')
        plt.bar(np.arange(10)+0.3, output_numbers_10, 0.3, label='CIFAR10')
        plt.xticks([i for i in range(10)], [i for i in range(10)])
        plt.title('Number of Unique Incorrect Predictions')
        plt.xlabel("Unique Image Class Predictions")
        plt.ylabel("Number of Images")
        plt.legend(loc='upper right')
        plt.savefig('./images/bin'+ str(count) +'_datasets_norm_unique.png')
        plt.clf() 
        count += 1

def close_distant_split(cifar10_1_preds, name, cifar10_1_outputs, bin1_idx, bin2_idx, bin3_idx):
    # close set -> max, mode of prediction vectors between model i, j within some epsilon
    # distant set -> else
    close, distant = [], []
    epsilon = 0.5
    mode_threshold = 1
    num_models = len(cifar10_1_preds)

    #cifar10_outputs = np.transpose(np.array(cifar10_outputs, dtype=np.int64))
    cifar10_1_outputs = np.transpose(np.array(cifar10_1_outputs, dtype=np.int64))
    
    #cifar10_images = np.transpose(np.array(cifar10_preds, dtype=np.int64), (1, 0, 2))
    cifar10_1_images = np.transpose(np.array(cifar10_1_preds, dtype=np.int64), (1, 0, 2))

    # 2000 x num_models X num_classes
    print(cifar10_1_images.shape, num_models)

    distances, uniques = [], []
    avg_distances = []
    max2, avg2 = [], []
    for im in range(len(cifar10_1_images)):
        max_distance = 0
        avg_distance = 0
        max2_dist = 0
        avg2_dist = 0
        for i in range(num_models):
            for j in range(i, num_models):
                distance = np.linalg.norm(cifar10_1_images[im][i] - cifar10_1_images[im][j])
                max_distance = max(distance, max_distance)
                avg_distance += distance

            temp = cifar10_1_images[im][i]
            temp.sort()
            max2_dist = max(max2_dist, abs(temp[-1] - temp[-2]))
            avg2_dist += abs(temp[-1] - temp[-2])
                    
                        
        max2.append(max2_dist)
        avg2.append(avg2_dist/i)
        distances.append(max_distance)
        avg_distances.append(avg_distance/(i**2))
        uniques.append(len(list(set(cifar10_1_outputs[im]))))

    color = 'blue' if name == "Cifar10" else 'orange'
    plt.scatter(distances, uniques, color=color)
    plt.title('Max Distance vs Unique ' + name)
    plt.savefig("./images/max_norm" + name + ".png")
    plt.clf()

    plt.scatter(avg_distances, uniques, color=color)
    plt.title('Average Distance vs Unique ' + name)
    plt.savefig("./images/avg_norm" + name + ".png")
    plt.clf()

    plt.scatter(max2, uniques, color=color)
    plt.title('Max Distance 2 vs Unique ' + name)
    plt.savefig("./images/max2_norm" + name + ".png")
    plt.clf()

    plt.scatter(avg2, uniques, color=color)
    plt.title('Average Distance 2 vs Unique ' + name)
    plt.savefig("./images/avg2_norm" + name + ".png")
    plt.clf()

                


    # return close, distant



# print(model_acc1, model_acc2)
# model_acc2.sort()
# print(model_acc2[:len(model_acc2)//3], model_acc2[len(model_acc2)//3:2*len(model_acc2)//3], model_acc2[2*len(model_acc2)//3:])


# plt.bar(np.arange(10), output_numbers_10_1, 0.3, label='CIFAR10.1')
# plt.bar(np.arange(10)+0.3, output_numbers_10, 0.3, label='CIFAR10')
# plt.xticks([i for i in range(10)], [i for i in range(10)])
# plt.title('Number of Unique Incorrect Predictions')
# plt.xlabel("Unique Image Class Predictions")
# plt.ylabel("Number of Images")
# plt.legend(loc='upper right')
# plt.savefig('./images/datasets_unique.png')
# plt.clf() 







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


cifar10_outputs, cifar10_1_outputs, cifar10_labels, cifar10_1_labels, bin1_idx, bin2_idx, bin3_idx, cifar10_preds, cifar10_1_preds = parse_data()
# bin_norm_unique_output(cifar10_1_outputs, cifar10_outputs, cifar10_1_labels, cifar10_labels, bin1_idx, bin2_idx, bin3_idx)
close_distant_split(cifar10_1_preds, "Cifar10-1", cifar10_1_outputs, bin1_idx, bin2_idx, bin3_idx)
close_distant_split(cifar10_preds, "Cifar10", cifar10_outputs, bin1_idx, bin2_idx, bin3_idx)