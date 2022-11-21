import os
import numpy as np
import matplotlib.pyplot as plt


gradient = {17: '#ff0000',
    16: '#f30f05',
    15: '#e71e0a',
    14: '#da2d0e',
    13: '#ce3c13',
    12: '#c24b18',
    11: '#b65a1d',
    10: '#aa6922',
    9: '#9e7827',
    8: '#91872b',
    7: '#859630',
    6: '#79a535',
    5: '#6db43a',
    4: '#61c33f',
    3: '#55d244',
    2: '#48e148',
    1: '#3cf04d',
    0: '#30ff52'}

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
                    cifar10_1_preds.append(data['probs'])
                else:
                    cifar10_outputs.append(np.argmax(data['preds'], axis=1))
                    cifar10_labels = data['labels']
                    model_acc2.append(float(data['acc']))
                    cifar10_preds.append(data['probs'])
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

def close_distant_split(preds, name, outputs, labels, misclassifications, bin_idx=None, bin_label=None):
    # close set -> max, mode of prediction vectors between model i, j within some epsilon
    # distant set -> else
    close, distant = [], []
    epsilon = 0.5
    mode_threshold = 1
    num_models = len(preds)

    outputs = np.transpose(np.array(outputs, dtype=np.int64))
    images = np.transpose(np.array(preds), (1, 0, 2))

    # 2000 x num_models X num_classes
    print(images.shape, num_models)

    distances, uniques = [], []
    avg_distances = []
    max2, avg2 = [], []
    all_avg2 = []
    all_uniques = []
    # colors = []
    accuracies = []
    for im in range(len(images)):
        max_distance = 0
        avg_distance = 0
        max2_dist = 0
        avg2_dist = 0
        num_comps = 0
        for i in range(num_models):
            #print(cifar10_1_images[im][i], np.sum(cifar10_1_images[im][i]))
            if bin_idx is not None and i not in bin_idx:
                continue
            for j in range(i, num_models):
                if bin_idx is not None and j not in bin_idx:
                    continue
                num_comps += 1
                distance = np.linalg.norm(images[im][i] - images[im][j])
                max_distance = max(distance, max_distance)
                avg_distance += distance

            temp = images[im][i]
            temp.sort()
            max2_dist = max(max2_dist, abs(temp[-1] - temp[-2]))
            avg2_dist += abs(temp[-1] - temp[-2])
            all_avg2.append(abs(temp[-1] - temp[-2]))
            #print(cifar10_1_images[im][i], np.sum(cifar10_1_images[im][i]), abs(temp[-1] - temp[-2]))
            all_uniques.append(len(list(set(outputs[im]))))
            # if outputs[im][i] == labels[im]:
            #     # colors.append('g')

            # else:
            #     # colors.append('r')
        accuracies.append(list(outputs[im]).count(labels[im])/num_models)
                    
        max2.append(max2_dist)
        if bin_idx is not None:
            avg2_dist /= len(bin_idx)
        else:
            avg2_dist /= num_models
        avg2.append(avg2_dist)
        distances.append(max_distance)
        avg_distances.append(avg_distance/num_comps)
        uniques.append(len(list(set(outputs[im]))))

    if bin_idx is not None:
        name += "_bin" + str(bin_label)

    color = 'blue' if name == "Cifar10" else 'orange'
    color2 = 'red'
    # plt.scatter(distances, uniques, color=color)
    for i in range(len(distances)):
        plt.plot(distances[i], uniques[i], '.', color=gradient[num_models - misclassifications[i]])
    plt.plot(1, 7, 'b,')
    plt.title('Max Distance vs Unique ' + name)
    plt.savefig("./images/max_norm_" + name + ".png")
    plt.clf()

    for i in range(len(avg_distances)):
        plt.plot(avg_distances[i], uniques[i], '.', color=gradient[num_models - misclassifications[i]])
    # plt.scatter(avg_distances, uniques, color=color)
    plt.plot(1, 7, 'b,')
    plt.title('Average Distance vs Unique ' + name)
    plt.savefig("./images/avg_norm_" + name + ".png")
    plt.clf()

    for i in range(len(max2)):
        plt.plot(max2[i], uniques[i], '.', color=gradient[num_models - misclassifications[i]])
    #plt.scatter(max2, uniques, color=color)
    plt.plot(1, 7, 'b,')
    plt.title('Max Image Top 2 Distance vs Unique ' + name)
    plt.savefig("./images/max_top2_" + name + ".png")
    plt.clf()
    
    for i in range(len(avg2)):
        plt.plot(avg2[i], uniques[i], '|', color=gradient[num_models - misclassifications[i]])
    plt.plot(1, 7, 'b,')
    #plt.scatter(avg2, uniques, color=color)
    plt.title('Average Image Top 2 vs Unique ' + name)
    plt.savefig("./images/avg_top2_" + name + ".png")
    plt.clf()

    # for i in range(len(all_avg2)):
    #     if colors[i] == 'g':
    #         plt.plot(all_avg2[i], all_uniques[i], 'g|')
    #     else:
    #         plt.plot(all_avg2[i], all_uniques[i], 'r|')
    # plt.plot(1, 7, 'b,')
    # plt.title("Top 2 Distance for all Images, all Models " + name)
    # plt.savefig("./images/Color_all_top2_" + name + ".png")
    # plt.clf()
                
    # return close, distant

    # TODO: plot num_images_accepted vs accuracy.

    avg_norm_threshold = np.arange(0.01, 0.85, 0.01)
    avg_norm_accuracies = np.zeros(len(avg_norm_threshold))
    avg_norm_rejects = np.zeros(len(avg_norm_threshold))
    avg_norm_r = np.zeros(len(avg_norm_threshold))
    avg_norm_a = np.zeros(len(avg_norm_threshold))
    avg_top2_threshold = np.arange(0.01, 1.0, 0.01)
    avg_top2_accuracies = np.zeros(len(avg_top2_threshold))
    avg_top2_rejects = np.zeros(len(avg_top2_threshold))
    avg_top2_r = np.zeros(len(avg_top2_threshold))
    avg_top2_a = np.zeros(len(avg_top2_threshold))
    max_norm_threshold = np.arange(0.01, 1.42, 0.01)
    max_norm_accuracies = np.zeros(len(max_norm_threshold))
    max_norm_rejects = np.zeros(len(max_norm_threshold))
    max_norm_r = np.zeros(len(max_norm_threshold))
    max_norm_a = np.zeros(len(max_norm_threshold))
    max_top2_threshold = np.arange(0.01, 1.00, 0.01)
    max_top2_accuracies = np.zeros(len(max_top2_threshold))
    max_top2_rejects = np.zeros(len(max_top2_threshold))
    max_top2_r = np.zeros(len(max_top2_threshold))
    max_top2_a = np.zeros(len(max_top2_threshold))

    num_images = len(images)
    
    for i, t in enumerate(avg_norm_threshold):
        num_images_accepted = 0
        num_images_rejected = 0
        for im in range(len(images)):
            if avg_distances[im] >= t:
                avg_norm_accuracies[i] += accuracies[im]
                num_images_accepted += 1
            else:
                avg_norm_rejects[i] += accuracies[im]
                num_images_rejected += 1
        if num_images_accepted > 0:
            avg_norm_accuracies[i] /= num_images_accepted
        if num_images_rejected > 0:
            avg_norm_rejects[i] /= num_images_rejected
        avg_norm_r[i] = num_images_accepted / num_images
        avg_norm_a[i] = num_images_rejected / num_images

    for i, t in enumerate(avg_top2_threshold):
        num_images_accepted = 0
        num_images_rejected = 0
        for im in range(len(images)):
            if avg2[im] >= t:
                avg_top2_accuracies[i] += accuracies[im]
                num_images_accepted += 1
            else:
                avg_top2_rejects[i] += accuracies[im]
                num_images_rejected += 1
        if num_images_accepted > 0:
            avg_top2_accuracies[i] /= num_images_accepted
        if num_images_rejected > 0:
            avg_top2_rejects[i] /= num_images_rejected
        avg_top2_r[i] = num_images_accepted / num_images
        avg_top2_a[i] = num_images_rejected / num_images

    for i, t in enumerate(max_norm_threshold):
        num_images_accepted = 0
        num_images_rejected = 0
        for im in range(len(images)):
            if distances[im] >= t:
                max_norm_accuracies[i] += accuracies[im]
                num_images_accepted += 1
            else:
                max_norm_rejects[i] += accuracies[im]
                num_images_rejected += 1
        if num_images_accepted > 0:
            max_norm_accuracies[i] /= num_images_accepted
        if num_images_rejected > 0:
            max_norm_rejects[i] /= num_images_rejected
        max_norm_r[i] = num_images_accepted / num_images
        max_norm_a[i] = num_images_rejected / num_images

    for i, t in enumerate(max_top2_threshold):
        num_images_accepted = 0
        num_images_rejected = 0
        for im in range(len(images)):
            if max2[im] >= t:
                max_top2_accuracies[i] += accuracies[im]
                num_images_accepted += 1
            else:
                max_top2_rejects[i] += accuracies[im]
                num_images_rejected += 1
        if num_images_accepted > 0:
            max_top2_accuracies[i] /= num_images_accepted
        if num_images_rejected > 0:
            max_top2_rejects[i] /= num_images_rejected
        max_top2_r[i] = num_images_accepted / num_images
        max_top2_a[i] = num_images_rejected / num_images
        
    plt.plot(avg_norm_threshold, avg_norm_accuracies, '.', color=color)
    plt.plot(avg_norm_threshold, avg_norm_rejects, '.', color=color2)
    plt.xlabel("Distance Threshold (avg distance)")
    plt.ylabel("Accuracy")
    plt.savefig("./images/pipeline_avg_dist_" + name + ".png")
    plt.clf()

    plt.plot(avg_top2_threshold, avg_top2_accuracies, '.', color=color)
    plt.plot(avg_top2_threshold, avg_top2_rejects, '.', color=color2)
    plt.xlabel("Distance Threshold (avg top2 distance)")
    plt.ylabel("Accuracy")
    plt.savefig("./images/pipeline_avg_top2_" + name + ".png")
    plt.clf()

    plt.plot(max_norm_threshold, max_norm_accuracies, '.', color=color)
    plt.plot(max_norm_threshold, max_norm_rejects, '.', color=color2)
    plt.xlabel("Distance Threshold (max distance)")
    plt.ylabel("Accuracy")
    plt.savefig("./images/pipeline_max_dist_" + name + ".png")
    plt.clf()

    plt.plot(max_top2_threshold, max_top2_accuracies, '.', color=color)
    plt.plot(max_top2_threshold, max_top2_rejects, '.', color=color2)
    plt.xlabel("Distance Threshold (max top2 distance)")
    plt.ylabel("Accuracy")
    plt.savefig("./images/pipeline_max_top2_" + name + ".png")
    plt.clf()




    plt.plot(avg_norm_threshold, avg_norm_a, '.', color=color)
    plt.plot(avg_norm_threshold, avg_norm_r, '.', color=color2)
    plt.ylabel("Reject/Accept Rate")
    plt.xlabel("Accuracy of images")
    plt.savefig("./images/a_r_avg_norm_" + name + ".png")
    plt.clf()

    plt.plot(avg_top2_threshold, avg_top2_a, '.', color=color)
    plt.plot(avg_top2_threshold, avg_top2_r, '.', color=color2)
    plt.ylabel("Reject/Accept Rate")
    plt.xlabel("Accuracy of images")
    plt.savefig("./images/a_r_avg_top2_" + name + ".png")
    plt.clf()

    plt.plot(max_norm_threshold, max_norm_a, '.', color=color)
    plt.plot(max_norm_threshold, max_norm_r, '.', color=color2)
    plt.ylabel("Reject/Accept Rate")
    plt.xlabel("Accuracy of images")
    plt.savefig("./images/a_r_max_norm_" + name + ".png")
    plt.clf()

    plt.plot(max_top2_threshold, max_top2_a, '.', color=color)
    plt.plot(max_top2_threshold, max_top2_r, '.', color=color2)
    plt.ylabel("Reject/Accept Rate")
    plt.xlabel("Accuracy of images")
    plt.savefig("./images/a_r_max_top2_" + name + ".png")
    plt.clf()
# print(model_acc1, model_acc2)
# model_acc2.sort()
# print(model_acc2[:len(model_acc2)//3], model_acc2[len(model_acc2)//3:2*len(model_acc2)//3], model_acc2[2*len(model_acc2)//3:])


def misclassifications(cifar10_1_outputs, cifar10_outputs):
    num_models = 18
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

    cifar10_1_outputs = np.transpose(np.array(cifar10_1_outputs, dtype=np.int64))
    misclassification_counts = np.zeros(num_models+1, dtype=float)
    incorrect_label_counts = []
    mis = []
    for i in range(10):
        incorrect_label_counts.append(np.zeros(10, dtype=np.int64))

    for i, label in enumerate(cifar10_1_labels):
        misclassification_counts[np.sum(cifar10_1_outputs[i] == label)] += 1
        mis.append(np.sum(cifar10_1_outputs[i] == label))
        for model in range(len(cifar10_1_outputs[i])):
            if cifar10_1_outputs[i][model] != label:
                incorrect_label_counts[label][cifar10_1_outputs[i][model]] += 1

    # plt.stairs(np.flip(misclassification_counts), fill=True)
    # ticks = [(2*i + 1)/2 for i in range(len(misclassification_counts))]
    # ticklabels = [i for i in range(len(misclassification_counts))]
    # plt.xticks(ticks, ticklabels)
    # plt.xlabel("Number of Models that incorrectly classify image")
    # plt.ylabel("Number of Images")
    # plt.title("CIFAR10.1 Misclassifications")
    # plt.savefig('./images/all_classes_cifar10_1.png')
    # plt.clf()
    cifar10_outputs = np.transpose(np.array(cifar10_outputs, dtype=np.int64))
    misclassification_counts2 = np.zeros(num_models+1, dtype=float)
    mis2 = []
    incorrect_label_counts2 = []
    for i in range(10):
        incorrect_label_counts2.append(np.zeros(10, dtype=np.int64))

    for i, label in enumerate(cifar10_labels):
        misclassification_counts2[np.sum(cifar10_outputs[i] == label)] += 1
        mis2.append(np.sum(cifar10_outputs[i] == label))
        for model in range(len(cifar10_outputs[i])):
            if cifar10_outputs[i][model] != label:
                incorrect_label_counts2[label][cifar10_outputs[i][model]] += 1

    # plt.stairs(np.flip(misclassification_counts2), fill=True, color='orange')
    ticks = [(2*i + 1)/2 for i in range(len(misclassification_counts2))]
    ticklabels = [i for i in range(len(misclassification_counts2))]
    plt.xticks(ticks, ticklabels)
    # plt.xlabel("Number of Models that incorrectly classify image")
    # plt.ylabel("Number of Images")
    # plt.title("CIFAR10 Misclassifications")
    # plt.savefig('./images/all_classes_cifar10.png')
    # plt.clf()
    # misclassification_counts /= len(cifar10_1_labels)
    # misclassification_counts2 /= len(cifar10_labels)

    plt.bar(np.arange(num_models+1), np.flip(misclassification_counts/len(cifar10_1_labels)), 0.3, label='CIFAR10.1')
    plt.bar(np.arange(num_models+1)+0.3, np.flip(misclassification_counts2/len(cifar10_labels)), 0.3, label='CIFAR10')
    plt.ylim((0, 0.2))
    plt.legend(loc='upper right')
    plt.xlabel("Number of Models that incorrectly classify image")
    plt.ylabel("Number of Images")
    plt.savefig('./images/all_classes.png')
    plt.clf()

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
    return mis, mis2

cifar10_outputs, cifar10_1_outputs, cifar10_labels, cifar10_1_labels, bin1_idx, bin2_idx, bin3_idx, cifar10_preds, cifar10_1_preds = parse_data()
bin_norm_unique_output(cifar10_1_outputs, cifar10_outputs, cifar10_1_labels, cifar10_labels, bin1_idx, bin2_idx, bin3_idx)
cifar10_1_mis, cifar10_mis = misclassifications(cifar10_1_outputs, cifar10_outputs)
# count = 1
# for bin_idx in [bin1_idx, bin2_idx, bin3_idx]:
#     close_distant_split(cifar10_1_preds, "Cifar10-1", cifar10_1_outputs, cifar10_1_labels, bin_idx, count)
#     close_distant_split(cifar10_preds, "Cifar10", cifar10_outputs, cifar10_labels, bin_idx, count)
#     count += 1
close_distant_split(cifar10_1_preds, "Cifar10-1", cifar10_1_outputs, cifar10_1_labels, cifar10_1_mis)
close_distant_split(cifar10_preds, "Cifar10", cifar10_outputs, cifar10_labels, cifar10_mis)


# [-6.0571203 -5.200472 -3.5986066 -2.924473 -2.828505 -0.9731233 1.3326535 2.582334 3.8859973 12.898695 ]