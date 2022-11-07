import torch
import sys
import numpy as np
import torch.distributed as dist
import time
import tqdm
import torch.nn.functional as F

sys.path.append("pytorch_image_classification/")
from pytorch_image_classification import (
    get_default_config,
    create_model,
    create_loss,
    create_transform,
    create_dataloader,
    update_config
)

from pytorch_image_classification.utils import (
    create_logger,
    AverageMeter,
    get_rank,
)


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    with torch.no_grad():
        for data, targets in test_loader:
            
            data = data.type(torch.float32)
            targets = targets.type(torch.int64)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            
            correct_ = preds.eq(targets).sum().item()
            #print(preds, targets, correct_)
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy


class Cifar10_1(torch.utils.data.Dataset):
    def __init__(self, transform):
        super(Cifar10_1, self).__init__()
        self.transform=transform
        self._x = np.load("CIFAR-10.1/datasets/cifar10.1_v6_data.npy")
        self._y = np.load("CIFAR-10.1/datasets/cifar10.1_v6_labels.npy")

    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return self.transform(x), y

def get_cifar_10_1_loader(config):
    
    transform = create_transform(config, is_train=False)
    cifar_10_1_dataset = Cifar10_1(transform)

    test_loader = torch.utils.data.DataLoader(cifar_10_1_dataset, 
                batch_size=16,
                num_workers=config.test.dataloader.num_workers,
                sampler=None,
                shuffle=False,
                drop_last=False,
                pin_memory=config.test.dataloader.pin_memory)
    
    return test_loader


# sys.path.insert(0, '..')

if torch.cuda.is_available():
    device = 'cuda'

with open("model_list.txt") as file:
    models = [line.rstrip() for line in file]

config = get_default_config()
cifar10_1_loader = get_cifar_10_1_loader(config)
update_config(config)
cifar10_loader = create_dataloader(config, is_train=False)
logger = create_logger(name=__name__, distributed_rank=get_rank())

for model_name in models:

    config = get_default_config()
    config.merge_from_file("config_files/" + model_name + ".yaml")
    model = create_model(config)
    checkpoint = torch.load("trained_models/" + model_name + ".pth")
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print("Loaded model from " + model_name + ".pth")
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc = evaluate(config, model, cifar10_1_loader, test_loss, logger)

    np.savez("model_outputs/" + model_name + "_cifar10-1.npz",
                preds=preds,
                probs=probs,
                labels=labels,
                loss=loss,
                acc=acc)

    preds, probs, labels, loss, acc = evaluate(config, model, cifar10_loader, test_loss, logger)

    np.savez("model_outputs/" + model_name + "_cifar10.npz",
                preds=preds,
                probs=probs,
                labels=labels,
                loss=loss,
                acc=acc)