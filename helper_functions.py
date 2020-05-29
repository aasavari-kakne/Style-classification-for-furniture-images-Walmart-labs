import torch
import os
import random
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

def dot_norm_exp(a,b):
    dot = torch.sum(a * b, dim=1)
    aa = torch.sum((a**2),dim=1)**0.5
    bb = torch.sum((b**2),dim=1)**0.5
    dot_norm = dot/(aa*bb)
    ret = torch.exp(dot_norm)
    return ret

def dot_norm(a,b):
    dot = torch.sum(a * b, dim=1)
    aa = torch.sum((a**2),dim=1)**0.5
    bb = torch.sum((b**2),dim=1)**0.5
    dot_norm = dot/(aa*bb)
    return dot_norm

def dot(a,b):
    dot = torch.sum(a * b, dim=1)
    return dot

def norm_euclidian(a,b):
    aa = (torch.sum((a**2),dim=1)**0.5).unsqueeze(dim=1)
    bb = (torch.sum((b**2),dim=1)**0.5).unsqueeze(dim=1)
    return (torch.sum(((a/aa-b/bb)**2),dim=1)**0.5)

def inspect_model(model):
    param_count = 0
    for param_tensor_str in model.state_dict():
        tensor_size = model.state_dict()[param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size} = {model.state_dict()[param_tensor_str].numel()} params")
        param_count += model.state_dict()[param_tensor_str].numel()

    print(f"Number of parameters: {param_count}")


def get_next_model_folder(prefix, path = ''):

    model_folder = lambda prefix, run_idx: "{}_model_run_{}".format(prefix,run_idx)

    run_idx = 1
    while os.path.isdir(os.path.join(path, model_folder(prefix, run_idx))):
        run_idx += 1

    model_path = os.path.join(path, model_folder(prefix, run_idx))
    print(f"STARTING {prefix} RUN {run_idx}! Storing the models at {model_path}")

    return model_path


def get_random_patches(random_patch_loader, num_random_patches):

        is_data_loader_finished = False

        try:
            img_batch = next(iter(random_patch_loader))['image']
        except StopIteration:
            is_data_loader_finished = True
            # random_patch_loader = DataLoader(dataset_train, num_random_patches, shuffle=True)

        if len(img_batch) < num_random_patches:
            is_data_loader_finished = True

        patches = []

        for i in range(num_random_patches):
            x = random.randint(0,6)
            y = random.randint(0,6)

            patches.append(img_batch[i:i+1,:,x*32:x*32+64,y*32:y*32+64])

            # plt.imshow(np.transpose(patches[-1][0],(1,2,0)))
            # plt.show()

        patches_tensor = torch.cat(patches, dim=0)

        return dict(
            patches_tensor = patches_tensor,
            is_data_loader_finished = is_data_loader_finished)


def get_patch_tensor_from_image_batch(img_batch):

    # Input of the function is a tensor [B, C, H, W]
    # Output of the functions is a tensor [B * 49, C, 64, 64]

    patch_batch = None
    all_patches_list = []

    for y_patch in range(7):
        for x_patch in range(7):

            y1 = y_patch * 32
            y2 = y1 + 64

            x1 = x_patch * 32
            x2 = x1 + 64

            img_patches = img_batch[:,:,y1:y2,x1:x2] # Batch(img_idx in batch), channels xrange, yrange
            img_patches = img_patches.unsqueeze(dim=1)
            all_patches_list.append(img_patches)

            # print(patch_batch.shape)
    all_patches_tensor = torch.cat(all_patches_list, dim=1)

    patches_per_image = []
    for b in range(all_patches_tensor.shape[0]):
        patches_per_image.append(all_patches_tensor[b])

    patch_batch = torch.cat(patches_per_image, dim = 0)
    return patch_batch


def write_csv_stats(csv_path, stats_dict):

    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(stats_dict.keys())

    for key, value in stats_dict.items():
        if isinstance(value, float):
            precision = 0.001
            stats_dict[key] =  ((value / precision ) // 1.0 ) * precision

    with open(csv_path, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(stats_dict.values())

def plot_confusion_matrix(cm, classes,name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(name)

def plot_acc_loss_classificator(data_csv,data_csv_train,data_path='./'):
  df = pd.read_csv(data_csv)
  epoch_ls=[int(val) for val in list(df['epoch'])]
  train_acc_ls=[float(val) for val in list(df['train_acc'])]
  train_loss_ls=[float(val) for val in list(df['train_loss'])]
  test_acc_ls=[float(val) for val in list(df['test_acc'])]
  test_loss_ls=[float(val) for val in list(df['test_loss'])]

  #plt.figure(figsize=(16,10))
  plt.plot(epoch_ls,train_acc_ls)
  plt.plot(epoch_ls,test_acc_ls)
  plt.title('classificator accuracy (epoch)')
  plt.ylabel('accuracy')
  plt.xlabel("epoch")
  plt.legend(['train acc', 'test acc'], loc='upper left')
  plt.show()
  # plt.savefig(data_path+'classificator_acc.png')
  plt.plot(epoch_ls,train_loss_ls)
  plt.plot(epoch_ls,test_loss_ls)
  plt.title('classificator loss (epoch)')
  plt.ylabel('loss')
  plt.xlabel("epoch")
  plt.legend(['train loss', 'test loss'], loc='upper left')
  #plt.savefig(data_path+'classificator_loss.png')
  plt.show()

  df_train= pd.read_csv(data_csv_train)
  train_loss_ls = [int(val) for val in list(df_train['batch_train_loss'])]
  train_acc_ls = [int(val) for val in list(df_train['batch_train_accuracy'])]

  interval=len(train_loss_ls)//100
  plt.plot(train_loss_ls[::interval])
  plt.title(f'classificator training loss (every {interval} iteration)')
  plt.ylabel('train loss')
  plt.xlabel(f"every {interval} iteration")
  plt.show()

  plt.plot(train_acc_ls[::interval])
  plt.title(f'classificator training accuracy (every {interval} iteration)')
  plt.ylabel('train acc')
  plt.xlabel(f"every {interval} iteration")
  plt.show()





def plot_loss_for_cpc(data_loss_csv,data_path='./'):
  df = pd.read_csv(data_loss_csv)
  mean_batch_loss=[float(val[:-1].split("(")[1]) for val in list(df['batch_loss'])]
  sum_batch_loss=[float(val[:-1].split("(")[1]) for val in list(df['sum_batch_loss'])]

  plt.plot(mean_batch_loss)
  plt.title('mean batch loss of cpc (iteration)')
  plt.ylabel('mean batch loss')
  plt.xlabel("iteration")
  plt.show()
  # plt.savefig(os.path.join(data_path,'cpc_mean_batch_loss.jpg'))
  plt.plot(sum_batch_loss)
  plt.title('sum batch loss of cpc (iteration)')
  plt.ylabel('sum batch loss')
  plt.xlabel("iteration")
  plt.show()
  #plt.savefig(os.path.join(data_path,'cpc_sum_batch_loss.jpg'))

