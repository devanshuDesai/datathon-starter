from model import *
from data_loader import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import warnings as wn
import torch
wn.filterwarnings('ignore')

batch_size_train = 60
batch_size_val = 30
batch_size_test = 30

# Datasets And Augmentations
augment = tf.Compose([tf.Resize(224),
                      tf.RandomCrop(224),
                      tf.RandomAffine(degrees=15, scale=(0.7,1.3),
                                      shear=5),
                      tf.ColorJitter(0.5,0.2,0.1,0.1),
                      tf.ToTensor(),
                      tf.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
train_dataset = KaggleDataset('train', augment)
val_dataset = KaggleDataset('val')
test_dataset = KaggleDataset('test')

# Get Dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          num_workers=4,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size_val,
                          num_workers=4,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size_test,
                          num_workers=4,
                          shuffle=False)

writer = SummaryWriter('runs/exp3')

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def train():
    train_epoch_loss = []
    train_epoch_acc = []
    val_epoch_loss = []
    val_epoch_acc = []
    best_model = './checkpoints/best_model_2.ct'

    best_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        scheduler.step()
        train_loss_per_iter = []
        train_acc_per_iter = []
        ts = time.time()
        for iter, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                labels = Y.long().cuda()
            else:
                inputs, labels = X, Y.long()
            outputs = kgm(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # save loss per iteration
            train_loss_per_iter.append(loss.item())
            t_acc = (torch.argmax(outputs,1)==labels).float().mean().item()
            train_acc_per_iter.append(t_acc)

        (print("Finish epoch {}, time elapsed {}, train acc {}".format(epoch,
               time.time() - ts, np.mean(train_acc_per_iter))))

        # calculate validation loss and accuracy
        val_loss, val_acc = val()
        print("Val loss {}, Val Acc {}".format(val_loss, val_acc))
        # Early Stopping
        if loss < best_loss:
            best_loss = loss
            # TODO: Consider switching to state dict instead
            torch.save(kgm, best_model)
        train_epoch_loss.append(np.mean(train_loss_per_iter))
        train_epoch_acc.append(np.mean(train_acc_per_iter))
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)

        writer.add_scalar('Loss/train', np.mean(train_loss_per_iter), epoch)
        writer.add_scalar('Accuracy/train', np.mean(train_acc_per_iter), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)


def val():
    batch_loss = []
    batch_acc = []
    for iter, (X, Y) in tqdm(enumerate(val_loader), total=len(val_loader)):
        '''
        y -> Labels (Used for pix acc and IOU)
        tar -> One-hot encoded labels (used for loss)
        '''
        if use_gpu:
            inputs = X.cuda()
            labels = Y.long().cuda()
        else:
            inputs, labels = X, Y.long()
        outputs = kgm(inputs)
        loss = criterion(outputs, labels)
        batch_loss.append(loss.item())
        batch_acc.append((torch.argmax(outputs,1)==labels).float().mean().item())

    return np.mean(batch_loss), np.mean(batch_acc)


def test():
    best_model = torch.load('./checkpoints/best_model_1.ct')
    all_fnames = []
    all_preds = []

    for iter, (fnames, imgs) in tqdm(enumerate(test_loader), total=len(test_loader)):
        if use_gpu:
            inputs = imgs.cuda()
        else:
            inputs = imgs
        outputs = best_model(inputs)
        preds = torch.argmax(outputs, 1)
        all_fnames.extend(list(fnames))
        all_preds.extend(preds.cpu().numpy().tolist())

    with open('submission.csv', 'w') as f:
        f.write('fname,breedID\n')       # header
        for row in zip(all_fnames, all_preds):
            f.write(','.join(map(str, row))+'\n')


if __name__ == "__main__":
    # Define model parameters
    epochs    = 200
    criterion = nn.CrossEntropyLoss()
    kgm = KaggleModel(37)
    kgm.apply(init_weights)
    params = kgm.parameters()
    optimizer = optim.Adam(params, 1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        kgm = kgm.cuda()

#     train()
    test()

