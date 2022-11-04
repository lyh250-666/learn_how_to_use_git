
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import os
from util.my_dataset import my_dataset
import h5py
import argparse
import time
from sklearn.metrics import roc_curve,auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics

#新增一行注释

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir',  type=str, default='/work/lt187/yhli/stDNN_v2/output')
parser.add_argument('--data_dir', type=str, default='/work/lt187/yhli/stDNN_v2/data')
parser.add_argument('--data_filename', type=str, default='MCAD_cut_catg2.hdf5') # MCAD_pad_catg2.hdf5
#parser.add_argument('--data_filename', type=str, default='ABIDE_pad.hdf5')  #ABIDE_cut.hdf5
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--total_epochs', default=200, type=int)
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
parser.add_argument('--n_fold', default=10, type=int)
parser.add_argument('--group_nfold', default=7, type=int)
parser.add_argument('--n_category', default=2, type=int)
parser.add_argument('--n_channal', default=392, type=int)
parser.add_argument('--avgpool_kernelsize', default=43, type=int)

args = parser.parse_args()  # for .py
#args = parser.parse_args("") # for .ipynb

print(args)


# In[15]:


# load data
data = h5py.File(os.path.join(args.data_dir, args.data_filename), 'r')

list(data.keys())     # ['X', 'age', 'sex', 'site', 'y']
X = data["X"]
y = data["y"]
site = data["site"]
sex = data["sex"]
age = data["age"]
print("X: ", X.shape)  # (809, 263, 169)
print("y: ", y.shape)  # <HDF5 dataset "y": shape (809, 1), type "<i4">
print("site: ", site.shape)
print("sex: ", sex.shape)
print("age: ", age.shape)


dataset = my_dataset(X, y,site, sex, age)



args.n_channal = X.shape[1]  # 263 for MCAD, 392 for ABIDE


class init_net(nn.Module):
    def __init__(self, n_channal):
        super(init_net, self).__init__()
        self.conv1d_1 = nn.Conv1d(n_channal, 256, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)  #[B, 256, 86]
        x = self.conv1d_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)  #[B, 512, 43]
        return x.shape[2]

def getAvgpoolKernelSize(s, n_channal):
    # feed the data to the network to get the feature dim after flatten layer
    mynet = init_net(n_channal)
    s = mynet(x_syn)
    return s

class init_net2(nn.Module):
    def __init__(self, n_channal, avgpool_kernelsize):
        super(init_net2, self).__init__()
        self.conv1d_1 = nn.Conv1d(n_channal, 256, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size = avgpool_kernelsize)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        x = self.conv1d_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x.shape[1]

def getFlattenDim(s, n_channal, avgpool_kernelsize):
    # feed the data to the network to get the feature dim after flatten layer
    return s

args.avgpool_kernelsize = (X.shape, args)


# In[16]:


# synthtic an example that has the same number of channels and features with the input X
x_syn = torch.randn((1, X.shape[1], X.shape[2]))
# print(x_syn.shape) #torch.Size([1, 392, 175])

# get kernel size for avgpool layer
args.avgpool_kernelsize = getAvgpoolKernelSize(x_syn, args.n_channal)

# get dim for linear layer
flatten_dim = getFlattenDim(x_syn, args.n_channal, args.avgpool_kernelsize)

args.linear_dim = flatten_dim + site.shape[1] + sex.shape[1] + 1  # the last "1" dim is for age



print(args.linear_dim)


# In[18]:


def train_stDNN(net, train_iter, test_iter, fold_index, args): #num_epochs, lr, device,  batch_size, out_dir):
    def init_weigths(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.kaiming_normal_(m.weight)

    net.apply(init_weigths)
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion_cls = nn.CrossEntropyLoss()

    avg_perf_mat = []  # num_epochs * 6
    max_test_acc = 0.5   #二分类，模型大于0.5之后再比较
    for epoch in range(args.total_epochs):
        perf_mat = np.zeros(6, dtype=np.float32)
        t0 = time.time()
        net.train()
        for _, data in enumerate(train_iter, 0):
            X, y, site_, sex_, age_ = data

            # the last batch only has 2 samples (for a batch_size of 128), discard it to reduce risk for accuracy calculation
            if X.shape[0] != args.batch_size:
#                 print("Discard this minibatch.Only ", str(X.shape[0]), " samples.")
                continue
            else:
                optimizer.zero_grad()
                X, y, site_, sex_, age_ = X.to(args.device), y.to(args.device), site_.to(args.device), sex_.to(args.device), age_.to(args.device)
                age_ = age_.unsqueeze(1)

                others = torch.cat((site_, sex_, age_), dim=1) # torch.Size([64, 3])
                y_ = net(X.float(), others.float())

                loss_cls = criterion_cls(y_, y.long())

                loss_cls.backward()

                optimizer.step()

                perf_mat[0] += loss_cls.item()
                perf_mat[1] += compute_acc(y_, y)
                perf_mat[2] += compute_auc(y_, y)

        ## Testing
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(test_iter, 0):
                X, y, site_, sex_, age_ = data

                # the last batch only has 2 samples (for a batch_size of 128), discard it to reduce risk for accuracy calculation
                if X.shape[0] != args.batch_size:
#                     print("Discard this minibatch.Only ", str(X.shape[0]), " samples.")
                    continue
                else:
                    X, y, site_, sex_, age_ = X.to(args.device), y.to(args.device), site_.to(args.device),sex_.to(args.device), age_.to(args.device)
                    age_ = age_.unsqueeze(1)
                    others = torch.cat([site_, sex_, age_], dim=1)

                    y_ = net(X.float(), others.float())

                    loss_cls = criterion_cls(y_, y.long())

                    perf_mat[3] += loss_cls.item()
                    perf_mat[4] += compute_acc(y_, y)
                    perf_mat[5] += compute_auc(y_, y)

        perf_mat[0:3] /= len(train_iter)
        perf_mat[3:6] /= len(test_iter)
        if perf_mat[4] > max_test_acc : 
            max_test_acc = perf_mat[4]
            # save model
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
                print("A new directory {} is created: ".format(args.out_dir))

            fn = os.path.join(args.out_dir, 'newstDNN_' + str(fold) + '.params')
            print('Saving trained model to {} at {} epoch'.format(fn, epoch))
            torch.save(net.state_dict(), fn)
            avg_perf_mat.append(perf_mat)

        print('EPOCH: {:03d} {:.2f}s || TRAIN LOSS: {:.4f} ACC {:.4f} AUC {:.4f} || TEST LOSS: {:.4f} ACC {:.4f} AUC {:.4f}'
              .format(epoch, time.time()-t0, perf_mat[0], perf_mat[1], perf_mat[2], perf_mat[3], perf_mat[4], perf_mat[5]))

    # plot the performance metrics for the fold
    plot_perf_mat(avg_perf_mat, args.out_dir, fold_index)

    return np.mean(avg_perf_mat, axis=0), max_test_acc


# In[ ]:


def Kfold_varify(dataset, args):
    kfold = KFold(n_splits=args.n_fold, shuffle=True, random_state=42)
    splits = kfold.split(dataset)
    perf = np.zeros((args.n_fold, 6), dtype=np.float32) # holding performance of all folds
    max_test_acc = np.zeros(args.n_fold)
    for fold, (train_index, test_index) in enumerate(splits):
        t1 = time.time()
        # dividing data into folds
        train = torch.utils.data.Subset(dataset, train_index)
        test = torch.utils.data.Subset(dataset, test_index)
        # load data
        train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(dataset=test,  batch_size=args.batch_size, shuffle=True)

        # train the model
        net = stDNN_class(args)
        perf[fold], max_test_acc[fold] = train_stDNN(net, train_loader, test_loader, fold, args) #num_epochs, lr, device,  batch_size, out_dir)

        M = perf[fold]
        print('FOLD: {:d} {:.2f}s || AVG TRAIN LOSS: {:.4f} ACC {:.4f} AUC {:.4f} || AVG TEST LOSS: {:.4f} ACC {:.4f} AUC {:.4f}'
             .format(fold, time.time()-t1, M[0], M[1], M[2], M[3], M[4], M[5]))
        print("MAX ACC:{:.4f}".format(max_test_acc[fold]))

    avg_perf = np.mean(perf, axis=0)
    print('AVG over FOLDs: AVG TRAIN LOSS: {:.4f} ACC {:.4f} AUC {:.4f} || AVG TEST LOSS: {:.4f} ACC {:.4f} AUC {:.4f}'
             .format(avg_perf[0], avg_perf[1], avg_perf[2], avg_perf[3], avg_perf[4], avg_perf[5]))
    print('AVG MAX TEST ACC:{}'.format(np.mean(max_test_acc)))


# In[19]:


from util.calculate_IG import process_attribution, inte_gradient

def Group_Kfold(dataset, args):
    group_kfold = GroupKFold(n_splits=args.group_nfold)
    splits = group_kfold.split(dataset.data, dataset.label, np.where(np.array(dataset.site))[1])
    perf = np.zeros((args.group_nfold, 6), dtype=np.float32) # holding performance of all folds
    max_test_acc = np.zeros(args.group_nfold)
    for fold, (train_index, test_index) in enumerate(splits):
        t1 = time.time()
        # dividing data into folds
        train = torch.utils.data.Subset(dataset, train_index)
        test = torch.utils.data.Subset(dataset, test_index)
        # load data
        train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(dataset=test,  batch_size=args.batch_size, shuffle=True)

        # train the model
        net = stDNN_class(args)
        perf[fold], max_test_acc[fold] = train_stDNN(net, train_loader, test_loader, fold, args) #num_epochs, lr, device,  batch_size, out_dir)

        M = perf[fold]
        print('FOLD: {:d} {:.2f}s || AVG TRAIN LOSS: {:.4f} ACC {:.4f} AUC {:.4f} || AVG TEST LOSS: {:.4f} ACC {:.4f} AUC {:.4f}'
             .format(fold, time.time()-t1, M[0], M[1], M[2], M[3], M[4], M[5]))
        print("MAX ACC:{:.4f}".format(max_test_acc[fold]))

        inte_gradient(dataset, net, fn)

    avg_perf = np.mean(perf, axis=0)
    print('AVG over FOLDs: AVG TRAIN LOSS: {:.4f} ACC {:.4f} AUC {:.4f} || AVG TEST LOSS: {:.4f} ACC {:.4f} AUC {:.4f}'
             .format(avg_perf[0], avg_perf[1], avg_perf[2], avg_perf[3], avg_perf[4], avg_perf[5]))
    print('AVG MAX TEST ACC:{}'.format(np.mean(max_test_acc)))


# In[20]:


class stDNN_class(nn.Module):
    def __init__(self, args):
        super(stDNN_class, self).__init__()
        self.conv1d_1 = nn.Conv1d(args.n_channal, 256, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AvgPool1d(kernel_size = args.avgpool_kernelsize)
        self.drop = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.liner = nn.Linear(args.linear_dim, args.n_category)
    def forward(self, x, others):
        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)  #[B, 256, 86]
        x = self.maxpool_2(x)  #[B, 512, 43]
        x = self.avgpool(x)
        x = self.flatten(x)  #[B, 512]
        x = self.liner(x) #[B, 2]
        return x


# In[21]:


import torch
import os
import numpy as np

# plot a figure that contains 3 subplots: loss, acc and auc of the train and test together
def plot_perf_mat(perf_mat, out_dir, fold_index):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    perf_mat = np.array(perf_mat)

    x = list(range(0, perf_mat.shape[0]))

    axs[0].plot(x, perf_mat[:,0], 'tab:orange', label='train')
    axs[0].plot(x, perf_mat[:,3], 'tab:blue', label='validation')
    axs[0].set_title('Loss')
    axs[1].plot(x, perf_mat[:,1], 'tab:orange', label='train')
    axs[1].plot(x, perf_mat[:,4], 'tab:blue', label='validation')
    axs[1].set_title('Accuracy')
    axs[2].plot(x, perf_mat[:,2], 'tab:orange', label='train')
    axs[2].plot(x, perf_mat[:,5], 'tab:blue', label='validation')
    axs[2].set_title('AUC')
    plt.legend()
    # save the plot to file instead of showing it
    fig_out_dir = os.path.join(out_dir, "plot-imgs")

    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)
        print("A new directory {} is created: ".format(fig_out_dir))

    fig_fn = os.path.join(fig_out_dir, "plot-imgs-" + str(fold_index) + ".png")

    print("Save plot images to: ", fig_fn)
    plt.savefig(fig_fn)



def compute_acc(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)

def compute_auc(y_pred, y_true):
    predictions = torch.argmax(y_pred, dim=1)

    #for binary classes
    if (y_pred.shape[1] == 2):

        ##one way. (while labels only contains one class (the data very imbalanced), say error: "ValueError: Only one c
        ##one way. (while labels only contains one class (the data very imbalanced), say error: "ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.")
        y_true = y_true.cpu().detach().numpy()
        y_score = predictions.cpu().detach().numpy()
#         s = roc_auc_score(y_true, y_score)

        # #a second way
        # lt = y_true.cpu()
        # yt = predictions.cpu()
        # fpr,tpr,_ = roc_curve(lt.detach().numpy(),yt.detach().numpy())
        # s = auc(fpr,tpr)

    # for multiclass
    # https://blog.51cto.com/u_15346769/3706658
    else:
        y_true = y_true.cpu().detach().numpy()
        y_score = predictions.cpu().detach().numpy()

        lb = LabelBinarizer()
        lb.fit(y_true)
        y_true = lb.transform(y_true)
        y_score = lb.transform(y_score)

    try:
        s = roc_auc_score(y_true, y_score, average='micro')
    except:
        #print("Class imbalance.") # one or more class is not in this minibatch
        s = 0

    return s

#Kfold_varify(dataset, args)
Group_Kfold(dataset, args)
