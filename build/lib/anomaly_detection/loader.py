import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from anomaly_detection import synthetic_data, detect_anomaly

class FeatureClusterDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, transform=None):
        """
        Arguments:
            np_array (np.array): data array.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datapoints_frame = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.datapoints_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feats = self.datapoints_frame[idx][0]
        labs = self.datapoints_frame[idx][1]
        sample = (feats, labs)

        if self.transform:
            sample = self.transform(sample)

        return sample

class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Linear(D_in, H)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# synth_data = synthetic_data.generateClusters(3, np.random.normal(0,2,[3000,3]), np.repeat([0.01*np.ones([3000, 3000])], 3) \
#             .reshape(3000,3000,3),3000)
# X,y = synth_data.generate_data()
# # print(X.shape,y.shape, np.vstack([X[:,:,0],X[:,:,1],X[:,:,2]]).shape)
# nn_model = NeuralNetwork(D_in=3000,H=30)
# x = torch.from_numpy(X[:,:,0]).type(torch.float)
# # y = torch.from_numpy(y[:,0])
# # x = torch.randn(100, 3000)
# y = torch.from_numpy(np.repeat(0,100))
# print(x.shape, y.shape)
# X_y = [(x[i],y[i]) for i in range(len(x))]
# dataset_ = FeatureClusterDataset(X_y)
# train_dataloader = DataLoader(dataset_, batch_size=64)

# # training_data = datasets.FashionMNIST(
# #     root="data",
# #     train=True,
# #     download=True,
# #     transform=ToTensor()
# # )
# # train_dataloader = DataLoader(training_data, batch_size=64)

# size = len(train_dataloader.dataset)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(nn_model.parameters(), lr=1e-3)
# def train_loop(dataloader, model, loss_fn, optimizer):
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = nn_model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, nn_model, loss_fn, optimizer)
# print("Done!")
# # print(nn_model(x).shape)
# # logits = nn_model(x)
# # pred_probab = nn.Softmax(dim=1)(logits)
# # y_pred = pred_probab.argmax(1)
# # opt = detect_anomaly.optimizeNN(train_dataloader, nn_model, learning_rate=1e-3, batch_size=64, epochs=5)
# # opt.train_loop()