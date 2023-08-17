from model import Model
from torch import nn 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import IrisDataset

def main():
    path = './Iris.csv'
    model = Model()
    IrisDataset(path)

    def train(batch_size): # training loop 
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        model.train()
        model.double()
        dataloader = DataLoader(IrisDataset(path), batch_size=batch_size)
        size = len(dataloader.dataset)

        for batch, dict_data in enumerate(dataloader):
            X, Y = dict_data['input'], dict_data['label']
            
            pred_y = model(X)
            loss = loss_fn(pred_y, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, current = loss.item(), (batch + 1) * len(X)
            print(batch)
            print(f"loss: {loss}  [{current}/{size}]")

    epochs = 10

    for _ in range(epochs):
        train(32)

        
    pass

if __name__ == '__main__':
    main()