from model import Model
from torch import nn 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import IrisDataset
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    path = './Iris.csv'
    model = Model()
    dataset = IrisDataset(path)
    # dataset.analyse_data()

    def train(batch_size): # training loop 
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        model.train()
        model.double()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        size = len(dataloader.dataset)

        for batch, dict_data in enumerate(dataloader):
            X, Y = dict_data['input'], dict_data['label']
            
            pred_y = model(X)
            loss = loss_fn(pred_y, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, current = loss.item(), (batch + 1) * len(X)
            # print(batch)
            # print(f"loss: {loss}  [{current}/{size}]")

    epochs = 200

    for _ in range(epochs):
        train(32)

    with torch.no_grad():
        X = dataset.get_data()
        Y = dataset.get_labels()
        model.eval()
        predicts = model(torch.tensor(X.values))
        Y_pred = np.argmax(predicts, axis=1)
        print(Y_pred)
        print(accuracy_score(Y, Y_pred))
    
    pass

if __name__ == '__main__':
    main()