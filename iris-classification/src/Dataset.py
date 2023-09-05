import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch

class IrisDataset(Dataset):
    def __init__(self, path) -> None:
        self.raw_data = None
        self.labelEncoder = LabelEncoder()
        self.path = path
        self.load_data()
        pass
    
    def load_data(self):
        with open(self.path, newline='') as csvfile:
          self.raw_data = pd.read_csv(csvfile)
          self.raw_data = self.raw_data.drop(columns=['Id'])
        
        self.preprocessing()
        # self.analyse_data()
    
        return self.raw_data.drop(columns=['Species']), self.raw_data['Species']

    def analyse_data(self):
        print(self.raw_data.describe())
        print(self.raw_data.isnull().sum())
        print('Shape of data: ', self.raw_data.shape)
        print(self.raw_data.info())
        print(self.raw_data['Species'].value_counts())
        pass

    def get_data(self):
        return self.raw_data.drop(columns=['Species'])
    
    def get_labels(self):
        return self.raw_data['Species']
  
    def preprocessing(self):
        # Transform labels to numbers 
        self.raw_data['Species'] = self.labelEncoder.fit_transform(self.raw_data['Species'])

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        features = self.raw_data.drop(columns=['Species']).loc[index, :]
        # print(features)
        label = self.raw_data.loc[index, 'Species']
        return {'input': torch.tensor(features), 
                'label': torch.tensor(label)}