import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader: 
  def __init__(self) -> None:
      self.X = None
      self.Y = None
      self.raw_data = None
      self.data = None

      self.encoder = LabelEncoder()
      pass
  def load_data(self, path): 
      with open(path, newline='') as csvfile:
        self.raw_data = pd.read_csv(csvfile)
        # Preprocess data 

        self.data = self.raw_data.copy()

        new_feature_names = [col.lower() for col in self.data.columns]
        self.data.columns = new_feature_names

        self.fill_item_weight()
        self.fill_outlet_size()
        self.add_item_category()
        self.clean_item_fat_content()
        self.add_year()

  def preprocessing(self):
      # Encode the values  
      cols_to_encode = ['item_identifier', 'item_type', 'outlet_identifier']
      for col in cols_to_encode:
        self.data[col] = self.encoder.fit_transform(self.data[col])
      
      # One hot encode
      self.data = pd.get_dummies(self.data, columns=['item_fat_content', 'outlet_size', 'outlet_location_type', 'outlet_type', 'item_category'])

  def get_X_Y(self):
     self.preprocessing()
     return self.data.drop(columns=['outlet_establishment_year', 'item_outlet_sales']), self.data['item_outlet_sales']
      
  def categorical_features(self):
    return self.raw_data.select_dtypes(exclude='number').columns.to_list()
  
  def numerical_features(self):
    return self.raw_data.select_dtypes(include='number').columns.to_list()
  
  def fill_item_weight(self):
     self.data['item_weight'].fillna(self.data['item_weight'].mean(), inplace=True)

  def fill_outlet_size(self):
    # get mode based on outlet_type 
    outlet_type_mode = self.data.pivot_table(values='outlet_size', columns='outlet_type', aggfunc=lambda x: x.mode())
    missing_values = self.data['outlet_size'].isnull() # Get the missing values in data 
    self.data.loc[missing_values, 'outlet_size'] = self.data.loc[missing_values, 'outlet_type'].apply(lambda x: outlet_type_mode[x].outlet_size)

  def clean_item_fat_content(self):
     self.data.replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
     self.data.loc[self.data['item_category']== 'Non-Consumable', 'item_fat_content'] = 'No edible'
  
  def add_item_category(self):
     self.data['item_category'] = self.data['item_identifier'].apply(lambda x: x[:2])
     self.data['item_category'] = self.data['item_category'].replace({'FD': 'Food', 'DR': 'Drink', 'NC': 'Non-Consumable'})

  def add_year(self):
     self.data['outlet_years'] = 2023 - self.data['outlet_establishment_year']

