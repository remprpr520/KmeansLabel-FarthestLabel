from .datasource import DataSource
from erasure.utils.config.global_ctx import Global
from erasure.data.datasets.Dataset import DatasetWrapper 
from erasure.utils.config.local_ctx import Local
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import ConcatDataset
import torch
import pandas as pd
import numpy as np
from datasets import Dataset

class UCIWrapper(DatasetWrapper):
    def __init__(self, data, preprocess,label, data_columns):
        super().__init__(data,preprocess)
        self.label = label
        self.data_columns = data_columns

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        X = torch.Tensor([value for key,value in sample.items() if key in self.data_columns])

        y = sample[self.label]
     
        return X,y


class UCIRepositoryDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.id = self.local_config['parameters']['id']
        self.dataset = None
        self.label = self.local_config['parameters']['label']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.to_encode = self.local_config['parameters']['to_encode']

    def get_name(self):
        return self.name

    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

        pddataset = pd.DataFrame(self.dataset.data.original)

        if not self.data_columns:
            self.data_columns = [col for col in pddataset if col != self.label]
        else:
            self.data_columns = [col for col in pddataset if col in self.data_columns and col != self.label]
            
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'


        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )

        self.dataset.classes = pddataset[self.label].unique()

        return self.get_simple_wrapper(self.dataset)

    
    def get_simple_wrapper(self, data):
        return UCIWrapper(data, self.preprocess, self.label, self.data_columns)
    

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['label'] = self.local_config['parameters'].get('label','')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns',[])
        self.local_config['parameters']['to_encode'] = self.local_config['parameters'].get('to_encode',[])

##Adult has a lot of errors in its data, therefore it's best to handle them in a different loader.
import os
import pandas as pd
from datasets import Dataset
from torch.utils.data import ConcatDataset


class UCI_Adult_DataSource(UCIRepositoryDataSource):

    def create_data(self):
        local_csv_path = "resources/data/adult.csv"

        if os.path.exists(local_csv_path):
            df = pd.read_csv(local_csv_path)

            df.columns = df.columns.str.replace('.', '-')

            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].replace('?', 'Other')

            def process_country(x):
                if pd.isna(x) or x == 'Other':
                    return 'Other'
                return 'United-States' if x == 'United-States' else 'Other'

            df['native-country'] = df['native-country'].apply(process_country)

            df = pd.get_dummies(df, columns=self.to_encode)

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                            'capital-loss', 'hours-per-week']
            for col in numeric_cols:
                if col in df.columns:
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            df[self.label] = df[self.label].apply(lambda x: 0 if '<=50K' in str(x) else 1)

            self.name = "Adult (UCI Census Income)"

        else:
            raise FileNotFoundError(f"找不到文件: {local_csv_path}")

        hfdataset = Dataset.from_pandas(df)
        self.dataset = ConcatDataset([hfdataset])
        self.dataset.classes = df[self.label].unique()

        return self.get_simple_wrapper(self.dataset)


class UCI_Iris_DataSource(UCIRepositoryDataSource):

    def create_data(self):
        local_csv_path = "resources/data/iris.csv"

        if os.path.exists(local_csv_path):
            df = pd.read_csv(local_csv_path)

            if 'Unnamed: 0' in df.columns or df.columns[0] == '':
                df = df.iloc[:, 1:]

            df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

            species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
            df[self.label] = df['species'].map(species_map)

            df = df.drop('species', axis=1)

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            for col in self.data_columns:
                if df[col].dtype in ['float64', 'int64']:
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            self.name = "Iris Dataset"

        else:
            if self.dataset is None:
                self.dataset = fetch_ucirepo(id=self.id)
            df = pd.DataFrame(self.dataset.data.original)

            if 'species' in df.columns:
                species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
                df[self.label] = df['species'].map(species_map)
                df = df.drop('species', axis=1)

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Iris'

        hfdataset = Dataset.from_pandas(df)
        self.dataset = ConcatDataset([hfdataset])
        self.dataset.classes = df[self.label].unique()

        return self.get_simple_wrapper(self.dataset)


class UCI_DryBean_DataSource(UCIRepositoryDataSource):

    def create_data(self):
        local_csv_path = "resources/data/Dry_Bean_Dataset.xlsx"

        if local_csv_path.endswith('.xlsx'):
            df = pd.read_excel(local_csv_path, engine='openpyxl')
        else:
            df = pd.read_csv(local_csv_path)

        df.columns = df.columns.str.strip()

        if 'Class' in df.columns:
            label_col = 'Class'
        elif 'class' in df.columns:
            label_col = 'class'
        else:
            raise KeyError(f"找不到列: {df.columns.tolist()}")

        classes = df[label_col].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        df[self.label] = df[label_col].map(class_to_idx)

        df = df.drop(label_col, axis=1)

        if not self.data_columns:
            self.data_columns = [col for col in df.columns if col != self.label]

        if df.isnull().sum().sum() > 0:
            for col in self.data_columns:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        for col in self.data_columns:
            if df[col].dtype in ['float64', 'int64']:
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std

        self.name = "Dry Bean Dataset"

        hfdataset = Dataset.from_pandas(df)
        self.dataset = ConcatDataset([hfdataset])
        self.dataset.classes = df[self.label].unique()

        return self.get_simple_wrapper(self.dataset)


class UCI_Covertype_DataSource(UCIRepositoryDataSource):
    def create_data(self):
        local_csv_path = "resources/data/covtype.csv"

        if os.path.exists(local_csv_path):

            df = pd.read_csv(local_csv_path)

            label_col = df.columns[-1]

            df[self.label] = df[label_col] - 1
            df = df.drop(label_col, axis=1)

            quantitative_cols = df.columns[:10].tolist()
            print(f"   定量特征列: {quantitative_cols[:5]}...")

            for col in quantitative_cols:
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            self.name = "Covertype Forest Cover Type"

        else:
            if self.dataset is None:
                self.dataset = fetch_ucirepo(id=self.id)

            if hasattr(self.dataset, 'data'):
                df = pd.DataFrame(self.dataset.data.original)
            else:
                df = pd.DataFrame(self.dataset['data']['original'])

            label_col = df.columns[-1]
            df[self.label] = df[label_col] - 1
            df = df.drop(label_col, axis=1)

            quantitative_cols = df.columns[:10].tolist()
            for col in quantitative_cols:
                if df[col].dtype in ['float64', 'int64']:
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            self.name = self.dataset.metadata.name if hasattr(self.dataset, 'metadata') else 'Covertype'

        hfdataset = Dataset.from_pandas(df)
        self.dataset = ConcatDataset([hfdataset])
        self.dataset.classes = df[self.label].unique()

        return self.get_simple_wrapper(self.dataset)


class UCI_WineQualityWhite_DataSource(UCIRepositoryDataSource):
    def create_data(self):
        local_csv_path = "resources/data/winequality-white.csv"

        if os.path.exists(local_csv_path):
            df = pd.read_csv(local_csv_path, sep=';')

            df[self.label] = df['quality'] - 3
            df = df.drop('quality', axis=1)

            if not self.data_columns:
                self.data_columns = [col for col in df.columns if col != self.label]

            for col in self.data_columns:
                if df[col].dtype in ['float64', 'int64']:
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std

            self.name = "White Wine Quality"

        else:
            raise FileNotFoundError(f"找不到文件")

        hfdataset = Dataset.from_pandas(df)
        self.dataset = ConcatDataset([hfdataset])
        self.dataset.classes = df[self.label].unique()

        return self.get_simple_wrapper(self.dataset)