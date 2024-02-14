import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE, ADASYN
import category_encoders as ce
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ColumnManipulatorDrop(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_df = ['item', 'cash_price', 'make']
        self.category_todelete = ['model', 'goods_code', 'Nbr_of_prod_purchas']
        self.delete_columns = []

    def generate_delete_columns(self):
        for i in self.category_df:
            for j in range(4, 25):
                self.delete_columns.append(i + str(j))
        for i in self.category_todelete:
            for j in range(1, 25):
                self.delete_columns.append(i + str(j))
        # Uncomment the line below if you want to include 'ID' in delete_columns
        # self.delete_columns.append('ID')
        return self.delete_columns
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        delete_columns = self.generate_delete_columns()
        df_drop = df.drop(columns=delete_columns)
        return df_drop

class featureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df["Total_price"] = df.loc[:, [col for col in df.columns if "cash_price" in col]].sum(axis=1)
        df["Total_items"] = df.loc[:, [f"Nbr_of_prod_purchas{i}" for i in range(1, 25)]].sum(axis=1)
        df["Price_per_total_purchases"] = df["Total_price"] / df["Total_items"]
        df["Price_per_unique_item"] = df["Total_price"] / df["Nb_of_items"]
        return df

class CleanItems(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        item = ['item1', 'item2', 'item3']
        for i in item:
            df[i] = df[i].str.replace("\s&", "", regex=True).str.replace(",", "", regex=True)
        return df

    
class preprocess(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        self.numerical_col = [col for col in df.columns
                 if df[col].dtypes in ['float64']]
        self.categorical_col = [col for col in df.columns
                   if df[col].dtypes in ['object']]
        self.onehot = OneHotEncoder()
        return self
    
    def transform(self, df):
        df_num = df[self.numerical_col]
        df_cat = df[self.categorical_col]

        #impute NaN 
        df_num = df_num.fillna(0)
        df_cat = df_cat.fillna('None')

        #Keep the categories that are more related to fraud cases
        make = ['make1', 'make2', 'make3']
        categoryToKeep = ['APPLE', 'SONY', 'LG', 'SAMSUNG', 'None']
        for i in make:
            for j in df_cat.index:
                if df_cat[i][j] not in categoryToKeep:
                    df_cat[i][j] = 'OTHER'
        
        #Keep the items that are more related to fraud cases
        item = ['item1', 'item2', 'item3']
        itemsToKeep = ['COMPUTERS', 'COMPUTER PERIPHERALS ACCESSORIES', 'TELEPHONES FAX MACHINES TWO-WAY RADIOS', 'TELEVISIONS HOME CINEMA']
        for i in item:
            for j in df_cat.index:
                if df_cat[i][j] not in itemsToKeep:
                    df_cat[i][j] = 'OTHER'

        #One Hot Encoding the categorical features
        if not self.fit:
            cat_oh = self.onehot.transform(df_cat)
        else:
            cat_oh = self.onehot.fit_transform(df_cat)
        df_cat_oh = pd.DataFrame(cat_oh.toarray(), columns=self.onehot.get_feature_names_out(self.categorical_col), index=df.index)

        #Concatenate df_num and df_cat
        df_prep = pd.concat([df_num, df_cat_oh], axis=1)

        return df_prep