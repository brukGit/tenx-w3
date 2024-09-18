import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StatisticalModeling:
    def __init__(self, df):
        self.df = df

    def handle_missing_data(self):
        self.df = self.df.dropna(subset=['TotalPremium', 'TotalClaims'])
        return self.df.fillna(0)  # Impute missing values with 0

    def feature_engineering(self):
        self.df['TotalPremium/Claims_Ratio'] = self.df['TotalPremium'] / (self.df['TotalClaims'] + 1)
        return self.df

    def encode_categorical_data(self):
        self.handle_datetime_columns()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = label_encoder.fit_transform(self.df[col])
        return self.df

    def handle_datetime_columns(self):
        datetime_columns = self.df.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            self.df[col + '_year'] = self.df[col].dt.year
            self.df[col + '_month'] = self.df[col].dt.month
            self.df[col + '_day'] = self.df[col].dt.day
            self.df[col + '_timestamp'] = self.df[col].astype(int) / 10**9
        self.df = self.df.drop(datetime_columns, axis=1)
        return self.df

    def select_important_features(self, target_column):
        correlation_matrix = self.df.corr()
        important_features = correlation_matrix[target_column].abs().sort_values(ascending=False).index
        return important_features[:10]  # Select top 10 features including the target

    def scale_features(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_test_split(self, target_column):
        features = self.select_important_features(target_column)
        X = self.df[features].drop([target_column], axis=1)
        y = self.df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, model, X_train, X_test, y_train, y_test):
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return predictions, mse, r2

    def linear_regression(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        return self.train_model(model, X_train, X_test, y_train, y_test)

    def random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestRegressor(n_estimators=100)
        return self.train_model(model, X_train, X_test, y_train, y_test)

    def xgboost(self, X_train, X_test, y_train, y_test):
        model = XGBRegressor()
        return self.train_model(model, X_train, X_test, y_train, y_test)
