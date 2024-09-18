import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

class StatisticalModeling:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.preprocessor = None

    def prepare_data(self, target='TotalPremium'):
        # Handle missing data
        # print(f"Data shape before dropna: {self.data.shape}")
        # self.data = self.data.dropna()
        # print(f"Data shape after dropna: {self.data.shape}")
        

        # Feature engineering
        self.data['VehicleAge'] = pd.to_datetime('today').year - self.data['RegistrationYear']
        self.data['IsNewVehicle'] = (self.data['VehicleAge'] <= 1).astype(int)

        # Split features and target
        X = self.data.drop([target, 'TotalClaims'], axis=1)
        y = self.data[target]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def build_models(self):
        # Linear Regression
        lr_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
        ])
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model

        # Random Forest
        rf_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model

        # XGBoost
        xgb_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, random_state=42))
        ])
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model

    def evaluate_models(self):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results[name] = {'MSE': mse, 'R2': r2}
        return results

    def feature_importance(self, model_name='Random Forest'):
        model = self.models[model_name]
        feature_names = self.preprocessor.get_feature_names_out()
        importances = model.named_steps['regressor'].feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def shap_analysis(self, model_name='XGBoost'):
        model = self.models[model_name]
        X_processed = self.preprocessor.transform(self.X_test)
        explainer = shap.TreeExplainer(model.named_steps['regressor'])
        shap_values = explainer.shap_values(X_processed)
        return shap_values, self.preprocessor.get_feature_names_out()