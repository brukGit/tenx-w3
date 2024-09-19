import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from scipy.stats import uniform, randint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedStatisticalModeling:
    def __init__(self, df):
        self.df = self.convert_to_numeric(df, ['SumInsured', 'RegistrationYear', 'CapitalOutstanding', 'CustomValueEstimate', 'CalculatedPremiumPerTerm'])
        self.numeric_features = ['SumInsured', 'RegistrationYear', 'CapitalOutstanding', 'CustomValueEstimate', 'CalculatedPremiumPerTerm']
        self.categorical_features = ['VehicleType', 'make', 'Model', 'CoverType', 'CoverCategory', 'CoverGroup', 'Country', 
                                     'Province', 'MainCrestaZone', 'SubCrestaZone', 'IsVATRegistered', 'MaritalStatus', 
                                     'Gender', 'Citizenship', 'Product', 'ExcessSelected', 'TermFrequency']
    
    def convert_to_numeric(self, df, columns):
        for column in columns:
            df[column] = df[column].replace(',', '', regex=True).astype(float)
        return df
    
    def preprocess_data(self, target_column):
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Handle outliers in target variable
        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        y = np.clip(y, lower_bound, upper_bound)

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

    def train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test, preprocessor):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return pipeline, mse, r2

    def feature_importance(self, pipeline, X):
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
        elif hasattr(pipeline.named_steps['model'], 'coef_'):
            importances = np.abs(pipeline.named_steps['model'].coef_)
        else:
            raise ValueError("Model doesn't have feature importances or coefficients")

        feature_names = (pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names(self.categorical_features).tolist() + self.numeric_features)
        
        return pd.DataFrame({'feature': feature_names, 'importance': importances})

    def run_analysis(self, target_column):
        (X_train, X_test, y_train, y_test), preprocessor = self.preprocess_data(target_column)

        models = {
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }

        results = {}

        for name, model in models.items():
            logging.info(f"Training {name}")
            pipeline, mse, r2 = self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor)
            feature_importance = self.feature_importance(pipeline, X_train)
            
            results[name] = {
                'pipeline': pipeline,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }

            logging.info(f"{name} - MSE: {mse}, R2: {r2}")

        best_model = min(results, key=lambda x: results[x]['mse'])
        logging.info(f"Best model: {best_model}")

        # Hyperparameter tuning for the best model
        best_pipeline = results[best_model]['pipeline']
        param_distributions = {
            'model__n_estimators': randint(100, 1000),
            'model__max_depth': randint(5, 30),
            'model__learning_rate': uniform(0.01, 0.3),
        }

        random_search = RandomizedSearchCV(best_pipeline, param_distributions, n_iter=20, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        logging.info("Best hyperparameters:")
        logging.info(random_search.best_params_)

        y_pred = random_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Tuned model - MSE: {mse}, R2: {r2}")

        return results, random_search
