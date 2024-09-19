import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import shap
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StatisticalModeling:
    def __init__(self, df):
        self.df = df
        self.columns_total_claim = ['VehicleType', 'make', 'Model', 'SumInsured', 'RegistrationYear', 'CoverType', 
                                    'ExcessSelected', 'CoverCategory', 'CoverGroup', 'Country', 'Province', 
                                    'MainCrestaZone', 'SubCrestaZone', 'IsVATRegistered', 'MaritalStatus', 
                                    'Gender', 'Citizenship']
        self.columns_total_premium = ['VehicleType', 'make', 'Model', 'SumInsured', 'RegistrationYear', 'CoverType', 
                                      'ExcessSelected', 'TermFrequency', 'CoverCategory', 'CoverGroup', 'Product', 
                                      'Country', 'Province', 'MainCrestaZone', 'CapitalOutstanding', 
                                      'CustomValueEstimate', 'CalculatedPremiumPerTerm']
        
        self.cat_vars = ['VehicleType', 'make', 'Model', 'CoverType', 'CoverCategory', 'CoverGroup', 'Country', 
                         'Province', 'MainCrestaZone', 'SubCrestaZone', 'IsVATRegistered', 'MaritalStatus', 
                         'Gender', 'Citizenship', 'Product', 'ExcessSelected', 'TermFrequency']
        self.numeric_vars = ['SumInsured', 'RegistrationYear', 'CapitalOutstanding', 'CustomValueEstimate', 
                             'CalculatedPremiumPerTerm']

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        # Handle missing values
        self.df = self.df.dropna()  # or use appropriate imputation method

        # Encode categorical variables
        for col in self.cat_vars:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))

        # Convert numeric columns and scale them
        scaler = StandardScaler()
        self.df[self.numeric_vars] = scaler.fit_transform(self.df[self.numeric_vars].apply(pd.to_numeric, errors='coerce'))

    def prepare_data(self, target_column):
        logging.info(f"Preparing data for target column: {target_column}")
        if target_column == 'TotalClaims':
            columns = self.columns_total_claim
        elif target_column == 'TotalPremium':
            columns = self.columns_total_premium
        else:
            raise ValueError("Invalid target column. Choose 'TotalClaims' or 'TotalPremium'.")

        # Prepare features and target variable
        X = self.df[columns]
        y = self.df[target_column]
         # Log data statistics
        logging.info(f"X shape: {X.shape}")
        logging.info(f"y shape: {y.shape}")
        logging.info(f"X columns: {X.columns}")
        logging.info(f"X description:\n{X.describe()}")
        logging.info(f"y description:\n{y.describe()}")

        # Check for infinite or NaN values
        logging.info(f"Infinite values in X: {np.isinf(X.values).sum()}")
        logging.info(f"NaN values in X: {np.isnan(X.values).sum()}")
        logging.info(f"Infinite values in y: {np.isinf(y.values).sum()}")
        logging.info(f"NaN values in y: {np.isnan(y.values).sum()}")

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def handle_missing_data(self):
        logging.info("Handling missing data")
        # Drop rows with critical missing data
        self.df = self.df.dropna(subset=['TotalPremium', 'TotalClaims'])
        # Impute missing values with 0
        return self.df.fillna(0)

    def train_model(self, model, X_train, X_test, y_train, y_test):
        logging.info(f"Training model: {model.__class__.__name__}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f"Model training complete: {model.__class__.__name__}")
        logging.info(f"MSE: {mse}")
        logging.info(f"R2 Score: {r2}")

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted - {model.__class__.__name__}')
        plt.tight_layout()
        plt.show()

        return model, predictions, mse, r2

    def linear_regression(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        return self.train_model(model, X_train, X_test, y_train, y_test)

    def random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        return self.train_model(model, X_train, X_test, y_train, y_test)

    def xgboost(self, X_train, X_test, y_train, y_test):
        model = XGBRegressor(random_state=42, n_jobs=-1)
        return self.train_model(model, X_train, X_test, y_train, y_test)

    def feature_importance(self, model, X_train, y_train):
        logging.info(f"Calculating feature importance for model: {model.__class__.__name__}")
        if isinstance(model, LinearRegression):
            feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
            importance = pd.DataFrame({'feature': feature_names, 'importance': np.abs(model.coef_)})
        elif isinstance(model, (RandomForestRegressor, XGBRegressor)):
            importance = self.permutation_importance(model, X_train, y_train)
        else:
            raise ValueError("Unsupported model type for feature importance")
        
        importance_sorted = importance.sort_values('importance', ascending=False)
        
        logging.info("Feature Importance:")
        for idx, row in importance_sorted.iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return importance_sorted

    def select_important_features(self, X_train, y_train, threshold=0.01):
        logging.info("Selecting important features based on RandomForest feature importance")
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        X_important_train = selector.transform(X_train)

        selected_features = X_train.columns[selector.get_support()]
        logging.info(f"Selected important features: {selected_features}")
        logging.info(f"Number of features selected: {len(selected_features)}")
        
        return pd.DataFrame(X_important_train, columns=selected_features), selected_features
        
        
    def run_analysis(self, target_column):
        logging.info(f"Running analysis for target column: {target_column}")
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1)
        }

        # Prepare initial data
        X_train, X_test, y_train, y_test = self.prepare_data(target_column)

        # Select important features before full training
        X_important_train, selected_features = self.select_important_features(X_train, y_train)
        X_important_test = X_test[selected_features]

        results = {}
        for name, model in models.items():
            logging.info(f"Starting training for {name}")
            model, predictions, mse, r2 = self.train_model(model, X_important_train, X_important_test, y_train, y_test)
            feature_importance = self.feature_importance(model, X_important_train, y_train)
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }
            
            logging.info(f"Top 10 important features for {name}:")
            logging.info(feature_importance.head(10).to_string(index=False))
        
        return results
    def partial_dependence_plot(self, model, X, features, n_cols=3):
        fig, ax = plt.subplots(figsize=(20, 20))
        PartialDependenceDisplay.from_estimator(model, X, features, n_cols=n_cols, ax=ax)
        plt.tight_layout()
        return fig

    def permutation_importance(self, model, X, y):
        
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        return pd.DataFrame({'feature': feature_names, 'importance': r.importances_mean}).sort_values('importance', ascending=False)

 