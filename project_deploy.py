import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.stats import boxcox
from sklearn.metrics import make_scorer
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
import joblib
import pickle

# Load the dataset
df = pd.read_csv("qsar_fish_toxicity.csv")

# Separate the target variable from the features
X = df.drop(columns=['LC50 [-LOG(mol/L)]'])
y = df['LC50 [-LOG(mol/L)]']
print(y.info())
# Impute missing values in the target variable using IterativeImputer
y_imputer = IterativeImputer(verbose=False)
y_imputed = y_imputer.fit_transform(y.values.reshape(-1, 1))

# Convert the imputed data back to a 1D Pandas Series
y_imputed = pd.Series(y_imputed.squeeze(), name='LC50 [-LOG(mol/L)]')
print(y_imputed.info())
# Print the information about the dataset
print(X.info())

# Plot a pair plot to see the relationships between the different variables
sns.pairplot(X)
plt.show()

# Plot a heatmap to show the missing values in the dataset
plt.figure(figsize=(8, 6))
sns.heatmap(X.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# Calculate the skewness of each column
skewness = X.skew()
print(skewness)

# Check the distribution of the data in each column
for column in X.columns:
    print(column, X[column].describe())

# Calculate the missing-value percentage
missing_value_percent = X.isna().sum() / len(X) * 100
print("Missing Value Percentage:")
print(missing_value_percent)

# Impute the missing values using IterativeImputer
imputed_data = IterativeImputer(verbose=False).fit_transform(X)

# Convert the imputed data to a Pandas DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=X.columns)
print(imputed_df.info())

# boxplot before imputing outliers
plt.figure(figsize=(10, 6))
imputed_df.boxplot()
plt.title('Box Plot (Before Outlier Handling)')
plt.show()

# Perform winsorization on each column to handle outliers
winsorized_df = imputed_df.apply(lambda x: winsorize(x, limits=[0.1, 0.1]))

# boxplot after imputing outliers
plt.figure(figsize=(10, 6))
winsorized_df.boxplot()
plt.title('Box Plot (After Outlier Handling)')
plt.show()

# Calculate the correlation matrix before box-cox
correlation_matrix_before_boxcox = winsorized_df.corr()

# Visualize the correlation matrix before box-cox using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_before_boxcox, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix (Before Box-Cox Transformation)')
plt.show()

# skewness before box-cox
skewness_before_boxcox = winsorized_df.skew()
print("Skewness Before Box-Cox Transformation:")
print(skewness_before_boxcox)

# Apply Box-Cox transformation on each feature
for column in winsorized_df.columns:
    winsorized_df[column], _ = boxcox(
        winsorized_df[column] + 0.001)  # Add a small constant to handle non-positive values

# Calculate the correlation matrix after box-cox
correlation_matrix_after_boxcox = winsorized_df.corr()

# Visualize the correlation matrix after box-cox using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_after_boxcox, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Correlation Matrix (After Box-Cox Transformation)')
plt.show()

# skewness after box-cox
skewness_after_boxcox = winsorized_df.skew()
print("Skewness After Box-Cox Transformation:")
print(skewness_after_boxcox)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(winsorized_df)

# Apply PCA for dimensionality reduction
# pca = PCA(n_components=5)
# X_pca = pca.fit_transform(X_scaled)

# One-hot encode the categorical features: "CIC0" and "SM1_Dz(Z)"
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(winsorized_df[['CIC0', 'SM1_Dz(Z)']]))
X_encoded.columns = encoder.get_feature_names_out(['CIC0', 'SM1_Dz(Z)'])

# Concatenate the one-hot encoded categorical columns with the remaining features
X_final = pd.concat([X_encoded, pd.DataFrame(X_scaled, columns=winsorized_df.columns)], axis=1)

# Descriptive Statistics
print(X_final.describe())

# Create a scatter plot
plt.scatter(X_final['MLOGP'], y_imputed, label='Data Points', color='blue', marker='o')
# Set plot labels and title
plt.xlabel('MLOGP')
plt.ylabel('LC50 [-LOG(mol/L)]')
plt.title('Scatter Plot Between Molecular prop(MLOGP) & Target value(LC50)')
# Show the legend
plt.legend()
# Display the plot
plt.show()

# Pairwise Scatter Plots
features_to_plot = ['GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
sns.pairplot(data=pd.concat([X_final[features_to_plot], y_imputed], axis=1), hue='LC50 [-LOG(mol/L)]')
plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_imputed, test_size=0.1, random_state=42)
# Set up regression models and their hyperparameter grids for tuning
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'Random Forest Regressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10]
        }
    },
    'Decision Tree Regressor': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [None, 5, 10]
        }
    },
    'XGBoost Regressor': {
        'model': XGBRegressor(),
        'params': {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [3, 5]
        }
    },
    'KNN Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7]
        }
    },
    'Gaussian Process Regressor': {
        'model': GaussianProcessRegressor(),
        'params': {}
    },
    'Bayesian Ridge Regressor': {
        'model': BayesianRidge(),
        'params': {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4]
        }
    },

}


# Function to perform GridSearchCV and return the best model and parameters
def find_best_model(model, params):
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_final, y_imputed)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params


# Cross-validation for each model
for model_name, model_info in models.items():
    best_model, best_params = find_best_model(model_info['model'], model_info['params'])
    print(f"Best hyperparameters for {model_name}: {best_params}")

    # Perform cross-validation on the entire dataset
    cv_scores = cross_val_score(best_model, X_final, y_imputed, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)

    print(f"{model_name} - Cross-Validation RMSE: {cv_rmse_scores.mean()}, Cross-Validation R2: {cv_scores.mean()}")
    print("-----------------------------------")

    # Train the best model on the entire dataset
    best_model.fit(X_final, y_imputed)

    # Make predictions on the testing data
    y_pred = best_model.predict(X_final)

    # Evaluate the model's performance
    mse = mean_squared_error(y_imputed, y_pred)
    mae = mean_absolute_error(y_imputed, y_pred)
    r2 = r2_score(y_imputed, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_imputed) - 1) / (len(y_imputed) - X_final.shape[1] - 1)

    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R-squared: {r2}, Adjusted R-squared: {adjusted_r2}")
    print("-----------------------------------")
    # Train the best models on the entire dataset
    best_models = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=None),
        'XGBoost Regressor': XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=5),
        'Gaussian Process Regressor': GaussianProcessRegressor()
    }

    for model_name, model in best_models.items():
        model.fit(X_final, y_imputed)

    # Make predictions using each individual model
    predictions = {}
    for model_name, model in best_models.items():
        predictions[model_name] = model.predict(X_final)

    # Take the mean of the individual model predictions for ensemble
    ensemble_prediction = np.mean(list(predictions.values()), axis=0)

    # Evaluate the ensemble performance
    ensemble_mse = mean_squared_error(y_imputed, ensemble_prediction)
    ensemble_r2 = r2_score(y_imputed, ensemble_prediction)

    print(f"Averaging Ensemble - MSE: {ensemble_mse}, R-squared: {ensemble_r2}")
# Assuming X_final and y_imputed are available and the Gaussian Process Regressor model is already trained

# Train the Gaussian Process Regressor on the entire dataset
best_gaussian_process_model = GaussianProcessRegressor()
best_gaussian_process_model.fit(X_final, y_imputed)

# Save the trained model to a file
model_filename = 'gaussian_process_model.pkl'
joblib.dump(best_gaussian_process_model, model_filename)

# Function to make predictions using the trained model
def predict_fish_toxicity(features):
    # Load the saved model
    loaded_model = joblib.load(model_filename)

    # Impute missing values using IterativeImputer
    features_imputed = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(features))

    # Winsorize outliers
    features_winsorized = features_imputed.apply(lambda x: winsorize(x, limits=[0.1, 0.1]))

    # Apply Box-Cox transformation
    for column in features_winsorized.columns:
        features_winsorized[column], _ = boxcox(features_winsorized[column] + 0.001)

    # Scale the data using StandardScaler
    features_scaled = scaler.transform(features_winsorized)

    # One-hot encode the categorical features: "CIC0" and "SM1_Dz(Z)"
    features_encoded = pd.DataFrame(encoder.transform(features[['CIC0', 'SM1_Dz(Z)']]))
    features_encoded.columns = encoder.get_feature_names_out(['CIC0', 'SM1_Dz(Z)'])

    # Concatenate the one-hot encoded categorical columns with the scaled features
    features_final = pd.concat([features_encoded, pd.DataFrame(features_scaled, columns=winsorized_df.columns)], axis=1)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(features_final)
    return predictions
     # Enter the compound name , and get it to predict as to how toxic it is for the fish

