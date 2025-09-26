#%% md
# # Instruction to lab work # 1
# 
# > Student name    - Volodymyr
# 
# > Student surname - Donets
# 
# > Group           - KU-31
# 
#%% md
# 
#%% md
# # Task description
# 
# ### Overall work description
# 
# The work will contain 2 main parts:
# 1. In the first part you'll build intuition behind using various regression models using only artificially generated (with noise & outliers) data. I've generated 2 options with `f(x) = A*sin(x) + B` and `f(x) = a*x + b` functions. (you free to add more complex functions (even more dimensional).
# 2. In the second part you'll solve regression problem on the real data.
# 
# Do NOT forget to change your `STUDENT_NO` -- this variable defines random state (it just makes experimental data for every one slighty different).
# 
# Do NOT forget to set you contact data in the top cell.
# 
# 
# ### I. Experiments on artificially generated data:
# 
# 1. Manually tune linear model by changing `a` and `b` weights of it. Observe position of the line and values of loss function.
# 2. **More advanced task** (+2 point): modify my code to work with more complex functions as regression models (linear, polynomial, sin, cos, and their combination). Feel free to use as complex function as you could found. Generate some random non-linear data and use your function to manually adjust its weights.
# 3. Play with default regression methods from `sklearn` library on non-linear data. Search, llm-prompt information on each model to understand how to make the model fit the data without overfitting.
# 
# 
# ### II. Experiments on real data. (you can keep it in this notebook or in a separate one).
# 
# 1. Choose any DataSet you like for you experiments (if you've chosen the same, consider that work must differ, otherwise both students will get 0 points for work).
# 2. Choose the top-3 methods (from `sklearn` library, or you can use other libraries (like `xgboost`) from the previous part.
# 2. Solve the regression problem in the same manner, as you used for.
#     1. Load the data
#     2. Do data visualization: correlation, feature distribution, etc.
#     3. Do data analysis
#     4. Do data correction
#     5. Prepare data on usage with ML model: train, validation if necessary, test split; data convertion (to fix distribution or change data type to numeric); remove outliers.
#     6. Tune hyperparameters of your model to get the best one.
#     7. Train & test the final version of the model. Do conclusion.
# 3. Your main goal is tune hyperparameters of the chosen models.
# 4. Examples and template you can find in `ML_basic_course/lab_works/lab2/lab_2_example_plus_task.ipynb`. Or in [my GitHub repo's folder](https://github.com/VolDonets/ML_basics_course/tree/master/lab_works/lab_2)
# 5. Use that notebook as template, but remember your main goal is to tune hyperparameters of chosen models.
# 
# ### III. Important.
# 
# 1. Students, who solved the problem in a single code cell will get 0 points for your work. It's hard to work with your messy code, we have limited time. Use this notebook as your template, create as many cells as you need. Also you can conduct experiments in .py files but in that case prepare normal report.
# 2. Experiments means you have multiple cells with EXPERIMENTS and your CONCLUSION after that.
# 3. You can have multiple notebooks if you need, but name it correctly and add `ReadMe.md`.
# 
#%% md
# ## Proposition of the real data for experiments
# 
# 0. Your own data (the main idea here that target value is a number, not a class)
# 1. [Taxi Price Regression 🚕](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction)
# 2. [Car Pricing Regression Dataset](https://www.kaggle.com/datasets/amjadzhour/car-price-prediction)
# 3. [Second Hand Car Price Prediction
# ](https://www.kaggle.com/datasets/sujithmandala/second-hand-car-price-prediction)
# 4. [Indian Rental House Price](https://www.kaggle.com/datasets/bhavyadhingra00020/india-rental-house-price)
# 5. *** [Google Stock Prediction](https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction) -- this is stock data, so probably for prediction you'll need the previous values (for previous days, or day).
# 6. [Forest Fire Regression](https://www.kaggle.com/datasets/nimapourmoradi/forest-fire-regression)
# 7. [California House Price](https://www.kaggle.com/datasets/shibumohapatra/house-price)
# 8. *** [Asteroid Dataset](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset) -- this is hard problem, try to predict any continuous feature (like asteroid radius).
# 9. [Salary Prediction](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor)
# 10. [January Flight Delay Prediction](https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction)
# 11. [Restaurants Revenue Prediction](https://www.kaggle.com/datasets/mrsimple07/restaurants-revenue-prediction)
# 12. [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
# 13. [Furniture Price Prediction](https://www.kaggle.com/datasets/shawkyelgendy/furniture-price-prediction)
#%%
STUDENT_NO = 23
#%% md
# # Import dependencies
#%%
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
#%% md
# # Helping functions
#%%
def generate_regression_data(
    n_samples=50, 
    n_features=1, 
    mode='linear', 
    noise=2.0, 
    n_outliers=5, 
    random_seed=42, 
    return_coef=False
):
    """
    Generates data for regression tasks, compatible with scikit-learn.

    :param n_samples: The total number of data points to generate.
    :param n_features: The number of features for the dataset.
    :param mode: The underlying pattern of the data. Can be 'linear' or 'nonlinear'.
    :param noise: The standard deviation of Gaussian noise added to the y-values.
    :param n_outliers: The number of outlier points to add to the dataset.
    :param random_seed: A seed for the random number generator for reproducibility.
    :param return_coef: If True, also returns the true coefficients of the generative model.
    :returns: A tuple (X, y) or (X, y, coefficients) if return_coef is True.
              X is the feature matrix of shape (n_samples, n_features).
              y is the target vector of shape (n_samples,).
              coefficients is a dictionary with 'weights' and 'intercept'.
    """
    
    np.random.seed(random_seed)

    X = np.random.rand(n_samples, n_features) * 10 - 5

    true_intercept = 0
    true_weights = np.zeros(n_features)
    
    if mode == 'linear':
        true_intercept = np.random.uniform(-3, 3)
        true_weights = np.random.uniform(-5, 5, size=n_features)
        y_true = np.dot(X, true_weights) + true_intercept
    elif mode == 'nonlinear':
        if n_features < 1:
            raise ValueError("Nonlinear mode requires at least 1 feature.")
        
        true_intercept = np.random.uniform(-15, 15)
        true_weights = np.random.uniform(-5, 5, size=n_features)
        
        y_true = true_weights[0] * 20 * np.sin(X[:, 0]*1.5) + true_intercept
        
        if n_features > 1:
             y_true += np.dot(X[:, 1:], true_weights[1:])
    else:
        raise ValueError("Mode must be either 'linear' or 'nonlinear'")

    y = y_true + np.random.normal(scale=noise, size=n_samples)

    if n_outliers > 0:
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        outlier_offset = (np.random.rand(n_outliers) - 0.5) * 30 * (noise + 1)
        y[outlier_indices] += outlier_offset

    if return_coef:
        coefficients = {'weights': true_weights, 'intercept': true_intercept}
        return X, y, coefficients
    else:
        return X, y
#%%
def plot_regression_model(
    X, 
    y, 
    weights=None,
    intercept=None, 
    title="Regression Model Visualization",
):
    """
    Visualizes regression data and model performance.

    If the data has one feature, it plots the data points and the regression line.
    If the data has multiple features, it plots the model's predicted vs. actual values.

    :param X: The feature matrix, shape (n_samples, n_features).
    :param y: The target vector, shape (n_samples,).
    :param weights: The model's feature weights (coefficients).
    :param intercept: The model's intercept.
    :param title: The title for the plot.
    :returns: None. Displays a matplotlib plot.
    """
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    model_provided = weights is not None and intercept is not None

    if n_features == 1:
        ax.scatter(X, y, c='cornflowerblue', alpha=0.7, edgecolors='k', label='Data Points')
        
        if model_provided:
            x_line = np.linspace(X.min(), X.max(), 200)
            y_line = x_line * weights[0] + intercept
            label = f'Model: y = {weights[0]:.2f}x + {intercept:.2f}'
            ax.plot(x_line, y_line, color='crimson', linewidth=2.5, label=label)

        ax.set_xlabel("Feature (X)", fontsize=12)
        ax.set_ylabel("Target (y)", fontsize=12)

    else: # n_features > 1
        if not model_provided:
            print("Cannot visualize raw multi-feature data. Please provide model weights and intercept to generate a plot.")
            plt.close(fig)
            return

        y_pred = np.dot(X, weights) + intercept
        
        ax.scatter(y, y_pred, c='cornflowerblue', alpha=0.7, edgecolors='k')
        
        perfect_fit_line = np.linspace(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()), 100)
        ax.plot(perfect_fit_line, perfect_fit_line, color='crimson', linestyle='--', linewidth=2.5, label='Perfect Fit (y_pred = y_true)')
        
        ax.set_xlabel("Actual Values (y_true)", fontsize=12)
        ax.set_ylabel("Predicted Values (y_pred)", fontsize=12)
        ax.legend(fontsize=11)

    ax.set_title(title, fontsize=14, weight='bold')
    plt.show()
#%%
def plot_sklearn_regression(model, X, y, title="Model Performance", step=0.01):
    """
    Visualizes the performance of a trained scikit-learn regression model.

    If X has one feature, it plots the data and the model's line point-by-point.
    If X has multiple features, it plots the model's predicted vs. actual values.

    :param model: A trained scikit-learn regressor object (e.g., LinearRegression, Ridge).
    :param X: The feature matrix, shape (n_samples, n_features).
    :param y: The true target vector, shape (n_samples,).
    :param title: The title for the plot.
    :param step: The precision delta for drawing the model's line in the 1D case.
    :returns: None. Displays a matplotlib plot.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    if n_features == 1:
        ax.scatter(X, y, c='cornflowerblue', alpha=0.6, edgecolors='k', label='Actual Data')
        
        # --- Updated Section ---
        x_min, x_max = X.min(), X.max()
        padding = (x_max - x_min) * 0.05
        
        # Create X-records point-by-point using the specified step/precision
        x_line = np.arange(x_min - padding, x_max + padding, step).reshape(-1, 1)
        
        # Get the corresponding y-values from the model
        y_line = model.predict(x_line)
        # --- End Updated Section ---
        
        ax.plot(x_line, y_line, color='crimson', linewidth=1.5, label='Model Prediction Line')
        ax.set_xlabel("Feature (X)", fontsize=12)
        ax.set_ylabel("Target (y)", fontsize=12)

    else: # Multi-feature case
        y_pred = model.predict(X)
        
        ax.scatter(y, y_pred, c='cornflowerblue', alpha=0.6, edgecolors='k')
        
        perfect_fit_line = np.linspace(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()), 100)
        ax.plot(perfect_fit_line, perfect_fit_line, color='crimson', linestyle='--', linewidth=2, label='Perfect Fit')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)

    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=14, weight='bold')
    plt.show()
#%%
def evaluate_regression_model(y_true, X, model, print_results=True):
    """
    Calculates and optionally prints metrics for a given regression model.

    This function can accept a scikit-learn model, a np.poly1d object,
    or a list of coefficients.

    :param y_true: The actual target values.
    :param X: The feature matrix.
    :param model: A trained sklearn model, a np.poly1d object, or a list of coefficients.
    :param print_results: If True, prints the metrics to the console.
    :returns: A dictionary containing the calculated metrics.
    """
    
    y_pred = None
    
    # --- Determine prediction method based on model type ---
    if hasattr(model, 'predict'): # Handles scikit-learn models
        y_pred = model.predict(X)
    elif isinstance(model, (list, np.ndarray, np.poly1d)): # Handles weights and poly1d objects
        poly_model = np.poly1d(model)
        if X.ndim > 1 and X.shape[1] > 1:
            print("Warning: Manual weights are being applied to the first feature of X only.")
        y_pred = poly_model(X[:, 0] if X.ndim > 1 else X)
    else:
        raise TypeError("Model type not supported. Please provide a scikit-learn model or a list of weights.")

    # --- Calculate metrics ---
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

    if print_results:
        print("----- 📈 Model Evaluation -----")
        print(pd.Series(metrics).to_string(float_format="%.4f"))
        print("-----------------------------")

    return metrics
#%% md
# # Section 1: experiments on artificial data
# 
# **More advanced task** (+2 point): modify my code to work with more complex functions for regression (linear, polynomial, sin, cos, and their combination). Feel free to use as complex function as you could found.
#%% md
# ## 1.1. Tuning manual Linear Regression 
# 
# Just change coefficients of the Linear model and observe changes in loss function
#%%
# Generating data
X_lin, y_lin = generate_regression_data(
    n_samples=250, 
    n_features=1, 
    mode='linear', 
    noise=5.0, 
    n_outliers=7, 
    random_seed=STUDENT_NO, 
    return_coef=False
)
#%%
# Example of performing experiments
# f(x) = a*x + b
# ! For additional task !: f(x) = a*sin(b * log_2(x * c) + d) + e

lin_model_weights = [-5, -20]
evaluate_regression_model(X_lin, y_lin, model=lin_model_weights, print_results=True)
plot_regression_model(X_lin, y_lin, 
                      weights=[lin_model_weights[0], ], 
                      intercept=lin_model_weights[1],
                      title='Manual Linear Regression')
#%%

#%%

#%%

#%%

#%% md
# ## 1.2. Tuning Linear Regression on non-linear data
# 
# Just change coefficients of the Linear model and observe changes in loss function
#%%
# Generating data
X_nonlin, y_nonlin = generate_regression_data(
    n_samples=300, 
    n_features=1, 
    mode='nonlinear', 
    noise=5.0, 
    n_outliers=10, 
    random_seed=STUDENT_NO, 
    return_coef=False
)
#%%
# Example of performing experiments
lin_model_weights = [-5, -20]
evaluate_regression_model(X_nonlin, y_nonlin, model=lin_model_weights, print_results=True)
plot_regression_model(X_nonlin, y_nonlin, 
                      weights=[lin_model_weights[0], ], 
                      intercept=lin_model_weights[1],
                      title='Manual Linear Regression')
#%%

#%%

#%%

#%% md
# ## 1.3. Configuring various linear models from sklearn library
# 
# Try to configure hyperparameters of various linear regression models from sklearn
#%%
from sklearn.linear_model import LinearRegression, HuberRegressor, PoissonRegressor
#%% md
# ### 1.3.1. Playing with `HuberRegressor`
#%%
hub_model = HuberRegressor(
    epsilon = 1.35,
    max_iter = 100,
    alpha = 0.0001,
    warm_start = False,
    fit_intercept = True,
    tol = 1e-05,
)
#%%
hub_model.fit(X_lin, y_lin)

evaluate_regression_model(y_lin, X_lin, model=hub_model, print_results=True)
plot_sklearn_regression(hub_model, X_lin.reshape(-1, 1), y_lin, title="Huber Model Performance on Test Data")
#%%

#%%

#%%

#%% md
# ### 1.3.2. Playing with `PoissonRegressor`
#%%
pois_model = PoissonRegressor(
    alpha = 1.0,
    fit_intercept = True,
    solver = "lbfgs",
    max_iter = 100,
    tol= 1e-4,
    warm_start = True,
    verbose = 0
)
#%%
pois_model.fit(X_lin, y_lin - np.min(y_lin))

evaluate_regression_model(y_lin - np.min(y_lin), X_lin, model=pois_model, print_results=True)
plot_sklearn_regression(pois_model, X_lin.reshape(-1, 1), y_lin - np.min(y_lin), title="Huber Model Performance on Test Data")
#%%

#%%

#%%

#%%

#%%

#%% md
# ## 1.4. Configure non-linear models from sklearn library
#%%
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
#%% md
# ### 1.4.1. Playing with `KNeighborsRegressor`
#%%

#%%

#%% md
# ### 1.4.2. Playing with `RadiusNeighborsRegressor`
#%%

#%%

#%% md
# ### 1.4.3. Playing with `DecisionTreeRegressor`
#%%
dt_model = DecisionTreeRegressor(
     criterion = "squared_error",
     splitter = "best",
     max_depth = None,
     min_samples_split = 2,
     min_samples_leaf = 1,
     min_weight_fraction_leaf = 0.0,
     max_features = None,
     random_state = None,
     max_leaf_nodes = None,
     min_impurity_decrease = 0.0,
     ccp_alpha = 0.0,
     monotonic_cst = None
)
dt_model.fit(X_nonlin, y_nonlin)

evaluate_regression_model(y_nonlin, X_nonlin, model=dt_model, print_results=True)
plot_sklearn_regression(dt_model, X_nonlin, y_nonlin, title="Decision Tree Model Performance on Test Data")
#%%

#%%

#%%

#%%

#%%

#%% md
# ### 1.4.4. Playing with `AdaBoostRegressor`
#%%

#%%

#%% md
# ### 1.4.5. Playing with `RandomForestRegressor`
#%%

#%%

#%% md
# ### 1.4.6. Playing with `GradientBoostingRegressor`
#%%

#%%

#%% md
# ### 1.4.7. Playing with `VotingRegressor`
#%%

#%%

#%% md
# ### 1.4.8. Playing with `SVR (Support Vector Regression)`
#%%

#%%

#%% md
# # Section 2: experiments on real data
#%% md
# 1. Choose any DataSet you like for you experiments (if you've chosen the same, consider that work must differ, otherwise both students will get 0 points for work).
# 2. Choose the top-3 methods (from `sklearn` library, or you can use other libraries (like `xgboost`) from the previous part.
# 2. Solve the regression problem in the same manner, as you used for.
#     1. Load the data
#     2. Do data visualization: correlation, feature distribution, etc.
#     3. Do data analysis
#     4. Do data correction
#     5. Prepare data on usage with ML model: train, validation if necessary, test split; data convertion (to fix distribution or change data type to numeric); remove outliers.
#     6. Tune hyperparameters of your model to get the best one.
#     7. Train & test the final version of the model. Do conclusion.
# 3. Your main goal is tune hyperparameters of the chosen models.
# 4. Examples and template you can find in `ML_basic_course/lab_works/lab2/lab_2_example_plus_task.ipynb`. Or in [my GitHub repo's folder](https://github.com/VolDonets/ML_basics_course/tree/master/lab_works/lab_2)
# 5. Use that notebook as template, but remember your main goal is to tune hyperparameters of chosen models.
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
