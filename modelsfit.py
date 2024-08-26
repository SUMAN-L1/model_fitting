import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Function for Logistic Model
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k*(x-x0)))

# Function for Modified Exponential Model
def mod_exp(x, a, b, c):
    return a * np.exp(b * x) + c

# Function for Cobb-Douglas Model
def cobb_douglas(x, a, b):
    return a * (x ** b)

# Function to Fit Models and Get Results
def fit_model(X, y, model_type):
    X = sm.add_constant(X)
    if model_type == 'linear':
        model = sm.OLS(y, X).fit()
    elif model_type == 'quadratic':
        X = np.column_stack((X, X[:,1]**2))
        model = sm.OLS(y, X).fit()
    elif model_type == 'cubic':
        X = np.column_stack((X, X[:,1]**2, X[:,1]**3))
        model = sm.OLS(y, X).fit()
    elif model_type == 'quartic':
        X = np.column_stack((X, X[:,1]**2, X[:,1]**3, X[:,1]**4))
        model = sm.OLS(y, X).fit()
    elif model_type == 'exponential':
        X = X[:,1]
        model = np.polyfit(X, np.log(y), 1)
        y_pred = np.exp(model[1]) * np.exp(model[0] * X)
        r_squared = r2_score(y, y_pred)
        return {'model': model, 'y_pred': y_pred, 'r_squared': r_squared}
    elif model_type == 'logistic':
        popt, _ = curve_fit(logistic, X[:,1], y, p0=[max(y),1,np.median(X[:,1])])
        y_pred = logistic(X[:,1], *popt)
        r_squared = r2_score(y, y_pred)
        return {'model': popt, 'y_pred': y_pred, 'r_squared': r_squared}
    elif model_type == 'mod_exp':
        popt, _ = curve_fit(mod_exp, X[:,1], y, p0=[1,1,1])
        y_pred = mod_exp(X[:,1], *popt)
        r_squared = r2_score(y, y_pred)
        return {'model': popt, 'y_pred': y_pred, 'r_squared': r_squared}
    elif model_type == 'cobb_douglas':
        popt, _ = curve_fit(cobb_douglas, X[:,1], y, p0=[1,1])
        y_pred = cobb_douglas(X[:,1], *popt)
        r_squared = r2_score(y, y_pred)
        return {'model': popt, 'y_pred': y_pred, 'r_squared': r_squared}
    else:
        return None
    y_pred = model.predict(X)
    r_squared = model.rsquared
    return {'model': model, 'y_pred': y_pred, 'r_squared': r_squared}

# Function to Plot Actual vs Fitted
def plot_actual_vs_fitted(y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y, label="Actual")
    plt.plot(y_pred, label="Fitted/Estimated", linestyle="--")
    plt.title(title)
    plt.legend()
    st.pyplot()

# Load Data
st.title("Crop Production Modeling")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    X = df['Area'].values.reshape(-1, 1)
    y = df['Production'].values

    # Model Fitting and Plotting
    models = ['linear', 'quadratic', 'cubic', 'quartic', 'exponential', 'logistic', 'mod_exp', 'cobb_douglas']
    summary_data = []

    for model_type in models:
        result = fit_model(X, y, model_type)
        if result:
            model = result['model']
            y_pred = result['y_pred']
            r_squared = result['r_squared']

            # Get coefficients and p-values
            if model_type in ['linear', 'quadratic', 'cubic', 'quartic']:
                coefficients = model.params
                p_values = model.pvalues
            else:
                coefficients = model
                p_values = None  # Not applicable for non-linear models in this context

            # Plot
            plot_actual_vs_fitted(y, y_pred, f"Actual vs Fitted - {model_type.capitalize()} Model")

            # Summary Data
            summary_data.append({
                'Model': model_type.capitalize(),
                'Coefficients': coefficients,
                'R-Squared': r_squared,
                'P-Values': p_values
            })

    # Summary Table
    st.write("### Summary Table")
    summary_df = pd.DataFrame(summary_data)
    st.write(summary_df)

    # Interpretation
    st.write("### Interpretations")
    st.write("The model with the highest R-Squared value indicates the best fit. Review the coefficients and p-values for each model to determine statistical significance and practical implications.")
