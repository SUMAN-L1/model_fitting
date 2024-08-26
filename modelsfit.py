import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Function to load the data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            return df
    st.error("Please upload a valid .xls or .xlsx file.")
    return None

# Function to fit models and return metrics
def fit_models(df, x_col, y_col):
    results = {}
    
    # Linear model
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    results['Linear'] = model.summary2().tables[1]
    
    # Quadratic model
    df['x2'] = df[x_col] ** 2
    X_quad = sm.add_constant(df[['x_col', 'x2']])
    model = sm.OLS(df[y_col], X_quad).fit()
    results['Quadratic'] = model.summary2().tables[1]
    
    # Cubic model
    df['x3'] = df[x_col] ** 3
    X_cubic = sm.add_constant(df[['x_col', 'x2', 'x3']])
    model = sm.OLS(df[y_col], X_cubic).fit()
    results['Cubic'] = model.summary2().tables[1]
    
    # Exponential model
    df['log_y'] = np.log(df[y_col])
    model = sm.OLS(df['log_y'], sm.add_constant(df[x_col])).fit()
    results['Exponential'] = model.summary2().tables[1]
    
    # Logistic model
    df['logit_y'] = np.log(df[y_col] / (1 - df[y_col]))
    model = sm.OLS(df['logit_y'], sm.add_constant(df[x_col])).fit()
    results['Logistic'] = model.summary2().tables[1]
    
    # Quartic model
    df['x4'] = df[x_col] ** 4
    X_quartic = sm.add_constant(df[['x_col', 'x2', 'x3', 'x4']])
    model = sm.OLS(df[y_col], X_quartic).fit()
    results['Quartic'] = model.summary2().tables[1]
    
    # Modified Exponential model
    model = smf.ols(f"{y_col} ~ np.exp({x_col})", data=df).fit()
    results['Modified Exponential'] = model.summary2().tables[1]
    
    # Cobb-Douglas model
    df['log_x'] = np.log(df[x_col])
    model = smf.ols(f"{y_col} ~ log_x", data=df).fit()
    results['Cobb-Douglas'] = model.summary2().tables[1]
    
    return results

# Function to plot model results
def plot_model_results(df, x_col, y_col, model_name, predictions):
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x_col], df[y_col], label='Actual Data')
    plt.plot(df[x_col], predictions, color='red', label='Fitted Data')
    plt.title(f'{model_name} Model')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    st.pyplot(plt)

# Streamlit app interface
def main():
    st.title("Crop Area and Production Regression Analysis")
    
    uploaded_file = st.file_uploader("Upload your .xls or .xlsx file", type=['xls', 'xlsx'])
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data Preview:")
            st.write(df.head())
            
            x_col = 'Area'  # Fixed to 'Area'
            y_col = 'Production'  # Fixed to 'Production'
            
            if x_col in df.columns and y_col in df.columns:
                results = fit_models(df, x_col, y_col)
                
                # Display model results and plots
                for model_name, summary in results.items():
                    st.write(f"**{model_name} Model**")
                    st.write(summary)
                    
                    # Plot fitted vs actual data
                    if model_name in ['Linear', 'Quadratic', 'Cubic', 'Quartic']:
                        X = sm.add_constant(df[x_col])
                        if model_name == 'Linear':
                            predictions = sm.OLS(df[y_col], X).fit().predict(X)
                        elif model_name == 'Quadratic':
                            predictions = sm.OLS(df[y_col], sm.add_constant(df[['x_col', 'x2']])).fit().predict(sm.add_constant(df[['x_col', 'x2']]))
                        elif model_name == 'Cubic':
                            predictions = sm.OLS(df[y_col], sm.add_constant(df[['x_col', 'x2', 'x3']])).fit().predict(sm.add_constant(df[['x_col', 'x2', 'x3']]))
                        elif model_name == 'Quartic':
                            predictions = sm.OLS(df[y_col], sm.add_constant(df[['x_col', 'x2', 'x3', 'x4']])).fit().predict(sm.add_constant(df[['x_col', 'x2', 'x3', 'x4']]))
                        plot_model_results(df, x_col, y_col, model_name, predictions)
                    elif model_name == 'Exponential':
                        predictions = np.exp(sm.OLS(df['log_y'], sm.add_constant(df[x_col])).fit().predict(sm.add_constant(df[x_col])))
                        plot_model_results(df, x_col, y_col, model_name, predictions)
                    elif model_name == 'Logistic':
                        predictions = model.predict()
                        plot_model_results(df, x_col, y_col, model_name, predictions)
                    elif model_name == 'Modified Exponential':
                        predictions = smf.ols(f"{y_col} ~ np.exp({x_col})", data=df).fit().predict()
                        plot_model_results(df, x_col, y_col, model_name, predictions)
                    elif model_name == 'Cobb-Douglas':
                        predictions = smf.ols(f"{y_col} ~ log_x", data=df).fit().predict()
                        plot_model_results(df, x_col, y_col, model_name, predictions)
                
                # Summary table
                summary_df = pd.DataFrame({
                    "Model": [],
                    "Coefficients": [],
                    "R-Squared": [],
                    "P-Values": [],
                    "Interpretation": []
                })
                
                for model_name, summary in results.items():
                    coeffs = summary[summary['index'].str.contains('const') | summary['index'].str.contains(x_col)]
                    coeffs_str = ', '.join(f"{row['index']}: {row['Coef.']:.3f}" for _, row in coeffs.iterrows())
                    
                    r_squared = summary['R-squared'].values[0] if 'R-squared' in summary.columns else 'N/A'
                    p_values = ', '.join(f"{row['index']}: {row['P>|t|']:.3f}" for _, row in coeffs.iterrows())
                    
                    interpretation = f"Model {model_name} with R-squared = {r_squared}"
                    summary_df = summary_df.append({
                        "Model": model_name,
                        "Coefficients": coeffs_str,
                        "R-Squared": r_squared,
                        "P-Values": p_values,
                        "Interpretation": interpretation
                    }, ignore_index=True)
                
                st.write("**Model Summary Table**")
                st.write(summary_df)
                
if __name__ == "__main__":
    main()
