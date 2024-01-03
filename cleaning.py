import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import os

# Function to load and display the uploaded data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to handle univariate summary for continuous variables
def univariate_continuous_summary(data, continuous_var):
    summary = data[continuous_var].describe()
    missing_values = data[continuous_var].isnull().sum()
    outliers = detect_outliers(data[continuous_var])
    
    st.write(f"**Summary for {continuous_var}**")
    st.write(f"Count: {summary['count']}")
    st.write(f"Mean: {summary['mean']}")
    st.write(f"Standard Deviation: {summary['std']}")
    st.write(f"Standard Error: {stats.sem(data[continuous_var])}")
    st.write(f"Confidence Interval (95%): {confidence_interval(data[continuous_var])}")
    st.write(f"Number of Missing Values: {missing_values}")
    st.write(f"Outlier Range: {outliers}")
    st.write(f"Number of Outliers: {len(outliers)}")
    st.write(f"Outlier IDs: {data.index[data[continuous_var].isin(outliers)].tolist()}")

# Function to detect outliers
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return data[(data < lower_limit) | (data > upper_limit)]

# Function to calculate confidence interval
def confidence_interval(data):
    z = 1.96  # 95% confidence interval
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    margin_of_error = z * (std / np.sqrt(n))
    return f"({mean - margin_of_error}, {mean + margin_of_error})"

# Function to handle outlier treatment
def handle_outliers(data, continuous_var):
    option = st.radio("Do you want to handle outliers?", ('Yes', 'No'))
    if option == 'Yes':
        method = st.radio("Do you want to exclude or cap outliers?", ('Exclude', 'Cap'))
        if method == 'Exclude':
            data = data[~data[continuous_var].isin(detect_outliers(data[continuous_var]))]
        elif method == 'Cap':
            lower_limit = np.percentile(data[continuous_var], 5)
            upper_limit = np.percentile(data[continuous_var], 95)
            data[continuous_var] = np.where(data[continuous_var] < lower_limit, lower_limit, data[continuous_var])
            data[continuous_var] = np.where(data[continuous_var] > upper_limit, upper_limit, data[continuous_var])
    return data

# Function to handle missing value treatment
def handle_missing_values(data, continuous_var):
    option = st.radio("Do you want to treat missing values?", ('Yes', 'No'))
    if option == 'Yes':
        method = st.radio("How do you want to handle missing values?", ('Mean', 'Median', 'Specify Value'))
        if method == 'Mean':
            data[continuous_var].fillna(data[continuous_var].mean(), inplace=True)
        elif method == 'Median':
            data[continuous_var].fillna(data[continuous_var].median(), inplace=True)
        elif method == 'Specify Value':
            value = st.number_input("Enter the value to replace missing values", value=0.0)
            data[continuous_var].fillna(value, inplace=True)
    return data

# Function to create bins for continuous variables
def create_bins(data, continuous_var):
    option = st.radio("Do you want to create bins for this continuous variable?", ('Yes', 'No'))
    if option == 'Yes':
        num_bins = st.number_input("How many bins do you want?", min_value=2)
        st.write("Enter bin definitions (comma-separated) and labels (comma-separated) in the format: bin1,bin2,... and label1,label2,...")
        bin_definitions_labels = st.text_input("Bin Definitions and Labels", '')
        
        bin_definitions_labels = bin_definitions_labels.split(" and ")
        
        if len(bin_definitions_labels) != 2:
            st.warning("Please enter both bin definitions and labels.")
            return data
        
        bin_definitions = bin_definitions_labels[0].split(',')
        bin_labels = bin_definitions_labels[1].split(',')
        
        if len(bin_definitions) != num_bins + 1 or len(bin_labels) != num_bins:
            st.warning("Please provide the correct number of bin definitions and labels.")
            return data

        bin_edges = [float(x.strip()) for x in bin_definitions]
        labels = [x.strip() for x in bin_labels]
        data[f'{continuous_var}_binned'] = pd.cut(data[continuous_var], bins=bin_edges, labels=labels)
    return data



# Function to handle categorical variable summary
def categorical_summary(data, categorical_var):
    summary = data[categorical_var].value_counts(normalize=True)
    st.write(f"**Summary for {categorical_var}**")
    st.write(summary)
    st.write("Categories with less than 5%:")
    st.write(summary[summary < 0.05])
    st.write(f"Category IDs with less than 5%: {data.index[data[categorical_var].isin(summary[summary < 0.05].index)].tolist()}")

# Function to download the revised data
def download_revised_data(data):
    csv_file = data.to_csv(index=False)
    
    # Create a temporary file name
    tmp_file_name = "revised_data.csv"
    
    # Write the data to the temporary file
    with open(tmp_file_name, "w") as tmp_file:
        tmp_file.write(csv_file)
    
    # Provide a download link to the temporary file
    st.markdown(f'<a href="{tmp_file_name}" download="{tmp_file_name}">Download Revised Data</a>', unsafe_allow_html=True)
# Main function
def main():
    st.title("Interactive Data Cleaning Pipeline")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        st.sidebar.header("Select Options")
        continuous_var = st.sidebar.selectbox("Select Continuous Variable", data.columns)
        
        univariate_continuous_summary(data, continuous_var)
        
        data = handle_outliers(data, continuous_var)
        st.write("**Data after handling outliers**")
        st.write(data)
        
        data = handle_missing_values(data, continuous_var)
        st.write("**Data after handling missing values**")
        st.write(data)
        
        data = create_bins(data, continuous_var)
        st.write("**Data after creating bins**")
        st.write(data)
        
        categorical_var = st.sidebar.selectbox("Select Categorical Variable", data.select_dtypes(include=['object']).columns)
        categorical_summary(data, categorical_var)

        download_revised_data(data)  # Call the function to download the revised data


          

if __name__ == "__main__":
    main()
