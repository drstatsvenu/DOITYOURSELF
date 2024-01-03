import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title('Interactive Chart Builder')

# File Upload
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Variable Selection
    continuous_vars = st.multiselect('Select Continuous Variable(s)', df.select_dtypes(include=['float64', 'int64']).columns.tolist())
    categorical_vars = st.multiselect('Select Categorical Variable(s)', df.select_dtypes(include=['object']).columns.tolist())
    
    # Data Visualization
    
    ## Continuous Variable
    if len(continuous_vars) == 1:
        st.subheader('Univariate Continuous Summary')
        var_cont = continuous_vars[0]
        
        # Histogram
        st.subheader(f'Histogram of {var_cont}')
        fig, ax = plt.subplots()
        sns.histplot(df[var_cont], kde=False, bins=30, ax=ax)
        st.pyplot(fig)
        
        # Density Plot
        st.subheader(f'Density Plot of {var_cont}')
        fig, ax = plt.subplots()
        sns.kdeplot(df[var_cont], fill=True, ax=ax)
        st.pyplot(fig)
        
        # Boxplot
        st.subheader(f'Box Plot of {var_cont}')
        fig, ax = plt.subplots()
        sns.boxplot(x=df[var_cont], ax=ax)
        st.pyplot(fig)
        
    elif len(continuous_vars) == 2:
        st.subheader('Bivariate Continuous Summary')
        var_cont1, var_cont2 = continuous_vars
        
        # Scatter Plot
        st.subheader(f'Scatter Plot of {var_cont1} and {var_cont2}')
        fig, ax = plt.subplots()
        sns.scatterplot(x=var_cont1, y=var_cont2, data=df, ax=ax)
        st.pyplot(fig)
        
        # Line Plot
        st.subheader(f'Line Plot of {var_cont1} and {var_cont2}')
        fig, ax = plt.subplots()
        sns.lineplot(x=var_cont1, y=var_cont2, data=df, ax=ax)
        st.pyplot(fig)
        
    ## Categorical Variable
    if len(categorical_vars) == 1:
        st.subheader('Univariate Categorical Summary')
        var_cat = categorical_vars[0]
        
        # Pie Chart
        st.subheader(f'Pie Chart of {var_cat}')
        fig, ax = plt.subplots()
        df[var_cat].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
        
        # Bar Plot
        st.subheader(f'Bar Plot of {var_cat}')
        fig, ax = plt.subplots()
        sns.countplot(y=df[var_cat], ax=ax)
        st.pyplot(fig)
        
    elif len(categorical_vars) == 2:
        st.subheader('Bivariate Categorical Summary')
        var_cat1, var_cat2 = categorical_vars
        
        # Horizontal Stacked Bar Chart
        st.subheader(f'Horizontal Stacked Bar Chart of {var_cat1} and {var_cat2}')
        crosstab = pd.crosstab(df[var_cat1], df[var_cat2])
        fig, ax = plt.subplots()
        crosstab.plot(kind='barh', stacked=True, ax=ax)
        st.pyplot(fig)
        
        # Vertical Stacked Bar Chart
        st.subheader(f'Vertical Stacked Bar Chart of {var_cat1} and {var_cat2}')
        fig, ax = plt.subplots()
        crosstab.plot(kind='bar', stacked=True, ax=ax)
        st.pyplot(fig)
    if len(continuous_vars) == 1 and len(categorical_vars) == 1:
        st.subheader('Mixed Variable Summary')
        var_cont = continuous_vars[0]
        var_cat = categorical_vars[0]
        
        # Box Plot
        st.subheader(f'Box Plot of {var_cont} by {var_cat}')
        fig, ax = plt.subplots()
        sns.boxplot(x=var_cat, y=var_cont, data=df, ax=ax)
        st.pyplot(fig)
    if len(continuous_vars) == 1 and len(categorical_vars) == 2:
        st.subheader('Mixed Variable Summary (1 Continuous, 2 Categorical)')
        var_cont = continuous_vars[0]
        var_cat1, var_cat2 = categorical_vars
        
        # Grouped Box Plot
        st.subheader(f'Box Plot of {var_cont} by {var_cat1} and {var_cat2}')
        fig, ax = plt.subplots()
        sns.boxplot(x=var_cat1, y=var_cont, hue=var_cat2, data=df, ax=ax)
        st.pyplot(fig)          
    if len(continuous_vars) == 2 and len(categorical_vars) == 1:
        st.subheader('Mixed Variable Summary (2 Continuous, 1 Categorical)')
        var_cont1, var_cont2 = continuous_vars
        var_cat = categorical_vars[0]
        
        # Scatter Plot
        st.subheader(f'Scatter Plot of {var_cont1} and {var_cont2} colored by {var_cat}')
        fig, ax = plt.subplots()
        sns.scatterplot(x=var_cont1, y=var_cont2, hue=var_cat, data=df, ax=ax)
        st.pyplot(fig)
 ## Mixed Variable (2 Continuous, 2 Categorical)
    if len(continuous_vars) == 2 and len(categorical_vars) == 2:
        st.subheader('Mixed Variable Summary (2 Continuous, 2 Categorical)')
        var_cont1, var_cont2 = continuous_vars
        var_cat1, var_cat2 = categorical_vars
        
        # Relational Plot
        st.subheader(f'Relational Plot of {var_cont1} and {var_cont2} colored by {var_cat1} and styled by {var_cat2}')
        fig = sns.relplot(x=var_cont1, y=var_cont2, hue=var_cat1, style=var_cat2, data=df)
        st.pyplot(fig)
    ## Correlation Plot (More than 2 Continuous)
    if len(continuous_vars) > 2 and len(categorical_vars) == 1:
        st.subheader('Correlation Plot')
        
        # Calculating Correlation Matrix
        corr_matrix = df[continuous_vars].corr()

        # Plotting a Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
## Treemap (More than 2 Categorical)
    if len(categorical_vars) > 2 and len(continuous_vars) == 1:
        st.subheader('Treemap for Multiple Categorical Variables')
        
        fig = px.treemap(df, path=categorical_vars)
        st.plotly_chart(fig)
        
          

if __name__ == "__main__":
    main()        
