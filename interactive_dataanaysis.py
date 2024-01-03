import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

st.title("Medeva Interactive Data Analysis App")

# 1. File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # 2. Variable Selection
    continuous_vars = st.multiselect("Select Continuous Variables", df.columns)
    categorical_vars = st.multiselect("Select Categorical Variables", df.columns)
    stratification_vars = st.multiselect("Select Stratification Variables", df.columns)
    
    # 3. Univariate Analysis
    if st.button("Show Univariate Analysis for Continuous Variables"):
        st.write(df[continuous_vars].describe())
        
        # Visualization: Histograms & Boxplots for Continuous Vars
        for var in continuous_vars:
            st.subheader(f"Histogram and Boxplot for {var}")
            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            sns.histplot(data=df, x=var, kde=True, ax=axs[0])
            sns.boxplot(x=df[var], ax=axs[1])
            st.pyplot(fig)
            plt.clf()  # Clear the current figure
    
    # 4. Bivariate Analysis
    if st.button("Show Bivariate Analysis"):
        for strat_var in stratification_vars:
            st.subheader(f"Bivariate Analysis stratified by {strat_var}")
            
            # Continuous Variables
            for cont_var in continuous_vars:
                st.write(f"Summary Statistics for {cont_var} by {strat_var}")
                groupby_data = df.groupby(strat_var)[cont_var].agg(['count', 'mean', 'std'])
                groupby_data['SE'] = groupby_data['std'] / np.sqrt(groupby_data['count'])
                st.write(groupby_data)
                
            # Categorical Variables
            for cat_var in categorical_vars:
                st.write(f"Cross-Tab for {cat_var} by {strat_var}")
                cross_tab = pd.crosstab(df[strat_var], df[cat_var], margins=True, margins_name="Total")
                st.write(cross_tab)
                
                st.write(f"Bar Chart of {cat_var} by {strat_var}")
                plt.figure(figsize=(10, 6))
                sns.countplot(x=df[cat_var], hue=df[strat_var])
                plt.xticks(rotation=45)
                st.pyplot()
                plt.clf()
    # 6. Normality Test and Q-Q Plot
    if st.button("Perform Normality Test"):
        # Allow user to select a variable for normality test
        test_var = st.selectbox("Select a Variable for Normality Test", continuous_vars)
        
        # Perform Shapiro-Wilk test for normality
        from scipy.stats import shapiro, probplot
        
        shapiro_stat, shapiro_p_value = shapiro(df[test_var].dropna())
        
        # Display results
        st.subheader("Shapiro-Wilk Normality Test Results")
        st.write(f"Statistic: {shapiro_stat}, p-value: {shapiro_p_value}")
        
        # Inference
        if shapiro_p_value > 0.05:
            st.write(f"The p-value is {shapiro_p_value:.2f}, which is greater than 0.05, suggesting the data is normally distributed.")
        else:
            st.write(f"The p-value is {shapiro_p_value:.2f}, which is less than or equal to 0.05, suggesting the data is not normally distributed.")
        
        # Q-Q Plot
        st.subheader(f"Q-Q Plot for {test_var}")
        fig, ax = plt.subplots(figsize=(8, 8))
        probplot(df[test_var].dropna(), plot=ax)
        st.pyplot(fig)
    # 7. Statistical Inference for Continuous Variables
    if st.button("Perform Advanced Statistical Inference"):
        # Allow user to select variables for inference
        inference_var = st.selectbox("Select a Continuous Variable for Advanced Inference", continuous_vars)
        stratification_var = st.selectbox("Select a Stratification Variable", stratification_vars)
        
        from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Check normality
        _, p_normal = shapiro(df[inference_var].dropna())
        is_normal = p_normal > 0.05
        
        # Check number of levels in stratification var
        n_levels = df[stratification_var].nunique()
        
        # Two-level Stratification Variable
        if n_levels == 2:
            group1, group2 = df[stratification_var].unique()
            data_group1 = df[df[stratification_var] == group1][inference_var].dropna()
            data_group2 = df[df[stratification_var] == group2][inference_var].dropna()
            
            # Perform t-test or Mann-Whitney U test
            if is_normal:
                stat, p_value = ttest_ind(data_group1, data_group2)
                test_name = 'T-Test'
            else:
                stat, p_value = mannwhitneyu(data_group1, data_group2)
                test_name = 'Mann-Whitney U Test'
            
            # Display results and inference
            st.write(f"{test_name} Results: Statistic = {stat}, p-value = {p_value}")
            
        # More than Two-level Stratification Variable
        elif n_levels > 2:
            # Get data per group
            groups = [df[df[stratification_var] == group][inference_var].dropna().to_numpy() for group in df[stratification_var].unique()]
            
            # Perform ANOVA or Kruskal-Wallis
            if is_normal:
                stat, p_value = f_oneway(*groups)
                test_name = 'ANOVA'
            else:
                stat, p_value = kruskal(*groups)
                test_name = 'Kruskal-Wallis Test'
            
            # Display results and inference
            st.write(f"{test_name} Results: Statistic = {stat}, p-value = {p_value}")
            
            # If ANOVA is significant, perform Tukey’s HSD
            if is_normal and p_value < 0.05:
                tukey_result = pairwise_tukeyhsd(df[inference_var].dropna(), df[stratification_var].dropna())
                st.write("Tukey HSD Result:")
                st.write(tukey_result)
    # 10. Categorical and Stratification Variable Inference
    if st.button("Perform Inference between Categorical and Stratification Variables"):
        # Allow user to select variables for inference
        cat_var = st.selectbox("Select a Categorical Variable", categorical_vars)
        strat_var = st.selectbox("Select a Stratification Variable", stratification_vars)
        
        from scipy.stats import chi2_contingency
        
        # Cross-tabulation
        cross_tab = pd.crosstab(df[cat_var], df[strat_var], margins=False)
        st.write("Cross-tabulation:")
        st.write(cross_tab)
        
        # Column Percentage
        col_percent = cross_tab.apply(lambda r: r/r.sum(), axis=0)
        st.write("Column Percentages:")
        st.write(col_percent)
        
        # Chi-Square Test
        chi2, p_value, _, _ = chi2_contingency(cross_tab)
        st.write(f"Chi-Square Test Results: Chi2 = {chi2}, p-value = {p_value}")
        
        # Interpretation
        if p_value < 0.05:
            st.write("The p-value is less than 0.05, suggesting a significant association between the variables.")
        else:
            st.write("The p-value is greater than 0.05, suggesting no significant association between the variables.")
   
    # 12. Multiple Variable Correlation Analysis
    if len(continuous_vars) > 1 and st.button("Perform Multiple Variable Correlation Analysis"):
        # Allow user to select variables for correlation
        selected_vars = st.multiselect("Select at least 2 Variables", continuous_vars, default=continuous_vars[:2])
        
        if len(selected_vars) == 1:
            st.write("Please select at least two variables for correlation analysis.")
        else:
            from scipy.stats import pearsonr
            import seaborn as sns
            
            # Subset dataframe
            subset_df = df[selected_vars].dropna()
            
            # Correlation Matrix
            corr_matrix = subset_df.corr()
            st.write("Correlation Matrix:")
            st.write(corr_matrix)
            
            # P-Value Matrix
            p_value_matrix = subset_df.apply(lambda x: subset_df.apply(lambda y: pearsonr(x, y)[1]))
            st.write("P-Value Matrix:")
            st.write(p_value_matrix)
            
            # Interpretation
            st.write("Interpretation:")
            for var1 in selected_vars:
                for var2 in [v for v in selected_vars if v != var1]:
                    corr = corr_matrix.loc[var1, var2]
                    p_value = p_value_matrix.loc[var1, var2]
                    st.write(f"Correlation between {var1} and {var2}: Coefficient = {corr}, p-value = {p_value}")
                    if p_value < 0.05:
                        st.write(f"-> Significant correlation: {'Positive' if corr > 0 else 'Negative'} linear relationship.")
                    else:
                        st.write("-> No significant correlation.")
                    
            # Pairplot
            st.write("Pairplot:")
            pair_plot = sns.pairplot(data=subset_df, kind='reg')
            st.pyplot(pair_plot.fig)
