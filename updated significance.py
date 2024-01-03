import streamlit as st
import pandas as pd
from tableone import TableOne
import base64

def create_download_link(df, filename):
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Convert dataframe to CSV and encode
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()

    # Create download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.title("Interactive Statistical Significance Analysis")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Step 2: Variable Selection
        variables = st.multiselect("Select Variables to Summarize", data.columns.tolist())
        
        # Step 3: Specify Categorical Variables
        cat_vars = st.multiselect("Specify Binary Categorical Variables", variables)

        # Step 4: Specify Non-Normal Variables
        non_normal_vars = st.multiselect("Specify Non-Normal Variables", variables)

        # Step 5: GroupBy Variable
        groupby_var = st.selectbox("Specify GroupBy Variable", variables)

        if st.button("Generate TableOne"):
            try:
                # Creating a TableOne instance and displaying it
                table_one = TableOne(data, columns=variables, categorical=cat_vars, nonnormal=non_normal_vars, groupby=groupby_var, pval=True,smd=True,htest_name=True)
                
                table_one.to_excel('Signifcance for differance.xlsx')
                
                # Displaying the table
                st.write(table_one.tableone)
                
                # Creating a download link for the CSV file
                st.markdown(create_download_link(table_one.tableone, "tableone_output.csv"), unsafe_allow_html=True)
            except Exception as e:
                st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
