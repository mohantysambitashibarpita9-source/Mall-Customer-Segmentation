
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mall Customer Segmentation")
st.write("An interactive Business Analytics application")

st.sidebar.header("Upload Customer Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = [
        'CustomerID',
        'Genre',
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("Uploaded file does not have the required columns.")

    elif df[required_columns].isnull().any().any():
        st.error("Dataset contains missing values. Please clean it first.")

    else:

        df = df.drop_duplicates().copy()

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(df))
        col2.metric("Average Age", round(df['Age'].mean(), 1))
        col3.metric(
            "Average Income",
            round(df['Annual Income (k$)'].mean(), 2)
        )

        st.subheader("Customer Dataset")

        st.dataframe(df, use_container_width=True)

        st.subheader("Customer Demographics")

        fig1, ax1 = plt.subplots()

        sns.countplot(data=df, x='Genre', ax=ax1)

        ax1.set_title("Gender Distribution")

        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("Income Analysis")

        fig2, ax2 = plt.subplots()

        sns.histplot(
            data=df,
            x='Annual Income (k$)',
            kde=True,
            ax=ax2
        )

        ax2.set_title("Annual Income Distribution")

        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Customer Segmentation")

        n_clusters = st.sidebar.slider(
            "Select Number of Clusters",
            min_value=2,
            max_value=10,
            value=5
        )

        X = df[
            ['Annual Income (k$)', 'Spending Score (1-100)']
        ]

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        df['Cluster'] = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, df['Cluster'])

        st.metric(
            "Silhouette Score",
            round(score, 3)
        )

        fig3, ax3 = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Cluster',
            palette='Set1',
            s=100,
            ax=ax3
        )

        ax3.set_title("K-Means Customer Segmentation")

        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Customer Segment Summary")

        summary = df.groupby('Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(2)

        summary.rename(
            columns={'CustomerID': 'Customer Count'},
            inplace=True
        )

        st.dataframe(summary, use_container_width=True)

        st.subheader("Business Recommendations")

        st.write(
            "Businesses can use these customer segments "
            "to design targeted marketing campaigns, "
            "improve customer retention and offer "
            "personalised promotions."
        )

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download Segmented Customer Data",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

else:

    st.info("Upload your Mall Customers CSV file to begin.")
