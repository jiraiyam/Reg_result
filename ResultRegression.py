import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
from sklearn import metrics
import plotly.express as px
# Sidebar
st.sidebar.title("Upload Data")
result_file = st.sidebar.file_uploader("Upload Result Data (Excel file)", type=["xlsx"])
prediction_file = st.sidebar.file_uploader("Upload Prediction Data (Excel file)", type=["xlsx"])

# Main Page
st.title("Regression Model Analysis")

# Check if files are uploaded
if result_file is not None and prediction_file is not None:
    # Load Data
    result = pd.read_excel(result_file)
    prediction = pd.read_excel(prediction_file)

    if result.columns[0]=="Unnamed: 0":
        result.drop(["Unnamed: 0"], axis=1, inplace=True)


    # Drop "Fitted Time" column
    result.drop(["Fitted Time"], axis=1, inplace=True)
    result = result.set_index(" Models")

    # Sidebar
    st.sidebar.title("Select Visualization")
    visualization_choice = st.sidebar.selectbox(
        "Choose a visualization",
        ["Q-Q Plot", "Heatmap", "Mean Performance Metrics", "Radar Plot", "Parallel Coordinates", "Scatter Matrix", "Scatter Plot", "Box Plot", "Violin Plot", "Residual Plot", "MSE Comparison", "Parallel Coordinates (Plotly)", "Correlation Heatmap", "R2 Polar Plot"]
    )

    # Plot Functions
    def plot_qq_plots(result):
        st.subheader("Q-Q Plot")
        for metric in result.columns:
            st.subheader(f"{metric.upper()}")
            fig = plt.figure(figsize=(6, 4))
            stats.probplot(result[metric], dist="norm", plot=plt)
            st.pyplot(fig)

    def plot_heatmap(result):
        st.subheader("Performance Metric Comparison (Heatmap)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(result, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)


    def plot_mean_performance_metrics(result):
        st.subheader("Mean Performance Metrics with Error Bars")
        mean_metrics = result.mean()
        std_metrics = result.std()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(mean_metrics.index, mean_metrics, yerr=std_metrics, capsize=5)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_title('Mean Performance Metrics with Error Bars')
        st.pyplot(fig)


    def plot_radar_plot(result):
        st.subheader("Radar Plot of Performance Metrics")
        normalized_metrics = (result - result.min()) / (result.max() - result.min())
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        for model in result.index:
            values = normalized_metrics.loc[model].tolist()
            values += values[:1]
            angles = [n / float(len(result.columns)) * 2 * math.pi for n in range(len(result.columns))]
            angles += angles[:1]
            ax.plot(angles, values, label=model)
        ax.set_title('Radar Plot of Performance Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        st.pyplot(fig)


    def plot_parallel_coordinates(result):
        st.subheader("Parallel Coordinates Plot of Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Reverse the order of x-axis labels
        reversed_models = result.index[::-1]
        pd.plotting.parallel_coordinates(result.loc[reversed_models].reset_index(), ' Models', colormap='viridis',
                                         ax=ax)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_title('Parallel Coordinates Plot of Model Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), title='Models', fontsize=8)

        st.pyplot(fig)


    def plot_scatter_matrix(result):
        st.subheader("Scatter Matrix Plot of Model Comparison")
        fig, ax = plt.subplots(figsize=(12, 12))
        pd.plotting.scatter_matrix(result, alpha=0.8, ax=ax, diagonal='hist')
        fig.suptitle('Scatter Matrix Plot of Model Comparison', y=1.02)
        fig.tight_layout(h_pad=0.5, w_pad=0.5)
        st.pyplot(fig)


    def plot_scatter_plot(prediction):
        st.subheader("Scatter Plot of Model Predictions vs. Original Values")
        fig, ax = plt.subplots(figsize=(12, 8))
        for model in prediction.columns:
            if model != 'Original':
                ax.scatter(prediction['Original'], prediction[model], label=model)
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Scatter Plot of Model Predictions vs. Original Values')
        ax.legend()
        ax.grid()
        st.pyplot(fig)


    def plot_box_plot(prediction):
        st.subheader("Box Plot of Prediction Errors for Each Model")
        fig, ax = plt.subplots(figsize=(15, 8))
        prediction_errors = prediction.drop(columns='Original').subtract(prediction['Original'], axis=0)
        prediction_errors.boxplot(ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Prediction Errors')
        ax.set_title('Box Plot of Prediction Errors for Each Model')
        ax.grid()
        st.pyplot(fig)


    def plot_violin_plot(prediction):
        st.subheader("Violin Plot of Prediction Errors for Each Model")
        fig, ax = plt.subplots(figsize=(15, 8))
        prediction_errors = prediction.drop(columns='Original').subtract(prediction['Original'], axis=0)
        prediction_errors = prediction_errors.melt(var_name='Model', value_name='Error')
        sns.violinplot(data=prediction_errors, x='Model', y='Error', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Prediction Errors')
        ax.set_title('Violin Plot of Prediction Errors for Each Model')
        ax.grid()
        st.pyplot(fig)


    def plot_residual_plot(prediction):
        st.subheader("Residual Plot of Model Predictions")
        fig, ax = plt.subplots(figsize=(15, 8))
        for model in prediction.columns:
            if model != 'Original':
                residuals = prediction[model] - prediction['Original']
                ax.plot(prediction.index, residuals, label=model)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot of Model Predictions')
        ax.legend()
        ax.grid()
        st.pyplot(fig)


    def plot_mse_comparison(result):
        st.subheader("Mean Squared Error (MSE) Comparison")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=result.mean(axis=1), y=result.index, palette='viridis', ax=ax)
        ax.set_title('Mean Squared Error (MSE) Comparison')
        ax.set_xlabel('Mean Squared Error (MSE)')
        ax.set_ylabel('Models')
        st.pyplot(fig)


    def plot_parallel_coordinates_plotly(result):
        st.subheader("Parallel Coordinates Plot of Regression Metrics by Model (Plotly)")
        fig = px.parallel_coordinates(result.reset_index(), color="R2", dimensions=list(result.columns),
                                      labels={'index': result.index.name},
                                      color_continuous_scale=px.colors.sequential.Viridis,
                                      title='Parallel Coordinates Plot of Regression Metrics by Model')

        # Update layout to set background color to white
        fig.update_layout(
            title_font=dict(family="Arial", size=20),
            font=dict(family="Arial", size=15),
            plot_bgcolor='white',
        )


        st.plotly_chart(fig)


    def plot_correlation_heatmap(prediction):
        st.subheader("Correlation Heatmap of Predicted Values by Model")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(prediction.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap of Predicted Values by Model')
        st.pyplot(fig)


    def plot_r2_polar_plot(prediction):
        st.subheader("R2 Polar Plot")
        for model in prediction.columns[:-1]:
            mse = metrics.mean_squared_error(prediction[prediction.columns[-1]], prediction[model])
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(prediction[prediction.columns[-1]], prediction[model])
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            ax.scatter(np.arccos(r2), np.sqrt(mse), s=30, c='red', label='Prediction')
            ax.set_title(model)
            ax.legend()
            st.pyplot(fig)




    # ... (rest of the plot functions remain the same)

    # Choose the plot based on the user's selection
    if visualization_choice == "Q-Q Plot":
        plot_qq_plots(result)
    elif visualization_choice == "Heatmap":
        plot_heatmap(result)
    elif visualization_choice == "Mean Performance Metrics":
        plot_mean_performance_metrics(result)
    elif visualization_choice == "Radar Plot":
        plot_radar_plot(result)
    elif visualization_choice == "Parallel Coordinates":
        plot_parallel_coordinates(result)
    elif visualization_choice == "Scatter Matrix":
        plot_scatter_matrix(result)
    elif visualization_choice == "Scatter Plot":
        plot_scatter_plot(prediction)
    elif visualization_choice == "Box Plot":
        plot_box_plot(prediction)
    elif visualization_choice == "Violin Plot":
        plot_violin_plot(prediction)
    elif visualization_choice == "Residual Plot":
        plot_residual_plot(prediction)
    elif visualization_choice == "MSE Comparison":
        plot_mse_comparison(result)
    elif visualization_choice == "Parallel Coordinates (Plotly)":
        plot_parallel_coordinates_plotly(result)
    elif visualization_choice == "Correlation Heatmap":
        plot_correlation_heatmap(prediction)
    elif visualization_choice == "R2 Polar Plot":
        plot_r2_polar_plot(prediction)


else:
    st.warning("Please upload both Result and Prediction files.")
