import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import metrics
import plotly.express as px
from pylab import rcParams
rcParams["figure.figsize"]=(30,18)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 15
#st.sidebar.title("Upload Data")
result_file = st.sidebar.file_uploader("Upload Result Data (Excel file)", type=["xlsx"])
prediction_file = st.sidebar.file_uploader("Upload Prediction Data (Excel file)", type=["xlsx"])

# Main Page
st.title("Regression Model Analysis")

# Check if files are uploaded
if result_file is not None or prediction_file is not None:
    # Load Data
    result = None
    prediction = None

    if result_file is not None:
        result = pd.read_excel(result_file)
        if result.columns[0] == "Unnamed: 0":
            result.drop(["Unnamed: 0"], axis=1, inplace=True)
        result.drop(["Fitted Time"], axis=1, inplace=True)
        result = result.set_index(" Models")

    if prediction_file is not None:
        prediction = pd.read_excel(prediction_file)

    # Sidebar
    st.sidebar.title("Select Visualization")
    visualization_choice = st.sidebar.selectbox(
        "Choose a visualization",
        [
            "Q-Q Plot",
            "Heatmap",
            "Mean Performance Metrics",
            "Radar Plot",
            "Parallel Coordinates",
            "Scatter Matrix",
            "Scatter Plot",
            "Box Plot",
            "Violin Plot",
            "Residual Plot",
            "MSE Comparison",
            "Parallel Coordinates (Plotly)",
            "Correlation Heatmap",
            "R2 Polar Plot",
            "Pairplot with Hue",
            "Distribution Plot for Each Metric",
            "Jointplot of Model Predictions vs. Original Values",
            "Density Contour Plot of Two Metrics",
            "3D Scatter Plot of Three Metrics",
            "Histogram with Marginal KDE",
            "PairGrid with Regression Fits",
            "Clustermap for Metric Correlation",
            "Error Bar Plot of Mean Performance Metrics",
            "Scatter Plot Matrix with Regression Fits"
        ]
    )


    # Plot Functions
    def plot_qq_plots(result):
        st.subheader("Q-Q Plot")
        num_metrics = len(result.columns)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

        for i in range(3):
            for j in range(3):
                if i * 3 + j < num_metrics:
                    metric = result.columns[i * 3 + j]
                    stats.probplot(result[metric], dist="norm", plot=axes[i, j])
                    axes[i, j].set_title(f"Q-Q Plot for {metric}", weight='bold')  # Title in bold

                    axes[i, j].set_xlabel(f"{metric}", weight='bold')  # X-axis label in bold
                    axes[i, j].set_ylabel("Probability", weight='bold')  # Y-axis label in bold

        fig.tight_layout()
        st.pyplot(fig)

    def plot_heatmap(result):
        st.subheader("Performance Metric Comparison (Heatmap)")
        fig, ax = plt.subplots(figsize=(8, 6))
        result.drop(['RRMSE'] , axis=1  , inplace=True)

        sns.heatmap(result, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    def plot_mean_performance_metrics(result):
        st.subheader("Mean Performance Metrics with Error Bars")
        result.drop(['RRMSE'] , axis=1  , inplace=True)
        mean_metrics = result.mean()
        std_metrics = result.std()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(mean_metrics.index, mean_metrics, yerr=std_metrics, capsize=5)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_title('Mean Performance Metrics with Error Bars' , weight='bold')
        st.pyplot(fig)
##################################################################################################
    def plot_radar_plot(result):
        st.subheader("Radar Plot of Performance Metrics")
        normalized_metrics = (result - result.min()) / (result.max() - result.min())
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        for model in result.index:
            values = normalized_metrics.loc[model].tolist()
            values += values[:1]
            angles = [n / float(len(result.columns)) * 2 * np.pi for n in range(len(result.columns))]
            angles += angles[:1]
            ax.plot(angles, values, label=model)
        ax.set_title('Radar Plot of Performance Metrics' , weight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        st.pyplot(fig)
##########################################################################################################
    def plot_parallel_coordinates(result):
        st.subheader("Parallel Coordinates Plot of Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        result.drop(['RRMSE'] , axis=1  , inplace=True)

        reversed_models = result.index[::-1]
        pd.plotting.parallel_coordinates(result.loc[reversed_models].reset_index(), ' Models', colormap='viridis', ax=ax)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_title('Parallel Coordinates Plot of Model Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), title='Models', fontsize=8)
        st.pyplot(fig)


    def plot_scatter_matrix(result):
        st.subheader("Scatter Matrix Plot of Model Comparison")
        fig, axes = plt.subplots(figsize=(12, 12))

        scatter_matrix = pd.plotting.scatter_matrix(result, alpha=0.8, diagonal='hist', ax=axes)

        font = {'weight': 'bold'}

        for ax in scatter_matrix.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontdict=font)
            ax.set_ylabel(ax.get_ylabel(), fontdict=font)
            ax.title.set_fontweight('bold')

        fig.suptitle('Scatter Matrix Plot of Model Comparison', y=1.02, weight='bold')
        fig.tight_layout(h_pad=0.5, w_pad=0.5)
        st.pyplot(fig)


    def plot_scatter_plot(prediction):
        st.subheader("Scatter Plot of Model Predictions vs. Original Values")
        fig, ax = plt.subplots(figsize=(20, 12))

        for model in prediction.columns:
            if model != 'Original':
                ax.scatter(prediction['Original'], prediction[model], label=model)

        # Set the font properties
        font = {'weight': 'bold'}

        # Update the font properties for the axis labels and title
        ax.set_xlabel('Original Values', fontdict=font)
        ax.set_ylabel('Predicted Values', fontdict=font)
        ax.set_title('Scatter Plot of Model Predictions vs. Original Values', fontdict=font)

        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
        ax.grid()

        st.pyplot(fig)


    def plot_box_plot(prediction):
        st.subheader("Box Plot of Prediction Errors for Each Model")
        fig, ax = plt.subplots(figsize=(15, 8))
        prediction_errors = prediction.drop(columns='Original').subtract(prediction['Original'], axis=0)
        prediction_errors.boxplot(ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Prediction Errors' , weight='bold')
        ax.set_title('Box Plot of Prediction Errors for Each Model' , weight='bold')
        ax.grid()
        st.pyplot(fig)

    def plot_violin_plot(prediction):
        st.subheader("Violin Plot of Prediction Errors for Each Model")
        fig, ax = plt.subplots(figsize=(15, 8))
        prediction_errors = prediction.drop(columns='Original').subtract(prediction['Original'], axis=0)
        prediction_errors = prediction_errors.melt(var_name='Model', value_name='Error')
        sns.violinplot(data=prediction_errors, x='Model', y='Error', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('Prediction Errors' , weight='bold')
        ax.set_title('Violin Plot of Prediction Errors for Each Model' , weight='bold')
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
        ax.set_xlabel('Data Points' , weight='bold')
        ax.set_ylabel('Residuals' , weight='bold')
        ax.set_title('Residual Plot of Model Predictions'  , weight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
        ax.grid()
        st.pyplot(fig)

    def plot_mse_comparison(result):
        st.subheader("Mean Squared Error (MSE) Comparison")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=result.mean(axis=1), y=result.index, palette='viridis', ax=ax)
        ax.set_title('Mean Squared Error (MSE) Comparison' ,weight='bold')
        ax.set_xlabel('Mean Squared Error (MSE)' , weight='bold')
        ax.set_ylabel('Models' , weight='bold')
        st.pyplot(fig)

    def plot_parallel_coordinates_plotly(result):
        st.subheader("Parallel Coordinates Plot of Regression Metrics by Model (Plotly)")
        fig = px.parallel_coordinates(result.reset_index(), color="R2", dimensions=list(result.columns),
                                      labels={'index': result.index.name},
                                      color_continuous_scale=px.colors.sequential.Viridis,
                                      title='Parallel Coordinates Plot of Regression Metrics by Model')
        fig.update_layout(
            title_font=dict(family="Arial", size=20),
            font=dict(family="Arial", size=15),
            plot_bgcolor='white',
        )
        st.plotly_chart(fig)

    def plot_correlation_heatmap(prediction):
        st.subheader("Correlation Heatmap of Predicted Values by Model")
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.heatmap(prediction.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap of Predicted Values by Model', weight='bold')
        st.pyplot(fig)


    def plot_r2_polar_plot(prediction):
        st.subheader("R2 Polar Plot")
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        for model in prediction.columns[:-1]:
            mse = metrics.mean_squared_error(prediction[prediction.columns[-1]], prediction[model])
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(prediction[prediction.columns[-1]], prediction[model])

            ax.scatter(np.arccos(r2), np.sqrt(mse), s=30, label=model)

        ax.set_title("R2 Polar Plot", weight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))  # Place legend outside the plot
        st.pyplot(fig)


    def plot_pairplot(result):
        st.subheader("Pairplot with Hue")
        pairplot = sns.pairplot(result, palette='viridis')
        font = {'weight': 'bold'}
        for ax in pairplot.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontdict=font)
            ax.set_ylabel(ax.get_ylabel(), fontdict=font)
            ax.title.set_fontweight('bold')

        st.pyplot(pairplot)


    def plot_distplot(result):
        st.subheader("Distribution Plot for Each Metric")
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
        metrics = result.columns

        # Set the font properties
        font = {'weight': 'bold'}

        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            sns.histplot(result[metric], kde=True, ax=axes[row, col])
            axes[row, col].set_title(f"Distribution of {metric}", fontdict=font)
            axes[row, col].set_xlabel(metric, fontdict=font)
            axes[row, col].set_ylabel('Frequency', fontdict=font)

        fig.tight_layout()
        st.pyplot(fig)


    def plot_jointplot(prediction):
        st.subheader("Jointplot of Model Predictions vs. Original Values")

        fig = sns.jointplot(x='Original', y=prediction.columns[1], data=prediction, kind='reg', height=8)

        font = {'weight': 'bold'}

        fig.ax_joint.set_xlabel('Original Values', fontdict=font)
        fig.ax_joint.set_ylabel('Predicted Values', fontdict=font)
        fig.fig.suptitle("Jointplot of Model Predictions vs. Original Values", fontdict=font)

        fig.fig.tight_layout()
        fig.fig.subplots_adjust(top=0.95)

        st.pyplot(fig)


    def plot_density_contour(result):
        st.subheader("Density Contour Plot of Two Metrics")
        fig = px.density_contour(result, x=result.columns[0], y=result.columns[1], marginal_x="histogram",
                                 marginal_y="histogram")
        st.plotly_chart(fig)


    def plot_3d_scatter(result):
        st.subheader("3D Scatter Plot of Three Metrics")
        # Remove any leading or trailing spaces from column names
        result.columns = result.columns.str.strip()
        fig = px.scatter_3d(result, x=result.columns[0], y=result.columns[1], z=result.columns[2])
        st.plotly_chart(fig)


    def plot_histogram_kde(result):
        st.subheader("Histogram with Marginal KDE")

        fig, axes = plt.subplots(nrows=len(result.columns), ncols=len(result.columns))

        font = {'weight': 'bold'}

        for i, col1 in enumerate(result.columns):
            for j, col2 in enumerate(result.columns):
                if i == j:
                    sns.histplot(result[col1], kde=True, ax=axes[i, j])
                else:
                    sns.scatterplot(x=result[col1], y=result[col2], ax=axes[i, j])

                if i == len(result.columns) - 1:
                    axes[i, j].set_xlabel(col2, fontdict=font)
                if j == 0:
                    axes[i, j].set_ylabel(col1, fontdict=font)

                axes[i, j].set_title(f"{col1} vs {col2}" if i != j else f"Distribution of {col1}", fontdict=font)

        fig.tight_layout()
        st.pyplot(fig)


    def plot_pairgrid(result):
        st.subheader("PairGrid with Regression Fits")

        # Create the PairGrid
        g = sns.PairGrid(result)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_diag(sns.histplot, kde=True)
        g.map_lower(sns.regplot)

        # Set the font properties
        font = {'weight': 'bold'}

        # Update the font properties for axis labels and titles
        for ax in g.axes.flatten():
            ax.set_title(ax.get_title(), fontdict=font ,weight='bold')
            ax.set_xlabel(ax.get_xlabel(), fontdict=font ,weight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontdict=font , weight='bold')

        g.fig.suptitle("PairGrid with Regression Fits", fontdict=font, y=1.05, weight='bold')


        st.pyplot(g)


    def plot_clustermap(result):
        st.subheader("Clustermap for Metric Correlation")

        # Compute the correlation matrix
        corr = result.corr()

        # Create the clustermap
        g = sns.clustermap(corr, cmap='coolwarm', linewidths=.5)

        # Set the font properties
        font = {'weight': 'bold'}

        # Update the font properties for axis labels
        for label in g.ax_heatmap.get_xticklabels():
            label.set_fontweight('bold')
        for label in g.ax_heatmap.get_yticklabels():
            label.set_fontweight('bold')

        # Add a title
        g.fig.suptitle("Clustermap for Metric Correlation", fontdict=font,weight='bold')


        st.pyplot(g)


    def plot_error_bar(result):
        st.subheader("Error Bar Plot of Mean Performance Metrics")
        result.drop(['RRMSE'] , axis=1  , inplace=True)
        mean_metrics = result.mean()
        std_metrics = result.std()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(mean_metrics.index, mean_metrics, yerr=std_metrics, fmt='o', capsize=5)
        ax.set_xlabel('Metrics',weight='bold')
        ax.set_ylabel('Metric Values',weight='bold')
        ax.set_title('Mean Performance Metrics with Error Bars' , weight='bold')
        ax.grid(True)
        st.pyplot(fig)


    def plot_scatter_matrix_regression(result):
        st.subheader("Scatter Plot Matrix with Regression Fits")

        # Create the scatter plot matrix with regression fits
        fig = sns.pairplot(result, kind="reg", plot_kws={'line_kws': {'color': 'red'}})

        # Set the font properties
        font = {'weight': 'bold'}

        # Update the font properties for axis labels and title
        for ax in fig.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontdict=font , weight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontdict=font , weight='bold')
            ax.set_title(ax.get_title(), fontdict=font ,  weight='bold')

        fig.fig.suptitle("Scatter Plot Matrix with Regression Fits", fontdict=font, y=1.02 , weight='bold')
        fig.fig.tight_layout()

        st.pyplot(fig)




    # Choose the plot based on the user's selection
    if visualization_choice == "Q-Q Plot":
        if result is not None:
            plot_qq_plots(result)
    elif visualization_choice == "Heatmap":
        if result is not None:
            plot_heatmap(result)
    elif visualization_choice == "Mean Performance Metrics":
        if result is not None:
            plot_mean_performance_metrics(result)
    elif visualization_choice == "Radar Plot":
        if result is not None:
            plot_radar_plot(result)
    elif visualization_choice == "Parallel Coordinates":
        if result is not None:
            plot_parallel_coordinates(result)
    elif visualization_choice == "Scatter Matrix":
        if result is not None:
            plot_scatter_matrix(result)
    elif visualization_choice == "Scatter Plot":
        if prediction is not None:
            plot_scatter_plot(prediction)
    elif visualization_choice == "Box Plot":
        if prediction is not None:
            plot_box_plot(prediction)
    elif visualization_choice == "Violin Plot":
        if prediction is not None:
            plot_violin_plot(prediction)
    elif visualization_choice == "Residual Plot":
        if prediction is not None:
            plot_residual_plot(prediction)
    elif visualization_choice == "MSE Comparison":
        if result is not None:
            plot_mse_comparison(result)
    elif visualization_choice == "Parallel Coordinates (Plotly)":
        if result is not None:
            plot_parallel_coordinates_plotly(result)
    elif visualization_choice == "Correlation Heatmap":
        if prediction is not None:
            plot_correlation_heatmap(prediction)
    elif visualization_choice == "R2 Polar Plot":
        if prediction is not None:
            plot_r2_polar_plot(prediction)

    elif visualization_choice == "R2 Polar Plot":
        if prediction is not None:
            plot_r2_polar_plot(prediction)

    elif visualization_choice == "Pairplot with Hue":
        if result is not None:
            plot_pairplot(result)

    elif visualization_choice == "Distribution Plot for Each Metric":
        if result is not None:
            plot_distplot(result)

    elif visualization_choice == "Jointplot of Model Predictions vs. Original Values":
        if prediction is not None:
            plot_jointplot(prediction)

    elif visualization_choice == "Density Contour Plot of Two Metrics":
        if result is not None and len(result.columns) >= 2:
            plot_density_contour(result)

    elif visualization_choice == "3D Scatter Plot of Three Metrics":
        if result is not None and len(result.columns) >= 3:
            plot_3d_scatter(result)

    elif visualization_choice == "Histogram with Marginal KDE":
        plot_histogram_kde(result)
    elif visualization_choice == "PairGrid with Regression Fits":
        plot_pairgrid(result)
    elif visualization_choice == "Clustermap for Metric Correlation":
        plot_clustermap(result)
    elif visualization_choice == "Error Bar Plot of Mean Performance Metrics":
        plot_error_bar(result)
    elif visualization_choice == "Scatter Plot Matrix with Regression Fits":
        plot_scatter_matrix_regression(result)

else:
    st.warning("Please upload both Result and Prediction files.")
