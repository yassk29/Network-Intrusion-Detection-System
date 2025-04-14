import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fast import main as fast_main
from slow import main as slow_main
from final6 import main as final6_main
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Network Traffic Analysis", layout="wide")

# Initialize session state for saved results
if 'saved_results' not in st.session_state:
    st.session_state.saved_results = {}

def run_model(model_type):
    if model_type == "Fast Model":
        return fast_main()
    elif model_type == "Slow Model":
        return slow_main()
    else:
        return final6_main()

def create_metrics_visualization(metrics):
    # Create a bar chart for key metrics
    fig = go.Figure(data=[
        go.Bar(name='Value', x=list(metrics.keys()), y=list(metrics.values()))
    ])
    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Value',
        showlegend=False
    )
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def create_comparison_chart(saved_results):
    if not saved_results:
        return None
    
    metrics = ['accuracy', 'f1', 'avg_latency', 'packets_per_second', 'flow_miss_rate']
    data = []
    
    for model_name, results in saved_results.items():
        for metric in metrics:
            data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': results[metric]
            })
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Metric', y='Value', color='Model', barmode='group',
                 title='Model Comparison')
    return fig

def main():
    st.title("Network Traffic Analysis Dashboard")
    st.write("Select a model to analyze network traffic patterns")

    # Create three columns for the model selection buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Fast Model", use_container_width=True):
            st.session_state['selected_model'] = "Fast Model"
    
    with col2:
        if st.button("Slow Model", use_container_width=True):
            st.session_state['selected_model'] = "Slow Model"
    
    with col3:
        if st.button("Combined Model", use_container_width=True):
            st.session_state['selected_model'] = "Combined Model"

    # Add a tab for model comparison
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Detailed Analysis", "Model Comparison"])

    if 'selected_model' in st.session_state:
        st.write(f"Running {st.session_state['selected_model']}...")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Run the selected model
        with st.spinner('Processing...'):
            results = run_model(st.session_state['selected_model'])
            progress_bar.progress(100)
            
            # Automatically save results
            st.session_state.saved_results[st.session_state['selected_model']] = results
            st.success(f"Results for {st.session_state['selected_model']} saved successfully!")

        with tab1:
            st.subheader("Confusion Matrix")
            fig = create_confusion_matrix_plot(results['true_labels'], results['predicted_labels'])
            st.pyplot(fig)

        with tab2:
            st.subheader("Detailed Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Model Statistics")
                st.write(f"Total Packets Processed: {results['total_packets']}")
                st.write(f"Packets Processed per Second: {results['packets_per_second']:.2f}")
                st.write(f"Fast Model Usage: {results['fast_model_count']}")
                st.write(f"Slow Model Usage: {results['slow_model_count']}")
                st.write(f"Average Confidence: {results['avg_confidence']:.2f}")
            
            with col2:
                st.write("Performance Metrics")
                st.write(f"Average Latency per Packet: {results['avg_latency']:.6f} seconds")
                st.write(f"End-to-End Latency: {results['avg_latency'] * 1000:.2f} ms")
                # st.write(f"3-Packet Flow Miss Rate: {results['flow_miss_rate']:.4f}")
                st.write(f"F1 Score: {results['f1']:.4f}")
                st.write(f"Accuracy: {results['accuracy']:.4f}")

    with tab3:
        st.subheader("Model Comparison")
        if st.session_state.saved_results:
            # Display saved results in a detailed table
            st.subheader("Saved Results Summary")
            comparison_data = []
            for model_name, results in st.session_state.saved_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Total Packets': results['total_packets'],
                    'Packets/Second': f"{results['packets_per_second']:.2f}",
                    'Fast Model Usage': results['fast_model_count'],
                    'Slow Model Usage': results['slow_model_count'],
                    'Average Confidence': f"{results['avg_confidence']:.2f}",
                    'Average Latency (ms)': f"{results['avg_latency'] * 1000:.2f}",
                    # 'Flow Miss Rate': f"{results['flow_miss_rate']:.4f}",
                    'F1 Score': f"{results['f1']:.4f}",
                    'Accuracy': f"{results['accuracy']:.4f}"
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
            
            # Add clear results button
            if st.button("Clear All Saved Results"):
                st.session_state.saved_results = {}
                st.success("All saved results have been cleared!")
        else:
            st.info("No saved results to compare. Run and save results from different models to enable comparison.")

if __name__ == "__main__":
    main() 