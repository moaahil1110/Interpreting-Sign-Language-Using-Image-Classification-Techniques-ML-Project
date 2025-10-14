
"""
🤟 ASL Alphabet Recognition Dashboard - Enhanced with Neural Network Visualizer
Real-time American Sign Language Recognition System
Powered by Deep Learning & PyTorch
Enhanced with CNN Layer Visualization
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import torch
from asl_inference_engine import ASLInferenceEngine
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict

# Page configuration
st.set_page_config(
    page_title="ASL Alphabet Recognition - Enhanced",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #333;
        color: #fff;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #555;
        transform: scale(1.05);
    }
    .prediction-box {
        background: #f0f0f0;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #333;
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .confidence-box {
        background: #e0e0e0;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #333;
        font-size: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stats-box {
        background: #d0d0d0;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #333;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #c0c0c0;
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
    }
    .neural-section {
        background: #b0b0b0;
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
    }
    h1 {
        color: #333;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f0f0;
        border-radius: 10px;
        color: #333; /* Ensure visible text/icon color on light bg */
    }
    /* Inherit color for inner elements to ensure icons/text are visible */
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
        fill: currentColor !important;
    }
    /* Hover state: slightly darker bg and darker text */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
        color: #222;
    }
    /* Active/selected tab: high-contrast background with white text */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #333;
        color: #ffffff !important;
    }
    /* Hide the sidebar header */
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }
""", unsafe_allow_html=True)

# Neural Network Visualizer Class
class CNNVisualizer:
    """
    Comprehensive CNN visualizer for ASL recognition model
    Shows layer-by-layer processing, feature maps, and network architecture
    """

    def __init__(self, model, device='cuda', class_names=None):
        """
        Initialize CNN Visualizer

        Args:
            model: Trained EfficientASLNet model
            device: Computing device ('cuda' or 'cpu')
            class_names: List of ASL alphabet class names
        """
        self.model = model.eval()  # Set to evaluation mode
        self.device = device
        self.class_names = class_names or [chr(i) for i in range(ord('A'), ord('Z')+1)]

        # Hook storage for intermediate activations
        self.activations = {}
        self.gradients = {}
        self.hooks = []

        # Layer information
        self.layer_info = self._get_layer_info()

    def _get_layer_info(self):
        """Extract information about each layer in the network"""
        layer_info = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, 
                                 torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)):
                info = {
                    'name': name,
                    'type': type(module).__name__,
                    'module': module
                }

                if hasattr(module, 'in_channels'):
                    info['in_channels'] = module.in_channels
                if hasattr(module, 'out_channels'):
                    info['out_channels'] = module.out_channels
                if hasattr(module, 'kernel_size'):
                    info['kernel_size'] = module.kernel_size
                if hasattr(module, 'in_features'):
                    info['in_features'] = module.in_features
                if hasattr(module, 'out_features'):
                    info['out_features'] = module.out_features

                layer_info.append(info)

        return layer_info

    def register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        self.clear_hooks()

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
            return hook

        # Register hooks for convolutional and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        """Clear all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def visualize_network_architecture(self):
        """Create an interactive network architecture visualization"""
        # Create network architecture diagram using Plotly
        fig = go.Figure()

        layer_positions = []
        y_pos = 0
        colors = {
            'Conv2d': '#FF6B6B',
            'BatchNorm2d': '#4ECDC4', 
            'MaxPool2d': '#45B7D1',
            'AdaptiveAvgPool2d': '#96CEB4',
            'Linear': '#FFEAA7',
            'Dropout': '#DDA0DD',
            'Dropout2d': '#DDA0DD'
        }

        for i, layer in enumerate(self.layer_info):
            layer_type = layer['type']
            layer_name = layer['name']

            # Determine layer description
            if layer_type == 'Conv2d':
                desc = f"Conv2d\n{layer.get('in_channels', '?')}→{layer.get('out_channels', '?')}\nK:{layer.get('kernel_size', '?')}"
            elif layer_type == 'Linear':
                desc = f"Linear\n{layer.get('in_features', '?')}→{layer.get('out_features', '?')}"
            else:
                desc = layer_type

            # Add layer box
            fig.add_trace(go.Scatter(
                x=[i],
                y=[y_pos],
                mode='markers+text',
                marker=dict(
                    size=50,
                    color=colors.get(layer_type, '#BDC3C7'),
                    line=dict(width=2, color='white')
                ),
                text=desc,
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                name=layer_name,
                hovertemplate=f"<b>{layer_name}</b><br>{desc}<extra></extra>"
            ))

            layer_positions.append((i, y_pos))

        # Add connections between layers
        for i in range(len(layer_positions) - 1):
            x0, y0 = layer_positions[i]
            x1, y1 = layer_positions[i + 1]

            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=2, color='rgba(100,100,100,0.5)'),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title="EfficientASLNet Architecture",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def extract_feature_maps(self, image, layers_to_visualize=None):
        """
        Extract feature maps from specified layers

        Args:
            image: Input image tensor
            layers_to_visualize: List of layer names to visualize

        Returns:
            Dictionary of feature maps for each layer
        """
        if layers_to_visualize is None:
            layers_to_visualize = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        self.register_hooks()

        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.model(image.to(self.device))

        # Extract requested feature maps
        feature_maps = {}
        for layer_name in layers_to_visualize:
            if layer_name in self.activations:
                feature_maps[layer_name] = self.activations[layer_name]

        self.clear_hooks()
        return feature_maps

    def visualize_feature_maps(self, feature_maps, max_channels=16):
        """
        Create visualization of feature maps for each layer

        Args:
            feature_maps: Dictionary of feature maps from extract_feature_maps
            max_channels: Maximum number of channels to display per layer

        Returns:
            Plotly figure with feature map visualizations
        """
        num_layers = len(feature_maps)
        if num_layers == 0:
            return None

        # Create subplots
        fig = make_subplots(
            rows=num_layers,
            cols=1,
            subplot_titles=list(feature_maps.keys()),
            vertical_spacing=0.02
        )

        for row, (layer_name, feature_map) in enumerate(feature_maps.items(), 1):
            if feature_map is None:
                continue

            # Get first batch and limit channels
            fmap = feature_map[0]  # Remove batch dimension
            num_channels = min(fmap.shape[0], max_channels)

            # Create a grid of feature maps
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            combined_map = np.zeros((
                grid_size * fmap.shape[1],
                grid_size * fmap.shape[2]
            ))

            for i in range(num_channels):
                row_idx = i // grid_size
                col_idx = i % grid_size

                start_row = row_idx * fmap.shape[1]
                end_row = start_row + fmap.shape[1]
                start_col = col_idx * fmap.shape[2]
                end_col = start_col + fmap.shape[2]

                combined_map[start_row:end_row, start_col:end_col] = fmap[i].numpy()

            # Add heatmap to subplot
            fig.add_trace(
                go.Heatmap(
                    z=combined_map,
                    colorscale='Viridis',
                    showscale=False,
                    hovertemplate=f"Layer: {layer_name}<br>Value: %{{z:.3f}}<extra></extra>"
                ),
                row=row, col=1
            )

        fig.update_layout(
            title="Feature Map Activations",
            height=200 * num_layers,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Remove axis labels for cleaner look
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig

    def create_activation_statistics(self, feature_maps):
        """
        Create statistical analysis of layer activations

        Args:
            feature_maps: Dictionary of feature maps

        Returns:
            Plotly figure with activation statistics
        """
        stats_data = []

        for layer_name, feature_map in feature_maps.items():
            if feature_map is None:
                continue

            fmap = feature_map[0].numpy()  # Remove batch dimension

            stats_data.append({
                'Layer': layer_name,
                'Mean Activation': np.mean(fmap),
                'Std Activation': np.std(fmap),
                'Max Activation': np.max(fmap),
                'Min Activation': np.min(fmap),
                'Active Neurons (>0)': np.sum(fmap > 0) / fmap.size * 100,
                'Shape': str(fmap.shape)
            })

        if not stats_data:
            return None

        # Create bar charts for different statistics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean Activation', 'Standard Deviation', 
                          'Max Activation', 'Active Neurons (%)'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        layers = [data['Layer'] for data in stats_data]

        # Mean activation
        fig.add_trace(
            go.Bar(x=layers, y=[data['Mean Activation'] for data in stats_data],
                   name='Mean', marker_color='#FF6B6B'),
            row=1, col=1
        )

        # Standard deviation
        fig.add_trace(
            go.Bar(x=layers, y=[data['Std Activation'] for data in stats_data],
                   name='Std', marker_color='#4ECDC4'),
            row=1, col=2
        )

        # Max activation
        fig.add_trace(
            go.Bar(x=layers, y=[data['Max Activation'] for data in stats_data],
                   name='Max', marker_color='#45B7D1'),
            row=2, col=1
        )

        # Active neurons percentage
        fig.add_trace(
            go.Bar(x=layers, y=[data['Active Neurons (>0)'] for data in stats_data],
                   name='Active %', marker_color='#FFEAA7'),
            row=2, col=2
        )

        fig.update_layout(
            title="Layer Activation Statistics",
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return fig

    def create_confidence_visualization(self, prediction_result):
        """Create detailed confidence visualization"""
        if 'probabilities' not in prediction_result:
            return None

        probs = prediction_result['probabilities']
        letters = list(probs.keys())
        confidences = [probs[letter] * 100 for letter in letters]

        # Sort by confidence
        sorted_data = sorted(zip(letters, confidences), key=lambda x: x[1], reverse=True)
        top_10 = sorted_data[:10]

        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            x=[item[1] for item in top_10],
            y=[item[0] for item in top_10],
            orientation='h',
            marker=dict(
                color=[item[1] for item in top_10],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence (%)")
            ),
            text=[f'{item[1]:.2f}%' for item in top_10],
            textposition='auto',
        ))

        fig.update_layout(
            title="Top 10 Predictions - Neural Network Output",
            xaxis_title="Confidence (%)",
            yaxis_title="ASL Letter",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model(model_path, device='cuda', smoothing=5, threshold=0.7):
    """Load the ASL model"""
    try:
        engine = ASLInferenceEngine(
            model_path=model_path,
            device=device,
            smoothing_window=smoothing,
            confidence_threshold=threshold
        )
        return engine, True, "✅ Model loaded successfully!"
    except Exception as e:
        return None, False, f"❌ Error loading model: {str(e)}"

def draw_hand_roi(frame, roi_size=300):
    """Draw region of interest for hand placement"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    roi_x1 = center_x - roi_size // 2
    roi_y1 = center_y - roi_size // 2
    roi_x2 = center_x + roi_size // 2
    roi_y2 = center_y + roi_size // 2

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
    cv2.putText(frame, "Place hand here", (roi_x1, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Extract ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    return frame, roi

def create_confidence_chart(probabilities, top_n=5):
    """Create a horizontal bar chart for top predictions"""
    # Get top N predictions
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    letters = [item[0] for item in sorted_probs]
    probs = [item[1] * 100 for item in sorted_probs]

    # Create bar chart
    fig = go.Figure(go.Bar(
        x=probs,
        y=letters,
        orientation='h',
        marker=dict(
            color=probs,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f'{p:.1f}%' for p in probs],
        textposition='auto',
    ))

    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Letter",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def create_history_chart(history):
    """Create line chart for prediction history"""
    if len(history) == 0:
        return None

    df = pd.DataFrame(history)

    fig = px.line(df, x='timestamp', y='confidence', 
                  title='Prediction Confidence Over Time',
                  labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'})

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

# Main App Header
st.markdown("<h1>ASL Alphabet Recognition - Enhanced with Neural Network Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Real-time American Sign Language Interpreter with AI Visualization</p>", unsafe_allow_html=True)

# Sidebar Configuration (same as original)
with st.sidebar:
    # st.image("https://raw.githubusercontent.com/microsoft/vscode-icons/main/icons/file_type_ai.svg", width=100)  # Removed as requested
    st.header("Configuration")

    # Model settings
    st.subheader("Model Settings")
    model_path = st.text_input(
        "Model Path",
        value="../best_asl_model_rtx4060_optimized.pth",
        help="Path to your trained .pth model file"
    )

    device = st.selectbox(
        "Device",
        options=['cuda', 'cpu'],
        index=0 if torch.cuda.is_available() else 1,
        help="Select GPU (cuda) or CPU for inference"
    )

    smoothing_window = st.slider(
        "Prediction Smoothing",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of frames to smooth predictions (higher = more stable)"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence to accept predictions"
    )

    # Load model button
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            engine, success, message = load_model(
                model_path, device, smoothing_window, confidence_threshold
            )
            if success:
                st.session_state.inference_engine = engine
                st.session_state.model_loaded = True
                st.success(message)
            else:
                st.error(message)

    # Camera settings
    st.subheader("Camera Settings")
    camera_id = st.number_input(
        "Camera ID",
        min_value=0,
        max_value=10,
        value=0,
        help="Camera device ID (usually 0 for default camera)"
    )

    roi_size = st.slider(
        "ROI Size",
        min_value=200,
        max_value=600,
        value=500,
        step=50,
        help="Size of the hand detection region"
    )

    # Model info
    if st.session_state.model_loaded and st.session_state.inference_engine:
        st.divider()
        st.subheader("Model Information")
        engine = st.session_state.inference_engine
        st.info(f"**Device:** {engine.device}")
        st.info(f"**Classes:** {len(engine.class_names)}")
        st.info(f"**Letters:** {', '.join(list(engine.class_names)[:5])}...")

# Main Content Area
if not st.session_state.model_loaded:
    # Welcome screen (same as original)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h2 style='text-align: center; margin-top: 0;'>Welcome!</h2>
            <p style='text-align: center; font-size: 1.1rem; margin-bottom: 0;'>
                Please load the ASL recognition model from the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### How to Use:
        1. **Load Model**: Click "Load Model" in the sidebar
        2. **Upload Image**: Navigate to the "Upload & Analyze" tab
        3. **View Neural Network**: See how the AI processes your image
        4. **Explore Features**: Check live camera and statistics tabs

        ### Enhanced Features:
        - Real-time ASL alphabet recognition
        - **Neural network visualization** (NEW!)
        - Live confidence metrics and visualizations
        - Enhanced image upload with AI analysis
        - Prediction history tracking
        - Beautiful, intuitive interface
        """)
else:
    # Enhanced tabs with neural network visualizer
    tab1, tab2, tab3 = st.tabs(["Live Camera", "Upload & Analyze Neural Network", "Statistics"])

    with tab1:
        # Same live camera functionality as original
        st.subheader("Real-time ASL Recognition")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Camera control buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                start_camera = st.button("Start Camera", width="stretch")
            with btn_col2:
                stop_camera = st.button("Stop Camera", width="stretch")
            with btn_col3:
                reset_btn = st.button("Reset Buffer", width="stretch")

            # Camera feed placeholder
            camera_placeholder = st.empty()

        with col2:
            # Prediction display
            prediction_display = st.empty()
            confidence_display = st.empty()
            chart_display = st.empty()
            top5_display = st.empty()

        # Handle camera controls (same as original implementation)
        if start_camera:
            st.session_state.camera_active = True

        if stop_camera:
            st.session_state.camera_active = False

        if reset_btn and st.session_state.inference_engine:
            st.session_state.inference_engine.reset_buffer()
            st.session_state.prediction_history = []
            st.success("Buffer reset!")

        # Camera loop
        if st.session_state.camera_active and st.session_state.inference_engine:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                st.error(f"Could not open camera {camera_id}. Please check camera permissions and device ID.")
                st.session_state.camera_active = False
            else:
                frame_count = 0
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from camera")
                        break
                    
                    frame_count += 1
                    
                    # Draw ROI and get hand region
                    frame, roi = draw_hand_roi(frame, roi_size)
                    
                    # Make prediction every 3 frames to improve performance
                    if frame_count % 3 == 0 and roi.size > 0:
                        try:
                            result = st.session_state.inference_engine.predict(roi, return_probabilities=True)
                            
                            # Display prediction
                            letter = result['smoothed_letter']
                            confidence = result['confidence'] * 100
                            
                            # Update prediction display
                            prediction_display.markdown(
                                f"<div class='prediction-box'>{letter}</div>",
                                unsafe_allow_html=True
                            )
                            
                            confidence_display.markdown(
                                f"<div class='confidence-box'>{confidence:.1f}%</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Update chart
                            chart_display.plotly_chart(
                                create_confidence_chart(result['probabilities']),
                                width="stretch"
                            )
                            
                            # Display top 5 predictions
                            top_5_probs = sorted(zip(result['class_names'], result['probabilities']), 
                                               key=lambda x: x[1], reverse=True)[:5]
                            top5_html = "<div style='margin-top: 20px;'><h4 style='color: #333; margin-bottom: 10px;'>📊 Top 5 Predictions:</h4>"
                            for i, (cls, prob) in enumerate(top_5_probs, 1):
                                bar_width = int(prob * 100)
                                color = '#4CAF50' if i == 1 else '#2196F3' if i == 2 else '#FF9800' if i == 3 else '#9C27B0' if i == 4 else '#607D8B'
                                top5_html += f"""
                                <div style='margin-bottom: 8px;'>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 2px;'>
                                        <span style='font-weight: 500; color: #333;'>{i}. {cls}</span>
                                        <span style='font-weight: 600; color: {color};'>{prob*100:.1f}%</span>
                                    </div>
                                    <div style='background-color: #f0f0f0; border-radius: 10px; overflow: hidden;'>
                                        <div style='background-color: {color}; height: 8px; width: {bar_width}%; transition: width 0.3s;'></div>
                                    </div>
                                </div>
                                """
                            top5_html += "</div>"
                            top5_display.markdown(top5_html, unsafe_allow_html=True)
                            
                            # Add to history
                            st.session_state.prediction_history.append({
                                'timestamp': datetime.now(),
                                'letter': letter,
                                'confidence': confidence
                            })
                            
                            # Overlay prediction on frame
                            cv2.putText(frame, f"Prediction: {letter}", (10, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.03)
                
                cap.release()

    with tab2:
        # ENHANCED UPLOAD TAB WITH NEURAL NETWORK VISUALIZER
        st.subheader("Upload Image & Analyze Neural Network")

        st.markdown("""
        <div class="neural-section">
            <h3 style='margin-top: 0; color: white;'>AI Brain Analysis</h3>
            <p style='margin-bottom: 0; color: white;'>
                Upload an ASL hand sign image to see exactly how the neural network processes it layer by layer!
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose an ASL hand sign image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image to see both prediction results and neural network analysis"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", width="stretch")

                col_analyze, col_neural = st.columns(2)

                with col_analyze:
                    if st.button("Analyze Image", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            result = st.session_state.inference_engine.predict(
                                image, return_probabilities=True
                            )
                            st.session_state.upload_result = result

                with col_neural:
                    if st.button("Visualize Neural Network", use_container_width=True):
                        with st.spinner("Analyzing neural network layers..."):
                            # Get prediction results
                            result = st.session_state.inference_engine.predict(
                                image, return_probabilities=True
                            )

                            # Create visualizer
                            visualizer = CNNVisualizer(
                                model=st.session_state.inference_engine.model,
                                device=st.session_state.inference_engine.device,
                                class_names=st.session_state.inference_engine.class_names
                            )

                            # Prepare image tensor for visualization
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(image).unsqueeze(0)

                            # Extract feature maps
                            feature_maps = visualizer.extract_feature_maps(img_tensor)

                            # Store results
                            st.session_state.neural_result = result
                            st.session_state.neural_visualizer = visualizer
                            st.session_state.neural_feature_maps = feature_maps
                            st.session_state.upload_result = result

        with col2:
            if 'upload_result' in st.session_state:
                result = st.session_state.upload_result

                # Display prediction
                st.markdown(
                    f"<div class='prediction-box'>{result['predicted_letter']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='confidence-box'>{result['confidence']*100:.1f}%</div>",
                    unsafe_allow_html=True
                )

                # Display confidence chart
                st.plotly_chart(
                    create_confidence_chart(result['probabilities']),
                    use_container_width=True
                )

        # NEURAL NETWORK VISUALIZATION SECTION
        if 'neural_feature_maps' in st.session_state and 'neural_visualizer' in st.session_state:
            st.divider()

            # Network Architecture
            st.subheader("Neural Network Architecture")
            st.markdown("**EfficientASLNet Structure** - How your image flows through the network:")

            arch_fig = st.session_state.neural_visualizer.visualize_network_architecture()
            if arch_fig:
                st.plotly_chart(arch_fig, use_container_width=True)

            st.divider()

            # Feature Maps Visualization
            st.subheader("Layer-by-Layer Feature Detection")
            st.markdown("""
            **What Each Layer Sees:**
            - **Conv1 (Early)**: Basic edges, lines, and contrasts
            - **Conv2-Conv3 (Middle)**: Hand shapes and finger patterns
            - **Conv4-Conv5 (Deep)**: Complex ASL gesture features
            """)

            feature_maps = st.session_state.neural_feature_maps
            feature_fig = st.session_state.neural_visualizer.visualize_feature_maps(feature_maps)
            if feature_fig:
                st.plotly_chart(feature_fig, use_container_width=True)

            st.divider()

            # Activation Statistics
            st.subheader("Neural Activation Analysis")
            st.markdown("**Layer Performance Metrics** - Statistical analysis of neuron activations:")

            stats_fig = st.session_state.neural_visualizer.create_activation_statistics(feature_maps)
            if stats_fig:
                st.plotly_chart(stats_fig, use_container_width=True)

            st.divider()

            # Final Prediction Analysis
            st.subheader("Final Layer Output Analysis")
            st.markdown("**Decision Process** - How the network arrives at its final prediction:")

            if 'neural_result' in st.session_state:
                conf_fig = st.session_state.neural_visualizer.create_confidence_visualization(
                    st.session_state.neural_result
                )
                if conf_fig:
                    st.plotly_chart(conf_fig, use_container_width=True)

                # Technical Summary
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Predicted Letter", st.session_state.neural_result['predicted_letter'])

                with col2:
                    st.metric("Confidence", f"{st.session_state.neural_result['confidence']*100:.1f}%")

                with col3:
                    st.metric("Network Depth", "5 Conv + 3 FC Layers")

                with col4:
                    st.metric("Parameters", "~2.5M")

    with tab3:
        # Same statistics functionality as original
        st.subheader("Performance Statistics")

        if st.session_state.inference_engine:
            stats = st.session_state.inference_engine.get_statistics()

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"<div class='stats-box'><h3>{stats['total_predictions']}</h3><p>Total Predictions</p></div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"<div class='stats-box'><h3>{stats['confident_predictions']}</h3><p>Confident Predictions</p></div>",
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"<div class='stats-box'><h3>{stats['confidence_rate']:.1f}%</h3><p>Confidence Rate</p></div>",
                    unsafe_allow_html=True
                )

            with col4:
                st.markdown(
                    f"<div class='stats-box'><h3>{stats['buffer_size']}</h3><p>Buffer Size</p></div>",
                    unsafe_allow_html=True
                )

            # Prediction history (same as original)
            if len(st.session_state.prediction_history) > 0:
                st.subheader("Prediction History")

                # Show chart
                history_chart = create_history_chart(st.session_state.prediction_history[-50:])
                if history_chart:
                    st.plotly_chart(history_chart, use_container_width=True)

                # Show recent predictions table
                st.subheader("Recent Predictions")
                recent_df = pd.DataFrame(st.session_state.prediction_history[-20:])
                recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(recent_df, use_container_width=True, hide_index=True)

                # Download button
                csv = recent_df.to_csv(index=False)
                st.download_button(
                    label="Download History as CSV",
                    data=csv,
                    file_name=f"asl_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No predictions yet. Start using the camera to see statistics!")

# Footer
st.divider()
st.markdown("""
    <p style='text-align: center; color: #666;'>
        ASL Alphabet Recognition Project - Enhanced with Neural Network Visualizer | 
        Powered by PyTorch & Streamlit | Optimized for RTX 4060
    </p>
""", unsafe_allow_html=True)
