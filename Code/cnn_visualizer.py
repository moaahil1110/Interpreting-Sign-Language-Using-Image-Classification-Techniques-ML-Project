
"""
Neural Network Visualizer for ASL CNN Model
Enhanced visualization of CNN layers, feature maps, and activations
Specifically designed for EfficientASLNet architecture
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from torchvision import transforms
import streamlit as st
from collections import OrderedDict
import matplotlib.cm as cm

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

    def visualize_prediction_flow(self, image, prediction_result):
        """
        Create a comprehensive visualization showing the prediction flow

        Args:
            image: Input image
            prediction_result: Prediction results from inference engine

        Returns:
            Dictionary containing all visualization figures
        """
        # Prepare image tensor
        if isinstance(image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)
        else:
            img_tensor = image

        # Extract feature maps
        feature_maps = self.extract_feature_maps(img_tensor)

        # Create visualizations
        visualizations = {
            'architecture': self.visualize_network_architecture(),
            'feature_maps': self.visualize_feature_maps(feature_maps),
            'activation_stats': self.create_activation_statistics(feature_maps),
            'prediction_confidence': self.create_confidence_visualization(prediction_result)
        }

        return visualizations

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

def create_enhanced_visualizer_tab():
    """
    Create the enhanced neural network visualizer tab content
    This function will be integrated into the main Streamlit app
    """
    st.subheader("🧠 Neural Network Visualizer")

    if not st.session_state.model_loaded or not st.session_state.inference_engine:
        st.warning("Please load the model first to use the neural network visualizer.")
        return

    st.markdown("""
    ### How the CNN Processes Your Image
    This visualizer shows you exactly how the EfficientASLNet model analyzes your uploaded image:

    - **Network Architecture**: Visual representation of all layers
    - **Feature Maps**: What each convolutional layer "sees" 
    - **Activation Statistics**: Numerical analysis of layer outputs
    - **Prediction Flow**: Complete journey from image to prediction
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Image upload for visualization
        viz_uploaded_file = st.file_uploader(
            "Upload image for neural network analysis",
            type=['jpg', 'jpeg', 'png'],
            key="viz_uploader",
            help="Upload an ASL hand sign image to see how the CNN processes it"
        )

        if viz_uploaded_file is not None:
            image = Image.open(viz_uploaded_file).convert('RGB')
            st.image(image, caption="Input Image", width="stretch")

            if st.button("🔬 Analyze Neural Network", key="analyze_nn"):
                with st.spinner("Analyzing neural network processing..."):
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

                    # Generate all visualizations
                    visualizations = visualizer.visualize_prediction_flow(image, result)

                    # Store in session state
                    st.session_state.nn_visualizations = visualizations
                    st.session_state.nn_prediction_result = result

    with col2:
        if 'nn_visualizations' in st.session_state and 'nn_prediction_result' in st.session_state:
            result = st.session_state.nn_prediction_result

            # Display prediction
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        color: white; font-size: 3rem; font-weight: bold; margin: 1rem 0;'>
                {result['predicted_letter']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem; border-radius: 15px; text-align: center; 
                        color: white; font-size: 1.5rem; margin: 1rem 0;'>
                Confidence: {result['confidence']*100:.2f}%
            </div>
            """, unsafe_allow_html=True)

    # Display visualizations
    if 'nn_visualizations' in st.session_state:
        visualizations = st.session_state.nn_visualizations

        st.divider()

        # Network Architecture
        st.subheader("🏗️ Network Architecture")
        st.markdown("This shows the structure of the EfficientASLNet model with each layer:")
        if visualizations['architecture']:
            st.plotly_chart(visualizations['architecture'], use_container_width=True)

        st.divider()

        # Feature Maps
        st.subheader("🎯 Feature Maps - What Each Layer Sees")
        st.markdown("""
        Feature maps show what patterns each convolutional layer detects:
        - **Early layers** (conv1, conv2): Detect edges, lines, and basic shapes
        - **Middle layers** (conv3, conv4): Detect hand shapes and finger positions  
        - **Later layers** (conv5): Detect complex hand gesture patterns
        """)

        if visualizations['feature_maps']:
            st.plotly_chart(visualizations['feature_maps'], use_container_width=True)

        st.divider()

        # Activation Statistics
        st.subheader("📊 Layer Activation Analysis")
        st.markdown("""
        Statistical analysis of neural network activations:
        - **Mean Activation**: Average activity level in each layer
        - **Standard Deviation**: How varied the activations are
        - **Max Activation**: Strongest response in each layer
        - **Active Neurons**: Percentage of neurons firing (>0)
        """)

        if visualizations['activation_stats']:
            st.plotly_chart(visualizations['activation_stats'], use_container_width=True)

        st.divider()

        # Prediction Confidence
        st.subheader("🎯 Neural Network Output Analysis")
        st.markdown("Final layer outputs showing confidence scores for each ASL letter:")

        if visualizations['prediction_confidence']:
            st.plotly_chart(visualizations['prediction_confidence'], use_container_width=True)

        # Technical details
        st.divider()
        st.subheader("🔧 Technical Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Type", "EfficientASLNet")
            st.metric("Total Parameters", "~2.5M")

        with col2:
            st.metric("Input Size", "224×224×3")
            st.metric("Output Classes", "26 (A-Z)")

        with col3:
            st.metric("Conv Layers", "5")
            st.metric("FC Layers", "3")

# Additional utility function
def get_layer_names_for_visualization():
    """Return the standard layer names for visualization"""
    return ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

print("✅ Neural Network Visualizer code created successfully!")
