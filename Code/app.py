"""
🤟 ASL Alphabet Recognition Dashboard - OPTIMIZED VERSION
Fixed: Camera lag, smoothing issues, flickering charts, warnings suppressed
Real-time American Sign Language Recognition System
Powered by Deep Learning & PyTorch
"""

# SUPPRESS ALL WARNINGS AND ERRORS
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.CRITICAL)

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
from collections import OrderedDict, deque
import os
from asl_evaluate_pytorch import evaluate

# Page configuration
st.set_page_config(
    page_title="ASL Alphabet Recognition - Optimized",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (same as before but with fixed chart containers)
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
    /* Fixed chart container to prevent resizing */
    .chart-container {
        height: 300px !important;
        overflow: hidden;
    }
    /* Stable plotly container */
    .js-plotly-plot {
        height: 300px !important;
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
        color: #333;
    }
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
        fill: currentColor !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
        color: #222;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #333;
        color: #ffffff !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ENHANCED PREDICTION SMOOTHING CLASS
class OptimizedPredictionSmoothing:
    """Enhanced prediction smoothing with temporal stability and reduced flickering"""
    
    def __init__(self, window_size=8, confidence_threshold=0.6, stability_threshold=0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.probability_history = deque(maxlen=5)  # For smooth chart transitions
        
    def add_prediction(self, letter, confidence, probabilities):
        """Add new prediction with enhanced smoothing"""
        self.predictions.append(letter)
        self.confidences.append(confidence)
        self.probability_history.append(probabilities)
        
        return self.get_stable_prediction()
    
    def get_stable_prediction(self):
        """Get smoothed prediction with stability check"""
        if len(self.predictions) < 3:
            return self.predictions[-1] if self.predictions else None
        
        # Count occurrences in recent predictions
        letter_counts = {}
        recent_predictions = list(self.predictions)[-5:]
        
        for letter in recent_predictions:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        # Get most stable letter
        most_common_letter = max(letter_counts, key=letter_counts.get)
        stability_ratio = letter_counts[most_common_letter] / len(recent_predictions)
        
        # Return stable prediction
        if stability_ratio >= self.stability_threshold:
            return most_common_letter
        else:
            return self.predictions[-1]
    
    def get_smooth_probabilities(self):
        """Get smoothed probabilities for stable chart updates"""
        if not self.probability_history:
            return {}
        
        # Average probabilities over recent frames for smoother charts
        all_letters = set()
        for probs in self.probability_history:
            all_letters.update(probs.keys())
        
        smoothed_probs = {}
        for letter in all_letters:
            values = [probs.get(letter, 0) for probs in self.probability_history]
            smoothed_probs[letter] = sum(values) / len(values)
        
        return smoothed_probs

# Initialize enhanced session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_smoother' not in st.session_state:
    st.session_state.prediction_smoother = OptimizedPredictionSmoothing()
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0
if 'chart_update_counter' not in st.session_state:
    st.session_state.chart_update_counter = 0

def load_model(model_path, device='cuda', smoothing=8, threshold=0.7):
    """Load the ASL model with error handling"""
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

def draw_hand_roi_optimized(frame, roi_size=300):
    """Optimized ROI drawing with better performance"""
    try:
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_x1 = max(0, center_x - roi_size // 2)
        roi_y1 = max(0, center_y - roi_size // 2)
        roi_x2 = min(w, center_x + roi_size // 2)
        roi_y2 = min(h, center_y + roi_size // 2)

        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(frame, "Hand Here", (roi_x1, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            
        return frame, roi
        
    except Exception:
        return frame, np.zeros((300, 300, 3), dtype=np.uint8)

def create_stable_confidence_chart(probabilities, top_n=5):
    """Create stable confidence chart that doesn't flicker"""
    try:
        # Get top N predictions
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        letters = [item[0] for item in sorted_probs]
        probs = [item[1] * 100 for item in sorted_probs]
        
        # Fixed color scheme
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=probs,
            y=letters,
            orientation='h',
            marker=dict(
                color=[colors[i % len(colors)] for i in range(len(probs))],
                line=dict(width=1, color='white')
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
            textfont=dict(size=11, color='white'),
            showlegend=False
        ))

        # Fixed layout to prevent resizing/flickering
        fig.update_layout(
            title=dict(
                text="Top 5 Predictions",
                font=dict(size=14, color='#333'),
                x=0.5
            ),
            xaxis=dict(
                title="Confidence (%)",
                range=[0, max(100, max(probs) + 10)],  # Dynamic but stable range
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="",
                categoryorder='total ascending'
            ),
            height=280,  # Fixed height
            margin=dict(l=50, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=10, color="#333"),
            transition=dict(duration=200, easing="cubic-in-out")  # Smooth transitions
        )
        
        return fig
        
    except Exception:
        # Return empty chart on error
        return go.Figure()

def create_history_chart(history):
    """Create line chart for prediction history"""
    try:
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
    except Exception:
        return None

# Main App Header
st.markdown("<h1>ASL Alphabet Recognition - Optimized</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Smooth Real-time ASL Interpreter with Enhanced Performance</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
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
        min_value=3,
        max_value=15,
        value=8,
        help="Higher values = more stable predictions"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
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
                st.session_state.prediction_smoother = OptimizedPredictionSmoothing(
                    window_size=smoothing_window,
                    confidence_threshold=confidence_threshold
                )
                st.success(message)
            else:
                st.error(message)

    # Camera settings
    st.subheader("Camera Settings")
    camera_id = st.number_input(
        "Camera ID",
        min_value=0,
        max_value=10,
        value=0
    )

    roi_size = st.slider(
        "ROI Size",
        min_value=200,
        max_value=600,
        value=400,
        step=50
    )
    
    # Performance settings
    st.subheader("Performance")
    frame_skip = st.slider(
        "Frame Skip (Higher = Less Lag)",
        min_value=5,
        max_value=20,
        value=10,
        help="Process every Nth frame"
    )

    # Model info
    if st.session_state.model_loaded and st.session_state.inference_engine:
        st.divider()
        st.subheader("Model Information")
        engine = st.session_state.inference_engine
        st.info(f"**Device:** {engine.device}")
        st.info(f"**Classes:** {len(engine.class_names)}")

# Main Content Area
if not st.session_state.model_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h2 style='text-align: center; margin-top: 0;'>Welcome to Optimized ASL Recognition!</h2>
            <p style='text-align: center; font-size: 1.1rem; margin-bottom: 0;'>
                Please load the model from the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### ✨ New Optimizations:
        - **Reduced Camera Lag**: Optimized frame processing
        - **Smooth Charts**: No more flickering bars
        - **Better Smoothing**: Enhanced prediction stability
        - **Error Suppression**: Clean, quiet operation
        - **Performance Controls**: Adjustable frame skip rates

        ### How to Use:
        1. **Load Model**: Click "Load Model" in the sidebar
        2. **Adjust Settings**: Use sliders to optimize performance
        3. **Start Camera**: Begin real-time recognition
        4. **Monitor Performance**: Check statistics tab
        """)
else:
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Camera", "🖼️ Upload Image", "📊 Statistics", "🧪 Testing"])

    with tab1:
        st.subheader("🎥 Real-time ASL Recognition - Optimized")

        col1, col2 = st.columns([2.2, 1])

        with col1:
            # Camera control buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                start_camera = st.button("▶️ Start Camera")
            with btn_col2:
                stop_camera = st.button("⏹️ Stop Camera")
            with btn_col3:
                reset_btn = st.button("🔄 Reset Buffer")

            # Camera feed placeholder
            camera_placeholder = st.empty()

        with col2:
            # Prediction displays
            prediction_display = st.empty()
            confidence_display = st.empty()
            
            # Chart container with fixed size
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            chart_display = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        # Handle camera controls
        if start_camera:
            st.session_state.camera_active = True

        if stop_camera:
            st.session_state.camera_active = False

        if reset_btn:
            st.session_state.prediction_smoother = OptimizedPredictionSmoothing()
            st.session_state.prediction_history = []
            st.success("✅ Buffer reset!")

        # OPTIMIZED CAMERA LOOP
        if st.session_state.camera_active and st.session_state.inference_engine:
            try:
                cap = cv2.VideoCapture(camera_id)
                
                # Optimize camera settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                cap.set(cv2.CAP_PROP_FPS, 20)  # Limit FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Fixed resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if not cap.isOpened():
                    st.error(f"❌ Could not open camera {camera_id}")
                    st.session_state.camera_active = False
                else:
                    frame_count = 0
                    last_chart_update = 0
                    
                    while st.session_state.camera_active:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_count += 1
                        current_time = time.time()
                        
                        # Draw ROI
                        frame, roi = draw_hand_roi_optimized(frame, roi_size)
                        
                        # Process predictions with reduced frequency
                        if (frame_count % frame_skip == 0 and roi.size > 0 and 
                            current_time - st.session_state.last_prediction_time > 0.25):
                            
                            try:
                                # Get prediction
                                result = st.session_state.inference_engine.predict(roi, return_probabilities=True)
                                
                                # Apply smoothing
                                smooth_letter = st.session_state.prediction_smoother.add_prediction(
                                    result['predicted_letter'], 
                                    result['confidence'],
                                    result['probabilities']
                                )
                                
                                confidence_pct = result['confidence'] * 100
                                
                                # Update displays
                                prediction_display.markdown(
                                    f"<div class='prediction-box'>{smooth_letter}</div>",
                                    unsafe_allow_html=True
                                )
                                
                                confidence_display.markdown(
                                    f"<div class='confidence-box'>{confidence_pct:.1f}%</div>",
                                    unsafe_allow_html=True
                                )
                                
                                # Update chart less frequently to prevent flickering
                                if current_time - last_chart_update > 0.5:  # Update every 0.5 seconds
                                    smooth_probs = st.session_state.prediction_smoother.get_smooth_probabilities()
                                    if smooth_probs:
                                        chart_fig = create_stable_confidence_chart(smooth_probs)
                                        
                                        # Use unique key to prevent flickering
                                        st.session_state.chart_update_counter += 1
                                        chart_display.plotly_chart(
                                            chart_fig, 
                                            use_container_width=True,
                                            key=f"stable_chart_{st.session_state.chart_update_counter}"
                                        )
                                        last_chart_update = current_time
                                
                                # Add overlay to frame
                                cv2.putText(frame, f"{smooth_letter}", (15, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                cv2.putText(frame, f"{confidence_pct:.1f}%", (15, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                                
                                st.session_state.last_prediction_time = current_time
                                
                                # Add to history less frequently
                                if frame_count % (frame_skip * 2) == 0:
                                    st.session_state.prediction_history.append({
                                        'timestamp': datetime.now(),
                                        'letter': smooth_letter,
                                        'confidence': confidence_pct
                                    })
                                    
                            except Exception:
                                pass  # Silently handle prediction errors
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", width=700)
                        
                        # Controlled frame rate
                        time.sleep(0.04)  # ~25 FPS max
                        
                cap.release()
                
            except Exception:
                st.session_state.camera_active = False

    with tab2:
        # Upload and analyze (simplified version)
        st.subheader("📤 Upload Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an ASL hand sign image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", width=300)
                
                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing..."):
                        result = st.session_state.inference_engine.predict(
                            image, return_probabilities=True
                        )
                        st.session_state.upload_result = result
            
            with col2:
                if 'upload_result' in st.session_state:
                    result = st.session_state.upload_result
                    
                    st.markdown(
                        f"<div class='prediction-box'>{result['predicted_letter']}</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        f"<div class='confidence-box'>{result['confidence']*100:.1f}%</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Display chart
                    chart_fig = create_stable_confidence_chart(result['probabilities'])
                    st.plotly_chart(chart_fig, use_container_width=True)

    with tab3:
        # Statistics tab
        st.subheader("📈 Performance Statistics")

        if st.session_state.inference_engine:
            stats = st.session_state.inference_engine.get_statistics()

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
                buffer_size = len(st.session_state.prediction_smoother.predictions)
                st.markdown(
                    f"<div class='stats-box'><h3>{buffer_size}</h3><p>Buffer Size</p></div>",
                    unsafe_allow_html=True
                )

            # Prediction history
            if len(st.session_state.prediction_history) > 0:
                st.subheader("📊 Prediction History")

                history_chart = create_history_chart(st.session_state.prediction_history[-50:])
                if history_chart:
                    st.plotly_chart(history_chart, use_container_width=True)

                st.subheader("📝 Recent Predictions")
                recent_df = pd.DataFrame(st.session_state.prediction_history[-20:])
                if not recent_df.empty:
                    recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
                    st.dataframe(recent_df, use_container_width=True, hide_index=True)
            else:
                st.info("📱 Start using the camera to see statistics!")

    with tab4:
        # Testing tab (simplified)
        st.subheader("🧪 Model Evaluation")
        
        test_data_dir = st.text_input(
            "Test dataset directory",
            value="validationResults"
        )
        
        if st.button("🚀 Run Evaluation"):
            try:
                with st.spinner("Running evaluation..."):
                    summary = evaluate(
                        model_path=model_path, 
                        test_data_path=test_data_dir, 
                        results_dir="validationResults"
                    )
                st.success(f"✅ Evaluation complete! Accuracy: {summary['accuracy']:.2f}%")
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")

# Footer
st.divider()
st.markdown("""
    <p style='text-align: center; color: #666;'>
        🚀 ASL Recognition - Optimized for Performance | Smooth • Fast • Accurate
    </p>
""", unsafe_allow_html=True)