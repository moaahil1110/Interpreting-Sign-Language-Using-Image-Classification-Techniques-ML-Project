"""
🤟 ASL Alphabet Recognition Dashboard
Real-time American Sign Language Recognition System
Powered by Deep Learning & PyTorch
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import torch
from asl_inference_engine import ASLInferenceEngine
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="ASL Alphabet Recognition",
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
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .confidence-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stats-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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
st.markdown("<h1>🤟 ASL Alphabet Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Real-time American Sign Language Interpreter powered by Deep Learning</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/vscode-icons/main/icons/file_type_ai.svg", width=100)
    st.header("⚙️ Configuration")
    
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
    if st.button("🚀 Load Model"):
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
        st.subheader("📊 Model Info")
        engine = st.session_state.inference_engine
        st.info(f"**Device:** {engine.device}")
        st.info(f"**Classes:** {len(engine.class_names)}")
        st.info(f"**Letters:** {', '.join(engine.class_names[:5])}...")

# Main Content Area
if not st.session_state.model_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h2 style='text-align: center; margin-top: 0;'>👋 Welcome!</h2>
            <p style='text-align: center; font-size: 1.1rem; margin-bottom: 0;'>
                Please load the ASL recognition model from the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📋 How to Use:
        1. **Load Model**: Click "🚀 Load Model" in the sidebar
        2. **Start Camera**: Navigate to the "Live Camera" tab
        3. **Show Hand Signs**: Place your hand in the green ROI box
        4. **View Results**: See real-time predictions and confidence scores
        
        ### 🎯 Features:
        - ✨ Real-time ASL alphabet recognition
        - 📊 Live confidence metrics and visualizations
        - 📸 Image upload support
        - 📈 Prediction history tracking
        - 🎨 Beautiful, intuitive interface
        """)
else:
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📹 Live Camera", "📸 Upload Image", "📊 Statistics"])
    
    with tab1:
        st.subheader("Real-time ASL Recognition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera control buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                start_camera = st.button("▶️ Start Camera", width="stretch")
            with btn_col2:
                stop_camera = st.button("⏹️ Stop Camera", width="stretch")
            with btn_col3:
                reset_btn = st.button("🔄 Reset Buffer", width="stretch")
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
        with col2:
            # Prediction display
            prediction_display = st.empty()
            confidence_display = st.empty()
            chart_display = st.empty()
            top5_display = st.empty()
        
        # Handle camera controls
        if start_camera:
            st.session_state.camera_active = True
        
        if stop_camera:
            st.session_state.camera_active = False
        
        if reset_btn and st.session_state.inference_engine:
            st.session_state.inference_engine.reset_buffer()
            st.session_state.prediction_history = []
            st.success("Buffer reset!")
        
        # Camera loop
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            engine = st.session_state.inference_engine
            frame_count = 0
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Draw ROI and get hand region
                frame, roi = draw_hand_roi(frame, roi_size)
                
                # Make prediction every 3 frames to improve performance
                if frame_count % 3 == 0 and roi.size > 0:
                    try:
                        result = engine.predict(roi, return_probabilities=True)
                        
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
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                
                frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
            
            cap.release()
    
    with tab2:
        st.subheader("Upload Image for Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image of an ASL hand sign"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", width="stretch")
                
                if st.button("🔍 Analyze Image", width="stretch"):
                    with st.spinner("Analyzing..."):
                        result = st.session_state.inference_engine.predict(
                            image, return_probabilities=True
                        )
                        
                        # Store result in session state
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
                    width="stretch"
                )
                
                # Display top 5 predictions
                st.subheader("Top 5 Predictions")
                for i, (letter, prob) in enumerate(result['top5'], 1):
                    st.progress(prob, text=f"{i}. **{letter}** - {prob*100:.2f}%")
    
    with tab3:
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
            
            # Prediction history
            if len(st.session_state.prediction_history) > 0:
                st.subheader("Prediction History")
                
                # Show chart
                history_chart = create_history_chart(st.session_state.prediction_history[-50:])
                if history_chart:
                    st.plotly_chart(history_chart, width="stretch")
                
                # Show recent predictions table
                st.subheader("Recent Predictions")
                recent_df = pd.DataFrame(st.session_state.prediction_history[-20:])
                recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(recent_df, width="stretch", hide_index=True)
                
                # Download button
                csv = recent_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download History as CSV",
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
        🎓 ASL Alphabet Recognition Project | Powered by PyTorch & Streamlit | Optimized for RTX 4060
    </p>
""", unsafe_allow_html=True)
