import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import io
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
import shutil
import glob

# Import the FOCAL model classes
from models.hrnet import FOCAL_HRNet
from models.vit import FOCAL_ViT

# Set page config
st.set_page_config(
    page_title="FOCAL - Multi-Image Forgery Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #155a8a;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .image-card {
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        background-color: #ffffff;
    }
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 3px;
        margin: 1rem 0;
    }
    .progress-fill {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for model management
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'uploaded_models' not in st.session_state:
    st.session_state.uploaded_models = []

# Model management functions
def get_available_models():
    """Get list of available model files"""
    models = []
    weights_dir = 'weights'
    if os.path.exists(weights_dir):
        model_files = glob.glob(os.path.join(weights_dir, '*.pth'))
        models = [os.path.basename(f) for f in model_files]
    return models

def save_uploaded_model(uploaded_file):
    """Save uploaded model file to weights directory"""
    try:
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        
        file_path = os.path.join(weights_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return True, f"Model '{uploaded_file.name}' uploaded successfully!"
    except Exception as e:
        return False, f"Error uploading model: {str(e)}"

def show_available_models():
    """Display available models"""
    available_models = get_available_models()
    
    if available_models:
        st.success(f"Found {len(available_models)} available models:")
        for i, model in enumerate(available_models, 1):
            st.write(f"**{i}.** {model}")
            
            # Show file size
            file_path = os.path.join('weights', model)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                st.caption(f"Size: {size_mb:.1f} MB")
    else:
        st.warning("No model files found in the weights directory.")

def on_model_change():
    """Callback when model selection changes"""
    st.session_state.selected_model = st.session_state.model_selector

def select_model_interface():
    """Interface for selecting a model"""
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No models available. Please upload a model first.")
        return None
    
    st.write("**Available Models:**")
    
    # Get current index for selectbox
    current_index = 0
    if st.session_state.selected_model and st.session_state.selected_model in available_models:
        current_index = available_models.index(st.session_state.selected_model)
    
    selected_model = st.selectbox(
        "Choose a model to use:",
        options=available_models,
        index=current_index,
        key="model_selector",
        on_change=on_model_change
    )
    
    # Show success message if model is selected
    if st.session_state.selected_model:
        st.success(f"Selected model: **{st.session_state.selected_model}**")
    
    return selected_model

# FOCAL class implementation based on original main.py
class FOCAL(nn.Module):
    def __init__(self, net_list=[('HRNet', '')]):
        super(FOCAL, self).__init__()
        self.network_list = []
        try:
            for net_name, net_weight in net_list:
                if net_name == 'HRNet':
                    cur_net = FOCAL_HRNet()
                elif net_name == 'ViT':
                    cur_net = FOCAL_ViT()
                else:
                    st.error('Error: Undefined Network.')
                    raise ValueError('Undefined Network')
                
                cur_net = nn.DataParallel(cur_net)
                if torch.cuda.is_available():
                    cur_net = cur_net.cuda()
                
                if net_weight != '':
                    self.load(cur_net, net_weight)
                
                self.network_list.append(cur_net)
            
            self.clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)
            
            
        except Exception as e:
            st.error(f'Error initializing FOCAL model: {str(e)}')
            raise e

    def process(self, Ii, isTrain=False):
        with torch.no_grad():
            Fo = self.network_list[0](Ii)
            Fo = Fo.permute(0, 2, 3, 1)
            B, H, W, C = Fo.shape
            Fo = F.normalize(Fo, dim=3)
            
            Mo = None
            Fo = torch.flatten(Fo, start_dim=1, end_dim=2)
            result = self.clustering(x=Fo, k=2)
            Lo_batch = result.labels
            
            for idx in range(B):
                Lo = Lo_batch[idx]
                if torch.sum(Lo) > torch.sum(1 - Lo):
                    Lo = 1 - Lo
                Lo = Lo.view(H, W)[None, :, :, None]
                Mo = torch.cat([Mo, Lo], dim=0) if Mo is not None else Lo
            
            Mo = Mo.permute(0, 3, 1, 2)
            return Mo

    def load(self, extractor, path=''):
        try:
            weights_path = os.path.join('weights', path)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            weights_file = torch.load(weights_path, map_location='cpu')
            cur_weights = extractor.state_dict()
            
            # Count loaded keys
            loaded_keys = 0
            for key in weights_file:
                if key in cur_weights.keys() and weights_file[key].shape == cur_weights[key].shape:
                    cur_weights[key] = weights_file[key]
                    loaded_keys += 1
            
            extractor.load_state_dict(cur_weights)
            st.success(f'Loaded [{extractor.module.name}] from [weights/{path}] - {loaded_keys} keys loaded')
            
        except Exception as e:
            st.error(f'Error loading weights from {path}: {str(e)}')
            raise e

def load_focal_model(model_name=None):
    """Load the FOCAL model with pre-trained weights"""
    try:
        # Use selected model or default
        if model_name is None:
            # Try to use session state selected model
            if st.session_state.selected_model:
                model_name = st.session_state.selected_model
            else:
                # Check if default model exists
                available_models = get_available_models()
                if 'new_small_cnn_model_eph_700.pth' in available_models:
                    model_name = 'new_small_cnn_model_eph_700.pth'
                elif available_models:
                    model_name = available_models[0]  # Use first available model
                else:
                    st.error("No model files found. Please upload a model first.")
                    return None
        
        model = FOCAL([
            ('HRNet', model_name),
        ])
        
        if model is None:
            st.error("Failed to initialize FOCAL model")
            return None
        
        # Set to evaluation mode
        for net in model.network_list:
            net.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            st.success(f"Model '{model_name}' loaded successfully on GPU")
        else:
            st.warning(f"Model '{model_name}' loaded on CPU (GPU not available)")
        
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Full error details: {e}")
        return None

def preprocess_image(image, target_size=1024):
    """Preprocess image for the model"""
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = np.array(image)
    
    # Ensure we have 3 channels (RGB)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB by compositing over white background
        alpha = image[:, :, 3:4] / 255.0
        rgb = image[:, :, :3] / 255.0
        white = np.ones_like(rgb)
        image = rgb * alpha + white * (1 - alpha)
        image = (image * 255).astype(np.uint8)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB, just normalize
        pass
    else:
        raise ValueError(f"Unsupported image format with shape: {image.shape}")
    
    # Store original RGB image for overlay
    original_rgb = image.copy()
    
    # Convert RGB to BGR for OpenCV (like original main.py)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize image
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_rgb

def process_single_image(model, image, target_size=1024):
    """Process a single image through the model"""
    try:
        # Preprocess image
        image_tensor, original_rgb = preprocess_image(image, target_size)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Run inference on single image
        mask_tensor = model.process(image_tensor, isTrain=False)
        
        return mask_tensor, original_rgb
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def thresholding(x, thres=0.5):
    """Threshold function from original main.py"""
    # Convert to uint8 and apply threshold
    x = x.astype(np.uint8)
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x

def postprocess_mask(mask_tensor, original_shape):
    """Postprocess the mask output from the model"""
    try:
        # Convert tensor to numpy using the same method as original main.py
        # The model output is in range [0,1], so multiply by 255 first
        mask = mask_tensor.squeeze().cpu().detach().numpy()
        
        # Convert from [0,1] to [0,255] like in original convert function
        mask = mask * 255.0
        
        # Apply thresholding like in original code
        mask = thresholding(mask)
        
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # Resize to original image size - cv2.resize expects (width, height)
        # original_shape should be (width, height) from PIL
        target_size = (original_shape[0], original_shape[1])  # (width, height)
        
        # Ensure mask is uint8 for OpenCV
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        mask = cv2.resize(mask, target_size)
        
        return mask
        
    except Exception as e:
        st.error(f"Error in postprocess_mask: {str(e)}")
        st.error(f"Mask tensor shape: {mask_tensor.shape if mask_tensor is not None else 'None'}")
        st.error(f"Original shape: {original_shape}")
        raise e

def create_overlay(original_image, mask):
    """Create overlay visualization"""
    # Ensure mask is binary
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Convert PIL image to numpy array if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Ensure original image is in RGB format
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        # Already RGB, create a copy for overlay
        overlay = original_image.copy()
    else:
        overlay = original_image.copy()
    
    # Create red overlay for detected regions (RGB format)
    red_color = np.array([255, 0, 0], dtype=np.uint8)  # RGB format - Red
    
    # Apply red color to detected regions
    for i in range(3):  # Apply to each channel
        overlay[:, :, i][binary_mask > 0] = red_color[i]
    
    # Blend with transparency
    alpha = 0.4
    blended = cv2.addWeighted(original_image.astype(np.float32), 1-alpha, overlay.astype(np.float32), alpha, 0)
    
    return blended.astype(np.uint8)



def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Image Tamper Detection</h1>', unsafe_allow_html=True)
    # st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 1rem;">Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering</p>', unsafe_allow_html=True)
    
    # Model Management Section
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">🤖 Model Management</h2>', unsafe_allow_html=True)
    
    # Create three columns for the buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Select Model", use_container_width=True, type="primary"):
            st.session_state.show_select_model = True
            st.session_state.show_available_models = False
            st.session_state.show_upload_model = False
    
    with col2:
        if st.button("📋 Available Models", use_container_width=True):
            st.session_state.show_available_models = True
            st.session_state.show_select_model = False
            st.session_state.show_upload_model = False
    
    with col3:
        if st.button("📤 Upload Model", use_container_width=True):
            st.session_state.show_upload_model = True
            st.session_state.show_select_model = False
            st.session_state.show_available_models = False
    
    # Initialize session state for button visibility
    if 'show_select_model' not in st.session_state:
        st.session_state.show_select_model = False
    if 'show_available_models' not in st.session_state:
        st.session_state.show_available_models = False
    if 'show_upload_model' not in st.session_state:
        st.session_state.show_upload_model = False
    
    # Show selected interface
    if st.session_state.show_select_model:
        st.markdown("### Select Model for Inference")
        select_model_interface()
        
    elif st.session_state.show_available_models:
        st.markdown("### Available Models")
        show_available_models()
        
    elif st.session_state.show_upload_model:
        st.markdown("### Upload Model File")
        uploaded_model = st.file_uploader(
            "Choose a .pth model file",
            type=['pth'],
            help="Upload a PyTorch model file (.pth format)"
        )
        
        if uploaded_model is not None:
            if st.button("Save Model", type="primary"):
                success, message = save_uploaded_model(uploaded_model)
                if success:
                    st.success(message)
                    # Clear the uploaded file after successful save
                    st.session_state.pop('uploaded_model_file', None)
                else:
                    st.error(message)
    
    # Show currently selected model
    if st.session_state.selected_model:
        st.info(f"**Currently selected model:** {st.session_state.selected_model}")
    else:
        st.warning("No model selected. Please select a model to run inference.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.markdown("### Model Configuration")
    
    # Model loading status in sidebar
    if st.session_state.selected_model:
        st.sidebar.success(f"✅ Model ready: {st.session_state.selected_model}")
    else:
        st.sidebar.warning("⚠️ No model selected")
    
    # Main content
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📤 Upload Multiple Images</h2>', unsafe_allow_html=True)
    st.markdown('<p>Upload multiple images to detect potential forgery regions in batch</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files is not None and len(uploaded_files) > 0:
        # Display uploaded images count
        st.markdown(f'<div class="result-section"><h3 class="sub-header">📷 Uploaded Images ({len(uploaded_files)} files)</h3></div>', unsafe_allow_html=True)
        
        # Load and display original images in a grid
        original_images = []
        filenames = []
        
        for uploaded_file in uploaded_files:
            original_image = Image.open(uploaded_file)
            original_images.append(original_image)
            filenames.append(uploaded_file.name)
        
        # Display images in a grid
        cols = st.columns(min(3, len(original_images)))
        for i, (img, filename) in enumerate(zip(original_images, filenames)):
            with cols[i % 3]:
                st.image(img, caption=filename, use_container_width=True)
        
        # Run batch inference button
        if st.button("🔍 Run Forgery Detection", type="primary"):
            # Check if model is selected
            if not st.session_state.selected_model:
                st.error("⚠️ Please select a model first using the 'Select Model' button above.")
                return
            
            # Load the selected model
            with st.spinner(f"Loading model: {st.session_state.selected_model}..."):
                model = load_focal_model()
                
            if model is None:
                st.error("Failed to load the selected model. Please check the model file.")
                return
                
            with st.spinner("Running forgery detection..."):
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    # Process each image individually to avoid memory issues
                    for i, (original_image, filename) in enumerate(zip(original_images, filenames)):
                        status_text.text(f"Processing image {i+1}/{len(original_images)}: {filename}")
                        
                        # Process single image
                        mask_tensor, processed_image = process_single_image(model, original_image)
                        
                        if mask_tensor is not None:
                            # Postprocess mask
                            mask = postprocess_mask(mask_tensor, original_image.size)
                            
                            # Create overlay
                            overlay = create_overlay(np.array(original_image), mask)
                            
                            results.append((original_image, mask, overlay, filename))
                        else:
                            st.error(f"Failed to process {filename}")
                            continue
                        
                        # Update progress
                        progress = int((i + 1) * 90 / len(original_images))
                        progress_bar.progress(progress)
                        
                        # Clear GPU memory after each image to prevent accumulation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    if results:
                        # Display results
                        st.markdown('<div class="result-section">', unsafe_allow_html=True)
                        st.markdown('<h3 class="sub-header">🎯 Detection Results</h3>', unsafe_allow_html=True)
                        
                        # Display results in a grid
                        for i, (original, mask, overlay, filename) in enumerate(results):
                            st.markdown(f'<h4>Image {i+1}: {filename}</h4>', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                                st.markdown('<h5>🖼️ Original</h5>', unsafe_allow_html=True)
                                st.image(original,caption='Original Image', use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                                st.markdown('<h5>🎭 Detection Mask</h5>', unsafe_allow_html=True)
                                st.image(mask, caption="Forgery Detection Mask", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                                st.markdown('<h5>🔴 Overlay Visualization</h5>', unsafe_allow_html=True)
                                st.image(overlay, caption="Overlay on Original Image", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("---")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("No images were processed successfully. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error during batch processing: {str(e)}")
                    st.error("Please check the image formats and try again.")
    

if __name__ == "__main__":
    main()
