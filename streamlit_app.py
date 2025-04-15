import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from PIL import Image

def bilinear_transform(b, a, fs):
    """Apply bilinear transformation to convert analog filter to digital filter"""
    bz, az = signal.bilinear(b, a, fs)
    return bz, az

def plot_frequency_response(bz, az, fs):
    """Plot frequency response of the digital filter"""
    w, h = signal.freqz(bz, az, worN=1024)
    freq = w * fs / (2 * np.pi)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq, 20 * np.log10(abs(h)), label='Magnitude Response', color='dodgerblue', linewidth=2)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title('Frequency Response of Digital Filter', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

def warp_image(image, scale_x, scale_y):
    """Apply bilinear transformation to warp an image"""
    height, width = image.shape[:2]
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    warped_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return warped_image

# Streamlit UI
st.set_page_config(page_title="Bilinear Transformation Calculator", layout="wide")
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .stSlider>div>div>div>div {
            background: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Interactive Bilinear Transformation Calculator ✨")
st.write("Dynamically adjust parameters and visualize transformations with an enhanced UI.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔢 Input Parameters")
    b = st.text_input("📌 Numerator Coefficients (comma-separated):", "1, 0")
    a = st.text_input("📌 Denominator Coefficients (comma-separated):", "1, 1")
    fs = st.slider("🎚️ Sampling Frequency (Hz):", min_value=1, max_value=1000, value=100, step=1)
    scale_x = st.slider("🔄 Image Warp Scale X", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    scale_y = st.slider("🔄 Image Warp Scale Y", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    #uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])
    
    if st.button("🚀 Compute & Visualize"):
        b = np.array([float(i) for i in b.split(',')])
        a = np.array([float(i) for i in a.split(',')])
        
        bz, az = bilinear_transform(b, a, fs)
        st.success("✅ Computation Successful!")
        st.write("**🔢 Digital Filter Coefficients:**")
        st.write(f"🔹 Numerator: {bz}")
        st.write(f"🔹 Denominator: {az}")
        
        with col2:
            st.header("📊 Frequency Response")
            plot_frequency_response(bz, az, fs)

#if uploaded_file is not None:
    #image = Image.open(uploaded_file)
    #image = np.array(image)
    #st.image(image, caption="🖼 Original Image", use_column_width=True)
    
    #warped_image = warp_image(image, scale_x, scale_y)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Initialize grid parameters
GRID_RESOLUTION = 30
x = np.linspace(-2, 2, GRID_RESOLUTION)
y = np.linspace(-2, 2, GRID_RESOLUTION)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

# Set up figure and axes
fig = plt.figure(figsize=(12, 8))
ax_original = fig.add_subplot(121)
ax_transformed = fig.add_subplot(122)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.4, top=0.9)

# Plot original grid
for i in range(len(y)):
    ax_original.plot(X[i,:], Y[i,:], color='blue', lw=0.7, alpha=0.6)
for j in range(len(x)):
    ax_original.plot(X[:,j], Y[:,j], color='blue', lw=0.7, alpha=0.6)

ax_original.set_title('Original Plane (Z)')
ax_original.set_xlabel('Real')
ax_original.set_ylabel('Imaginary')
ax_original.axis('equal')
ax_original.grid(True, alpha=0.3)

# Initialize transformed grid plot
transformed_lines = []
for i in range(len(y)):
    line, = ax_transformed.plot([], [], color='red', lw=0.7, alpha=0.6)
    transformed_lines.append(line)
for j in range(len(x)):
    line, = ax_transformed.plot([], [], color='red', lw=0.7, alpha=0.6)
    transformed_lines.append(line)

ax_transformed.set_title('Transformed Plane (W)')
ax_transformed.set_xlabel('Real')
ax_transformed.set_ylabel('Imaginary')
ax_transformed.set_xlim(-5, 5)
ax_transformed.set_ylim(-5, 5)
ax_transformed.grid(True, alpha=0.3)

# Create parameter sliders
slider_params = {
    'a_real': {'pos': [0.1, 0.35], 'init': 1.0},
    'a_imag': {'pos': [0.1, 0.3], 'init': 0.0},
    'b_real': {'pos': [0.1, 0.25], 'init': 0.0},
    'b_imag': {'pos': [0.1, 0.2], 'init': 0.0},
    'c_real': {'pos': [0.1, 0.15], 'init': 0.0},
    'c_imag': {'pos': [0.1, 0.1], 'init': 0.0},
    'd_real': {'pos': [0.1, 0.05], 'init': 1.0},
    'd_imag': {'pos': [0.1, 0.0], 'init': 0.0},
}

sliders = {}
for name, params in slider_params.items():
    ax = fig.add_axes(params['pos'] + [0.35, 0.03])
    slider = Slider(ax, name, -2.0, 2.0, valinit=params['init'])
    sliders[name] = slider

# Add reset button
reset_ax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', hovercolor='0.8')

def reset(event):
    for name, slider in sliders.items():
        slider.reset()

reset_button.on_clicked(reset)

# Transformation function with error handling
def bilinear_transform(z, a, b, c, d):
    denominator = c*z + d
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (a*z + b) / denominator
    return np.ma.masked_invalid(result)

# Update function
def update(val):
    a = sliders['a_real'].val + 1j*sliders['a_imag'].val
    b = sliders['b_real'].val + 1j*sliders['b_imag'].val
    c = sliders['c_real'].val + 1j*sliders['c_imag'].val
    d = sliders['d_real'].val + 1j*sliders['d_imag'].val
    
    W = bilinear_transform(Z, a, b, c, d)
    U = W.real
    V = W.imag
    
    # Update horizontal lines
    for i, line in enumerate(transformed_lines[:len(y)]):
        line.set_xdata(U[i,:])
        line.set_ydata(V[i,:])
    
    # Update vertical lines
    for j, line in enumerate(transformed_lines[len(y):]):
        line.set_xdata(U[:,j])
        line.set_ydata(V[:,j])
    
    # Auto-scale axes while maintaining reasonable bounds
    ax_transformed.set_xlim(np.nanmin(U)-0.5, np.nanmax(U)+0.5)
    ax_transformed.set_ylim(np.nanmin(V)-0.5, np.nanmax(V)+0.5)
    
    fig.canvas.draw_idle()

# Register update with all sliders
for slider in sliders.values():
    slider.on_changed(update)

# Initial update
update(None)

plt.show()
#st.image(warped_image, caption="🎨 Warped Image", use_column_width=True)