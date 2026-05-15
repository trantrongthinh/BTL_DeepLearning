import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import io
from PIL import Image

# Cấu hình trang
st.set_page_config(
    page_title="Nhận Dạng Nhạc Cụ",
    page_icon="🎵",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #6366F1;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Danh sách nhạc cụ
INSTRUMENTS = {
    0: 'Cello (Đàn Cello)',
    1: 'Clarinet (Kèn Clarinet)',
    2: 'Flute (Sáo)',
    3: 'Acoustic Guitar (Đàn Guitar thùng)',
    4: 'Electric Guitar (Đàn Guitar điện)',
    5: 'Organ (Đàn Organ)',
    6: 'Piano (Đàn Piano)',
    7: 'Saxophone (Kèn Saxophone)',
    8: 'Trumpet (Kèn Trumpet)',
    9: 'Violin (Đàn Violin)',
    10: 'Voice (Giọng hát)'
}

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('irmas_vgg_model.keras')
        return model
    except:
        st.error("⚠️ Không tìm thấy file model. Vui lòng đặt file 'irmas_vgg_model.keras' cùng thư mục.")
        return None

def preprocess_audio(audio_file, sr=22050, n_mels=128, target_shape=(128, 128)):
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr)
        
        # Xử lý độ dài - giống code training
        MAX_LEN = int(sr * 3.0)  # 3 giây
        
        if len(y) < MAX_LEN:
            # Pad nếu ngắn hơn
            y = np.pad(y, (0, MAX_LEN - len(y)), mode='constant')
        else:
            # Cắt nếu dài hơn
            y = y[:MAX_LEN]
        
        # Tạo Mel Spectrogram - GIỐNG CODE TRAINING
        mel = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Chuyển sang dB
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize về [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        # Resize về (128, 128)
        mel_db = librosa.util.fix_length(mel_db, size=target_shape[1], axis=1)
        mel_db = mel_db[:target_shape[0], :]            
        # Debug
        st.write(f"🔍 Shape: {mel_db.shape}")
        st.write(f"🔍 Value range: [{mel_db.min():.4f}, {mel_db.max():.4f}]")
        
        # Reshape cho model (1, 128, 128, 1)
        mel_spec_input = mel_db.reshape(1, 128, 128, 1)
        
        return mel_spec_input, mel_db, y, sr
        
    except Exception as e:
        st.error(f"Lỗi xử lý âm thanh: {str(e)}")
        return None, None, None, None

def plot_waveform(y, sr):
    """Vẽ dạng sóng âm thanh"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Dạng sóng âm thanh (Waveform)')
    ax.set_xlabel('Thời gian (s)')
    ax.set_ylabel('Biên độ')
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(mel_spec_db, sr):
    """Vẽ Mel Spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_predictions(predictions):
    """Vẽ biểu đồ dự đoán"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sắp xếp theo xác suất giảm dần
    sorted_idx = np.argsort(predictions[0])[::-1][:5]  # Top 5
    sorted_probs = predictions[0][sorted_idx] * 100
    sorted_labels = [INSTRUMENTS[i] for i in sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_labels)))
    bars = ax.barh(sorted_labels, sorted_probs, color=colors)
    
    ax.set_xlabel('Độ tin cậy (%)', fontsize=12)
    ax.set_title('Top 5 Dự đoán', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Thêm giá trị lên thanh
    for bar, prob in zip(bars, sorted_probs):
        ax.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# HEADER
st.markdown('<p class="main-header">🎵 Nhận Dạng Nhạc Cụ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sử dụng Deep Learning (CNN) & Mel Spectrogram</p>', unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2995/2995101.png", width=150)
    st.title("📚 Thông tin")
    st.info("""
    **Đề tài:** Nhận dạng nhạc cụ từ tín hiệu âm thanh
    
    **Phương pháp:**
    - Mel Spectrogram
    - CNN (4 Conv Blocks)
    - 11 loại nhạc cụ
    
    **Độ chính xác:** ~70%
    
    **Sinh viên thực hiện:**
    Trần Trọng Thịnh - 2251262647
    
    """)
    
    st.markdown("---")
    st.write("**🎼 Các nhạc cụ được hỗ trợ:**")
    for i, name in INSTRUMENTS.items():
        st.write(f"• {name}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload File Âm Thanh")
    
    uploaded_file = st.file_uploader(
        "Chọn file âm thanh (WAV, MP3, OGG)",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Độ dài khuyến nghị: 2-5 giây"
    )
    
    if uploaded_file is not None:
        # Reset trạng thái khi upload file mới
        if 'current_file' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
            st.session_state['analyzed'] = False
            st.session_state['current_file'] = uploaded_file.name
        
        st.success(f"✅ Đã tải lên: {uploaded_file.name}")
        
        # Phát âm thanh
        st.audio(uploaded_file, format='audio/wav')
        
        # Nút phân tích
        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang xử lý âm thanh..."):
                # Load model
                model = load_trained_model()
                
                if model is not None:
                    # QUAN TRỌNG: Reset file pointer về đầu
                    uploaded_file.seek(0)
                    
                    # Tiền xử lý
                    mel_input, mel_spec_db, y, sr = preprocess_audio(uploaded_file)
                    
                    if mel_input is not None:
                        # Dự đoán
                        predictions = model.predict(mel_input, verbose=0)
                        
                        # Lưu kết quả vào session state
                        st.session_state['predictions'] = predictions
                        st.session_state['mel_spec_db'] = mel_spec_db
                        st.session_state['y'] = y
                        st.session_state['sr'] = sr
                        st.session_state['analyzed'] = True
                        
                        st.success("✅ Phân tích hoàn tất!")
                        st.rerun()  # Rerun để hiển thị kết quả ngay
    else:
        # Xóa kết quả khi không có file
        if 'analyzed' in st.session_state:
            st.session_state['analyzed'] = False
with col2:
    st.subheader("📊 Kết quả phân tích")
    
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        predictions = st.session_state['predictions']
        
        # Kết quả chính
        top_idx = np.argmax(predictions[0])
        top_confidence = predictions[0][top_idx] * 100
        
        st.markdown(f"""
        <div class="metric-card">
            <h2>🎺 {INSTRUMENTS[top_idx]}</h2>
            <h1>{top_confidence:.1f}%</h1>
            <p>Độ tin cậy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Hiển thị top 3
        st.write("**Top 3 dự đoán:**")
        sorted_idx = np.argsort(predictions[0])[::-1][:3]
        
        for rank, idx in enumerate(sorted_idx, 1):
            conf = predictions[0][idx] * 100
            col_rank, col_name, col_conf = st.columns([1, 6, 2])
            with col_rank:
                st.write(f"**{rank}.**")
            with col_name:
                st.write(INSTRUMENTS[idx])
            with col_conf:
                st.write(f"_{conf:.1f}%_")
    else:
        st.info("👆 Upload file âm thanh và nhấn 'Bắt đầu phân tích' để xem kết quả")

# Phần trực quan hóa
if 'analyzed' in st.session_state and st.session_state['analyzed']:
    st.markdown("---")
    st.subheader("📈 Trực quan hóa")
    
    tab1, tab2, tab3 = st.tabs(["🌊 Dạng sóng", "🎨 Mel Spectrogram", "📊 Biểu đồ dự đoán"])
    
    with tab1:
        fig_wave = plot_waveform(st.session_state['y'], st.session_state['sr'])
        st.pyplot(fig_wave)
        st.caption("Biểu diễn biên độ âm thanh theo thời gian")
    
    with tab2:
        fig_mel = plot_mel_spectrogram(st.session_state['mel_spec_db'], st.session_state['sr'])
        st.pyplot(fig_mel)
        st.caption("Biểu diễn năng lượng theo thời gian và tần số trên thang Mel")
    
    with tab3:
        fig_pred = plot_predictions(st.session_state['predictions'])
        st.pyplot(fig_pred)
        st.caption("Xác suất dự đoán cho top 5 nhạc cụ")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.metric("Mô hình", "CNN 4 Blocks")
with col_f2:
    st.metric("Dataset", "IRMAS")
with col_f3:
    st.metric("Số lớp", "11 nhạc cụ")

st.markdown("""""", unsafe_allow_html=True)