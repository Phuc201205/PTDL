import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from wordcloud import WordCloud

# =============================================================================
# 1. CẤU HÌNH GIAO DIỆN & TỪ ĐIỂN
# =============================================================================
st.set_page_config(
    page_title="ABSA Sentiment Analyzer",
    page_icon="📱",
    layout="wide"
)

# CSS tùy chỉnh giao diện
st.markdown("""
<style>
    .stTextArea textarea {font-size: 16px;}
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        color: #333;
    }
    
    /* Định nghĩa màu sắc cho từng trạng thái */
    .positive {background-color: #28a745; color: white; border: none;}
    .negative {background-color: #dc3545; color: white; border: none;}
    .neutral {background-color: #6c757d; color: white; border: none;}
    
    /* Style cho nhãn không rõ ràng (Mới) */
    .not-mentioned {
        background-color: #f8f9fa; 
        color: #6c757d; 
        border: 1px dashed #ccc;
        opacity: 0.8;
    }
    
    .overall-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- CẤU HÌNH TỪ ĐIỂN ---
ASPECTS = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']

# [CẬP NHẬT] Map hiển thị bao gồm cả nhãn 0
SENTIMENT_MAP = {
    0: '⚪ Không được đề cập rõ ràng',
    1: '🔴 Tiêu cực', 
    2: '🔘 Trung tính', 
    3: '🟢 Tích cực'
} 
SENTIMENT_MAP_TRAIN = {'Negative': 1, 'Neutral': 2, 'Positive': 3}

STATIC_TEENCODE = {
    "mk": "mình", "mik": "mình", "mjk": "mình", "m": "mình", "t": "tôi", "tui": "tôi", "tao": "tôi", "b": "bạn", "bn": "bạn",
    "ad": "admin", "shop": "cửa hàng", "nv": "nhân viên", "ship": "giao hàng",
    "k": "không", "ko": "không", "kh": "không", "hok": "không", "not": "không", "chả": "chẳng",
    "yes": "có", "ye": "có", "uk": "ừ", "uhm": "ừ", "r": "rồi",
    "sp": "sản phẩm", "dt": "điện thoại", "đt": "điện thoại", "dế": "điện thoại", "mb": "máy", "mobile": "điện thoại",
    "ip": "iphone", "ss": "samsung", "sam": "samsung",
    "cam": "camera", "mic": "micro", "loa": "loa", "pin": "pin", "sac": "sạc",
    "dc": "được", "đc": "được", "ok": "tốt", "okie": "tốt", "oke": "tốt", "ổn": "tốt",
    "chê": "không thích", "khen": "thích", "good": "tốt", "bad": "tệ", "nice": "tốt",
    "thik": "thích", "iu": "yêu", "love": "yêu",
    "bth": "bình thường", "bt": "bình thường",
    "lag": "giật", "đơ": "đứng máy", "mượt": "nhanh",
    "nhìu": "nhiều", "wa": "quá", "wá": "quá", "qa": "quá", "mua": "mua", "ban": "bán",
    "wf": "wifi", "4g": "mạng", "net": "mạng", "app": "ứng dụng", "game": "trò chơi",
    "fb": "facebook", "zalo": "zalo", "mess": "tin nhắn", "ib": "nhắn tin",
    "bh": "bây giờ", "h": "giờ", "bit": "biết", "vs": "với", "tr": "triệu", "k": "nghìn"
}

STOPWORDS = set(["bị", "bởi", "cả", "các", "cái", "cần", "càng", "thì", "là", "mà"])

ASPECT_KEYWORDS = {
    'BATTERY': ['pin', 'bin', 'sạc', 'xạc', 'mah'],
    'CAMERA': ['cam', 'ảnh', 'chụp', 'selfie', 'quay', 'video', 'focus', 'nét'],
    'DESIGN': ['thiết kế', 'đẹp', 'xấu', 'mỏng', 'nhẹ', 'cầm', 'nắm', 'lưng', 'viền', 'nhựa', 'nhôm', 'kính', 'ngoại hình'],
    'FEATURES': ['wifi', '4g', '5g', 'sóng', 'vân tay', 'face id', 'loa', 'âm', 'sim', 'esim', 'bluetooth', 'kết nối'],
    'PERFORMANCE': ['game', 'liên quân', 'pubg', 'lác', 'lag', 'giật', 'mượt', 'nhanh', 'chậm', 'treo', 'đơ', 'nóng', 'nhiệt', 'chip', 'ram', 'tác vụ', 'hiệu năng'],
    'PRICE': ['giá', 'tiền', 'đắt', 'rẻ', 'hợp lý', 'mắc', 'chi phí', 'ví'],
    'SCREEN': ['màn', 'hình', 'hiển thị', 'nét', 'rỗ', 'ám', 'tối', 'sáng', 'tần số quét', 'hz', 'oled', 'lcd'],
    'SER&ACC': ['giao', 'ship', 'đóng gói', 'hộp', 'nhân viên', 'shop', 'tư vấn', 'bảo hành', 'phụ kiện', 'tai nghe', 'cáp', 'củ sạc'],
    'STORAGE': ['gb', 'tb', 'bộ nhớ', 'lưu', 'trữ', 'dung lượng'],
    'GENERAL': []
}

SENTIMENT_KEYWORDS = [
    'tốt', 'xấu', 'khen', 'chê', 'ngon', 'dở', 'tệ', 'kém', 'ổn', 'ok', 'được', 'thích', 'yêu', 'ghét',
    'mượt', 'lag', 'giật', 'đơ', 'nhanh', 'chậm', 'nóng', 'mát', 'ấm', 'trâu', 'yếu', 'bền', 'lởm',
    'nét', 'mờ', 'rõ', 'nhòe', 'rỗ', 'sắc', 'ảo', 'đẹp', 'xấu', 'sang', 'thô', 'mỏng', 'dày', 'nặng', 'nhẹ',
    'rẻ', 'đắt', 'hợp lý', 'mắc', 'chát', 'cao', 'thấp',
    'to', 'nhỏ', 'bé', 'lớn', 'rè', 'vọng', 'êm',
    'nhạy', 'ngu', 'thông minh', 'lỗi', 'xịn', 'dỏm', 'fake', 'hư', 'hỏng',
    'nhiệt tình', 'thân thiện', 'láo', 'cọc', 'nhanh', 'lâu', 'chậm', 'cẩn thận', 'móp', 'rách',
    'thất vọng', 'hài lòng', 'ưng', 'phê', 'chán', 'tiếc', 'phí', 'đáng', 'tuyệt'
]

# =============================================================================
# 2. HÀM XỬ LÝ TEXT
# =============================================================================
def clean_text_ultimate(text):
    if pd.isna(text): return ""
    text = str(text).lower()

    text = re.sub(r'\b\d+\s?(gb|tb|g|mb)\b', ' token_memory ', text)
    text = re.sub(r'bộ nhớ\s?(trong)?', ' token_memory ', text)
    text = re.sub(r'lưu trữ', ' token_memory ', text)
    text = re.sub(r'thẻ nhớ', ' token_memory ', text)
    text = re.sub(r'đầy\s?bộ\s?nhớ', ' token_memory_full ', text)

    text = re.sub(r'\b\d+\s?hz\b', ' token_hz ', text)
    text = re.sub(r'tần số quét', ' token_hz ', text)

    text = emoji.demojize(text, delimiters=(" ", " "))
    text = unicodedata.normalize('NFC', text)

    sorted_keys = sorted(STATIC_TEENCODE.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')
    text = pattern.sub(lambda x: STATIC_TEENCODE[x.group()], text)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = ViTokenizer.tokenize(text)

    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

# =============================================================================
# 3. HÀM HUẤN LUYỆN MODEL
# =============================================================================
@st.cache_resource
def train_model(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    if 'BATTERY' not in df.columns:
        def parse_labels(row):
            res = {asp: 0 for asp in ASPECTS}
            if pd.isna(row['label']): return pd.Series(res)
            tags = row['label'].split(';')
            for tag in tags:
                tag = tag.strip().replace('{', '').replace('}', '')
                if '#' in tag:
                    parts = tag.split('#')
                    asp, sent = parts[0], parts[1] if len(parts) > 1 else None
                    if asp in ASPECTS and sent in SENTIMENT_MAP_TRAIN: 
                        res[asp] = SENTIMENT_MAP_TRAIN[sent]
            return pd.Series(res)
        label_df = df.apply(parse_labels, axis=1)
        df = pd.concat([df, label_df], axis=1)

    df['comment_cleaned'] = df['comment'].apply(clean_text_ultimate)
    df_clean = df.dropna(subset=['comment_cleaned'])
    df_clean = df_clean[df_clean['comment_cleaned'].str.strip().astype(bool)]

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    X_vec_all = vectorizer.fit_transform(df_clean['comment_cleaned'].values)
    models = {}

    progress_bar = st.progress(0)
    
    for idx, aspect in enumerate(ASPECTS):
        y = df_clean[aspect].values
        mask = (y != 0)
        
        X_curr = X_vec_all[mask]
        y_curr = y[mask] - 1 

        if len(y_curr) < 10:
            base_svc = LinearSVC(class_weight='balanced', random_state=42)
            if len(y_curr) > 0:
                base_svc.fit(X_curr, y_curr)
                models[aspect] = base_svc
            else:
                models[aspect] = None
            continue

        X_train, _, y_train, _ = train_test_split(X_curr, y_curr, test_size=0.1, random_state=42, stratify=y_curr)

        try:
            rus = RandomUnderSampler(random_state=42)
            X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        except:
            X_train_res, y_train_res = X_train, y_train
            
        try:
            min_samples = sorted(dict(pd.Series(y_train_res).value_counts()).values())[0]
            k = min(3, min_samples - 1)
            if k > 0:
                smote = SMOTE(k_neighbors=k, random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train_res)
        except:
            pass

        base_svc = LinearSVC(class_weight='balanced', random_state=42, dual=False, max_iter=3000)
        model = BaggingClassifier(estimator=base_svc, n_estimators=10, random_state=42, n_jobs=-1)
        
        model.fit(X_train_res, y_train_res)
        models[aspect] = model
        
        progress_bar.progress((idx + 1) / len(ASPECTS))
    
    progress_bar.empty()
    return vectorizer, models, df_clean

# =============================================================================
# 4. HARD RULES & HYBRID LOGIC
# =============================================================================
def has_aspect_keyword(text, aspect):
    if aspect == 'GENERAL': return True
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    return any(kw in text for kw in keywords)

def has_sentiment_keyword(text):
    return any(kw in text for kw in SENTIMENT_KEYWORDS)

def check_strict_sentiment(raw_text, aspect):
    if aspect == 'GENERAL': return True
    segments = re.split(r'[.,;!]+', raw_text)
    for seg in segments:
        seg_clean = clean_text_ultimate(seg).lower().replace('_', ' ')
        if has_aspect_keyword(seg_clean, aspect):
            if has_sentiment_keyword(seg_clean):
                return True
    return False

def apply_hard_rules_hybrid(text, pred_vector):
    text_lower = text.lower()
    
    def set_force(asp_name, val):
        idx = ASPECTS.index(asp_name)
        pred_vector[idx] = val

    def has_kw(keywords):
        return any(kw in text_lower for kw in keywords)

    neg_dep = ['không đẹp', 'ko đẹp', 'k đẹp', 'chả đẹp', 'chẳng đẹp', 'xấu', 'thô']
    neg_net = ['không nét', 'ko nét', 'k nét', 'mờ', 'không rõ', 'k rõ']
    pos_design_strong = ['máy đẹp', 'đt đẹp', 'điện thoại đẹp', 'thiết kế đẹp', 'ngoại hình đẹp', 'nhìn đẹp']

    contrast_words = ['tuy nhiên', 'nhưng mà', 'có điều', 'mỗi tội', 'điểm trừ', 'tiếc là']
    for word in contrast_words:
        if word in text_lower:
            parts = text_lower.split(word)
            if len(parts) > 1:
                after_part = parts[1]
                if 'cam' in after_part and not has_kw(['nét', 'đẹp']): set_force('CAMERA', 1)
                if 'pin' in after_part: set_force('BATTERY', 1)
                if 'màn' in after_part: set_force('SCREEN', 1)
                if 'nóng' in after_part: set_force('PERFORMANCE', 1)

    if has_kw(['thiết kế', 'ngoại hình', 'kiểu dáng', 'máy', 'điện thoại']):
        if has_kw(pos_design_strong): set_force('DESIGN', 3)
        elif has_kw(neg_dep) or has_kw(['nhựa', 'ọp ẹp', 'lỏng lẻo', 'cấn']): set_force('DESIGN', 1)
        elif has_kw(['đẹp', 'sang', 'xịn', 'mỏng', 'nhẹ', 'cầm sướng']): set_force('DESIGN', 3)

    if has_kw(['pin', 'bin']):
        if has_kw(['trâu', 'khỏe', 'lâu', 'cả ngày', 'ngon']): set_force('BATTERY', 3)
        if has_kw(['tuột', 'tụt', 'yếu', 'hẻo', 'nhanh hết', 'sụt', 'kém']): set_force('BATTERY', 1)
        if has_kw(['trung bình', 'đủ dùng', 'bth', 'bình thường']): set_force('BATTERY', 2)

    if has_kw(['màn hình', 'màn']):
        if has_kw(neg_dep) or has_kw(neg_net) or has_kw(['rỗ', 'ám', 'tối', 'đơ', 'loạn', 'sọc']): set_force('SCREEN', 1)
        elif has_kw(['nét', 'đẹp', 'sắc', 'mượt', 'tươi']): set_force('SCREEN', 3)

    if has_kw(['cam', 'ảnh', 'chụp', 'selfie', 'quay']):
        if has_kw(neg_dep) or has_kw(neg_net) or has_kw(['mờ', 'bể', 'nhòe', 'tệ', 'kém', 'rung', 'bệt']): set_force('CAMERA', 1)
        elif has_kw(['nét', 'đẹp', 'ảo', 'ngon', 'rõ', 'xuất sắc', 'chi tiết']): set_force('CAMERA', 3)

    if has_kw(['nóng', 'ấm máy', 'tỏa nhiệt', 'loạn cảm ứng']): set_force('PERFORMANCE', 1)
    if has_kw(['lag', 'giật', 'treo logo', 'khựng', 'đứng hình']): set_force('PERFORMANCE', 1)
    if has_kw(['game', 'liên quân', 'pubg', 'tác vụ', 'hiệu năng']):
        if has_kw(['k ngon', 'không ngon', 'chán']): set_force('PERFORMANCE', 1)
        elif has_kw(['mượt', 'phê', 'nhanh', 'chiến', 'ngon']): set_force('PERFORMANCE', 3)
        elif has_kw(['bình thường', 'ổn', 'tạm']): set_force('PERFORMANCE', 2)

    idx_price = ASPECTS.index('PRICE')
    if pred_vector[idx_price] == 3:
        if not has_kw(['rẻ', 'tốt', 'hợp lý', 'ok', 'ngon', 'giảm', 'sale', 'đáng', 'mềm']):
            pred_vector[idx_price] = 0
    if has_kw(['giá', 'tiền']):
        if has_kw(['rẻ', 'tốt', 'hợp lý', 'mềm']): set_force('PRICE', 3)
        if has_kw(['đắt', 'cao', 'chát', 'mắc']): set_force('PRICE', 1)

    if has_kw(['nhân viên', 'tư vấn', 'shop', 'phục vụ']):
        if has_kw(['nhiệt tình', 'tốt', 'dễ thương', 'thân thiện']): set_force('SER&ACC', 3)
        if has_kw(['thái độ', 'tệ', 'láo', 'cọc']): set_force('SER&ACC', 1)

    if has_kw(['thất vọng', 'đừng mua', 'phí tiền']): set_force('GENERAL', 1)
    if has_kw(['nhìn chung', 'tổng thể']):
        if has_kw(['đẹp', 'tốt', 'ok']): set_force('GENERAL', 3)

    return pred_vector

# =============================================================================
# 5. GIAO DIỆN STREAMLIT CHÍNH
# =============================================================================
st.sidebar.title("⚙️ Bảng điều khiển")
uploaded_file = st.sidebar.file_uploader("Upload file Training (CSV)", type=['csv'])

if uploaded_file is not None:
    st.sidebar.success("File đã tải lên!")
    if st.sidebar.button("Huấn luyện Mô hình 🚀"):
        with st.spinner("Đang huấn luyện mô hình Bagging SVC + SMOTE..."):
            try:
                vectorizer, models, df_visual = train_model(uploaded_file)
                st.session_state['vectorizer'] = vectorizer
                st.session_state['models'] = models
                st.session_state['df_visual'] = df_visual
                st.sidebar.success("Huấn luyện hoàn tất!")
            except Exception as e:
                st.sidebar.error(f"Có lỗi xảy ra: {e}")
else:
    st.sidebar.info("Vui lòng tải file CSV để bắt đầu.")

st.title("📱 Hệ Thống Phân Tích Cảm Xúc Điện Thoại")

tab1, tab2 = st.tabs(["🔍 Phân Tích Bình Luận", "📊 Trực Quan Hóa Dữ Liệu"])

# --- TAB 1: PHÂN TÍCH ---
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Nhập bình luận của khách hàng:", height=150, placeholder="Ví dụ: Máy dùng tốt, pin trâu nhưng camera hơi mờ...")
        analyze_btn = st.button("Phân tích ngay ✨", type="primary")

    if analyze_btn and user_input:
        if 'models' not in st.session_state:
            st.error("⚠️ Vui lòng huấn luyện mô hình trước!")
        else:
            cleaned_text = clean_text_ultimate(user_input)
            vec_input = st.session_state['vectorizer'].transform([cleaned_text])
            
            text_lower_cleaned = cleaned_text.lower().replace('_', ' ')
            text_raw_lower = user_input.lower()
            
            mentioned_aspects = [asp for asp in ASPECTS if asp != 'GENERAL' and has_aspect_keyword(text_lower_cleaned, asp)]
            is_multi_aspect = len(mentioned_aspects) > 1

            ml_preds_vector = []
            for aspect in ASPECTS:
                if st.session_state['models'][aspect] is None:
                    pred_label = 0
                else:
                    pred_label = st.session_state['models'][aspect].predict(vec_input)[0] + 1
                
                if pred_label != 0:
                    if not has_aspect_keyword(text_lower_cleaned, aspect):
                        pred_label = 0
                    elif aspect != 'GENERAL':
                        if is_multi_aspect:
                            if not check_strict_sentiment(text_raw_lower, aspect):
                                pred_label = 0
                        else:
                            if not has_sentiment_keyword(text_lower_cleaned):
                                pred_label = 0
                ml_preds_vector.append(pred_label)
            
            final_preds = apply_hard_rules_hybrid(user_input, np.array(ml_preds_vector))
            
            active_sentiments = [p for p in final_preds if p != 0]
            
            st.markdown("---")
            
            # Tính toán tổng quan
            if not active_sentiments:
                st.warning("Hệ thống chưa tìm thấy khía cạnh nào rõ ràng để kết luận tổng quan.")
            else:
                n_pos = active_sentiments.count(3)
                n_neg = active_sentiments.count(1)
                
                if n_pos > n_neg:
                    overall_html = f"""<div class="overall-card positive">🌟 KẾT LUẬN: KHÁCH HÀNG HÀI LÒNG</div>"""
                elif n_neg > n_pos:
                    overall_html = f"""<div class="overall-card negative">😡 KẾT LUẬN: KHÁCH HÀNG KHÔNG HÀI LÒNG</div>"""
                else:
                    overall_html = f"""<div class="overall-card neutral">⚖️ KẾT LUẬN: ĐÁNH GIÁ TRUNG TÍNH / TRÁI CHIỀU</div>"""
                
                st.markdown(overall_html, unsafe_allow_html=True)

            # [CẬP NHẬT GIAO DIỆN] Hiển thị tất cả nhãn, bao gồm cả nhãn 0
            st.subheader("📝 Chi tiết phân tích:")
            cols = st.columns(4)
            col_idx = 0
            
            for i, aspect in enumerate(ASPECTS):
                sentiment = final_preds[i]
                
                # Class CSS tương ứng
                if sentiment == 3: color_class = "positive"
                elif sentiment == 1: color_class = "negative"
                elif sentiment == 2: color_class = "neutral"
                else: color_class = "not-mentioned" # Class mới cho nhãn 0
                
                label_text = SENTIMENT_MAP[sentiment]
                
                with cols[col_idx % 4]:
                    st.markdown(f"""
                    <div class="metric-card {color_class}">
                        <div>{aspect}</div>
                        <div style="font-size: 1.1em; font-weight: normal;">{label_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1

    with col2:
        st.markdown("### ℹ️ Hướng dẫn")
        st.info("""
        **Quy trình:**
        1. Tải file CSV huấn luyện.
        2. Nhấn nút "Huấn luyện".
        3. Nhập bình luận và xem kết quả.
        
        **Chú thích:**
        - 🟢 Xanh: Tích cực
        - 🔴 Đỏ: Tiêu cực
        - 🔘 Xám Đậm: Trung tính
        - ⚪ Xám Nhạt: Không được đề cập
        """)
        if 'models' in st.session_state:
            st.success("✅ Hệ thống đã sẵn sàng!")

# --- TAB 2: TRỰC QUAN HÓA ---
with tab2:
    if 'df_visual' not in st.session_state:
        st.warning("⚠️ Vui lòng huấn luyện mô hình ở Tab 'Phân Tích' để tải dữ liệu!")
    else:
        df = st.session_state['df_visual']
        st.header("📊 Dashboard Phân Tích Dữ Liệu")
        
        # 1. Phân phối Sao
        st.subheader("1. Phân phối đánh giá sao (1-5)")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        if 'n_star' in df.columns:
            sns.countplot(x=df["n_star"], color="#33CCFF", ax=ax1)
            st.pyplot(fig1)
        else:
            st.write("Không tìm thấy cột 'n_star'.")

        # 2. Tổng quan Sentiment
        st.subheader("2. Tỷ lệ Cảm xúc Toàn hệ thống")
        polarity_counts = {
            "Negative": (df[ASPECTS] == 1).sum().sum(),
            "Neutral":  (df[ASPECTS] == 2).sum().sum(),
            "Positive": (df[ASPECTS] == 3).sum().sum(),
        }
        fig2, ax2 = plt.subplots()
        ax2.pie(polarity_counts.values(), labels=polarity_counts.keys(), autopct='%1.1f%%', colors=['#dc3545', '#6c757d', '#28a745'])
        st.pyplot(fig2)

        # 3. Bar Chart
        st.subheader("3. Chi tiết Cảm xúc theo Khía cạnh")
        aspect_sentiment = pd.DataFrame({
            "Aspect": ASPECTS,
            "Negative": [(df[a] == 1).sum() for a in ASPECTS],
            "Neutral":  [(df[a] == 2).sum() for a in ASPECTS],
            "Positive": [(df[a] == 3).sum() for a in ASPECTS],
        })
        fig3 = aspect_sentiment.set_index("Aspect").plot(kind="bar", figsize=(12, 6), color=['#dc3545', '#6c757d', '#28a745']).figure
        st.pyplot(fig3)

        # 4. Heatmap
        st.subheader("4. Ma trận Tương quan giữa các Khía cạnh")
        corr = df[ASPECTS].replace({0: np.nan}).corr()
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

        # 5. WordCloud (ĐÃ SỬA LỖI CRASH)
        st.subheader("5. Từ khóa nổi bật (WordCloud)")
        col_wc1, col_wc2 = st.columns(2)
        
        positive_text = " ".join(df[df[ASPECTS].eq(3).any(axis=1)]["comment_cleaned"])
        negative_text = " ".join(df[df[ASPECTS].eq(1).any(axis=1)]["comment_cleaned"])
        
        with col_wc1:
            st.write("**Từ khóa Tích cực**")
            # [FIX LỖI] Kiểm tra độ dài text để tránh crash
            if len(positive_text.strip()) > 0:
                try:
                    wc_pos = WordCloud(width=400, height=300, background_color="white").generate(positive_text)
                    fig_p, ax_p = plt.subplots()
                    ax_p.imshow(wc_pos, interpolation='bilinear')
                    ax_p.axis("off")
                    st.pyplot(fig_p)
                except ValueError:
                    st.info("Dữ liệu không đủ để tạo WordCloud.")
            else:
                st.info("Không có dữ liệu tích cực.")
        
        with col_wc2:
            st.write("**Từ khóa Tiêu cực**")
            if len(negative_text.strip()) > 0:
                try:
                    wc_neg = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(negative_text)
                    fig_n, ax_n = plt.subplots()
                    ax_n.imshow(wc_neg, interpolation='bilinear')
                    ax_n.axis("off")
                    st.pyplot(fig_n)
                except ValueError:
                    st.info("Dữ liệu không đủ để tạo WordCloud.")
            else:
                st.info("Không có dữ liệu tiêu cực.")
