import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
from scipy.stats import norm
from collections import Counter
import itertools

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
        color: white;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .positive {background-color: #28a745;}
    .negative {background-color: #dc3545;}
    .neutral {background-color: #6c757d;}
    
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

ASPECTS = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL', 'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']
SENTIMENT_MAP = {1: 'Tiêu cực', 2: 'Trung tính', 3: 'Tích cực'}

STATIC_TEENCODE = {
    "mk": "mình", "mik": "mình", "mjk": "mình", "m": "mình", "t": "tôi", "tui": "tôi",
    "tao": "tôi", "tớ": "tôi", "b": "bạn", "bn": "bạn", "shop": "cửa hàng", "xốp": "cửa hàng",
    "nv": "nhân viên", "ship": "giao hàng", "shipper": "người giao hàng",
    "k": "không", "ko": "không", "kh": "không", "hok": "không", "khum": "không", "not": "không",
    "dt": "điện thoại", "đt": "điện thoại", "mb": "máy", "mobile": "điện thoại",
    "ip": "iphone", "ss": "samsung", "sam": "samsung", "táo": "apple",
    "cam": "camera", "mic": "micro", "loa": "loa", "pin": "pin", "bin": "pin",
    "sac": "sạc", "cap": "cáp",
    "dc": "được", "đc": "được", "dk": "được", "ok": "tốt", "oke": "tốt", "ổn": "tốt",
    "gud": "tốt", "good": "tốt", "bad": "tệ", "lag": "giật", "lác": "giật", "đơ": "đứng máy",
    "mượt": "nhanh", "nhìu": "nhiều", "wa": "quá", "wá": "quá", "mua": "mua", "xai": "xài",
    "app": "ứng dụng", "game": "trò chơi", "fb": "facebook", "mess": "tin nhắn",
    "tr": "triệu", "củ": "triệu"
}

STOPWORDS = set(["bị", "bởi", "cả", "các", "cái", "cần", "càng", "thì", "là", "mà"])
NEGATION_WORDS = ["không", "chẳng", "chả", "chưa", "đừng", "k", "ko", "kh", "nỏ", "not", "đếch", "éo"]

# =============================================================================
# 2. HÀM XỬ LÝ TEXT (CLEANING)
# =============================================================================
def resolve_ambiguity(text):
    text = " " + text + " "
    text = re.sub(r'(\d+)\s*k\b', r'\1 nghìn', text)
    text = re.sub(r'\bk\b', 'không', text)
    text = re.sub(r'\b(xin|gửi|tại|ở)\s+(dc|đc)\b', r'\1 địa chỉ', text)
    text = re.sub(r'\b(dc|đc)\b', 'được', text)
    return text.strip()

def normalize_repeated_characters(text):
    return re.sub(r'(\w)\1{2,}', r'\1', text)

def merge_negation(text):
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in NEGATION_WORDS and i < len(words) - 1:
            new_words.append(f"{word}_{words[i+1]}")
            i += 2
        else:
            new_words.append(word)
            i += 1
    return " ".join(new_words)

def clean_text_ultimate(text):
    if pd.isna(text): return ""
    text = str(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = unicodedata.normalize('NFC', text).lower()
    text = resolve_ambiguity(text)
    
    sorted_keys = sorted(STATIC_TEENCODE.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')
    text = pattern.sub(lambda x: STATIC_TEENCODE[x.group()], text)
    
    text = normalize_repeated_characters(text)
    text = re.sub(r'[^\w\s_:]', ' ', text)
    text = ViTokenizer.tokenize(text)
    text = merge_negation(text)
    
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

# =============================================================================
# 3. HÀM HUẤN LUYỆN MODEL
# =============================================================================
@st.cache_resource
def train_model(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Tách nhãn (Label Parsing)
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
                    if asp in ASPECTS and sent in {'Negative': 1, 'Neutral': 2, 'Positive': 3}: 
                        res[asp] = {'Negative': 1, 'Neutral': 2, 'Positive': 3}[sent]
            return pd.Series(res)
        label_df = df.apply(parse_labels, axis=1)
        df = pd.concat([df, label_df], axis=1)

    df['comment_cleaned'] = df['comment'].apply(clean_text_ultimate)
    df_clean = df.dropna(subset=['comment_cleaned'])
    df_clean = df_clean[(df_clean['comment_cleaned'].apply(lambda x: len(str(x).split())) >= 3)]

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), min_df=2, sublinear_tf=True)
    X_all_vec = vectorizer.fit_transform(df_clean['comment_cleaned'])
    models = {}

    progress_bar = st.progress(0)
    for idx, aspect in enumerate(ASPECTS):
        y = df_clean[aspect]
        try:
            sampler = SMOTE(random_state=42, k_neighbors=1)
            X_res, y_res = sampler.fit_resample(X_all_vec, y)
        except:
            X_res, y_res = X_all_vec, y
            
        svm = LinearSVC(dual='True', class_weight='balanced', random_state=42)
        lr = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        ensemble = VotingClassifier(estimators=[('svm', svm), ('lr', lr)], voting='hard')
        ensemble.fit(X_res, y_res)
        models[aspect] = ensemble
        progress_bar.progress((idx + 1) / len(ASPECTS))
    
    progress_bar.empty()
    return vectorizer, models, df_clean

# =============================================================================
# 4. HARD RULES V4.1
# =============================================================================
def apply_hard_rules_hybrid(text, pred_vector):
    text_lower = text.lower()
    def set_sent(asp_name, val):
        idx = ASPECTS.index(asp_name)
        pred_vector[idx] = val
    def has_kw(keywords):
        return any(kw in text_lower for kw in keywords)

    # 1. Cấu trúc câu
    contrast_words = ['tuy nhiên', 'nhưng mà', 'có điều', 'mỗi tội', 'điểm trừ', 'tiếc là']
    for word in contrast_words:
        if word in text_lower:
            parts = text_lower.split(word)
            if len(parts) > 1:
                after_part = parts[1]
                if 'cam' in after_part or 'ảnh' in after_part: set_sent('CAMERA', 1)
                if 'pin' in after_part: set_sent('BATTERY', 1)
                if 'màn' in after_part: set_sent('SCREEN', 1)
                if 'loa' in after_part: set_sent('FEATURES', 1)
                if 'nóng' in after_part: set_sent('PERFORMANCE', 1)

    # 2. Domain Rules
    if has_kw(['pin', 'bin']):
        if has_kw(['trâu', 'khỏe', 'lâu', 'cả ngày', 'ngon', 'mạnh', 'tốt', 'ổn', 'bền']): set_sent('BATTERY', 3)
        if has_kw(['tuột', 'tụt', 'yếu', 'hẻo', 'nhanh hết', 'sụt']): set_sent('BATTERY', 1)

    if has_kw(['màn hình', 'màn']):
        if has_kw(['nét', 'đẹp', 'sắc', 'nhạy', 'mượt']): set_sent('SCREEN', 3)
        if has_kw(['rỗ', 'ám vàng', 'tối', 'đơ', 'loạn', 'liệt', 'nhòe']): set_sent('SCREEN', 1)

    if has_kw(['cam', 'ảnh', 'chụp', 'selfie', 'quay']):
        if has_kw(['nét', 'đẹp', 'ảo', 'ngon', 'rõ', 'xuất sắc']): set_sent('CAMERA', 3)
        elif has_kw(['mờ', 'xấu', 'bể', 'nhòe', 'tệ', 'kém', 'rung']): set_sent('CAMERA', 1)

    if has_kw(['nóng', 'ấm máy', 'tỏa nhiệt']): set_sent('PERFORMANCE', 1)
    if has_kw(['game', 'liên quân', 'pubg', 'tác vụ', 'máy']):
        if has_kw(['mượt', 'ngon', 'phê', 'nhanh', 'mạnh']): set_sent('PERFORMANCE', 3)
        if has_kw(['lag', 'giật', 'khựng', 'đứng', 'văng']): set_sent('PERFORMANCE', 1)
    if has_kw(['lag', 'giật', 'treo logo']): set_sent('PERFORMANCE', 1)

    if has_kw(['giao hàng', 'ship', 'vận chuyển', 'đặt hàng']):
        if has_kw(['nhanh', 'lẹ', 'sớm', 'hỏa tốc']): 
            set_sent('SER&ACC', 3)
            pred_vector[ASPECTS.index('PERFORMANCE')] = 0 
        if has_kw(['lâu', 'chậm', 'lề mề']): set_sent('SER&ACC', 1)
    
    if has_kw(['đóng gói', 'hộp', 'tai nghe', 'sạc']):
        if has_kw(['cẩn thận', 'đẹp', 'kỹ']): set_sent('SER&ACC', 3)
        if has_kw(['móp', 'rách', 'thiếu']): set_sent('SER&ACC', 1)

    if has_kw(['nhân viên', 'shop', 'tư vấn']):
        if has_kw(['nhiệt tình', 'dễ thương', 'tốt']): set_sent('SER&ACC', 3)
        if has_kw(['lồi lõm', 'thái độ', 'bố láo']): set_sent('SER&ACC', 1)

    if has_kw(['giá', 'tiền', 'túi tiền']):
        if has_kw(['rẻ', 'tốt', 'hợp lý', 'ok', 'ngon']): set_sent('PRICE', 3)
        if has_kw(['đắt', 'cao', 'chát']): set_sent('PRICE', 1)
    if has_kw(['đáng đồng tiền', 'đáng tiền']): set_sent('PRICE', 3)

    if has_kw(['wifi', '4g', 'sóng', 'vân tay', 'face id']):
        if has_kw(['yếu', 'kém', 'chập chờn', 'lỗi']): set_sent('FEATURES', 1)
        if has_kw(['nhạy', 'khỏe', 'căng']): set_sent('FEATURES', 3)
    if has_kw(['loa', 'âm thanh']):
        if has_kw(['to', 'hay', 'lớn']): set_sent('FEATURES', 3)
        if has_kw(['bé', 'nhỏ', 'rè']): set_sent('FEATURES', 1)

    if has_kw(['thất vọng', 'đừng mua', 'tránh xa', 'phí tiền', 'hối hận']): set_sent('GENERAL', 1)
    if has_kw(['nên mua', 'tuyệt vời', 'xuất sắc', 'hài lòng', '10 điểm']):
        if not any(x == 3 for x in pred_vector): set_sent('GENERAL', 3)

    return pred_vector

# =============================================================================
# 5. GIAO DIỆN STREAMLIT CHÍNH
# =============================================================================
st.sidebar.title("⚙️ Bảng điều khiển")
uploaded_file = st.sidebar.file_uploader("Upload file Training (CSV)", type=['csv'])

if uploaded_file is not None:
    st.sidebar.success("File đã tải lên!")
    if st.sidebar.button("Huấn luyện Mô hình 🚀"):
        with st.spinner("Đang huấn luyện mô hình Ensemble..."):
            try:
                vectorizer, models, df_visual = train_model(uploaded_file)
                st.session_state['vectorizer'] = vectorizer
                st.session_state['models'] = models
                st.session_state['df_visual'] = df_visual # Lưu data để vẽ biểu đồ
                st.sidebar.success("Huấn luyện hoàn tất!")
            except Exception as e:
                st.sidebar.error(f"Có lỗi xảy ra: {e}")
else:
    st.sidebar.info("Vui lòng tải file CSV để bắt đầu.")

st.title("📱 Hệ Thống Phân Tích Cảm Xúc Điện Thoại")

# TẠO TABS
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
            # Predict
            cleaned_text = clean_text_ultimate(user_input)
            vec_input = st.session_state['vectorizer'].transform([cleaned_text])
            
            ml_preds = []
            for aspect in ASPECTS:
                ml_preds.append(st.session_state['models'][aspect].predict(vec_input)[0])
            
            final_preds = apply_hard_rules_hybrid(user_input, np.array(ml_preds))
            
            # --- TÍNH TOÁN KẾT LUẬN TỔNG QUAN ---
            # Lọc các aspect có nhắc đến (khác 0)
            active_sentiments = [p for p in final_preds if p != 0]
            
            st.markdown("---")
            
            if not active_sentiments:
                 st.warning("Không tìm thấy khía cạnh cụ thể nào trong bình luận.")
            else:
                n_pos = active_sentiments.count(3)
                n_neg = active_sentiments.count(1)
                n_neu = active_sentiments.count(2)

                # Logic kết luận
                if n_pos > n_neg:
                    overall_html = f"""
                    <div class="overall-card positive">
                        🌟 KẾT LUẬN: KHÁCH HÀNG HÀI LÒNG<br>
                        <span style="font-size: 16px; font-weight: normal;">(Tích cực: {n_pos} | Tiêu cực: {n_neg})</span>
                    </div>
                    """
                elif n_neg > n_pos:
                    overall_html = f"""
                    <div class="overall-card negative">
                        😡 KẾT LUẬN: KHÁCH HÀNG KHÔNG HÀI LÒNG<br>
                        <span style="font-size: 16px; font-weight: normal;">(Tích cực: {n_pos} | Tiêu cực: {n_neg})</span>
                    </div>
                    """
                else:
                    overall_html = f"""
                    <div class="overall-card neutral">
                        ⚖️ KẾT LUẬN: ĐÁNH GIÁ TRUNG TÍNH / TRÁI CHIỀU<br>
                        <span style="font-size: 16px; font-weight: normal;">(Tích cực: {n_pos} | Tiêu cực: {n_neg})</span>
                    </div>
                    """
                
                st.markdown(overall_html, unsafe_allow_html=True)

                # --- HIỂN THỊ CHI TIẾT TỪNG KHÍA CẠNH ---
                st.subheader("📝 Chi tiết phân tích:")
                cols = st.columns(4)
                col_idx = 0
                for i, aspect in enumerate(ASPECTS):
                    sentiment = final_preds[i]
                    if sentiment != 0:
                        color_class = "positive" if sentiment == 3 else "negative" if sentiment == 1 else "neutral"
                        label_text = SENTIMENT_MAP[sentiment]
                        with cols[col_idx % 4]:
                            st.markdown(f"""
                            <div class="metric-card {color_class}">
                                <div>{aspect}</div>
                                <div style="font-size: 1.2em;">{label_text}</div>
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
        
        **Màu sắc:**
        - 🟢 Xanh: Tích cực
        - 🔴 Đỏ: Tiêu cực
        - 🔘 Xám: Trung tính
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

        # 2. Tổng quan Sentiment (Pie Chart)
        st.subheader("2. Tỷ lệ Cảm xúc Toàn hệ thống")
        polarity_counts = {
            "Negative": (df[ASPECTS] == 1).sum().sum(),
            "Neutral":  (df[ASPECTS] == 2).sum().sum(),
            "Positive": (df[ASPECTS] == 3).sum().sum(),
        }
        fig2, ax2 = plt.subplots()
        ax2.pie(polarity_counts.values(), labels=polarity_counts.keys(), autopct='%1.1f%%', colors=['#dc3545', '#6c757d', '#28a745'])
        st.pyplot(fig2)

        # 3. Phân phối theo Khía cạnh (Bar Chart)
        st.subheader("3. Chi tiết Cảm xúc theo Khía cạnh")
        aspect_sentiment = pd.DataFrame({
            "Aspect": ASPECTS,
            "Negative": [(df[a] == 1).sum() for a in ASPECTS],
            "Neutral":  [(df[a] == 2).sum() for a in ASPECTS],
            "Positive": [(df[a] == 3).sum() for a in ASPECTS],
        })
        fig3 = aspect_sentiment.set_index("Aspect").plot(kind="bar", figsize=(12, 6), color=['#dc3545', '#6c757d', '#28a745']).figure
        st.pyplot(fig3)

        # 4. Heatmap Tương quan (Correlation)
        st.subheader("4. Ma trận Tương quan giữa các Khía cạnh")
        corr = df[ASPECTS].replace({0: np.nan}).corr()
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

        # 5. WordCloud
        st.subheader("5. Từ khóa nổi bật (WordCloud)")
        col_wc1, col_wc2 = st.columns(2)
        
        positive_text = " ".join(df[df[ASPECTS].eq(3).any(axis=1)]["comment_cleaned"])
        negative_text = " ".join(df[df[ASPECTS].eq(1).any(axis=1)]["comment_cleaned"])
        
        with col_wc1:
            st.write("**Từ khóa Tích cực**")
            if len(positive_text) > 0:
                wc_pos = WordCloud(width=400, height=300, background_color="white").generate(positive_text)
                fig_p, ax_p = plt.subplots()
                ax_p.imshow(wc_pos, interpolation='bilinear')
                ax_p.axis("off")
                st.pyplot(fig_p)
        
        with col_wc2:
            st.write("**Từ khóa Tiêu cực**")
            if len(negative_text) > 0:
                wc_neg = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(negative_text)
                fig_n, ax_n = plt.subplots()
                ax_n.imshow(wc_neg, interpolation='bilinear')
                ax_n.axis("off")

                st.pyplot(fig_n)
