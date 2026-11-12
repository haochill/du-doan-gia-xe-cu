"""App Streamlit: Dự đoán giá xe cũ (SVM) và phân cụm xe (K-means) - Nhóm 2.

Chạy:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Giá xe cũ - Nhóm 2", page_icon="🚗", layout="wide")
DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------- Nạp dữ liệu/mô hình
@st.cache_data
def load_data():
    full = pd.read_excel(DATA / "Nhom2_GiaXeCu.xlsx")
    train = pd.read_excel(DATA / "Nhom2_GiaXeCu_train.xlsx")
    test = pd.read_excel(DATA / "Nhom2_GiaXeCu_test.xlsx")
    return full, train, test


@st.cache_resource
def get_regressor(kind):
    _, train, test = load_data()
    return core.train_regressor(train, test, kind)


@st.cache_data
def get_wcss():
    full, _, _ = load_data()
    return core.wcss_curve(full)


@st.cache_resource
def get_kmeans(k):
    full, _, _ = load_data()
    return core.fit_kmeans(full, k)


full, train, test = load_data()
rng = full.agg(["min", "max"])  # khoảng giá trị của dữ liệu huấn luyện

# ------------------------------------------------------------------- Giao diện
st.title("🚗 Dự đoán giá xe cũ & Phân cụm xe")
st.caption(
    "Nhóm 2 • Dữ liệu: Kaggle *Used Car Price Prediction Dataset* (3.934 xe, giá USD, "
    "nhiều khả năng thuộc thị trường xe cũ Mỹ). Kết quả chỉ mang tính tham khảo, "
    "không áp dụng trực tiếp cho thị trường Việt Nam."
)

tab1, tab2, tab3 = st.tabs(["💰 Dự đoán giá (Thuật toán 1)",
                            "🧩 Phân cụm K-means (Thuật toán 2)",
                            "ℹ️ Về dự án"])

# ============================ TAB 1: DỰ ĐOÁN GIÁ ============================
with tab1:
    kind = st.selectbox("Mô hình dự đoán", list(core.REGRESSORS),
                        format_func=lambda k: core.REGRESSORS[k])
    model, scaler, columns, metrics, result = get_regressor(kind)

    st.subheader("Nhập thông tin xe")
    with st.form("form_price"):
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Hãng xe", core.BRANDS)
            fuel = st.selectbox("Loại nhiên liệu", core.FUELS)
            trans = st.selectbox("Hộp số", core.TRANSMISSIONS)
            accident = st.checkbox("Xe từng có tai nạn / hư hại")
            clean = st.checkbox("Có giấy tờ sạch (clean title)", value=True)
        with c2:
            year = st.number_input("Năm sản xuất", int(rng.loc["min", "model_year"]),
                                   int(rng.loc["max", "model_year"]), 2018, step=1)
            mil = st.number_input("Số dặm đã chạy (miles)", 0, 500000, 50000, step=1000)
            hp = st.number_input("Công suất (HP)", 50, 1200, 300, step=10)
            liter = st.number_input("Dung tích động cơ (lít)", 0.5, 9.0, 3.0, step=0.1)
        submitted = st.form_submit_button("Dự đoán giá", type="primary")

    if submitted:
        price = core.predict_price(
            model, scaler, columns, brand=brand, fuel_type=fuel, transmission=trans,
            model_year=year, milage=mil, hp=hp, liter=liter,
            accident=accident, clean_title=clean)
        if price < 0:
            st.warning("Mô hình cho giá âm với thông tin này (thường gặp ở hồi quy tuyến tính). "
                       "Hiển thị mức tối thiểu 0 USD.")
        st.metric("Giá dự đoán", f"{max(price, 0):,.0f} USD")
        # Cảnh báo khi đầu vào nằm ngoài vùng dữ liệu huấn luyện
        for name, val in [("milage", mil), ("hp", hp), ("liter", liter)]:
            if not rng.loc["min", name] <= val <= rng.loc["max", name]:
                st.info(f"Giá trị **{name}** nằm ngoài khoảng dữ liệu huấn luyện "
                        f"({rng.loc['min', name]:g} – {rng.loc['max', name]:g}), "
                        "độ tin cậy của dự đoán thấp hơn.")

    with st.expander("Độ chính xác mô hình trên tập test (787 xe)"):
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R2']:.3f}")
        m2.metric("MAE", f"{metrics['MAE']:,.0f} USD")
        m3.metric("RMSE", f"{metrics['RMSE']:,.0f} USD")
        st.write("5 dòng đầu của tập test: giá thực tế và giá dự đoán")
        st.dataframe(result.head(5), use_container_width=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(result["price_thuc_te"], result["price_du_doan"], s=8, alpha=0.5)
        lim = [0, result["price_thuc_te"].max()]
        ax.plot(lim, lim, "r--", label="Dự đoán hoàn hảo")
        ax.set_xlabel("Giá thực tế (USD)")
        ax.set_ylabel("Giá dự đoán (USD)")
        ax.legend()
        st.pyplot(fig)

# ============================ TAB 2: K-MEANS ================================
with tab2:
    st.markdown(
        "Phân cụm xe theo 5 thuộc tính: `model_year`, `milage`, `hp`, `liter`, `price` "
        "(đã chuẩn hóa) để nhận diện các phân khúc xe cũ."
    )
    st.subheader("1. Chọn số cụm bằng phương pháp Elbow")
    wcss = get_wcss()
    cA, cB = st.columns([1, 2])
    cA.dataframe(wcss, hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(wcss["k"], wcss["WCSS"], "o-")
    ax.axvline(3, color="red", linestyle="--", label="k = 3")
    ax.set_xlabel("Số cụm k")
    ax.set_ylabel("WCSS")
    ax.set_xticks(wcss["k"])
    ax.legend()
    ax.grid(alpha=0.3)
    cB.pyplot(fig)

    st.subheader("2. Kết quả phân cụm")
    k = st.slider("Số cụm k", 2, 8, 3)
    km_scaler, km, clustered, summary = get_kmeans(k)
    names = core.cluster_names(summary)
    table = summary.copy()
    table.insert(0, "phân khúc", [names[c] for c in table.index])
    st.dataframe(table, use_container_width=True)

    st.subheader("3. Xe của bạn thuộc cụm nào?")
    with st.form("form_cluster"):
        d1, d2, d3, d4, d5 = st.columns(5)
        y2 = d1.number_input("Năm SX", 1974, 2024, 2018, step=1)
        m2_ = d2.number_input("Số dặm", 0, 500000, 50000, step=1000)
        h2 = d3.number_input("HP", 50, 1200, 300, step=10)
        l2 = d4.number_input("Lít", 0.5, 9.0, 3.0, step=0.1)
        p2 = d5.number_input("Giá (USD)", 1000, 250000, 30000, step=1000)
        go = st.form_submit_button("Xác định cụm", type="primary")

    point = None
    if go:
        c = core.predict_cluster(km_scaler, km, model_year=y2, milage=m2_, hp=h2,
                                 liter=l2, price=p2)
        st.success(f"Xe thuộc **cụm {c}**: {names[c]}")
        point = (m2_, p2)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(clustered["milage"], clustered["price"], c=clustered["cluster"],
                    cmap="viridis", s=10, alpha=0.7)
    if point:
        ax.scatter(*point, marker="*", s=300, color="red", edgecolor="black", label="Xe của bạn")
        ax.legend()
    ax.set_xlabel("Số dặm (milage)")
    ax.set_ylabel("Giá (USD)")
    ax.set_title(f"Phân cụm K-means (k = {k})")
    fig.colorbar(sc, label="cụm")
    st.pyplot(fig)

# ============================ TAB 3: GIỚI THIỆU =============================
with tab3:
    st.markdown(
        """
**Bài toán:** dự đoán giá xe cũ và phân nhóm xe theo đặc điểm.

**Thuật toán 1 – Dự đoán giá (SVM / SVR):** biến mục tiêu là `price`; các biến đầu vào gồm năm
sản xuất, số dặm, công suất, dung tích động cơ, hãng, nhiên liệu, hộp số, tai nạn, giấy tờ.
Dữ liệu được chuẩn hóa trước khi huấn luyện; huấn luyện trên tập train (80%), đánh giá trên tập test (20%).

**Thuật toán 2 – Phân cụm K-means:** chia xe thành các phân khúc theo năm, số dặm, công suất,
dung tích và giá; số cụm chọn bằng phương pháp Elbow.

**Giới hạn:** dữ liệu từ một nguồn Kaggle, giá USD, nhiều khả năng của thị trường Mỹ; xe hiếm,
giá quá cao hoặc ngoài khoảng dữ liệu huấn luyện sẽ cho dự đoán kém tin cậy.
"""
    )
    st.write("Mẫu dữ liệu đã xử lý (10 dòng đầu):")
    st.dataframe(full.head(10), use_container_width=True)
