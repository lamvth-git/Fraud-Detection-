"""
app.py - Dashboard Phát hiện Gian lận Giao dịch Ngân hàng
===========================================================
TUẦN 9  : Giao diện Streamlit + Sidebar nhập tay + KPI + biểu đồ Plotly cụm KH
TUẦN 10 : Mô phỏng luồng giao dịch real-time bằng st.empty() + cảnh báo đỏ

CẤU TRÚC THƯ MỤC GIẢ ĐỊNH (app.py đặt ở thư mục gốc dự án):

    Fraud-Detection-/
    ├── Data/
    │   └── AIML Dataset.csv
    ├── Notebooks/
    │   ├── FraudDetectionPipeline_RF.pkl   <- có sẵn từ notebook
    │   ├── threshold.json                  <- có sẵn từ notebook
    │   ├── cluster_scaler.pkl              <- sinh ra từ prepare_extra_artifacts.py
    │   ├── kmeans_model.pkl                <- sinh ra từ prepare_extra_artifacts.py
    │   ├── pca_model.pkl                   <- sinh ra từ prepare_extra_artifacts.py
    │   └── pca_background_sample.csv       <- sinh ra từ prepare_extra_artifacts.py
    └── app.py   <- file này

CÁCH CHẠY (đứng tại thư mục gốc Fraud-Detection-):
    streamlit run app.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# 0. CẤU HÌNH CHUNG - sửa lại đường dẫn ở đây nếu cấu trúc thư mục khác đi
#    (hiện đang trỏ theo cấu trúc: app.py ở gốc, model nằm trong Notebooks/)
# --------------------------------------------------------------------------
DATA_PATH = "Data/AIML Dataset.csv"
MODEL_PATH = "Notebooks/FraudDetectionPipeline_RF.pkl"
THRESHOLD_PATH = "Notebooks/threshold.json"
CLUSTER_SCALER_PATH = "Notebooks/cluster_scaler.pkl"
KMEANS_PATH = "Notebooks/kmeans_model.pkl"
PCA_PATH = "Notebooks/pca_model.pkl"
PCA_BACKGROUND_PATH = "Notebooks/pca_background_sample.csv"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(
    page_title="Bank Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
)

# --------------------------------------------------------------------------
# 1. LOAD MODEL & DỮ LIỆU (chỉ load 1 lần nhờ cache)
# --------------------------------------------------------------------------


@st.cache_resource
def load_rf_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)
    return float(data["threshold"])


@st.cache_resource
def load_cluster_artifacts():
    """Trả về (scaler, kmeans, pca) hoặc (None, None, None) nếu chưa tạo."""
    try:
        scaler = joblib.load(CLUSTER_SCALER_PATH)
        kmeans = joblib.load(KMEANS_PATH)
        pca = joblib.load(PCA_PATH)
        return scaler, kmeans, pca
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_pca_background():
    try:
        return pd.read_csv(PCA_BACKGROUND_PATH)
    except FileNotFoundError:
        return None


@st.cache_data
def load_raw_dataset():
    """Dùng cho phần mô phỏng real-time (Tuần 10): lấy ngẫu nhiên 1 dòng."""
    return pd.read_csv(DATA_PATH)


rf_pipeline = load_rf_pipeline()
THRESHOLD = load_threshold()
cluster_scaler, kmeans_model, pca_model = load_cluster_artifacts()
pca_background = load_pca_background()


# --------------------------------------------------------------------------
# 2. HÀM TIỆN ÍCH: tạo feature & chấm điểm rủi ro cho 1 giao dịch
# --------------------------------------------------------------------------

def engineer_features(tx_type, amount, old_org, new_org, old_dest, new_dest):
    """Tính đúng các đặc trưng (feature) mà notebook đã tạo ra trước khi
    đưa vào pipeline Random Forest."""
    balance_diff_org = old_org - new_org
    balance_diff_dest = new_dest - old_dest
    amount_balance_ratio = amount / (old_org + 1)
    balance_error_org = abs(old_org - amount - new_org)
    balance_error_dest = abs(old_dest + amount - new_dest)

    row = pd.DataFrame([{
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "balanceDiffOrg": balance_diff_org,
        "balanceDiffDest": balance_diff_dest,
        "amountBalanceRatio": amount_balance_ratio,
        "balanceErrorOrg": balance_error_org,
        "balanceErrorDest": balance_error_dest,
    }])
    return row


def score_transaction(tx_type, amount, old_org, new_org, old_dest, new_dest):
    """Trả về (risk_score 0-1, is_fraud bool) cho 1 giao dịch."""
    row = engineer_features(tx_type, amount, old_org, new_org, old_dest, new_dest)
    risk_score = float(rf_pipeline.predict_proba(row)[:, 1][0])
    is_fraud = risk_score >= THRESHOLD
    return risk_score, is_fraud


def assign_cluster(amount, old_org, new_org, old_dest, new_dest):
    """
    Trả về:
        cluster_id, pca1, pca2

    Nếu chưa có artifact thì trả về:
        None, None, None
    """

    if cluster_scaler is None or kmeans_model is None or pca_model is None:
        return None, None, None

    # --------------------------------------------------
    # 1. Tạo dữ liệu đầu vào với float64
    # --------------------------------------------------
    feats = np.array(
        [[
            float(amount),
            float(old_org),
            float(new_org),
            float(old_dest),
            float(new_dest)
        ]],
        dtype=np.float64
    )

    # Đảm bảo mảng liên tục trong bộ nhớ + float64
    feats = np.ascontiguousarray(feats, dtype=np.float64)

    # --------------------------------------------------
    # 2. Scale dữ liệu
    # --------------------------------------------------
    feats_scaled = cluster_scaler.transform(feats)

    # Ép lại float64 sau khi scale
    feats_scaled = np.ascontiguousarray(
        feats_scaled,
        dtype=np.float64
    )

    # --------------------------------------------------
    # 3. KMeans dự đoán cluster
    # --------------------------------------------------
    cluster_id = int(
        kmeans_model.predict(feats_scaled)[0]
    )

    # --------------------------------------------------
    # 4. PCA
    # --------------------------------------------------
    pca_xy = pca_model.transform(feats_scaled)[0]

    return (
        cluster_id,
        float(pca_xy[0]),
        float(pca_xy[1])
    )


def make_cluster_chart(highlight_point=None):
    """Vẽ biểu đồ Plotly các cụm khách hàng (PCA1 vs PCA2).
    Nếu có highlight_point=(pca1, pca2) thì đánh dấu giao dịch hiện tại."""
    if pca_background is None:
        return None

    fig = px.scatter(
        pca_background,
        x="PCA1",
        y="PCA2",
        color=pca_background["Cluster"].astype(str),
        opacity=0.45,
        labels={"color": "Cụm (Cluster)"},
        title="Phân cụm khách hàng theo hành vi giao dịch (PCA 2D)",
    )

    if highlight_point is not None:
        fig.add_trace(go.Scatter(
            x=[highlight_point[0]],
            y=[highlight_point[1]],
            mode="markers",
            marker=dict(size=18, color="red", symbol="star",
                        line=dict(width=2, color="black")),
            name="Giao dịch hiện tại",
        ))

    fig.update_layout(legend_title_text="Cụm (Cluster)")
    return fig


# --------------------------------------------------------------------------
# 3. SIDEBAR - NHẬP DỮ LIỆU THỦ CÔNG (TUẦN 9)
# --------------------------------------------------------------------------
st.sidebar.header("🧾 Nhập giao dịch thủ công")

tx_type = st.sidebar.selectbox("Loại giao dịch", TRANSACTION_TYPES)
amount = st.sidebar.number_input("Số tiền giao dịch (amount)", min_value=0.0,
                                  value=50000.0, step=1000.0)
old_org = st.sidebar.number_input("Số dư TRƯỚC giao dịch - bên gửi (oldbalanceOrg)",
                                   min_value=0.0, value=100000.0, step=1000.0)
new_org = st.sidebar.number_input("Số dư SAU giao dịch - bên gửi (newbalanceOrig)",
                                   min_value=0.0, value=50000.0, step=1000.0)
old_dest = st.sidebar.number_input("Số dư TRƯỚC giao dịch - bên nhận (oldbalanceDest)",
                                    min_value=0.0, value=0.0, step=1000.0)
new_dest = st.sidebar.number_input("Số dư SAU giao dịch - bên nhận (newbalanceDest)",
                                    min_value=0.0, value=50000.0, step=1000.0)

check_button = st.sidebar.button("🔍 Chấm điểm rủi ro", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Ngưỡng cảnh báo (threshold) hiện tại: **{THRESHOLD:.2f}**")

# --------------------------------------------------------------------------
# 4. TIÊU ĐỀ + KPI TỔNG QUAN
# --------------------------------------------------------------------------
st.title("🏦 Dashboard Phát hiện Gian lận Giao dịch Ngân hàng")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Ngưỡng cảnh báo", f"{THRESHOLD:.2f}")
kpi2.metric("Số cụm khách hàng", "5" if kmeans_model is not None else "Chưa có")
kpi3.metric("Mô hình đang dùng", "Random Forest")
kpi4.metric("Trạng thái mô hình cụm",
            "Đã sẵn sàng" if cluster_scaler is not None else "Thiếu artifact")

st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Chấm điểm thủ công", "⏱️ Mô phỏng Real-time"])

# ==========================================================================
# TAB 1: CHẤM ĐIỂM THỦ CÔNG (TUẦN 9)
# ==========================================================================
with tab1:
    st.subheader("Kết quả chấm điểm giao dịch")

    # Luôn chấm điểm với giá trị hiện tại trong sidebar, kể cả khi chưa bấm nút,
    # để người dùng thấy kết quả ngay khi vừa mở trang.
    risk_score, is_fraud = score_transaction(
        tx_type, amount, old_org, new_org, old_dest, new_dest
    )
    cluster_id, pca1, pca2 = assign_cluster(amount, old_org, new_org, old_dest, new_dest)

    c1, c2, c3 = st.columns(3)
    c1.metric("Điểm rủi ro (Risk score)", f"{risk_score * 100:.2f} %")
    c2.metric("Kết luận", "🚨 GIAN LẬN" if is_fraud else "✅ Bình thường")
    c3.metric("Cụm khách hàng (Cluster)",
              cluster_id if cluster_id is not None else "N/A")

    if is_fraud:
        st.error(
            f"🚨 CẢNH BÁO: Giao dịch có dấu hiệu GIAN LẬN! "
            f"Điểm rủi ro = {risk_score * 100:.2f}% "
            f"(vượt ngưỡng {THRESHOLD * 100:.0f}%)"
        )
    else:
        st.success(
            f"✅ Giao dịch bình thường. Điểm rủi ro = {risk_score * 100:.2f}% "
            f"(dưới ngưỡng {THRESHOLD * 100:.0f}%)"
        )

    st.markdown("#### Biểu đồ phân cụm khách hàng")
    chart = make_cluster_chart(
        highlight_point=(pca1, pca2) if pca1 is not None else None
    )
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning(
            "Chưa tìm thấy file cluster_scaler.pkl / kmeans_model.pkl / "
            "pca_model.pkl / pca_background_sample.csv.\n\n"
            "Hãy chạy `python prepare_extra_artifacts.py` một lần trước khi "
            "mở dashboard để tạo các file này."
        )

# ==========================================================================
# TAB 2: MÔ PHỎNG REAL-TIME (TUẦN 10)
# ==========================================================================
with tab2:
    st.subheader("Mô phỏng luồng giao dịch ngân hàng theo thời gian thực")
    st.caption(
        "Mỗi 1-2 giây, hệ thống lấy ngẫu nhiên 1 giao dịch từ dữ liệu gốc, "
        "chấm điểm rủi ro tức thì và cảnh báo nếu vượt ngưỡng."
    )

    col_a, col_b, col_c = st.columns(3)
    n_transactions = col_a.number_input("Số giao dịch mô phỏng", min_value=5,
                                         max_value=500, value=30, step=5)
    speed = col_b.slider("Tốc độ (giây / giao dịch)", min_value=1.0, max_value=2.0,
                          value=1.0, step=0.5)
    start_sim = col_c.button("▶️ Bắt đầu mô phỏng", use_container_width=True)

    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []

    # Placeholder để cập nhật liên tục (st.empty) — đúng yêu cầu Tuần 10
    kpi_placeholder = st.empty()
    alert_placeholder = st.empty()
    table_placeholder = st.empty()

    def render_state(latest_alert=None):
        """Vẽ lại KPI + bảng log dựa trên st.session_state.sim_log."""
        log = st.session_state.sim_log
        total = len(log)
        n_fraud = sum(1 for r in log if r["Cảnh báo"] == "🚨 FRAUD")
        fraud_rate = (n_fraud / total * 100) if total else 0.0
        avg_score = (sum(r["Điểm rủi ro (%)"] for r in log) / total) if total else 0.0

        with kpi_placeholder.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Giao dịch đã xử lý", total)
            k2.metric("Số giao dịch bị gắn cờ gian lận", n_fraud)
            k3.metric("Điểm rủi ro trung bình", f"{avg_score:.2f} %")

        if latest_alert is not None:
            alert_placeholder.error(latest_alert)
        elif total > 0:
            alert_placeholder.info("Giao dịch mới nhất: bình thường, không có cảnh báo.")

        with table_placeholder.container():
            st.markdown("##### Nhật ký giao dịch gần nhất")
            if log:
                df_log = pd.DataFrame(log[::-1][:20])  # mới nhất lên trên
                st.dataframe(df_log, use_container_width=True, hide_index=True)
            else:
                st.caption("Chưa có giao dịch nào được mô phỏng.")

    # Vẽ trạng thái hiện có (nếu người dùng vừa chuyển qua lại giữa các tab)
    render_state()

    if start_sim:

        # Xóa kết quả cũ khi bắt đầu lần mô phỏng mới
        st.session_state.sim_log = []

        raw_df = load_raw_dataset()

        for i in range(int(n_transactions)):

            # --------------------------------------------------
            # 1. Lấy ngẫu nhiên 1 giao dịch
            # --------------------------------------------------
            sample = raw_df.sample(1).iloc[0]

            # --------------------------------------------------
            # 2. Chấm điểm giao dịch
            # --------------------------------------------------
            risk_score, is_fraud = score_transaction(
                sample["type"],
                sample["amount"],
                sample["oldbalanceOrg"],
                sample["newbalanceOrig"],
                sample["oldbalanceDest"],
                sample["newbalanceDest"]
            )

            # --------------------------------------------------
            # 3. Lưu kết quả
            # --------------------------------------------------
            record = {
                "STT": i + 1,
                "Loại GD": sample["type"],
                "Số tiền": round(float(sample["amount"]), 2),
                "Điểm rủi ro (%)": round(risk_score * 100, 2),
                "Cảnh báo": "🚨 FRAUD" if is_fraud else "Bình thường"
            }

            st.session_state.sim_log.append(record)

            # --------------------------------------------------
            # 4. Tính KPI
            # --------------------------------------------------
            total = len(st.session_state.sim_log)

            n_fraud = sum(
                1 for r in st.session_state.sim_log
                if r["Cảnh báo"] == "🚨 FRAUD"
            )

            avg_score = sum(
                r["Điểm rủi ro (%)"]
                for r in st.session_state.sim_log
            ) / total

            # --------------------------------------------------
            # 5. Cập nhật KPI
            # --------------------------------------------------
            with kpi_placeholder.container():

                k1, k2, k3 = st.columns(3)

                k1.metric(
                    "Giao dịch đã xử lý",
                    total
                )

                k2.metric(
                    "Số giao dịch bị gắn cờ gian lận",
                    n_fraud
                )

                k3.metric(
                    "Điểm rủi ro trung bình",
                    f"{avg_score:.2f}%"
                )

            # --------------------------------------------------
            # 6. Hiển thị cảnh báo
            # --------------------------------------------------
            if is_fraud:

                alert_placeholder.error(
                    f"🚨 CẢNH BÁO GIAN LẬN — "
                    f"Giao dịch #{i + 1} | "
                    f"Loại: {sample['type']} | "
                    f"Số tiền: {sample['amount']:.2f} | "
                    f"Điểm rủi ro: {risk_score * 100:.2f}%"
                )

            else:

                alert_placeholder.info(
                    f"✅ Giao dịch #{i + 1} bình thường — "
                    f"Điểm rủi ro: {risk_score * 100:.2f}%"
                )

            # --------------------------------------------------
            # 7. Cập nhật bảng
            # --------------------------------------------------
            with table_placeholder.container():

                st.markdown("##### Nhật ký giao dịch gần nhất")

                df_log = pd.DataFrame(
                    st.session_state.sim_log[::-1]
                )

                st.dataframe(
                    df_log,
                    use_container_width=True,
                    hide_index=True
                )

            # --------------------------------------------------
            # 8. Chờ số giây đã chọn
            # --------------------------------------------------
            time.sleep(speed)

    st.success("✅ Đã hoàn tất mô phỏng!")
    if st.button("🗑️ Xoá nhật ký mô phỏng"):
        st.session_state.sim_log = []
        st.rerun()