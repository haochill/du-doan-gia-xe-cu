"""Logic mô hình cho app Dự đoán giá xe cũ - Nhóm 2 (không phụ thuộc Streamlit)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Các lựa chọn của người dùng (khớp với cột one-hot trong file dữ liệu)
BRANDS = ["BMW", "Chevrolet", "Ford", "Jeep", "Lexus", "Mercedes-Benz",
          "Porsche", "Toyota", "Other"]
FUELS = ["Gasoline", "Diesel", "Hybrid", "Plug-In Hybrid", "E85 Flex Fuel", "Unknown"]
FUEL_BASELINE = "Diesel"          # drop_first=True nên Diesel không có cột riêng
TRANSMISSIONS = ["Automatic", "Manual", "Other"]
TRANS_BASELINE = "Automatic"      # tương tự, Automatic là mức cơ sở

TARGET = "price"
CLUSTER_COLS = ["model_year", "milage", "hp", "liter", "price"]
REGRESSORS = {"svr": "SVM - SVR (nhân RBF, C = 10000)",
              "linear": "Hồi quy tuyến tính (đối chứng)"}


# ---------------------------------------------------------------- Dự đoán giá
def train_regressor(train: pd.DataFrame, test: pd.DataFrame, kind: str = "svr"):
    """Huấn luyện trên file train, đánh giá trên file test.

    Trả về (model, scaler, danh_sách_cột_X, chỉ_số, bảng_kết_quả_test).
    """
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test[X_train.columns], test[TARGET]

    scaler = StandardScaler().fit(X_train)       # SVM bắt buộc phải chuẩn hóa
    if kind == "svr":
        model = SVR(kernel="rbf", C=10000, gamma="scale")
    else:
        model = LinearRegression()
    model.fit(scaler.transform(X_train), y_train)

    pred = model.predict(scaler.transform(X_test))
    metrics = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
    }
    result = X_test[["model_year", "milage", "hp", "liter"]].copy()
    result["price_thuc_te"] = y_test.values
    result["price_du_doan"] = pred.round(0)
    return model, scaler, list(X_train.columns), metrics, result


def build_input_row(columns, *, brand, fuel_type, transmission, model_year,
                    milage, hp, liter, accident, clean_title) -> pd.DataFrame:
    """Chuyển lựa chọn người dùng thành 1 dòng đúng định dạng one-hot của dữ liệu huấn luyện."""
    row = {c: 0 for c in columns}
    row.update(model_year=model_year, milage=milage, hp=hp, liter=liter,
               accident=int(accident), clean_title=int(clean_title))
    row[f"brand_{brand}"] = 1
    if fuel_type != FUEL_BASELINE:
        row[f"fuel_type_{fuel_type}"] = 1
    if transmission != TRANS_BASELINE:
        row[f"transmission_{transmission}"] = 1
    return pd.DataFrame([row], columns=columns)


def predict_price(model, scaler, columns, **inputs) -> float:
    x = build_input_row(columns, **inputs)
    return float(model.predict(scaler.transform(x))[0])


# ------------------------------------------------------------------- K-means
def wcss_curve(df: pd.DataFrame, k_max: int = 10) -> pd.DataFrame:
    Z = StandardScaler().fit_transform(df[CLUSTER_COLS])
    rows = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(Z)
        rows.append({"k": k, "WCSS": round(km.inertia_, 2)})
    return pd.DataFrame(rows)


def fit_kmeans(df: pd.DataFrame, k: int = 3):
    scaler = StandardScaler().fit(df[CLUSTER_COLS])
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(scaler.transform(df[CLUSTER_COLS]))
    out = df[CLUSTER_COLS].copy()
    out["cluster"] = labels
    summary = out.groupby("cluster")[CLUSTER_COLS].mean().round(1)
    summary["so_xe"] = out["cluster"].value_counts().sort_index()
    return scaler, km, out, summary


def cluster_names(summary: pd.DataFrame) -> dict:
    """Đặt tên cụm theo giá trung bình (thấp -> cao). Tên mô tả dùng cho k = 3."""
    order = summary["price"].sort_values().index.tolist()
    if len(order) == 3:
        labels = ["Xe đời cũ, chạy nhiều, giá rẻ",
                  "Xe đời mới, tầm trung",
                  "Xe hiệu suất cao, giá đắt"]
    else:
        labels = [f"Phân khúc giá #{i + 1} (thấp -> cao)" for i in range(len(order))]
    return {c: labels[i] for i, c in enumerate(order)}


def predict_cluster(scaler, km, *, model_year, milage, hp, liter, price) -> int:
    x = pd.DataFrame([[model_year, milage, hp, liter, price]], columns=CLUSTER_COLS)
    return int(km.predict(scaler.transform(x))[0])
