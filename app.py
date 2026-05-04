import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Business Dashboard", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.stApp {
    background-color: #F5F7FA;
}
div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #2E3A59;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

USER_FILE = "users.json"

# ------------------ USER FUNCTIONS ------------------
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# ------------------ SESSION ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ------------------ LOGIN ------------------
def login():
    st.title("🔐 Login")

    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ------------------ SIGNUP ------------------
def signup():
    st.title("📝 Signup")

    users = load_users()

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if new_user.strip() == "" or new_pass.strip() == "":
            st.warning("Username and password required")
        elif new_user in users:
            st.error("User already exists")
        else:
            users[new_user] = new_pass
            save_users(users)
            st.success("Account created! Please login.")

# ------------------ AUTH ------------------
if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])
    if menu == "Login":
        login()
    else:
        signup()
    st.stop()

# ------------------ LOGOUT ------------------
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------ DASHBOARD ------------------
st.title("📊 AI Business Insights Dashboard")

st.sidebar.header("⚙️ Filters")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    # Filters
    if "Region" in df.columns:
        region = st.sidebar.multiselect("Region", df["Region"].unique(), df["Region"].unique())
        df = df[df["Region"].isin(region)]

    if "Category" in df.columns:
        category = st.sidebar.multiselect("Category", df["Category"].unique(), df["Category"].unique())
        df = df[df["Category"].isin(category)]

    # KPIs
    st.subheader("📌 Key Metrics")
    total_revenue = df["Sales"].sum() if "Sales" in df else 0
    avg_sales = df["Sales"].mean() if "Sales" in df else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Revenue", f"{total_revenue:.2f}")
    col2.metric("📊 Avg Sales", f"{avg_sales:.2f}")
    col3.metric("📦 Orders", len(df))

    # Sales Trend
    if "Date" in df.columns and "Sales" in df.columns:
        st.subheader("📈 Sales Trend")
        df = df.sort_values("Date")
        fig = px.line(df, x="Date", y="Sales",
                      color_discrete_sequence=["#2196F3"])
        st.plotly_chart(fig, use_container_width=True)

    # Category
    if "Category" in df.columns and "Sales" in df.columns:
        st.subheader("📊 Category Performance")
        cat_df = df.groupby("Category")["Sales"].sum().reset_index()
        fig = px.bar(cat_df, x="Category", y="Sales", color="Category",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    # Region
    if "Region" in df.columns and "Sales" in df.columns:
        st.subheader("🌍 Region Distribution")
        reg_df = df.groupby("Region")["Sales"].sum().reset_index()
        fig = px.pie(reg_df, names="Region", values="Sales",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    # AI Insights
    st.subheader("🤖 AI Insights")
    if "Sales" in df.columns and len(df) > 5:
        if df["Sales"].iloc[-1] < df["Sales"].iloc[-5]:
            st.warning("Sales are decreasing")
        else:
            st.success("Sales are increasing")

    # Prediction
    if "Date" in df.columns and "Sales" in df.columns:
        st.subheader("🔮 Prediction")

        df["Days"] = (df["Date"] - df["Date"].min()).dt.days
        X = df[["Days"]]
        y = df["Sales"]

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(df["Days"].max()+1, df["Days"].max()+15).reshape(-1,1)
        preds = model.predict(future_days)

        future_dates = pd.date_range(df["Date"].max(), periods=14)

        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Prediction": preds
        })

        fig = px.line(pred_df, x="Date", y="Prediction",
                      color_discrete_sequence=["#FF5722"])
        st.plotly_chart(fig, use_container_width=True)

    # Clustering
    if "Sales" in df.columns:
        st.subheader("👥 Segmentation")

        features = df[["Sales"]].copy()
        if "Profit" in df.columns:
            features["Profit"] = df["Profit"]

        scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(scaled)

        y_axis = features.columns[-1]

        fig = px.scatter(df, x="Sales", y=y_axis,
                         color=df["Cluster"].astype(str),
                         color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

    # Download
    st.subheader("⬇️ Download")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "data.csv", "text/csv")

else:
    st.info("👈 Upload CSV from sidebar to start")