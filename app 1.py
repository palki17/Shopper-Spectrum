import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD SAVED ARTIFACTS
# =========================
scaler = joblib.load("artifacts/rfm_scaler.pkl")
kmeans = joblib.load("artifacts/kmeans_model.pkl")
label_map = joblib.load("artifacts/cluster_label_map.pkl")
item_similarity_df = joblib.load("artifacts/item_similarity_df.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation and Product Recommendation System")

# =========================
# PRODUCT RECOMMENDATION FUNCTION
# =========================
def recommend_products(product_name, similarity_df, top_n=5):
    if not isinstance(product_name, str):
        return ["Invalid input. Please enter a valid product name."]
    
    product_name = product_name.strip().upper()
    
    if product_name not in similarity_df.index:
        matches = [item for item in similarity_df.index if product_name in item]
        if len(matches) == 0:
            return ["Product not found in catalog."]
        product_name = matches[0]
    
    similar_scores = similarity_df[product_name].sort_values(ascending=False)
    similar_scores = similar_scores.drop(product_name, errors='ignore')
    return similar_scores.head(top_n).index.tolist()

# =========================
# CUSTOMER SEGMENT PREDICTION FUNCTION
# =========================
def predict_customer_segment(recency, frequency, monetary, scaler_obj, model_obj, cluster_to_label_map):
    input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    scaled_input = scaler_obj.transform(input_data)
    cluster = model_obj.predict(scaled_input)[0]
    segment = cluster_to_label_map.get(cluster, f"Cluster-{cluster}")
    return cluster, segment

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🎯 Product Recommendation", "👥 Customer Segmentation"])

# =========================
# TAB 1: PRODUCT RECOMMENDATION
# =========================
with tab1:
    st.markdown("## Product Recommendation Module")
    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:
            recommendations = recommend_products(product_name, item_similarity_df, top_n=5)
            st.success("Top 5 Recommended Products")
            for i, product in enumerate(recommendations, 1):
                st.write(f"{i}. {product}")

# =========================
# TAB 2: CUSTOMER SEGMENTATION
# =========================
with tab2:
    st.markdown("## Customer Segmentation Module")

    recency = st.number_input("Recency (in days)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Cluster"):
        cluster, segment = predict_customer_segment(
            recency, frequency, monetary, scaler, kmeans, label_map
        )
        st.success(f"Predicted Segment: {segment}")
        st.info(f"Cluster ID: {cluster}")
