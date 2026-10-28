import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set_theme(style="darkgrid")


@st.cache_data(show_spinner="Membaca all_data.csv...")
def load_data(csv_path, uploaded_file=None):
    """Membaca satu CSV gabungan, bukan dataset mentah terpisah."""
    source = uploaded_file if uploaded_file is not None else csv_path
    if uploaded_file is not None:
        uploaded_file.seek(0)
    df = pd.read_csv(source, dtype={
        "customer_zip_code_prefix": "string",
        "seller_zip_code_prefix": "string",
    }, encoding="utf-8-sig")
    required = {
        "order_id", "order_item_id", "order_status", "customer_unique_id",
        "customer_state", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date",
        "price", "freight_value", "category",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))
    if df["order_id"].isna().any():
        raise ValueError("Ada order_id kosong. Periksa all_data.csv.")
    if df.duplicated(["order_id", "order_item_id"]).any():
        raise ValueError("Ada pasangan order_id dan order_item_id berulang.")
    for col in ["order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["price", "freight_value"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["category"] = df["category"].fillna("Unknown")
    return df


def prepare_data(all_data):
    """Pisahkan tingkat pesanan dan item agar pesanan tidak dihitung berulang."""
    order_columns = [
        "order_id", "order_status", "customer_unique_id", "customer_state",
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = all_data[order_columns].drop_duplicates()
    if orders["order_id"].duplicated().any():
        raise ValueError("Informasi pesanan tidak konsisten untuk order_id yang sama.")
    items = all_data.loc[
        all_data["order_item_id"].notna(),
        ["order_id", "order_item_id", "price", "freight_value", "category"],
    ].copy()
    if items[["price", "freight_value"]].isna().any().any():
        raise ValueError("Ada item tanpa harga atau ongkir. Periksa all_data.csv.")
    order_value = items.groupby("order_id", as_index=False).agg(
        sales=("price", "sum"), freight=("freight_value", "sum")
    )
    delivered = orders.loc[orders["order_status"] == "delivered"].copy()
    order_analysis = delivered.merge(
        order_value, on="order_id", how="left", validate="one_to_one"
    )
    if order_analysis["sales"].isna().any():
        raise ValueError("Ada pesanan delivered tanpa item. Periksa kelengkapan data.")
    return order_analysis, items


def filter_dates(df, column, start_date, end_date):
    # Batas akhir eksklusif mencakup seluruh jam pada tanggal terakhir.
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return df[(df[column] >= start) & (df[column] < end)].copy()

def create_daily_orders_df(df):
    return df.resample('D', on='order_purchase_timestamp').agg(
        order_count=('order_id', 'nunique'), revenue=('sales', 'sum')).reset_index()

def create_monthly_orders_df(df, start_date, end_date):
    result = df.groupby(df['order_purchase_timestamp'].dt.to_period('M')).agg(
        sales=('sales', 'sum'), orders=('order_id', 'nunique'))
    months = pd.period_range(pd.Timestamp(start_date).to_period('M'),
                            pd.Timestamp(end_date).to_period('M'), freq='M')
    return result.reindex(months, fill_value=0)

def create_category_sales_df(items, orders):
    selected = pd.merge(items, orders[['order_id']], on='order_id',
                         how='inner', validate='many_to_one')
    result = selected.groupby('category').agg(sales=('price', 'sum'))
    result = result.sort_values('sales', ascending=False)
    total = result['sales'].sum()
    result['share_pct'] = result['sales'] / total * 100 if total else 0.0
    return result
