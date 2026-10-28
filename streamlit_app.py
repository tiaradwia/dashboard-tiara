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

def create_bystate_df(df):
    return df.groupby('customer_state', dropna=False).agg(sales=('sales', 'sum'),
        orders=('order_id', 'nunique'), customers=('customer_unique_id', 'nunique')
    ).sort_values('sales', ascending=False)

def create_delivery_df(df, start_date, end_date):
    scope = filter_dates(df, 'order_delivered_customer_date', start_date, end_date)
    delivery = scope.dropna(subset=['order_purchase_timestamp',
        'order_delivered_customer_date', 'order_estimated_delivery_date']).copy()
    delivery = delivery[delivery['order_delivered_customer_date'] >=
                        delivery['order_purchase_timestamp']].copy()
    delivery['delay_days'] = (delivery['order_delivered_customer_date'].dt.normalize()
        - delivery['order_estimated_delivery_date'].dt.normalize()).dt.days
    delivery['late'] = delivery['delay_days'] > 0
    return scope, delivery
def create_abc_df(category_sales):
    abc = category_sales.sort_values('sales', ascending=False).copy()
    abc['cumulative_pct'] = abc['share_pct'].cumsum()
    previous = abc['cumulative_pct'] - abc['share_pct']
    abc['segment'] = 'C'
    abc.loc[previous < 95, 'segment'] = 'B'
    abc.loc[previous < 80, 'segment'] = 'A'
    return abc

def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.set_page_config(
        page_title="Dashboard E-Commerce Tiara",
        layout="wide"
    )

    st.title("Dashboard Analisis E-Commerce")
    st.caption("Tiara Dwi Ardiyanti | Olist | Tahun 2017")

    # Menentukan folder kerja
    script_path = globals().get("__file__", "").replace("\\", "/")
    base = script_path.rsplit("/", 1)[0] if "/" in script_path else "."
    csv_path = base + "/all_data.csv"
    # Sidebar
    with st.sidebar:
        st.header("Sumber Data")

        csv_location = st.text_input(
            "Lokasi all_data.csv",
            value=str(csv_path)
        )

        upload = st.file_uploader(
            "Atau unggah all_data.csv",
            type=["csv"]
        )

        dates = st.date_input(
            "Rentang waktu tahun 2017",
            value=(
                pd.Timestamp("2017-01-01").date(),
                pd.Timestamp("2017-12-31").date()
            ),
            min_value=pd.Timestamp("2017-01-01").date(),
            max_value=pd.Timestamp("2017-12-31").date()
        )
         # Memastikan rentang tanggal sudah lengkap
        if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Pilih tanggal awal dan akhir.")
        return

    start_date, end_date = dates

    # Membaca dan menyiapkan data
    try:
        all_data = load_data(csv_location, upload)
        order_analysis, item_analysis = prepare_data(all_data)

    except (OSError, ValueError, KeyError) as error:
        st.error(f"Data belum dapat dibaca: {error}")
        st.info(
            "Letakkan all_data.csv dalam folder yang sama dengan streamlit_app.py "
            "atau unggah CSV melalui sidebar."
        )
        return

st.caption(
        f"Periode: {start_date:%d-%m-%Y} hingga {end_date:%d-%m-%Y}. "
        "Hanya pesanan berstatus delivered. Nilai penjualan adalah harga produk tanpa ongkir."
    )

    # Filter penjualan berdasarkan tanggal pembelian
    sales = filter_dates(
        order_analysis,
        "order_purchase_timestamp",
        start_date,
        end_date
    )

    st.subheader("1. Penjualan dan Jumlah Pesanan")

    if sales.empty:
        st.info("Tidak ada pesanan yang dibeli pada periode ini.")

    else:
        monthly = create_monthly_orders_df(
            sales, start_date, end_date
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Penjualan",
            format_currency(sales["sales"].sum(), "BRL", locale="pt_BR")
        )
        col2.metric(
            "Jumlah Pesanan",
            f"{sales['order_id'].nunique():,}"
        )
        col3.metric(
            "Pelanggan Unik",
            f"{sales['customer_unique_id'].nunique():,}"
        )

# Grafik penjualan dan pesanan
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True
        )

        axes[0].plot(
            monthly.index.astype(str),
            monthly["sales"] / 1_000_000,
            marker="o"
        )
        axes[0].set_title("Penjualan Tahun 2017")
        axes[0].set_ylabel("Nilai produk (juta BRL)")

        axes[1].bar(
            monthly.index.astype(str),
            monthly["orders"],
            color="teal"
        )
        axes[1].set_ylabel("Jumlah pesanan")
        axes[1].set_xlabel("Bulan pembelian")

        plt.xticks(rotation=45)
        plt.tight_layout()
        show_figure(fig)

        peak = monthly["sales"].idxmax()

        st.write(
            f"Penjualan tertinggi pada periode terpilih terjadi "
            f"pada {peak}, sebesar "
            f"{format_currency(monthly.loc[peak, 'sales'], 'BRL', locale='pt_BR')}, dari "
            f"{monthly.loc[peak, 'orders']:,.0f} pesanan."
        )

        st.caption(
            "Jika rentang tanggal hanya mencakup sebagian bulan, "
            "nilai bulan tersebut juga hanya mencakup tanggal terpilih."
        )

        # Kategori produk
        st.subheader("2. Lima Kategori dengan Penjualan Tertinggi")

        category_sales = create_category_sales_df(
            item_analysis, sales
        )
        top5 = category_sales.head(5)

        fig, ax = plt.subplots(figsize=(10, 5))

        top5["sales"].sort_values().div(1_000_000).plot(
            kind="barh",
            ax=ax,
            color="teal"
        )

        ax.set_xlabel("Nilai produk (juta BRL)")
        ax.set_ylabel("Kategori produk")
        plt.tight_layout()
        show_figure(fig)

        st.dataframe(top5)

        st.write(
            f"Lima kategori teratas menyumbang "
            f"{top5['share_pct'].sum():.2f}% dari total nilai "
            f"penjualan. Kontribusi terbesar berasal dari "
            f"{top5.index[0]}, yaitu "
            f"{top5['share_pct'].iloc[0]:.2f}%."
        )

        # Negara bagian
        st.subheader("3. Penjualan dan Pelanggan per Negara Bagian")

        state_summary = create_bystate_df(sales)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        state_summary["sales"].nlargest(10).sort_values().div(
            1_000_000
        ).plot(
            kind="barh",
            ax=axes[0],
            color="teal"
        )
        
        axes[0].set_title("Penjualan Tertinggi")
        axes[0].set_xlabel("Nilai produk (juta BRL)")
        axes[0].set_ylabel("Negara bagian")

        state_summary["customers"].nlargest(10).sort_values().plot(
            kind="barh",
            ax=axes[1],
            color="steelblue"
        )

        axes[1].set_title("Pelanggan Terbanyak")
        axes[1].set_xlabel("Jumlah pelanggan unik")
        axes[1].set_ylabel("Negara bagian")

        plt.tight_layout()
        show_figure(fig)

        st.dataframe(state_summary)

        st.write(
            f"Penjualan tertinggi berasal dari "
            f"{state_summary['sales'].idxmax()}, sedangkan "
            f"pelanggan terbanyak berasal dari "
            f"{state_summary['customers'].idxmax()}."
        )
        
        # Segmentasi ABC
        st.subheader("Analisis Lanjutan: Segmentasi ABC")

        abc = create_abc_df(category_sales)

        hasil_abc = abc.groupby("segment").agg(
            jumlah_kategori=("sales", "count"),
            total_penjualan=("sales", "sum"),
            kontribusi_persen=("share_pct", "sum")
        )

        st.dataframe(hasil_abc)

        with st.expander("Lihat kategori setiap kelompok"):
            st.dataframe(abc)

        st.caption(
            "Kelompok A menyumbang sekitar 80% pertama penjualan, "
            "B sekitar 15% berikutnya, dan C sisanya. "
            "Persentase dapat melewati batas karena kategori "
            "dimasukkan secara utuh."
        )

    # Pengiriman difilter berdasarkan tanggal penerimaan,
    # bukan tanggal pembelian
    st.subheader("4. Ketepatan Waktu Pengiriman")

    delivery_scope, delivery = create_delivery_df(
        order_analysis,
        start_date,
        end_date
    )

    st.caption(
        "Bagian ini menggunakan tanggal penerimaan pelanggan. "
        "Jumlah pesanan dapat berbeda dari bagian penjualan."
    )

    if delivery.empty:
        st.info(
            "Tidak ada pesanan dengan tanggal valid "
            "yang diterima pada periode ini."
        )
        return

    tepat_waktu = (delivery["delay_days"] <= 0).sum()
    terlambat = (delivery["delay_days"] > 0).sum()

    hari_terlambat = delivery.loc[
        delivery["delay_days"] > 0, "delay_days"
    ]

    on_time_pct = tepat_waktu / len(delivery) * 100
    avg_late = hari_terlambat.mean()
   col1, col2, col3 = st.columns(3)

    col1.metric("Pesanan Dianalisis", f"{len(delivery):,}")
    col2.metric("Tepat Waktu", f"{on_time_pct:.2f}%")
    col3.metric(
        "Rata-rata Keterlambatan",
        f"{avg_late:.2f} hari"
        if not hari_terlambat.empty
        else "Tidak ada"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        ["Tepat waktu", "Terlambat"],
        [tepat_waktu, terlambat],
        color=["teal", "coral"]
    )
    axes[0].set_title("Ketepatan Pengiriman")
    axes[0].set_ylabel("Jumlah pesanan")

    axes[1].hist(
        hari_terlambat,
        bins=25,
        color="coral",
        edgecolor="white"
    )
    axes[1].set_title("Distribusi Keterlambatan")
    axes[1].set_xlabel("Lama keterlambatan (hari)")
    axes[1].set_ylabel("Jumlah pesanan")

    plt.tight_layout()
    show_figure(fig)

    st.write(
        f"Dari {len(delivery):,} pesanan yang dianalisis, "
        f"{on_time_pct:.2f}% diterima tepat waktu."
    )
if not hari_terlambat.empty:
        st.write(
            f"Pesanan yang terlambat rata-rata diterima "
            f"{avg_late:.2f} hari setelah tanggal perkiraan penerimaan."
        )

    st.write(
        "Pesanan dalam cakupan penerimaan yang tidak disertakan:",
        len(delivery_scope) - len(delivery)
    )

    st.caption(
        "Pesanan yang diterima pada tanggal estimasi tetap "
        "dianggap tepat waktu."
    )

    st.caption("Tiara Dwi Ardiyanti | Proyek Analisis Data")










