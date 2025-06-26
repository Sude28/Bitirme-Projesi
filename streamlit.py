import streamlit as st
import pandas as pd
import plotly.express as px

# 🎨 Sayfa yapılandırması
st.set_page_config(page_title="İş ve Kariyer Trend Analizi", layout="wide")

# 📁 Pozisyon ve CSV dosyaları
csv_files = {
    "Backend Developer": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_backend_50.csv",
    "Frontend Developer": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_frontend_50.csv",
    "Mobile Developer": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_mobil.csv",
    "Full Stack Developer": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_full_stack.csv",
    "Data Scientist": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_datascientist.csv",
    "DevOps": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_devops.csv",
    "Siber Güvenlik Uzmanı": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_siber.csv",
    "Network Yönetimi": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_network.csv",
    "Software Engineer": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_software.csv",
    "Yapay Zeka Mühendisi": "C:\\Users\\Sude\\PycharmProjects\\BitirmeProjesiPlaywright\\.venv\\İlanlar\\output_yapayzeka.csv"
}

# 📊 Başlık
st.markdown("<h1 style='margin-top: -1em; margin-bottom: 0.5em;'>📊 İş ve Kariyer Trend Analizi Portalı</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-top: 0;'>Bu arayüzde, farklı pozisyonlara göre iş ilanlarından çıkarılmış <b>en sık geçen 15 teknolojiyi</b> interaktif olarak inceleyebilirsiniz. "
    "Veriler 2025 Mayıs ayında toplanmıştır ve analiz edilen ilan sayısı gösterilir.</p>",
    unsafe_allow_html=True
)

# 🔽 Pozisyon seçimi
selected_position = st.selectbox("Pozisyon seçin:", [""] + list(csv_files.keys()))

# 📈 Analiz bloğu
if selected_position:
    file_path = csv_files[selected_position]
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["Technology", "Frequency"])

        # ✔️ Analiz edilen ilan sayısı
        st.success(f"Bu pozisyon için analiz edilen ilan sayısı:  *{df['Frequency'].sum()}*")

        # 📊 Frekansa göre AZALAN sırala
        df = df.sort_values(by="Frequency", ascending=False)

        # 🔷 Plotly grafiği
        fig = px.bar(
            df,
            x="Frequency",
            y="Technology",
            orientation="h",
            title=f"{selected_position} - En Sık Geçen 15 Teknoloji",
            color="Frequency",
            color_continuous_scale="teal"
        )

        fig.update_layout(
            xaxis_title="Frekans",
            yaxis_title="Teknoloji",
            title_font_size=20,
            title_x=0.05,
            margin=dict(l=50, r=30, t=60, b=50),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=True
        )

        fig.update_traces(marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)

        # 📋 Veri tablosu
        st.subheader("📋 Veri Tablosu")
        st.dataframe(df)

        # 📥 CSV indir butonu
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 CSV dosyasını indir",
            data=csv_data,
            file_name=f"{selected_position.replace(' ', '_').lower()}_trend.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Hata: {e}")
else:
    st.info("Lütfen analiz edilecek bir pozisyon seçin.")