import streamlit as st
import pandas as pd
import joblib
from pyvis.network import Network
import streamlit.components.v1 as components

model = joblib.load("hastalik_tahmin_modeli.pkl")
semptom_listesi = joblib.load("semptom_listesi.pkl")
kurallar_df = pd.read_csv("birliktelik_kurallari.csv")

def metin_temizle(metin):
    return metin.replace("frozenset({", "").replace("})", "").replace(""", "").replace(""", "")

kurallar_df["antecedents"] = kurallar_df["antecedents"].apply(metin_temizle)
kurallar_df["consequents"] = kurallar_df["consequents"].apply(metin_temizle)

st.set_page_config(page_title="Yapay Zeka Destekli Tanı ve Analiz", layout="wide")
st.title("🩺 AI Sağlık Teşhis ve Semptom Analiz Aracı")
st.markdown("*Uyarı: Bu bir tıbbi teşhis aracı değil, makine öğrenmesi destekli akademik bir projedir.*")

tab1, tab2 = st.tabs(["Hastalık Tahmini", "Semptom İlişki Analizi"])

with tab1:
    st.header("Semptomlarınızı Seçin")
    secilen_semptomlar = st.multiselect(
        "Lütfen yaşadığınız semptomları aşağıdaki listeden seçiniz:",
        options=semptom_listesi
    )
    
    if st.button("Hastalığı Tahmin Et"):
        if len(secilen_semptomlar) == 0:
            st.warning("Lütfen en az bir semptom seçin.")
        else:
            input_vector = [1 if semptom in secilen_semptomlar else 0 for semptom in semptom_listesi]
            tahmin = model.predict([input_vector])[0]
            st.success(f"🤖 Yapay Zeka Modelinin Tahmini: **{tahmin}**")

with tab2:
    st.header("Semptomlar Arası Gizli İlişkiler")
    
    ilgi_semptomu = st.selectbox("İlişkisini görmek istediğiniz semptomu seçin:", options=semptom_listesi)
    
    filtreli_kurallar = kurallar_df[kurallar_df["antecedents"].str.contains(ilgi_semptomu, na=False)]
    
    if len(filtreli_kurallar) > 0:
        st.subheader("Semptom İlişki Ağı")

        ag = Network(height="500px", width="100%", bgcolor="black", font_color="white")
        ag.barnes_hut(gravity=-8000) 
        ag.add_node(ilgi_semptomu, label=ilgi_semptomu, color="red", size=25)

        for index, row in filtreli_kurallar.head(15).iterrows():
            hedef = row["consequents"]
            bag_gucu = row["lift"]

            ag.add_node(hedef, label=hedef, color="blue", size=15)
            ag.add_edge(ilgi_semptomu, hedef, value=bag_gucu, title=f"Bağ Gücü (Lift): {bag_gucu:.2f}")
            
        ag.save_graph("ag_grafigi.html")
        with open("ag_grafigi.html", "r", encoding="utf-8") as f:
            html_string = f.read()
        components.html(html_string, height=520)
        
        st.dataframe(filtreli_kurallar[["antecedents", "consequents", "confidence", "lift"]].head(10), use_container_width=True)

    else:
        st.info(f"'{ilgi_semptomu}' semptomu için yeterince güçlü bir birliktelik kuralı bulunamadı.")