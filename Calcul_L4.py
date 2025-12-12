# PARTIE 1/3 : Imports, chargement, calculs et utilitaires
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
st.title("Analyse d'impédance — (Partie 1/3 : données & calculs)")

# ---------------------------------------
# Upload du fichier
# ---------------------------------------
uploaded_file = st.file_uploader("Choisir un fichier CSV ou TXT (séparateur tab)", type=["csv", "txt"])
if uploaded_file is None:
    st.info("Dépose ou sélectionne ton fichier de mesures (colonnes: freq_Hz, Z_ohm, theta_deg).")
    st.stop()

# Lecture sécurisée du fichier (essayer plusieurs encodages si besoin)
try:
    df = pd.read_csv(
        uploaded_file,
        sep="\t",
        encoding="latin1",
        comment="!",
        engine="python"
    )
except Exception as e:
    st.error(f"Erreur lecture fichier : {e}")
    st.stop()

# Si les colonnes ne sont pas nommées, on force les noms attendus
if df.shape[1] >= 3:
    df = df.iloc[:, :3]
    df.columns = ["freq_Hz", "Z_ohm", "theta_deg"]
else:
    st.error("Fichier invalide : doit contenir au moins 3 colonnes (freq_Hz, Z_ohm, theta_deg).")
    st.stop()

# Convertir en numpy
f = np.asarray(df["freq_Hz"], dtype=float)
Z = np.asarray(df["Z_ohm"], dtype=float)
theta_deg = np.asarray(df["theta_deg"], dtype=float)
theta = np.deg2rad(theta_deg)

# ---------------------------------------
# Calculs principaux (avec protections)
# ---------------------------------------
# Partie réelle et imaginaire de Z
ReZ = Z * np.cos(theta)
ImZ = Z * np.sin(theta)

# Éviter divisions par zéro lors des calculs
eps = 1e-30

# Série
Rs = ReZ.copy()
# Cs = -1 / (2 * pi f ImZ)  -> protéger ImZ ~ 0
Cs = np.where(np.abs(ImZ) > eps, -1.0 / (2 * np.pi * f * ImZ), np.nan)

# Parallèle
den = ReZ**2 + ImZ**2
Gp = np.where(den > eps, ReZ / den, 0.0)   # conductance
Bp = np.where(den > eps, -ImZ / den, 0.0)  # susceptance
# Eviter division par zéro pour Rp
Rp = np.where(np.abs(Gp) > eps, 1.0 / Gp, np.nan)
Cp = np.where(np.abs(f) > eps, Bp / (2 * np.pi * f), np.nan)

# Quality factor Q (impédance simple)
Q_z = np.where(np.abs(ReZ) > eps, np.abs(ImZ) / np.abs(ReZ), np.nan)

# ---------------------------------------
# SRF et Ls
# ---------------------------------------
def compute_srf_and_ls(f_arr, ImZ_arr, ReZ_arr):
    """Retourne (f_srf or None, Ls_est or None, idx_min_absIm)"""
    idx_zero_cross = np.where(np.diff(np.sign(ImZ_arr)) != 0)[0]
    f_srf = None
    if idx_zero_cross.size > 0:
        i = idx_zero_cross[0]
        # interpolation linéaire pour la fréquence d'annulation de Im(Z)
        denom = (ImZ_arr[i+1] - ImZ_arr[i])
        if np.abs(denom) > eps:
            f_srf = f_arr[i] - ImZ_arr[i] * (f_arr[i+1] - f_arr[i]) / denom

    # Inductance série Ls : prendre le point où |ImZ| est minimal (proche de 0)
    idx_min = int(np.argmin(np.abs(ImZ_arr)))
    if np.abs(f_arr[idx_min]) > eps:
        Ls = np.abs(ImZ_arr[idx_min]) / (2 * np.pi * f_arr[idx_min])
    else:
        Ls = None

    return f_srf, Ls, idx_min

f_srf, Ls, idx_min_absIm = compute_srf_and_ls(f, ImZ, ReZ)

# ---------------------------------------
# Cp à 10 kHz (interpolation si dans la plage)
# ---------------------------------------
freq_target = 10_000.0
if (freq_target >= f.min()) and (freq_target <= f.max()):
    # interpolation en fréquence en ignorant NaN
    valid = np.isfinite(Cp)
    if valid.sum() >= 2:
        Cp_10k = float(np.interp(freq_target, f[valid], Cp[valid]))
    else:
        Cp_10k = np.nan
else:
    Cp_10k = None  # hors plage
    
# ----- Extraction ESR @ 10 kHz -----
target_freq = 10_000  # 10 kHz
idx_esr = (df["Freq (Hz)"] - target_freq).abs().idxmin()

ESR_10kHz = Rs[idx_esr]
freq_esr = df["Freq (Hz)"].iloc[idx_esr]

print(f"ESR @ 10 kHz : {ESR_10kHz:.4f} Ω (fréquence la plus proche : {freq_esr} Hz)")

# ---------------------------------------
# Préparer dictionnaire des données disponibles
# ---------------------------------------
data_options = {
    "Module Z": Z,
    "Phase θ (°)": theta_deg,
    "Ré(Z)": ReZ,
    "Im(Z)": ImZ,
    "Rs (série)": Rs,
    "Cs (série)": Cs,
    "Rp (parallèle)": Rp,
    "Cp (parallèle)": Cp,
    "Q (|Im|/Re)": Q_z
}

# ---------------------------------------
# Affichage rapide des résultats numériques
# ---------------------------------------
st.subheader("Valeurs calculées (aperçu)")

col1, col2, col3 = st.columns(3)
with col1:
    if Cp_10k is None:
        st.write("Cp @ 10 kHz : hors plage de mesure")
    elif np.isnan(Cp_10k):
        st.write("Cp @ 10 kHz : impossible à calculer (données manquantes)")
    else:
        st.metric("Cp @ 10 kHz", f"{Cp_10k:.3e} F")

with col2:
    if Ls is None:
        st.write("Ls : impossible à calculer")
    else:
        st.metric("Inductance série Ls", f"{Ls:.3e} H (à f={f[idx_min_absIm]:.3e} Hz)")

with col3:
    if f_srf is None:
        st.write("SRF : non détectée")
    else:
        st.metric("SRF (f_srf)", f"{f_srf:.3e} Hz")

# ---------------------------------------
# Sélection plage de fréquence (globale)
# ---------------------------------------
st.subheader("Sélection de la plage de fréquence pour affichage")
f_min, f_max = st.slider(
    "Plage (Hz)",
    min_value=float(f.min()),
    max_value=float(f.max()),
    value=(float(f.min()), float(f.max()))
)
mask = (f >= f_min) & (f <= f_max)

# ---------------------------------------
# Utilitaires pour tracer et préparer exports
# (Figures & DataFrames seront utilisés dans la PARTIE 2/3)
# ---------------------------------------
def make_dataframe_for_export(freq_array, selected_labels):
    """Renvoie un DataFrame avec freq + colonnes sélectionnées (appliqué au mask courant)."""
    df_out = pd.DataFrame({"freq_Hz": freq_array})
    for label in selected_labels:
        if label in data_options:
            df_out[label] = data_options[label][mask]
    return df_out

def create_matplotlib_figure(x, y_series: dict, xscale="log", yscale="lin", title=""):
    """Crée et renvoie une figure matplotlib (sans l'afficher). y_series est dict(label->array)."""
    fig, ax = plt.subplots()
    for label, y in y_series.items():
        ax.plot(x, y, label=label)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Valeur")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    return fig

# Indication prête pour PARTIE 2/3
st.info("Les fonctions et données sont prêtes. Passe à la PARTIE 2 pour les onglets de tracé (Graphique1/2, Nyquist, Q) et aux options d'export.")
# ------------------------------------------------------------
# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.title("Analyse d’impédance – Visualisation & Export")

st.sidebar.header("Paramètres d’affichage")

# ---- Sélection des deux courbes ----
available_curves = {
    "Z (Ω)": Z,
    "Re(Z) (Ω)": ReZ,
    "Im(Z) (Ω)": ImZ,
    "Phase (°)": theta_deg,
    "Cp (F)": Cp,
    "Cs (F)": Cs,
    "Rp (Ω)": Rp,
    "Rs (Ω)": Rs,
    "Q = |Im(Z)| / Re(Z)": np.abs(ImZ) / ReZ
}

curve1 = st.sidebar.selectbox("Courbe 1 (axe Y gauche)", list(available_curves.keys()))
curve2 = st.sidebar.selectbox("Courbe 2 (axe Y droite)", list(available_curves.keys()), index=1)

# ---- Fenêtre de fréquence (unique) ----
fmin, fmax = st.sidebar.slider(
    "Plage de fréquence (Hz)", 
    float(f.min()), float(f.max()),
    (float(f.min()), float(f.max()))
)

# ---- Log axes ----
log_x = st.sidebar.checkbox("Axe X logarithmique", True)
log_y1 = st.sidebar.checkbox("Axe Y1 logarithmique", False)
log_y2 = st.sidebar.checkbox("Axe Y2 logarithmique", False)

# ---- Filtrage des données ----
mask = (f >= fmin) & (f <= fmax)
f_plot = f[mask]
y1 = available_curves[curve1][mask]
y2 = available_curves[curve2][mask]

# ============================================================
#                   ONGLET : GRAPHIQUES (BODE)
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Bode", "🔵 Nyquist", "✨ Facteur Q", "📘 Valeurs clés"])

with tab1:
    st.header("Diagramme personnalisé – 2 axes Y")

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Courbe 1 = Axe Y gauche
    if log_x and log_y1:
        ax1.loglog(f_plot, y1, label=curve1, color="tab:blue")
    elif log_x:
        ax1.semilogx(f_plot, y1, label=curve1, color="tab:blue")
    elif log_y1:
        ax1.semilogy(f_plot, y1, label=curve1, color="tab:blue")
    else:
        ax1.plot(f_plot, y1, label=curve1, color="tab:blue")

    ax1.set_xlabel("Fréquence (Hz)")
    ax1.set_ylabel(curve1, color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True)

    # Courbe 2 = Axe Y droite
    ax2 = ax1.twinx()

    if log_x and log_y2:
        ax2.loglog(f_plot, y2, label=curve2, color="tab:red")
    elif log_x:
        ax2.semilogx(f_plot, y2, label=curve2, color="tab:red")
    elif log_y2:
        ax2.semilogy(f_plot, y2, label=curve2, color="tab:red")
    else:
        ax2.plot(f_plot, y2, label=curve2, color="tab:red")

    ax2.set_ylabel(curve2, color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    st.pyplot(fig)

# ============================================================
#                   ONGLET : NYQUIST
# ============================================================
with tab2:
    st.header("Nyquist – Im(Z) vs Re(Z)")

    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ReZ, -ImZ)
    ax.set_xlabel("Re(Z) (Ω)")
    ax.set_ylabel("-Im(Z) (Ω)")
    ax.grid(True)
    st.pyplot(fig2)

# ============================================================
#                   ONGLET : FACTEUR Q (avec filtrage & log)
# ============================================================
with tab3:
    st.header("Facteur de qualité Q")

    Q = np.abs(ImZ) / ReZ
    Q_plot = Q[mask]

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    if log_x and log_y1:
        ax3.loglog(f_plot, Q_plot)
    elif log_x:
        ax3.semilogx(f_plot, Q_plot)
    elif log_y1:
        ax3.semilogy(f_plot, Q_plot)
    else:
        ax3.plot(f_plot, Q_plot)

    ax3.set_xlabel("Fréquence (Hz)")
    ax3.set_ylabel("Facteur Q")
    ax3.grid(True)
    st.pyplot(fig3)

# ============================================================
#                   ONGLET : VALEURS CLÉS
# ============================================================
with tab4:
    st.header("Résumé des grandeurs extraites")

    idx_10k = np.argmin(np.abs(f - 10e3))
    Cp_10k = Cp[idx_10k]

    st.write(f"**Cp @ 10 kHz :** {Cp_10k:.3e} F")
    if f_srf is not None:
        st.write(f"**Fréquence de résonance (SRF) :** {f_srf:.3e} Hz")
    else:
        st.write("**SRF :** non détectée")

    if Ls is not None:
        st.write(f"**Inductance série Ls :** {Ls:.3e} H")
    else:
        st.write("**Ls :** non disponible")

# ------------------------------------------------------------
#                 EXPORT PDF
# ------------------------------------------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
import tempfile
import datetime


st.subheader("📄 Exporter le rapport PDF")

if st.button("Générer le rapport PDF"):

    # ---------------------------------------------
    # Sauvegarde TEMPORAIRE des 3 graphiques
    # ---------------------------------------------
    temp_dir = tempfile.mkdtemp()

    # Graphique Bode (2 courbes)
    bode_path = f"{temp_dir}/bode.png"
    fig.savefig(bode_path, dpi=200, bbox_inches="tight")

    # Graphique Nyquist
    nyquist_path = f"{temp_dir}/nyquist.png"
    fig2.savefig(nyquist_path, dpi=200, bbox_inches="tight")

    # Graphique Q
    q_path = f"{temp_dir}/Q.png"
    fig3.savefig(q_path, dpi=200, bbox_inches="tight")

    # ---------------------------------------------
    # Création du PDF
    # ---------------------------------------------
    pdf_path = f"{temp_dir}/rapport_impedance.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Titre
    flow.append(Paragraph(
        "<b>Analyse d'impédance – Rapport automatique</b>",
        styles["Title"]
    ))
    flow.append(Paragraph(
        f"Généré le : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["Normal"]
    ))
    flow.append(Spacer(1, 0.5*cm))

    # Résultats chiffrés
    flow.append(Paragraph("<b>Résumé des grandeurs</b>", styles["Heading2"]))

    if f_srf is not None:
        flow.append(Paragraph(f"Fréquence de résonance (SRF) : {f_srf:.3e} Hz", styles["Normal"]))
    else:
        flow.append(Paragraph("SRF : non détectée", styles["Normal"]))

    if Ls is not None:
        flow.append(Paragraph(f"Inductance série Ls : {Ls:.3e} H", styles["Normal"]))
    else:
        flow.append(Paragraph("Ls non disponible", styles["Normal"]))

    flow.append(Paragraph(f"Cp à 10 kHz : {Cp_10k:.3e} F", styles["Normal"]))
    flow.append(Spacer(1, 0.5*cm))

    # Ajout des images
    flow.append(Paragraph("<b>Graphiques</b>", styles["Heading2"]))
    flow.append(Spacer(1, 0.2*cm))

    flow.append(Paragraph("Diagramme Bode (2 courbes choisies)", styles["Heading3"]))
    flow.append(Image(bode_path, width=14*cm, height=9*cm))
    flow.append(Spacer(1, 0.5*cm))

    flow.append(Paragraph("Diagramme Nyquist", styles["Heading3"]))
    flow.append(Image(nyquist_path, width=14*cm, height=9*cm))
    flow.append(Spacer(1, 0.5*cm))

    flow.append(Paragraph("Facteur de qualité Q", styles["Heading3"]))
    flow.append(Image(q_path, width=14*cm, height=9*cm))

    doc.build(flow)

    # ---------------------------------------------
    # Téléchargement Streamlit
    # ---------------------------------------------
    with open(pdf_path, "rb") as fpdf:
        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=fpdf,
            file_name="rapport_impedance.pdf",
            mime="application/pdf"
        )

    st.success("PDF généré avec succès !")
