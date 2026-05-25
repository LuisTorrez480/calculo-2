import math
import time
import requests
import streamlit as st
import folium
import numpy as np
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from streamlit_folium import folium_static

API_KEY = "AIzaSyDk_Nm-NAp593GW_EORMNtleCyBgIG-3to"
genai.configure(api_key=API_KEY)


def obtener_modelo() -> genai.GenerativeModel | None:
    try:
        modelos = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        if modelos:
            seleccionado = next((m for m in modelos if "flash" in m), modelos[0])
            return genai.GenerativeModel(seleccionado)
        return None
    except Exception:
        return None


IA_MODEL = obtener_modelo()


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO 2 — GRADIENTE Y DESCENSO POR GRADIENTE
# ══════════════════════════════════════════════════════════════════════════════

def funcion_distancia(x: float, y: float, x2: float, y2: float) -> float:
    return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)


def gradiente(x: float, y: float, x2: float, y2: float) -> tuple[float, float]:
    f = funcion_distancia(x, y, x2, y2)
    if f == 0:
        return (0.0, 0.0)
    df_dx = (x - x2) / f
    df_dy = (y - y2) / f
    return (df_dx, df_dy)


def magnitud_gradiente(gx: float, gy: float) -> float:
    return math.sqrt(gx**2 + gy**2)


def descenso_gradiente(
    x0: float, y0: float,
    x_dest: float, y_dest: float,
    tasa: float = 0.1,
    iteraciones: int = 50,
    tolerancia: float = 1e-4,
) -> list[tuple[float, float]]:
    camino: list[tuple[float, float]] = [(x0, y0)]
    x, y = x0, y0
    for _ in range(iteraciones):
        f_val = funcion_distancia(x, y, x_dest, y_dest)
        if f_val < tolerancia:
            break
        gx, gy = gradiente(x, y, x_dest, y_dest)
        x = x - tasa * gx
        y = y - tasa * gy
        camino.append((x, y))
    camino.append((x_dest, y_dest))
    return camino


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def obtener_ruta_osrm(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[list[tuple[float, float]], float] | tuple[None, None]:
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}"
        f"?overview=full&geometries=geojson"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            return None, None
        ruta = data["routes"][0]
        distancia_km = ruta["distance"] / 1000.0
        coords = ruta["geometry"]["coordinates"]
        waypoints = [(c[1], c[0]) for c in coords]
        return waypoints, distancia_km
    except requests.RequestException:
        return None, None


# ── Mapa GPS — paleta negro / rojo / verde / blanco ──────────────────────────
def construir_mapa(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    waypoints_osrm: list[tuple[float, float]] | None,
    camino_gradiente: list[tuple[float, float]],
    origen_txt: str,
    destino_txt: str,
    dist_calles: float | None,
    dist_geodesica: float,
) -> folium.Map:
    centro = [(lat1 + lat2) / 2, (lon1 + lon2) / 2]

    mapa = folium.Map(location=centro, zoom_start=13, tiles=None)

    # Tile oscuro CartoDB Dark Matter
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="Dark GPS",
        max_zoom=20,
    ).add_to(mapa)

    # ── Ruta por calles — ROJA con borde negro ────────────────────────────
    if waypoints_osrm and len(waypoints_osrm) > 1:
        folium.PolyLine(          # borde / sombra negra
            waypoints_osrm, color="#000000", weight=10, opacity=0.7,
        ).add_to(mapa)
        folium.PolyLine(          # línea roja principal
            waypoints_osrm,
            color="#E8000D", weight=5, opacity=1.0,
            tooltip=f"🛣️ Ruta por calles — {dist_calles:.2f} km" if dist_calles else "🛣️ Ruta por calles",
        ).add_to(mapa)
    else:
        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color="#E8000D", weight=3, dash_array="10",
            tooltip="⚠️ Línea recta (OSRM no disponible)",
        ).add_to(mapa)

    # ── Trayectoria descenso ∇f — VERDE con borde negro ───────────────────
    if len(camino_gradiente) > 1:
        folium.PolyLine(          # borde negro
            camino_gradiente, color="#000000", weight=6, opacity=0.6,
        ).add_to(mapa)
        folium.PolyLine(          # línea verde
            camino_gradiente,
            color="#00C853", weight=3, dash_array="7 5", opacity=0.95,
            tooltip="📉 Descenso por gradiente ∇f",
        ).add_to(mapa)

    # ── Marcador ORIGEN — pin rojo con punto blanco ───────────────────────
    icono_origen = folium.DivIcon(
        html="""
        <div style="
            position:relative;
            width:20px; height:20px;
            background:#E8000D;
            border:2.5px solid #ffffff;
            border-radius:50% 50% 50% 0;
            transform:rotate(-45deg);
            box-shadow:0 0 0 2px #000, 0 0 10px rgba(232,0,13,0.6);
        ">
            <div style="
                position:absolute; top:50%; left:50%;
                transform:translate(-50%,-50%) rotate(45deg);
                width:6px; height:6px;
                background:#ffffff; border-radius:50%;
            "></div>
        </div>""",
        icon_size=(26, 26), icon_anchor=(13, 26),
    )
    folium.Marker([lat1, lon1], tooltip=f"📍 Origen: {origen_txt}", icon=icono_origen).add_to(mapa)

    # ── Marcador DESTINO — círculo verde pulsante ─────────────────────────
    icono_destino = folium.DivIcon(
        html="""
        <style>
        @keyframes gps-pulse {
            0%   { transform:scale(1);   opacity:.9; }
            60%  { transform:scale(2);   opacity:.1; }
            100% { transform:scale(1);   opacity:.9; }
        }
        </style>
        <div style="position:relative; width:34px; height:34px;">
            <div style="
                position:absolute; inset:0;
                background:rgba(0,200,83,0.22);
                border-radius:50%;
                animation:gps-pulse 1.8s ease-out infinite;
            "></div>
            <div style="
                position:absolute; top:7px; left:7px;
                width:20px; height:20px;
                background:#00C853;
                border:2px solid #ffffff;
                border-radius:50%;
                box-shadow:0 0 0 2px #000, 0 0 10px rgba(0,200,83,0.7);
                display:flex; align-items:center; justify-content:center;
                font-size:10px; line-height:1;
            ">🏁</div>
        </div>""",
        icon_size=(34, 34), icon_anchor=(17, 17),
    )
    folium.Marker([lat2, lon2], tooltip=f"🏁 Destino: {destino_txt}", icon=icono_destino).add_to(mapa)

    # ── HUD / Leyenda — fondo negro, texto blanco, acentos rojo y verde ───
    dist_label = f"{dist_calles:.1f} km" if dist_calles else f"{dist_geodesica:.1f} km"
    leyenda = f"""
    <div style="
        position:fixed; bottom:28px; left:28px; z-index:1000;
        background:rgba(8,8,8,0.92);
        border:1px solid #333;
        border-left:4px solid #E8000D;
        padding:13px 18px; border-radius:8px;
        box-shadow:0 4px 20px rgba(0,0,0,0.7);
        font-family:'Segoe UI',Arial,sans-serif; font-size:13px; color:#f0f0f0;
        min-width:190px;
    ">
        <div style="font-weight:700; color:#ffffff; margin-bottom:8px;
                    letter-spacing:1.5px; font-size:12px; text-transform:uppercase;">
            📡 GPS Rutas
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin:5px 0;">
            <div style="width:28px; height:4px; background:#E8000D;
                        border-radius:2px; box-shadow:0 0 6px #E8000D;"></div>
            <span style="color:#cccccc; font-size:12px;">Ruta por calles</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin:5px 0;">
            <div style="width:28px; height:0; border-top:3px dashed #00C853;
                        opacity:.9;"></div>
            <span style="color:#cccccc; font-size:12px;">Descenso ∇f</span>
        </div>
        <div style="margin-top:10px; padding-top:8px; border-top:1px solid #222;
                    color:#888; font-size:11px;">
            Distancia:
            <b style="color:#E8000D; font-size:13px;">{dist_label}</b>
        </div>
    </div>"""
    mapa.get_root().html.add_child(folium.Element(leyenda))

    try:
        from folium.plugins import MeasureControl
        MeasureControl(position="topright", primary_length_unit="kilometers").add_to(mapa)
    except Exception:
        pass

    return mapa


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Negro dominante · Rojo · Verde · Blanco  (sin fondo rojo)
# ══════════════════════════════════════════════════════════════════════════════
GPS_CSS = """
<style>
/* ── Paleta de variables ── */
:root {
    --bg-main:    #0a0a0a;
    --bg-card:    #111111;
    --bg-sidebar: #0d0d0d;
    --border:     #1e1e1e;
    --red:        #E8000D;
    --red-dim:    rgba(232,0,13,0.18);
    --green:      #00C853;
    --green-dim:  rgba(0,200,83,0.14);
    --white:      #f5f5f5;
    --muted:      #888888;
}

/* ── Fondo principal — negro puro ── */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
}
[data-testid="stHeader"]      { background: transparent !important; }
[data-testid="stToolbar"]     { background: transparent !important; }
.main .block-container        { padding-top: 2rem; }

/* ── Tipografía ── */
h1 {
    color: var(--white) !important;
    border-bottom: 2px solid var(--red);
    padding-bottom: 8px;
}
h2, h3 { color: var(--white) !important; }
p, li, span, div { color: var(--white); }
.stMarkdown p { color: #cccccc !important; }

/* ── Sidebar — negro muy oscuro con borde rojo fino ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--red) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #dddddd !important; }

/* Sliders — riel negro, thumb rojo ── */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: var(--red) !important;
    border-color: var(--red) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[class*="Track"] {
    background: #2a2a2a !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid*="stSliderThumb"] {
    background: var(--red) !important;
}

/* ── Botón CALCULAR ── */
button[kind="primary"], button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button {
    background: var(--red) !important;
    color: #ffffff !important;
    border: 1px solid #aa0008 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    border-radius: 6px !important;
    transition: all .2s;
}
button:hover {
    background: #c0000b !important;
    box-shadow: 0 0 14px rgba(232,0,13,0.45) !important;
}

/* ── Inputs de texto ── */
[data-testid="stTextInput"] input {
    background: #161616 !important;
    border: 1px solid #333 !important;
    color: var(--white) !important;
    border-radius: 6px;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 2px rgba(232,0,13,0.25) !important;
}

/* ── Tarjetas de métricas ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid #222 !important;
    border-top: 3px solid var(--red) !important;
    border-radius: 8px;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .8px; }
[data-testid="stMetricValue"] { color: var(--white) !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { color: var(--green) !important; }

/* ── Expander — borde verde sutil ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid #1e1e1e !important;
    border-left: 3px solid var(--green) !important;
    border-radius: 8px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: var(--white) !important; font-weight: 600; }

/* ── Alerts — success verde, warning rojo ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--bg-card) !important;
    border-radius: 6px;
}
div[data-testid="stAlert"] > div[role="alert"][class*="success"] {
    border-left: 4px solid var(--green) !important;
    background: var(--green-dim) !important;
}
div[data-testid="stAlert"] > div[role="alert"][class*="warning"],
div[data-testid="stAlert"] > div[role="alert"][class*="info"] {
    border-left: 4px solid var(--red) !important;
    background: var(--red-dim) !important;
}

/* Fallback genérico para alertas */
[data-testid="stAlertContainer"] { border-radius: 6px; }

/* ── Separadores ── */
hr { border-color: #1e1e1e !important; margin: 1.5rem 0; }

/* ── Tablas (markdown) ── */
table { border-collapse: collapse; width: 100%; }
th {
    background: #161616 !important;
    color: var(--red) !important;
    border: 1px solid #2a2a2a;
    padding: 8px 12px;
    font-size: 12px; text-transform: uppercase; letter-spacing: .7px;
}
td {
    background: var(--bg-card) !important;
    color: #cccccc !important;
    border: 1px solid #1e1e1e;
    padding: 8px 12px;
}
tr:hover td { background: #181818 !important; }

/* ── Sidebar header ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--white) !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GPS Rutas — Cálculo 2",
    layout="wide",
    page_icon="🛰️",
)

# Inyectar CSS
st.markdown(GPS_CSS, unsafe_allow_html=True)

st.title("🛰️ GPS Optimizador de Rutas — Cálculo 2")
st.markdown(
    """
    Aplica el concepto de **gradiente** (∇f) y **descenso por gradiente** de Cálculo 2
    para encontrar la trayectoria óptima que **minimiza** la función de distancia
    $f(x,y) = \\sqrt{(x-x_2)^2 + (y-y_2)^2}$.
    """
)

with st.sidebar:
    st.header("⚙️ Parámetros GPS")
    origen_txt = st.text_input("📍 Origen:", "Plaza Murillo, La Paz, Bolivia")
    destino_txt = st.text_input("🏁 Destino:", "Sopocachi, La Paz, Bolivia")
    eficiencia = st.slider("⛽ Eficiencia del vehículo (km/gal):", 10, 60, 40)
    tasa_aprendizaje = st.slider("α — Tasa de descenso:", 0.05, 0.5, 0.15, step=0.05)
    iteraciones = st.slider("Iteraciones de descenso:", 10, 100, 40)
    st.caption("💡 Escribe ciudad y país para mejores resultados")
    btn_calcular = st.button("🛰️ CALCULAR RUTA", use_container_width=True)


def geocodificar(geolocator: Nominatim, direccion: str, intentos: int = 3):
    """Geocodifica con reintentos y delay entre llamadas."""
    for intento in range(intentos):
        try:
            time.sleep(1)   # Nominatim exige mínimo 1 seg entre peticiones
            resultado = geolocator.geocode(
                direccion,
                timeout=12,
                language="es",
            )
            return resultado
        except GeocoderTimedOut:
            if intento == intentos - 1:
                raise
            time.sleep(2)
        except GeocoderServiceError as e:
            raise e
    return None

if btn_calcular:
    if not origen_txt.strip() or not destino_txt.strip():
        st.error("❌ Por favor ingresa ambas direcciones.")
        st.stop()

    geolocator = Nominatim(user_agent=f"calculo2_gps_{int(time.time())}")

    with st.spinner("🔍 Buscando origen..."):
        try:
            loc1 = geocodificar(geolocator, origen_txt)
        except Exception as e:
            st.error(f"❌ Error buscando el origen: {e}")
            st.stop()

    if not loc1:
        st.error(
            f"❌ No se encontró el origen: **{origen_txt}**  \n"
            "💡 Intenta ser más específico, por ejemplo: *Av. Arce, La Paz, Bolivia*"
        )
        st.stop()

    with st.spinner("🔍 Buscando destino..."):
        try:
            loc2 = geocodificar(geolocator, destino_txt)
        except Exception as e:
            st.error(f"❌ Error buscando el destino: {e}")
            st.stop()

    if not loc2:
        st.error(
            f"❌ No se encontró el destino: **{destino_txt}**  \n"
            "💡 Intenta ser más específico, por ejemplo: *El Alto, La Paz, Bolivia*"
        )
        st.stop()

    lat1, lon1 = loc1.latitude, loc1.longitude
    lat2, lon2 = loc2.latitude, loc2.longitude

    gx, gy = gradiente(lat1, lon1, lat2, lon2)
    norma_grad = magnitud_gradiente(gx, gy)
    f_origen = funcion_distancia(lat1, lon1, lat2, lon2)

    camino_grad = descenso_gradiente(
        lat1, lon1, lat2, lon2,
        tasa=tasa_aprendizaje,
        iteraciones=iteraciones,
    )

    dist_geodesica = haversine_km(lat1, lon1, lat2, lon2)

    with st.spinner("🛣️ Calculando ruta por calles..."):
        waypoints, dist_calles = obtener_ruta_osrm(lat1, lon1, lat2, lon2)

    dist_final = dist_calles if dist_calles else dist_geodesica
    consumo = dist_final / eficiencia
    ahorro = consumo * 0.20

    st.markdown("---")
    st.subheader("📊 Resultados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("∇f en el origen", f"⟨{gx:.4f}, {gy:.4f}⟩")
    c2.metric("‖∇f‖", f"{norma_grad:.4f}")
    c3.metric("Distancia por calles" if dist_calles else "Dist. geodésica", f"{dist_final:.2f} km")
    c4.metric("Ahorro estimado", f"{ahorro:.3f} gal")

    st.markdown("---")
    with st.expander("📘 Desarrollo matemático — Cálculo 2", expanded=True):

        st.markdown("### 1. Función a minimizar")
        st.latex(r"f(x,y) = \sqrt{(x - x_2)^2 + (y - y_2)^2}")
        st.markdown(
            f"**Valor en el origen:** $f({lat1:.4f},\\ {lon1:.4f}) = {f_origen:.6f}°$"
        )

        st.markdown("### 2. Gradiente analítico")
        st.latex(
            r"\nabla f(x,y) = \left(\frac{\partial f}{\partial x},\ "
            r"\frac{\partial f}{\partial y}\right) = "
            r"\left(\frac{x - x_2}{f},\ \frac{y - y_2}{f}\right)"
        )
        st.markdown(
            f"**Evaluado en el origen:**  \n"
            f"$\\nabla f = \\langle {gx:.6f},\\ {gy:.6f} \\rangle$  \n"
            f"$\\|\\nabla f\\| = {norma_grad:.6f}$  \n\n"
            "*Para la función distancia euclidiana, ‖∇f‖ = 1 siempre.*"
        )

        st.markdown("### 3. Descenso por gradiente")
        st.latex(
            r"\mathbf{p}_{n+1} = \mathbf{p}_n - \alpha \cdot \nabla f(\mathbf{p}_n)"
        )
        st.markdown(
            f"Con **α = {tasa_aprendizaje}** y **{len(camino_grad) - 1} iteraciones**, "
            f"el algoritmo converge al destino en la dirección de **máximo descenso** de $f$."
        )

        st.markdown("### 4. Interpretación geométrica")
        st.markdown(
            "- ∇f apunta *alejándose* del destino (máximo crecimiento).  \n"
            "- −∇f apunta *hacia* el destino (mínimo de f).  \n"
            "- Cada paso reduce f, acercando la posición al mínimo global."
        )

        st.markdown("### 5. Distancia real vs. lineal")
        if dist_calles:
            st.markdown(
                f"| Método | Distancia |  \n"
                f"|--------|-----------|  \n"
                f"| Línea recta (Haversine) | {dist_geodesica:.4f} km |  \n"
                f"| Ruta por calles (OSRM)  | {dist_calles:.4f} km |"
            )
        else:
            st.markdown(f"| Haversine (sin OSRM) | {dist_geodesica:.4f} km |")

    st.markdown("---")
    st.subheader("🤖 Análisis Gemini")
    if IA_MODEL:
        try:
            prompt = (
                f"Eres un profesor de Cálculo 2. Explica en 4 oraciones, "
                f"de forma didáctica, cómo se aplicó el gradiente para optimizar "
                f"la ruta de '{origen_txt}' a '{destino_txt}'. "
                f"El gradiente calculado fue ∇f = ⟨{gx:.5f}, {gy:.5f}⟩ con norma {norma_grad:.4f}. "
                f"La distancia es {dist_final:.2f} km y el consumo {consumo:.3f} gal "
                f"a {eficiencia} km/gal. Menciona brevemente el descenso por gradiente."
            )
            respuesta = IA_MODEL.generate_content(prompt)
            st.info(respuesta.text)
        except Exception:
            st.warning("IA no disponible. El gradiente negativo indica la dirección óptima de mínimo costo.")
    else:
        st.warning("Modelo de IA no configurado.")

    st.markdown("---")
    st.subheader("🗺️ Mapa GPS")
    if waypoints:
        st.success(f"✅ Ruta real obtenida: {len(waypoints)} puntos de trayectoria.")
    else:
        st.warning("⚠️ OSRM no respondió. Se muestra línea recta.")

    mapa = construir_mapa(
        lat1, lon1, lat2, lon2,
        waypoints, camino_grad,
        origen_txt, destino_txt,
        dist_calles, dist_geodesica,
    )
    folium_static(mapa, width=950, height=540)

else:
    st.info("📡 Configura los parámetros en la barra lateral y presiona **CALCULAR RUTA**.")