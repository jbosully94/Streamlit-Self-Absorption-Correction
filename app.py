import streamlit as st
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation
import xraylib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io

st.set_page_config(page_title="SAC Correction", layout="wide")
st.title("Self-Absorption Correction")

Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'K': 19, 'Ca': 20, 'Mn': 25, 'Fe': 26, 'Cu': 29, 'Zn': 30}

def show_map(arr, title, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig

uploaded = st.file_uploader("Upload XRF map (TIFF)", type=["tif", "tiff"])

if uploaded:
    data = tifffile.imread(uploaded).astype(float)
    if data.ndim == 3:
        data = data[0]
    data = np.nan_to_num(data)

    st.subheader("Mask")

    thresh = threshold_otsu(data)
    mask = (data > thresh).astype(bool)

    col1, col2 = st.columns(2)
    with col1:
        erosion_iter = st.slider("Erosion", 0, 10, 1)
    with col2:
        dilation_iter = st.slider("Dilation", 0, 10, 1)

    if erosion_iter > 0:
        mask = binary_erosion(mask, iterations=erosion_iter)
    if dilation_iter > 0:
        mask = binary_dilation(mask, iterations=dilation_iter)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(show_map(data, "original"))
    with c2:
        st.pyplot(show_map(mask.astype(float), "mask", cmap='gray'))

    st.subheader("Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        element = st.selectbox("Element", ['Fe', 'Zn', 'Ca', 'Mn', 'Cu', 'K'])
        pix = st.number_input("Pixel size (um)", value=5.0)
    with col2:
        exc_energy = st.number_input("Excitation energy (keV)", value=10.0)
        density = st.number_input("Density (g/cm3)", value=1.2)
    with col3:
        det_angle = st.number_input("Detector angle from vertical (deg)", value=25.0)
        exc_angle = st.number_input("Excitation angle from vertical (deg)", value=55.0)

    with st.expander("Matrix composition (mass fractions)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_C = st.number_input("C", value=0.45)
            f_O = st.number_input("O", value=0.45)
        with c2:
            f_H = st.number_input("H", value=0.06)
            f_N = st.number_input("N", value=0.03)
        with c3:
            f_P = st.number_input("P", value=0.005)
            f_K = st.number_input("K (matrix)", value=0.005)

    comp = {'C': f_C, 'O': f_O, 'H': f_H, 'N': f_N, 'P': f_P, 'K': f_K}

    if st.button("Run correction"):
        with st.spinner("tracing ray paths..."):

            xrf_energy = xraylib.LineEnergy(Z[element], xraylib.KA1_LINE)
            mu_exc = sum(f * xraylib.CS_Total(Z[e], exc_energy) for e, f in comp.items())
            mu_xrf = sum(f * xraylib.CS_Total(Z[e], xrf_energy) for e, f in comp.items())

            a_det = np.radians(90 - det_angle)
            a_exc = np.radians(90 - exc_angle)
            dx_det, dy_det = -np.cos(a_det), -np.sin(a_det)
            dx_exc, dy_exc = -np.cos(a_exc), np.sin(a_exc)

            rows, cols = mask.shape
            det_paths = np.zeros_like(mask, dtype=float)
            exc_paths = np.zeros_like(mask, dtype=float)
            step = 0.1

            for i in range(rows):
                for j in range(cols):
                    if not mask[i, j]:
                        continue

                    target_r, target_c = i + 0.5, j + 0.5

                    steps_right = (cols - 1 - target_c) / abs(dx_exc)
                    row_at_right = target_r - dy_exc * steps_right

                    if row_at_right >= 0:
                        start_r, start_c = row_at_right, cols - 1
                    else:
                        steps_top = target_r / dy_exc
                        start_r, start_c = 0, target_c - dx_exc * steps_top

                    r, c = start_r, start_c
                    path = 0.0
                    while abs(r - target_r) > step or abs(c - target_c) > step:
                        if 0 <= int(r) < rows and 0 <= int(c) < cols:
                            if mask[int(r), int(c)]:
                                path += step * pix
                        r += dy_exc * step
                        c += dx_exc * step
                        if r > rows or c < 0:
                            break
                    exc_paths[i, j] = path

                    r, c = i + 0.5, j + 0.5
                    path = 0.0
                    while 0 <= r < rows and 0 <= c < cols:
                        if mask[int(r), int(c)]:
                            path += step * pix
                        r += dy_det * step
                        c += dx_det * step
                    det_paths[i, j] = path

            exc_cm = exc_paths * 1e-4
            det_cm = det_paths * 1e-4
            corr = np.exp(density * (mu_exc * exc_cm + mu_xrf * det_cm))

            corrected = data.copy()
            corrected[mask] *= corr[mask]

        st.subheader("Results")
        st.write(f"max correction factor: **{corr[mask].max():.3f}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(show_map(data, "original"))
        with c2:
            st.pyplot(show_map(corrected, "corrected"))
        with c3:
            corr_display = np.ones_like(corr)
            corr_display[mask] = corr[mask]
            st.pyplot(show_map(corr_display, "correction factor", cmap='plasma'))

        buf = io.BytesIO()
        tifffile.imwrite(buf, corrected.astype(np.float32))
        st.download_button(
            label="Download corrected TIFF",
            data=buf.getvalue(),
            file_name=f"corrected_{element}.tif",
            mime="image/tiff"
        )
