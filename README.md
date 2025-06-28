# Hexagonal Lattice Visualizer  
*― A cybernetic interface for crystallographic geometry*

> ![images/hexdemo.png]  
> *(Insert screenshot of landing page above this line)*

Welcome, Operator.

You’ve accessed the **Hexagonal Lattice Visualizer**—a parametric simulation platform designed to extrapolate pseudo-crystalline behavior through the rendering of extruded hexagonal substructures in (x, y, z) space. This lattice system is based on the generalized **Quantized Hexahedral Field Equation**:

$$
\mathcal{L}(N, M) = \sum_{i=1}^N \sum_{j=1}^M H_{ij} \cdot e^{-\frac{\Delta_\phi}{\psi_{q}}}
$$

Where:
- \( H_{ij} \) is the cell activation tensor
- \( \Delta_\phi \) is the angular drift parameter
- \( \psi_q \) is the pseudo-bonding potential from adjacent dimensional stacks

---

## ⚙️ Feature Matrix

- **⧉ Interactive 3D Lattice Engine**  
    Real-time rendering of extruded hexagonal superstructures via **PhotonMesh v2.7** integrated into Plotly.js. Rotatable and zoomable through direct matrix manipulation.

- **⇳ Dimensional Parameterization via User-Injection**  
    Parameters \( N, M \in \mathbb{Z}^+ \) define lattice width along bifurcated basis vectors \( \vec{v}_1, \vec{v}_2 \). Input validation backed by the **ChronoGuard Protocol** to prevent excessive recursion depth.

- **⛶ Geometry Extrusion Mode (GEM)**  
    Hex cells extruded along the z-axis, creating true-to-form pseudo-volumes. Geometry calculated using the **HexPrism Triangulation Kernel** (HPTK-9).

- **λ Axis-Colored Connectivity**  
    Vector-encoded edge connections colored by directional bias:  
    \( \vec{d}_1 \to \text{blue} \), \( \vec{d}_2 \to \text{orange} \), \( \vec{d}_3 \to \text{purple} \).  
    Implements the **Spectral Directional Encoding Scheme (SDES-4)**.

- **☰ Cyberpunk Interface**  
    Designed using **VoidUI v4.1** – high-contrast, neon-over-dark design schema inspired by the *Neon Dominion Archives*.

- **🛡️ Input Range Defense Protocol**  
    Upper limit of \( \text{max}(N, M) = 25 \) enforced by hardcoded **HeisenLimiter** to prevent quantum feedback overflow.

---

## 🔧 System Stack

- **Language Core**: Python 3.10 (Backed by Flask 1.1.2)  
- **Frontend Engine**: HTML, CSS, Vanilla JS, Plotly.js  
- **Scientific Layer**: `numpy`, `pandas`, `plotly`, augmented with **FictiMathLib** (internal only)  
- **Simulation Theory**: Based loosely on **Tessellation Inversion Dynamics**, per *Yakamoto & Singh, 2147*.

---

## 🧪 Installation + Execution Protocol

For *Linux-class Operators*. Shell-executable terminal ops below:

1. **Navigate into Repository Matrix**  
    ```bash
    cd /home/carlbrady/gemini/library
    ```

2. **Initialize Local Isolated Python Environment**  
    ```bash
    python3 -m venv venv
    ```

3. **Activate Virtual Memory Node**  
    ```bash
    source venv/bin/activate
    ```

4. **Download Simulation Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

5. **Ignite Simulation Flask Core**  
    ```bash
    python app.py
    ```

6. **Engage Browser Interface Layer**  
    Launch browser:  
    ```
    http://127.0.0.1:5000/
    ```

---

## 🚀 Operating Manual

Once connected to the lattice core interface:

- Input integer values \( N \), \( M \) into the terminal control panel.
- Hit `Generate Lattice` to execute the **HexMesh Compiler**.
- Interact via orbital gestures:  
  - **Rotate** with left click + drag  
  - **Zoom** with scroll  
  - **Pan** with right click + drag

---

## 📁 Directory Topology

- `app.py` → Core Flask daemon. Connects frontend and simulation engine.
- `hexagonal_lattice_generator.py` → Contains **LatticeFieldEngine**, geometry generators, and render structs.
- `templates/index.html` → Quantum shell interface w/ embedded Plotly.js.
- `requirements.txt` → Dependency matrix. Feed to `pip`.
- `venv/` → Localized Python runtime hyperstructure (generated during setup).

---

## 📡 Future Work & Speculative Extensions

- Integration with **QuBitMesh™** GPU accelerator  
- Multilattice layering via **Z-Brane Overlays**  
- Import/export using `.dimfield` and `.lattx` formats  
- Support for **Edge-Entangled Topologies** (Ono & Ibrahim, 2192)

---

*Prepare the grid. Project the shell. Initialize geometry.*  
**You are now inside the lattice.**
