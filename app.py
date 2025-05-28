# Integrated Fluid Mechanics Application with Full Functionality
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

# Configuration with enhanced theming
st.set_page_config(
    page_title="Fluid Mechanics I",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .stProgress > div > div > div {
        background-color: #2ecc71;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #f0f2f6;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }
    .hover-tooltip {
        border-bottom: 2px dotted #3498db;
        cursor: help;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization with enhanced tracking
def init_session_state():
    if 'progress' not in st.session_state:
        st.session_state.progress = {
            "intro": False,
            "properties": False,
            "statics": False,
            "dynamics": False,
            "applications": False
        }
    if 'last_section' not in st.session_state:
        st.session_state.last_section = ""

# Enhanced Validation Decorator
def validate_positive(func):
    def wrapper(*args, **kwargs):
        try:
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float)) and arg <= 0:
                    raise ValueError(f"Parameter {func.__code__.co_varnames[i]} must be positive")
            return func(*args, **kwargs)
        except ValueError as e:
            st.error(f"🚨 {str(e)}")
            return None
    return wrapper

# Calculation Functions with tooltips
@validate_positive
def hydrostatic_pressure(depth: float, density: float = 1000.0) -> float:
    return density * 9.81 * depth

@validate_positive
def bernoulli_equation(p1: float, v1: float, z1: float, 
                      p2: float, v2: float, z2: float, 
                      density: float = 1000.0) -> float:
    energy1 = p1 + 0.5*density*v1**2 + density*9.81*z1
    energy2 = p2 + 0.5*density*v2**2 + density*9.81*z2
    if abs(energy1 - energy2) > 1e-5:
        st.warning("⚠️ Energy conservation violation - check inputs")
    return energy1 - energy2

# Enhanced Visualization Functions
def plot_moody(Re: float, rel_roughness: float):
    with st.expander("ℹ️ Moody Diagram Guide"):
        st.markdown("""
        **Understanding the Diagram:**
        - **Laminar Flow (Re < 2000):** Friction factor decreases linearly with Re
        - **Transition Zone (2000 < Re < 4000):** Unpredictable flow behavior
        - **Turbulent Flow (Re > 4000):** Friction factor depends on roughness
        """)
    
    Re_laminar = np.linspace(1e3, 2e3, 100)
    f_laminar = 64 / Re_laminar
    
    Re_turbulent = np.logspace(3.5, 8, 100)
    f_turbulent = []
    for re in Re_turbulent:
        f = fsolve(lambda x: 1/np.sqrt(x) + 2*np.log10((rel_roughness/3.7) + 2.51/(re*np.sqrt(x))), 0.02)[0]
        f_turbulent.append(f)
    
    flow_regime = "Laminar" if Re < 2000 else "Transitional" if Re < 4000 else "Turbulent"
    f_current = 64 / Re if Re < 2000 else fsolve(lambda x: 1/np.sqrt(x) + 2*np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(x))), 0.02)[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Re_laminar, y=f_laminar, name='Laminar Flow',
                            line=dict(color='#3498db', width=3)))
    fig.add_trace(go.Scatter(x=Re_turbulent, y=f_turbulent, name='Turbulent Flow',
                            line=dict(color='#e74c3c', width=3)))
    fig.add_trace(go.Scatter(x=[Re], y=[f_current], mode='markers+text',
                            marker=dict(size=12, color='#2ecc71'),
                            text=[f"Current: Re={Re/1e3:.1f}k<br>f={f_current:.4f}"],
                            textposition="bottom center"))
    
    fig.update_layout(
        title=f'Interactive Moody Diagram - {flow_regime} Flow',
        xaxis=dict(type='log', title='Reynolds Number (Re)'),
        yaxis=dict(type='log', title='Friction Factor (f)'),
        template='plotly_white',
        height=600,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bernoulli(diameter: float, velocity: float, elevation: float):
    areas = np.linspace(0.5*diameter, 2*diameter, 50)
    velocities = velocity * (diameter**2) / (areas**2)
    pressures = 101325 + 0.5*1000*(velocity**2 - velocities**2) + 1000*9.81*elevation

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=areas, y=pressures, name='Pressure',
                            line=dict(color='#e74c3c', width=3)))
    fig.add_trace(go.Scatter(x=areas, y=velocities, name='Velocity',
                            line=dict(color='#3498db', width=3, dash='dot')))
    
    fig.update_layout(
        title="Bernoulli Principle Visualization",
        xaxis_title="Cross-Sectional Area (m²)",
        yaxis_title="Pressure (Pa) / Velocity (m/s)",
        template='plotly_white',
        hoverlabel=dict(bgcolor="white")
    )
    st.plotly_chart(fig, use_container_width=True)

def water_tower_design(required_pressure: float, fluid_density: float):
    height = required_pressure / (fluid_density * 9.81)
    
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=height,
        number={'suffix': " m", 'font': {'size': 40}},
        title={'text': "Minimum Tower Height", 'font': {'size': 24}},
        delta={
            'reference': 40,
            'relative': True,
            'valueformat': ".0%",
            'increasing': {'color': "red"},
            'decreasing': {'color': "green"}
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, b=10, t=70)
    )
    with st.expander("🏗️ Design Recommendations"):
        st.markdown(f"""
        - Minimum safety factor: 3x → **{height*3:.1f} m**
        - Structural considerations for {fluid_density} kg/m³ fluid
        - Foundation requirements for {required_pressure/1e3:.1f} kPa pressure
        """)
    st.plotly_chart(fig, use_container_width=True)

def pipeline_network():
    G = nx.DiGraph()
    G.add_edges_from([("Reservoir", "Pump"), ("Pump", "Valve"), ("Valve", "City")])
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                           line=dict(width=0.5, color='#888')))
    
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', 
                           text=list(G.nodes()), marker_size=20))
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced Application Sections
def show_intro():
    st.header("📘 Fluid Mechanics I (CE221)")
    st.subheader("University of Tripoli - Civil Engineering Department")
    
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("📚 Course Book")
        st.markdown("""
        - **Fluid Mechanics with Engineering Applications**  
          Joseph B. Franzini and E. John Finnemore
        - **Fluid Mechanics**  
          R. C. Hibbeler
        """)
        
    with cols[1]:
        st.subheader("⏲️ Office Hours")
        st.markdown("""
        **Tuesday**  
        10:00am – 12:00pm  
        **Building C, Room 305**
        """)
    
    st.subheader("📈 Assessment Criteria")
    fig = go.Figure(go.Pie(
        labels=["Midterm Exam", "Laboratory Experiments", "Final Exam"],
        values=[30, 10, 60],
        marker=dict(colors=['#3498db', '#2ecc71', '#e74c3c']),
        hole=0.6,
        textinfo='percent+label'
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 Course Timeline")
    timeline = pd.DataFrame({
        "Week": ["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14"],
        "Topic": ["Introduction & Properties", "Fluid Statics", "Midterm Exam", 
                  "Fluid Dynamics", "Pipeline Systems", "Applications", "Review"]
    })
    
    fig = go.Figure()
    for i, row in timeline.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Week"]],
            y=[1],
            name=row["Topic"],
            text=row["Topic"],
            marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#34495e', '#95a5a6'][i]
        ))
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=200,
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🧠 Concept Map")
    nodes = [
        Node(id="Fluid Mechanics", label="Fluid Mechanics", size=25,
             color="#3498db", font={"color": "white"}),
        Node(id="Statics", label="Fluid Statics", size=20,
             color="#2ecc71", font={"color": "white"}),
        Node(id="Dynamics", label="Fluid Dynamics", size=20,
             color="#e74c3c", font={"color": "white"}),
        Node(id="Properties", label="Fluid Properties", size=20,
             color="#f1c40f", font={"color": "black"}),
        Node(id="Applications", label="Applications", size=20,
             color="#9b59b6", font={"color": "white"})
    ]
    edges = [
        Edge(source="Fluid Mechanics", target="Statics", color="#7f8c8d"),
        Edge(source="Fluid Mechanics", target="Dynamics", color="#7f8c8d"),
        Edge(source="Fluid Mechanics", target="Properties", color="#7f8c8d"),
        Edge(source="Fluid Mechanics", target="Applications", color="#7f8c8d")
    ]
    config = Config(width=700, height=500, nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6", collapsible=True,
                    directed=True, physics=True)
    agraph(nodes=nodes, edges=edges, config=config)
    
    st.subheader("📜 Historical Development")
    timeline = pd.DataFrame({
        "Year": [1628, 1663, 1738, 1755, 1822, 1827, 1883, 1934],
        "Event": [
            "Giovanni Borelli studies blood circulation",
            "Edme Mariotte discovers Boyle's Law",
            "Daniel Bernoulli publishes 'Hydrodynamica'",
            "Leonhard Euler formulates fluid equations",
            "Navier-Stokes equations developed",
            "Jean-Baptiste Poiseuille studies tube flow",
            "Osborne Reynolds discovers turbulence",
            "Theodore von Kármán studies vortex shedding"
        ]
    })
    fig = go.Figure(data=[go.Bar(
        x=timeline["Year"],
        y=[1]*len(timeline["Year"]),
        text=timeline["Event"],
        marker_color='#3498db'
    )])
    fig.update_layout(
        yaxis=dict(visible=False),
        title="Historical Milestones in Fluid Mechanics",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_properties():
    st.header("📊 Properties of Fluids")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔬 Fundamental Properties")
        st.markdown("""
        - **Density (ρ):** Mass per unit volume [kg/m³]
        - **Specific Weight (γ):** Weight per unit volume [N/m³]
        - **Specific Gravity (SG):** Ratio to water's density
        - **Viscosity:** Resistance to flow
        """)
        
        st.subheader("🌡️ Temperature-Viscosity Relationship")
        temp = np.linspace(0, 100, 20)
        viscosity_water = 0.001*(1 - 0.018*temp)
        viscosity_oil = 0.8 - 0.0075*temp
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temp, y=viscosity_water*1e6,
            mode='lines+markers', name='Water',
            line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(
            x=temp, y=viscosity_oil*1e6,
            mode='lines+markers', name='Oil',
            line=dict(color='#e74c3c')))
        fig.update_layout(
            title="Viscosity vs Temperature",
            xaxis_title="Temperature (°C)",
            yaxis_title="Viscosity (µPa·s)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🧪 Example Problem Solver")
        example = st.selectbox("Choose Example", 
                             ["Example 1", "Example 2", "Example 3"])
        
        with st.expander(f"🧩 {example}"):
            if example == "Example 1":
                st.write("A liquid weighs 4,905 N in a 0.315 m³ container.")
                if st.button("Solve Example 1"):
                    γ = 4905 / 0.315
                    ρ = γ / 9.81
                    sg = γ / 9810
                    st.markdown(f"""
                    **Solution:**
                    - Specific Weight (γ): {γ:.2f} N/m³
                    - Density (ρ): {ρ:.2f} kg/m³
                    - Specific Gravity: {sg:.2f}
                    """)
                    
            elif example == "Example 2":
                st.write("Water: 9.81 kN/m³, Mercury SG: 13.56")
                if st.button("Solve Example 2"):
                    γ_mercury = 9.81 * 13.56
                    ρ_mercury = γ_mercury / 9.81 * 1000
                    st.markdown(f"""
                    **Solution:**
                    - Mercury Specific Weight: {γ_mercury} kN/m³
                    - Mercury Density: {ρ_mercury:.2f} kg/m³
                    """)
                    
            elif example == "Example 3":
                st.write("Container mass: 550g empty, 8.5kg with water, 12.25kg with fluid")
                if st.button("Solve Example 3"):
                    mass_fluid = 12.25 - 0.550
                    vol = (8.5 - 0.550)/1000
                    ρ_fluid = mass_fluid / vol
                    st.markdown(f"""
                    **Solution:**
                    - Fluid Density: {ρ_fluid:.2f} kg/m³
                    - 100 cm³ Mass: {ρ_fluid*0.0001:.2f} kg
                    """)
    
    st.subheader("🌐 3D Fluid Property Explorer")
    col1, col2 = st.columns([1, 4])

    with col1:
        temperature = st.slider("Temperature (°C)", 0, 100, 25)
        pressure = st.slider("Pressure (atm)", 1, 10, 1)
        fluid = st.selectbox("Select Fluid", ["Water", "Air", "Oil"], key="fluid_selector")

    with col2:
        # Create grid of temperature and pressure values
        temp_grid = np.linspace(0, 100, 50)
        press_grid = np.linspace(1, 10, 50)
        T, P = np.meshgrid(temp_grid, press_grid)

        # Calculate density based on fluid selection
        if fluid == "Water":
            # Using simplified water density approximation
            density = 1000 * (1 - (T * 0.0002) + (P * 0.001))
        elif fluid == "Air":
            # Using ideal gas law approximation for air
            density = (P * 101325 / 10) / (287 * (273.15 + T))
        else:  # Oil
            # Simplified oil density model
            density = 850 * (1 - (T * 0.0001) + (P * 0.0005))

        fig = go.Figure(data=[go.Surface(
            z=density,
            x=temp_grid,
            y=press_grid,
            colorscale='Viridis',
            hovertemplate='Temperature: %{x} °C<br>Pressure: %{y} atm<br>Density: %{z} kg/m³'
        )])
        fig.update_layout(
            title=f"{fluid} Density Variation",
            scene=dict(
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (atm)",
                zaxis_title="Density (kg/m³)"
            ),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def show_statics():
    st.header("⚖️ Fluid Statics")
    
    tab1, tab2, tab3 = st.tabs(["Hydrostatic Pressure", "Buoyancy", "Manometry"])
    
    with tab1:
        st.subheader("🌊 Hydrostatic Pressure Calculator")
        col1, col2 = st.columns(2)
        with col1:
            depth = st.slider("Depth (m)", 0.0, 100.0, 10.0)
            density = st.selectbox("Fluid Density (kg/m³)", 
                                 [1000, 1025, 13600], 
                                 format_func=lambda x: f"{x} ({'Water' if x==1000 else 'Seawater' if x==1025 else 'Mercury'})")
        
        pressure = hydrostatic_pressure(depth, density)
        with col2:
            st.metric("Hydrostatic Pressure", 
                     f"{pressure/1e3:.2f} kPa" if pressure else "N/A",
                     help="P = ρgh")
        
        st.subheader("Pressure Distribution")
        depth_viz = st.slider("Visualization Depth (m)", 0.0, 50.0, 10.0)
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, depth_viz, 50)
        X, Y = np.meshgrid(x, y)
        P = density * 9.81 * Y
        
        fig = go.Figure(data=go.Contour(
            z=P, x=x, y=y, colorscale='Viridis',
            contours=dict(coloring='lines', showlabels=True)
        ))
        fig.update_layout(
            title=f"Pressure Distribution at {depth_viz}m Depth",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🛳️ Buoyancy Calculator")
        col1, col2 = st.columns(2)
        with col1:
            volume = st.number_input("Submerged Volume (m³)", 0.01, 1000.0, 1.0)
            fluid_density = st.selectbox("Fluid Density (kg/m³)", [1000, 1025, 13600])
        
        buoyant_force = fluid_density * 9.81 * volume
        with col2:
            st.metric("Buoyant Force", f"{buoyant_force:.2f} N",
                     help="F_b = ρ_fluid * g * V_displaced")
        
        st.subheader("Stability Analysis")
        object_density = st.number_input("Object Density (kg/m³)", 500.0, 20000.0, 2700.0)
        stability = "Floats" if object_density < fluid_density else "Sinks"
        st.metric("Stability", stability, 
                 delta=f"{abs(object_density-fluid_density)/fluid_density:.1%} {'lighter' if object_density < fluid_density else 'heavier'} than fluid")
    
    with tab3:
        st.subheader("📏 Manometer Simulator")
        col1, col2 = st.columns(2)
        with col1:
            pressure = st.slider("Applied Pressure (kPa)", -100, 100, 20)
            fluid = st.selectbox("Manometer Fluid", ["Water", "Mercury", "Oil"])
        with col2:
            density = 1000 if fluid == "Water" else 13595 if fluid == "Mercury" else 870
            height = (pressure * 1000) / (density * 9.81)
            
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=height,
                        fillcolor="#3498db", line=dict(color="#2c3e50"))
            fig.add_annotation(x=0.5, y=height/2, text=f"{height:.2f} m {fluid}",
                            showarrow=False, font=dict(color="white", size=14))
            fig.update_layout(
                height=300,
                yaxis=dict(range=[0, max(1, height*1.2)]), 
                xaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

def show_dynamics():
    st.header("🌪️ Fluid Dynamics")
    
    tab1, tab2, tab3 = st.tabs(["Moody Diagram", "Bernoulli", "Flow Measurement"])
    
    with tab1:
        st.subheader("📈 Interactive Moody Diagram")
        col1, col2 = st.columns([2, 1])
        with col1:
            Re = st.slider("Reynolds Number", 1e3, 1e7, 5e5, 
                          help="Ratio of inertial to viscous forces")
            roughness = st.select_slider("Relative Roughness", 
                                       options=[0.0001, 0.001, 0.01, 0.05],
                                       format_func=lambda x: f"{x:.4f}")
        with col2:
            st.metric("Flow Regime", 
                     "Laminar" if Re < 2000 else "Transitional" if Re < 4000 else "Turbulent",
                     delta=f"Re = {Re:.1e}")
        
        plot_moody(Re, roughness)
    
    with tab2:
        st.subheader("📉 Bernoulli Principle Simulator")
        col1, col2 = st.columns(2)
        with col1:
            diameter = st.slider("Diameter (m)", 0.1, 2.0, 0.5)
            velocity = st.slider("Velocity (m/s)", 0.1, 10.0, 2.0)
        with col2:
            elevation = st.slider("Elevation Change (m)", -10.0, 10.0, 0.0)
            fluid = st.selectbox("Fluid", ["Water", "Oil", "Mercury"])
        
        plot_bernoulli(diameter, velocity, elevation)
    with tab3:
        st.subheader("📏 Flow Measurement Devices")
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
            .device-info {
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            .param-container {
                margin-bottom: 15px;
            }
            .equation {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        device_type = st.radio("Select Device", ["Orifice", "Venturi", "Rotameter"])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f'<div class="device-info"><h4>{device_type} Information</h4></div>', unsafe_allow_html=True)
            
            if device_type == "Orifice":
                st.markdown("""
                **Orifice Plate** is a thin plate with a hole in the center, placed in a pipe to measure fluid flow rate.
                The pressure difference across the orifice is related to the flow rate.
                """)
                st.markdown('<div class="equation">$$Q = C_d A_0 \sqrt{\frac{2 \Delta P}{\rho (1 - \beta^4)}}$$</div>', unsafe_allow_html=True)
                st.markdown("- \(Q\) = Volumetric flow rate")
                st.markdown("- \(C_d\) = Discharge coefficient")
                st.markdown("- \(A_0\) = Orifice area")
                st.markdown("- \(\Delta P\) = Pressure difference")
                st.markdown("- \(\rho\) = Fluid density")
                st.markdown("- \(\beta\) = Ratio of orifice diameter to pipe diameter")
                
                beta = st.slider("Beta Ratio (d/D)", 0.2, 0.8, 0.5)
                Cd = st.slider("Discharge Coefficient", 0.5, 1.0, 0.62)
                
            elif device_type == "Venturi":
                st.markdown("""
                **Venturi Meter** uses a converging section followed by a diverging section to measure fluid flow rate.
                The pressure difference between the inlet and throat is related to the flow rate.
                """)
                st.markdown('<div class="equation">$$Q = C_d A_2 \sqrt{\frac{2 \Delta P}{\rho (1 - \beta^4)}}$$</div>', unsafe_allow_html=True)
                st.markdown("- \(Q\) = Volumetric flow rate")
                st.markdown("- \(C_d\) = Discharge coefficient")
                st.markdown("- \(A_2\) = Throat area")
                st.markdown("- \(\Delta P\) = Pressure difference")
                st.markdown("- \(\rho\) = Fluid density")
                st.markdown("- \(\beta\) = Ratio of throat diameter to inlet diameter")
                
                beta = st.slider("Beta Ratio (d/D)", 0.2, 0.8, 0.5)
                Cd = st.slider("Discharge Coefficient", 0.9, 0.98, 0.95)
                
            else:  # Rotameter
                st.markdown("""
                **Rotameter** is a variable area flow meter where a float rises in a tapered tube as flow increases.
                The position of the float indicates the flow rate.
                """)
                st.markdown('<div class="equation">$$Q = K (h)^{1/2}$$</div>', unsafe_allow_html=True)
                st.markdown("- \(Q\) = Volumetric flow rate")
                st.markdown("- \(K\) = Calibration constant")
                st.markdown("- \(h\) = Float position (height)")
                
                float_pos = st.slider("Float Position", 0.1, 1.0, 0.5)
                K = st.slider("Calibration Constant", 0.1, 1.0, 0.5)
        
        with col2:
            fig = go.Figure()
            
            if device_type == "Orifice":
                orifice = fig.add_shape(type="circle", x0=-beta, y0=-beta, x1=beta, y1=beta, 
                                    line=dict(color="red", width=2))
                pipe = fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1, 
                                    line=dict(color="blue", width=2))
                fig.add_annotation(x=0, y=1.3, text="Pipe", showarrow=False, font=dict(size=14))
                fig.add_annotation(x=0, y=beta+0.1, text="Orifice", showarrow=False, font=dict(size=14))
                
                fig.update_traces(hoverinfo='name')
                # orifice.hovertemplate = 'Orifice<br>Area: %{x:.2f} m²'
                # pipe.hovertemplate = 'Pipe<br>Diameter: 1.0 m'
                
            elif device_type == "Venturi":
                x = np.linspace(-1, 1, 100)
                y_upper = np.where(x < 0, 1, beta)
                y_lower = np.where(x < 0, -1, -beta)
                inlet = fig.add_trace(go.Scatter(x=x[x < 0], y=y_upper[x < 0], fill=None, 
                                            mode='lines', line=dict(color='red', width=2), name='Inlet'))
                throat = fig.add_trace(go.Scatter(x=x[x >= 0], y=y_upper[x >= 0], fill=None, 
                                            mode='lines', line=dict(color='red', width=2), name='Throat'))
                fig.add_trace(go.Scatter(x=x, y=y_lower, fill='tonexty', 
                                    mode='lines', line=dict(color='red', width=2)))
                fig.add_annotation(x=-0.5, y=1.3, text="Inlet", showarrow=False, font=dict(size=14))
                fig.add_annotation(x=0.5, y=beta+0.1, text="Throat", showarrow=False, font=dict(size=14))
                
                fig.update_traces(hoverinfo='name+text')
                # inlet.hovertemplate = 'Inlet<br>Diameter: 1.0 m'
                # throat.hovertemplate = f'Throat<br>Diameter: {beta:.2f} m'
                
            else:  # Rotameter
                tube = fig.add_shape(type="path", 
                                path="M 0,0 Q 0.5,1 1,0.5 L 1,-0.5 Q 0.5,-1 0,0 Z",
                                fillcolor="rgba(255,0,0,0.3)", line=dict(color="red", width=2))
                float_valve = fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=0, line=dict(color="black", width=2))
                fig.add_annotation(x=0.5, y=0, text="Float Position", showarrow=True, arrowhead=1, 
                                font=dict(size=14), ay=-30)
                
                fig.update_traces(hoverinfo='name')
                # tube.hovertemplate = 'Tube<br>Diameter: 1.0 m'
                # float_valve.hovertemplate = f'Float Position: {float_pos:.2f} m'
                
            fig.update_layout(
                width=600, height=500,
                xaxis_range=[-1.5, 1.5], yaxis_range=[-1.5, 1.5],
                xaxis_visible=False, yaxis_visible=False,
                title=f"{device_type} Meter",
                title_x=0.5,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_applications():
    st.header("🏗️ Engineering Applications")
    
    tab1, tab2 = st.tabs(["Water Systems", "Hydraulic Structures"])
    
    with tab1:
        st.subheader("🏙️ Municipal Water System Design")
        col1, col2 = st.columns(2)
        with col1:
            pressure = st.slider("Required Pressure (kPa)", 100, 500, 300)
            fluid = st.selectbox("Fluid", ["Water", "Mercury"])
        with col2:
            density = 1000 if fluid == "Water" else 13600
            water_tower_design(pressure*1000, density)
        
        st.subheader("📶 Pipeline Network Analysis")
        pipeline_network()
        
        st.subheader("🚰 Hydraulic Lift System")
        col1, col2 = st.columns(2)
        with col1:
            force1 = st.slider("Input Force (N)", 100, 10000, 500)
            area1 = st.slider("Small Piston Area (m²)", 0.01, 0.1, 0.02)
        with col2:
            area2 = st.slider("Large Piston Area (m²)", 0.1, 1.0, 0.2)
            force2 = force1 * (area2 / area1)
            
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=force1/1000,
                        fillcolor="#3498db", line=dict(color="#2c3e50"))
            fig.add_shape(type="rect", x0=2, y0=0, x1=3, y1=force2/1000,
                        fillcolor="#2ecc71", line=dict(color="#27ae60"))
            fig.add_annotation(x=0.5, y=force1/2000, text=f"Input: {force1} N",
                            showarrow=False, font=dict(color="white"))
            fig.add_annotation(x=2.5, y=force2/2000, text=f"Output: {force2:.0f} N",
                            showarrow=False, font=dict(color="white"))
            fig.update_layout(
                height=300, 
                yaxis=dict(visible=False),
                xaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🌊 Spillway Flow Visualization")
        col1, col2 = st.columns(2)
        with col1:
            spillway_type = st.radio("Spillway Type", ["Broad-crested", "Sharp-crested", "Ogee"])
            flow_rate = st.slider("Flow Rate (m³/s)", 1.0, 50.0, 10.0)
        with col2:
            x = np.linspace(-5, 5, 100)
            if spillway_type == "Broad-crested":
                y = np.where(np.abs(x) < 2, 0, np.exp(-(x**2)/5))
            elif spillway_type == "Sharp-crested":
                y = np.where(np.abs(x) < 1, 0, np.exp(-(x**2)/5))
            else:
                y = np.where(x < 0, np.exp(-(x**2)/5), np.exp(-((x-2)**2)/5))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy',
                                   fillcolor='rgba(52, 152, 219, 0.6)',
                                   line=dict(color='#2c3e50')))
            fig.add_shape(type="rect", x0=-2, y0=-0.5, x1=2, y1=0,
                         line=dict(color="#34495e"), fillcolor="#7f8c8d")
            fig.update_layout(
                title=f"{spillway_type} Spillway",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def show_resources():
    st.header("📚 Resources & Tools")
    
    tab1, tab2, tab3 = st.tabs(["Unit Converter", "Formulas", "Research Tools"])
    
    with tab1:
        st.subheader("📐 Unit Conversion System")
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.selectbox("Quantity Type", 
                                  ["Pressure", "Flow Rate", "Length", "Area", "Volume"])
            value = st.number_input("Input Value", 1.0)
        
        with col2:
            units = {
                "Pressure": ["Pa", "kPa", "bar", "psi", "atm"],
                "Flow Rate": ["m³/s", "L/s", "m³/h", "ft³/s", "gpm"],
                "Length": ["m", "km", "mm", "ft", "in"],
                "Area": ["m²", "km²", "mm²", "ft²", "in²"],
                "Volume": ["m³", "L", "gal", "ft³", "in³"]
            }
            from_unit = st.selectbox("From Unit", units[quantity])
            to_unit = st.selectbox("To Unit", units[quantity])
        
        if st.button("🔁 Convert"):
            conversions = {
                "Pressure": {
                    ("Pa", "kPa"): 0.001, ("Pa", "bar"): 0.00001,
                    ("kPa", "Pa"): 1000, ("kPa", "bar"): 0.01,
                    ("bar", "Pa"): 100000, ("bar", "kPa"): 1000
                },
                "Flow Rate": {
                    ("m³/s", "L/s"): 1000, ("L/s", "m³/s"): 0.001,
                    ("m³/h", "m³/s"): 1/3600, ("ft³/s", "m³/s"): 0.0283168
                },
                "Length": {
                    ("m", "km"): 0.001, ("km", "m"): 1000,
                    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048
                },
                "Area": {
                    ("m²", "km²"): 1e-6, ("km²", "m²"): 1e6,
                    ("m²", "ft²"): 10.7639, ("ft²", "m²"): 0.092903
                },
                "Volume": {
                    ("m³", "L"): 1000, ("L", "m³"): 0.001,
                    ("gal", "L"): 3.78541, ("ft³", "m³"): 0.0283168
                }
            }
            try:
                factor = conversions[quantity][(from_unit, to_unit)]
                result = value * factor
                st.success(f"✅ {value} {from_unit} = {result:.4f} {to_unit}")
            except KeyError:
                st.error("🚨 Conversion not supported for selected units")
        
    with tab2:
        st.subheader("🧮 Formula Reference")
        formula = st.selectbox("Select Formula", [
            "Bernoulli's Equation",
            "Hydrostatic Pressure",
            "Continuity Equation",
            "Reynolds Number",
            "Darcy-Weisbach Equation"
        ])

        if formula == "Bernoulli's Equation":
            st.markdown(r"""
            $$
            P_1 + \frac{1}{2}\rho v_1^2 + \rho g z_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho g z_2
            $$
            **Application:** Fluid flow energy conservation
            """)
            
            st.markdown("**Variables:**")
            st.markdown("- \(P\) = Pressure (Pa)")
            st.markdown("- \(\rho\) = Fluid density (kg/m³)")
            st.markdown("- \(v\) = Flow velocity (m/s)")
            st.markdown("- \(z\) = Elevation (m)")
            st.markdown("- \(g\) = Gravitational acceleration (9.81 m/s²)")

            st.markdown("**Key Concept:** The sum of pressure energy, kinetic energy, and potential energy remains constant in an ideal fluid flow.")

            st.markdown("**Real-world Applications:**")
            st.markdown("- Airfoil lift generation")
            st.markdown("- Venturi tube flow measurement")
            st.markdown("- Blood flow dynamics")

            # Interactive Visualization
            st.subheader("Interactive Bernoulli Equation")
            rho = st.slider("Fluid Density (kg/m³)", 900, 1100, 1000)
            v1 = st.slider("Velocity at Point 1 (m/s)", 0.0, 20.0, 5.0)
            z1 = st.slider("Elevation at Point 1 (m)", 0.0, 100.0, 0.0)
            v2 = st.slider("Velocity at Point 2 (m/s)", 0.0, 20.0, 10.0)
            P1 = 0.5 * rho * v1**2 + rho * 9.81 * z1
            P2 = 0.5 * rho * v2**2 + rho * 9.81 * z1  # Assuming same elevation for simplicity
            st.markdown(f"Pressure at Point 1: {P1:.2f} Pa")
            st.markdown(f"Pressure at Point 2: {P2:.2f} Pa")

        elif formula == "Hydrostatic Pressure":
            st.markdown(r"""
            $$
            P = \rho g h
            $$
            **Application:** Pressure at depth in static fluids
            """)
            
            st.markdown("**Variables:**")
            st.markdown("- \(P\) = Pressure (Pa)")
            st.markdown("- \(\rho\) = Fluid density (kg/m³)")
            st.markdown("- \(h\) = Depth (m)")
            st.markdown("- \(g\) = Gravitational acceleration (9.81 m/s²)")

            st.markdown("**Key Concept:** Pressure increases linearly with depth in a static fluid.")

            st.markdown("**Real-world Applications:**")
            st.markdown("- Dam design")
            st.markdown("- Submarine pressure hulls")
            st.markdown("- Scuba diving safety")

            # Interactive Visualization
            st.subheader("Hydrostatic Pressure Calculator")
            rho = st.slider("Fluid Density (kg/m³)", 900, 1100, 1000)
            h = st.slider("Depth (m)", 0.0, 100.0, 10.0)
            P = rho * 9.81 * h
            st.markdown(f"Pressure at {h:.1f} m depth: {P:.2f} Pa ({P/1000:.2f} kPa)")

        elif formula == "Continuity Equation":
            st.markdown(r"""
            $$
            \rho_1 A_1 v_1 = \rho_2 A_2 v_2
            $$
            **Application:** Mass conservation in fluid flow
            **(For incompressible fluids:**
            $$
            A_1 v_1 = A_2 v_2
            $$
            **)**
            """)
            
            st.markdown("**Variables:**")
            st.markdown("- \(A\) = Cross-sectional area (m²)")
            st.markdown("- \(v\) = Flow velocity (m/s)")
            st.markdown("- \(\rho\) = Fluid density (kg/m³)")

            st.markdown("**Key Concept:** Mass flow rate remains constant in a steady flow system.")

            st.markdown("**Real-world Applications:**")
            st.markdown("- Pipe flow systems")
            st.markdown("- Nozzle design")
            st.markdown("- Aircraft wing design")

            # Interactive Visualization
            st.subheader("Continuity Equation Calculator")
            A1 = st.slider("Area at Point 1 (m²)", 0.01, 0.1, 0.05)
            v1 = st.slider("Velocity at Point 1 (m/s)", 1.0, 20.0, 5.0)
            A2 = st.slider("Area at Point 2 (m²)", 0.01, 0.1, 0.025)
            v2 = (A1 * v1) / A2
            st.markdown(f"Velocity at Point 2: {v2:.2f} m/s")

        elif formula == "Reynolds Number":
            st.markdown(r"""
            $$
            Re = \frac{\rho v L}{\mu}
            $$
            **Application:** Dimensionless quantity for flow regime determination
            **(Where:**
            - \(\rho\) = Fluid density
            - \(v\) = Flow velocity
            - \(L\) = Characteristic length
            - \(\mu\) = Dynamic viscosity
            **)**
            """)
            
            st.markdown("**Key Concept:** Ratio of inertial forces to viscous forces in a fluid flow.")
            st.markdown("**Flow Regimes:**")
            st.markdown("- Laminar flow: Re < 2000")
            st.markdown("- Transitional flow: 2000 ≤ Re ≤ 4000")
            st.markdown("- Turbulent flow: Re > 4000")

            st.markdown("**Real-world Applications:**")
            st.markdown("- Pipe flow analysis")
            st.markdown("- Aircraft wing design")
            st.markdown("- Heat exchanger design")

            # Interactive Visualization
            st.subheader("Reynolds Number Calculator")
            rho = st.slider("Fluid Density (kg/m³)", 900, 1100, 1000)
            v = st.slider("Flow Velocity (m/s)", 0.1, 20.0, 1.0)
            L = st.slider("Characteristic Length (m)", 0.01, 0.1, 0.05)
            mu = st.slider("Dynamic Viscosity (Pa·s)", 0.0001, 0.001, 0.001)
            Re = (rho * v * L) / mu
            flow_regime = "Laminar" if Re < 2000 else "Transitional" if Re < 4000 else "Turbulent"
            st.markdown(f"Reynolds Number: {Re:.2f} ({flow_regime} flow)")

        elif formula == "Darcy-Weisbach Equation":
            st.markdown(r"""
            $$
            h_f = f \frac{L}{D} \frac{v^2}{2g}
            $$
            **Application:** Pressure loss due to friction in pipes
            **(Where:**
            - \(h_f\) = Head loss
            - \(f\) = Friction factor
            - \(L\) = Pipe length
            - \(D\) = Pipe diameter
            - \(v\) = Flow velocity
            - \(g\) = Gravitational acceleration
            **)**
            """)
            
            st.markdown("**Key Concept:** Calculates the head loss in a pipe due to friction.")
            st.markdown("**Friction Factor Determination:**")
            st.markdown("- Laminar flow: \(f = 64/Re\)")
            st.markdown("- Turbulent flow: Determined from Moody diagram")

            st.markdown("**Real-world Applications:**")
            st.markdown("- Pipeline design")
            st.markdown("- Pump selection")
            st.markdown("- Irrigation system design")

            # Interactive Visualization
            st.subheader("Darcy-Weisbach Calculator")
            f = st.slider("Friction Factor", 0.001, 0.1, 0.02)
            L = st.slider("Pipe Length (m)", 1.0, 1000.0, 100.0)
            D = st.slider("Pipe Diameter (m)", 0.01, 0.5, 0.1)
            v = st.slider("Flow Velocity (m/s)", 0.1, 10.0, 1.0)
            hf = f * (L / D) * (v**2 / (2 * 9.81))
            st.markdown(f"Head Loss: {hf:.2f} m")
    
    with tab3:
        st.subheader("🔬 Research Tools")
        tools = pd.DataFrame({
            "Tool": ["CFD", "PIV", "LDA", "Hot-Wire", "Pressure Sensors"],
            "Description": [
                "Numerical flow simulation",
                "Optical velocity measurement",
                "Laser-based measurements",
                "Turbulence analysis",
                "Pressure distribution studies"
            ]
        })
        st.dataframe(tools.style.set_properties(**{'background-color': '#f8f9fa'}))

# Main Application with Progress Tracking
def main():
    init_session_state()
    st.sidebar.title("🧭 Navigation")
    
    sections = {
        "🏠 Introduction": show_intro,
        "📦 Properties": show_properties,
        "⚖️ Statics": show_statics,
        "🌪️ Dynamics": show_dynamics,
        "🏗️ Applications": show_applications,
        "📚 Resources": show_resources
    }
    
    selection = st.sidebar.radio("Select Section", list(sections.keys()))
    
    if st.session_state.last_section != selection:
        st.session_state.last_section = selection
    
    sections[selection]()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Progress Tracker")
    for module in st.session_state.progress:
        st.session_state.progress[module] = st.sidebar.checkbox(
            f"{module.title()}",
            value=st.session_state.progress[module]
        )
    
    progress_value = sum(st.session_state.progress.values())/len(st.session_state.progress)
    st.sidebar.markdown(f"**Overall Progress:** {progress_value*100:.1f}%")
    st.sidebar.progress(progress_value)

if __name__ == "__main__":
    main()
