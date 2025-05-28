# Integrated Fluid Mechanics Application with Enhanced UX
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

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
    """
    Calculates hydrostatic pressure at given depth
    Formula: P = ρgh
    - ρ: Fluid density (kg/m³)
    - g: Gravitational acceleration (9.81 m/s²)
    - h: Depth (m)
    """
    return density * 9.81 * depth

@validate_positive
def bernoulli_equation(p1: float, v1: float, z1: float, 
                      p2: float, v2: float, z2: float, 
                      density: float = 1000.0) -> float:
    """
    Bernoulli's equation energy conservation check
    Formula: P₁ + ½ρv₁² + ρgz₁ = P₂ + ½ρv₂² + ρgz₂
    """
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

def show_properties():
    st.header("📊 Properties of Fluids")
    
    st.subheader("🔍 Fluid Comparison Matrix")
    fluids = pd.DataFrame({
        "Fluid": ["Water", "Mercury", "Oil", "Air"],
        "Density (kg/m³)": [1000, 13595, 870, 1.225],
        "Viscosity (Pa·s)": [0.001, 0.00155, 0.08, 0.000018],
        "Specific Heat (J/kg·K)": [4181, 139, 2000, 1005]
    })
    
    selected_fluids = st.multiselect("Select fluids to compare:", fluids.Fluid, default=["Water", "Mercury"])
    filtered_fluids = fluids[fluids.Fluid.isin(selected_fluids)]
    
    prop_choice = st.radio("Select property to visualize:", 
                         ["Density", "Viscosity", "Specific Heat"],
                         horizontal=True)
    
    fig = go.Figure()
    for fluid in filtered_fluids.itertuples():
        fig.add_trace(go.Bar(
            x=[fluid.Fluid],
            y=[getattr(fluid, prop_choice)],
            name=fluid.Fluid,
            marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'][fluid.Index]
        ))
    
    fig.update_layout(
        title=f"{prop_choice} Comparison",
        yaxis_title=prop_choice,
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧮 Property Calculator")
    col1, col2 = st.columns(2)
    with col1:
        calc_type = st.selectbox("Calculation Type", [
            "Density",
            "Specific Weight",
            "Specific Gravity"
        ])
    
    with col2:
        if calc_type == "Density":
            mass = st.number_input("Mass (kg)", min_value=0.0, format="%.3f")
            vol = st.number_input("Volume (m³)", min_value=0.0, format="%.5f")
            if st.button("Calculate Density"):
                density = mass / vol if vol !=0 else 0
                st.metric("Result", f"{density:.2f} kg/m³")
        
        elif calc_type == "Specific Gravity":
            liquid_weight = st.number_input("Weight (N)", min_value=0.0)
            volume = st.number_input("Volume (m³)", min_value=0.0)
            if st.button("Calculate SG"):
                sg = (liquid_weight / (volume * 9810)) if volume !=0 else 0
                st.metric("Specific Gravity", f"{sg:.2f}")

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
            z=P,
            x=x,
            y=y,
            colorscale='Viridis',
            contours=dict(
                coloring='lines',
                showlabels=True
            )
        ))
        fig.update_layout(
            title=f"Pressure Distribution at {depth_viz}m Depth",
            xaxis_title="Horizontal Position (m)",
            yaxis_title="Depth (m)",
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

def show_resources():
    st.header("📚 Resources & Tools")
    
    st.subheader("📐 Unit Converter")
    col1, col2 = st.columns(2)
    with col1:
        convert_from = st.number_input("Value", 1.0)
        unit_type = st.selectbox("Quantity", ["Pressure", "Length", "Volume"])
    
    with col2:
        units = {
            "Pressure": ["Pa", "kPa", "psi", "bar"],
            "Length": ["m", "cm", "mm", "ft"],
            "Volume": ["m³", "L", "gal", "ft³"]
        }
        from_unit = st.selectbox("From", units[unit_type])
        to_unit = st.selectbox("To", units[unit_type])
    
    # Conversion logic here
    
    st.subheader("📖 Formula Reference")
    formula = st.selectbox("Select Formula", [
        "Bernoulli Equation",
        "Hydrostatic Pressure",
        "Continuity Equation"
    ])
    
    if formula == "Bernoulli Equation":
        st.markdown(r"""
        $$
        P_1 + \frac{1}{2}\rho v_1^2 + \rho g z_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho g z_2
        $$
        """)
        st.write("**Application:** Fluid flow energy conservation")

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
        # st.experimental_rerun()
    
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



# # Integrated Fluid Mechanics Application with Presentation Additions
# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from scipy.optimize import fsolve
# import pandas as pd
# from streamlit_agraph import agraph, Node, Edge, Config
# import networkx as nx
# from sklearn.preprocessing import MinMaxScaler

# # Configuration
# st.set_page_config(
#     page_title="Fluid Mechanics I",
#     layout="wide",
#     page_icon="💧"
# )

# # Session State Initialization
# def init_session_state():
#     if 'progress' not in st.session_state:
#         st.session_state.progress = {
#             "intro": False,
#             "properties": False,
#             "statics": False,
#             "dynamics": False,
#             "applications": False
#         }

# # Validation Decorator
# def validate_positive(func):
#     def wrapper(*args, **kwargs):
#         try:
#             for arg in args:
#                 if isinstance(arg, (int, float)) and arg <= 0:
#                     raise ValueError("All values must be positive")
#             return func(*args, **kwargs)
#         except ValueError as e:
#             st.error(str(e))
#             return None
#     return wrapper

# # Calculation Functions
# @validate_positive
# def hydrostatic_pressure(depth: float, density: float = 1000.0) -> float:
#     return density * 9.81 * depth

# @validate_positive
# def bernoulli_equation(p1: float, v1: float, z1: float, 
#                       p2: float, v2: float, z2: float, 
#                       density: float = 1000.0) -> float:
#     energy1 = p1 + 0.5*density*v1**2 + density*9.81*z1
#     energy2 = p2 + 0.5*density*v2**2 + density*9.81*z2
#     if abs(energy1 - energy2) > 1e-5:
#         st.warning("Energy conservation violation - check inputs")
#     return energy1 - energy2

# # Visualization Functions
# def plot_moody(Re: float, rel_roughness: float):
#     Re_laminar = np.linspace(1e3, 2e3, 100)
#     f_laminar = 64 / Re_laminar
    
#     Re_turbulent = np.logspace(3.5, 8, 100)
#     f_turbulent = []
#     for re in Re_turbulent:
#         f = fsolve(lambda x: 1/np.sqrt(x) + 2*np.log10((rel_roughness/3.7) + 2.51/(re*np.sqrt(x))), 0.02)[0]
#         f_turbulent.append(f)
    
#     if Re < 2000:
#         f_current = 64 / Re
#     else:
#         f_current = fsolve(lambda x: 1/np.sqrt(x) + 2*np.log10((rel_roughness/3.7) + 2.51/(Re*np.sqrt(x))), 0.02)[0]

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=Re_laminar, y=f_laminar, name='Laminar Flow'))
#     fig.add_trace(go.Scatter(x=Re_turbulent, y=f_turbulent, name='Turbulent Flow'))
#     fig.add_trace(go.Scatter(x=[Re], y=[f_current], mode='markers', name='Current Parameters'))
    
#     fig.update_layout(
#         title='Interactive Moody Diagram',
#         xaxis=dict(type='log', title='Reynolds Number (Re)'),
#         yaxis=dict(type='log', title='Friction Factor (f)'),
#         template='plotly_white',
#         height=600
#     )
#     st.plotly_chart(fig)

# def plot_bernoulli(diameter: float, velocity: float, elevation: float):
#     areas = np.linspace(0.5*diameter, 2*diameter, 50)
#     velocities = velocity * (diameter**2) / (areas**2)
#     pressures = 101325 + 0.5*1000*(velocity**2 - velocities**2) + 1000*9.81*elevation

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=areas, y=pressures, name='Pressure'))
#     fig.add_trace(go.Scatter(x=areas, y=velocities, name='Velocity'))
    
#     fig.update_layout(
#         title="Bernoulli Principle Visualization",
#         xaxis_title="Cross-Sectional Area (m²)",
#         yaxis_title="Pressure (Pa) / Velocity (m/s)",
#         template='plotly_white'
#     )
#     st.plotly_chart(fig)

# def water_tower_design(required_pressure: float, fluid_density: float):
#     height = required_pressure / (fluid_density * 9.81)
    
#     fig = go.Figure(go.Indicator(
#         mode="number+delta",
#         value=height,
#         number={'suffix': " m", 'font': {'size': 40}},
#         title={'text': "Minimum Tower Height", 'font': {'size': 24}},
#         delta={
#             'reference': 40,
#             'relative': True,
#             'valueformat': ".0%",
#             'increasing': {'color': "red"},
#             'decreasing': {'color': "green"}
#         },
#         domain={'x': [0, 1], 'y': [0, 1]}
#     ))
#     fig.update_layout(
#         height=300,
#         margin=dict(l=10, r=10, b=10, t=70)
#     )
#     st.plotly_chart(fig)

# def pipeline_network():
#     G = nx.DiGraph()
#     G.add_edges_from([("Reservoir", "Pump"), ("Pump", "Valve"), ("Valve", "City")])
#     pos = nx.spring_layout(G)
    
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x += [x0, x1, None]
#         edge_y += [y0, y1, None]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')))
    
#     node_x = [pos[n][0] for n in G.nodes()]
#     node_y = [pos[n][1] for n in G.nodes()]
#     fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), marker_size=20))
    
#     st.plotly_chart(fig)

# # Application Sections
# def show_intro():
#     st.header("Fluid Mechanics I (CE221)")
#     st.subheader("University of Tripoli - Civil Engineering Department")
    
#     # Course Book and Assessment Criteria
#     st.subheader("Course Book")
#     st.write("Fluid Mechanics with Engineering Applications by Joseph B. Franzini and E. John Finnemore")
#     st.write("Fluid Mechanics by R. C. Hibbeler")
    
#     st.subheader("Assessment Criteria")
#     fig = go.Figure(go.Pie(
#         labels=["Midterm Exam", "Laboratory Experiments", "Final Exam"],
#         values=[30, 10, 60],
#         marker=dict(colors=['#3498db', '#2ecc71', '#e74c3c']),
#         hole=0.6
#     ))
#     fig.update_traces(textposition='inside', textinfo='percent+label')
#     st.plotly_chart(fig)
    
#     # Office Hours
#     st.subheader("Office Hours")
#     st.write("Tuesday 10:00am – 12:00pm")
    
#     # Telegram Group Channel
#     st.subheader("Resources")
#     st.markdown("[Join Telegram Group](https://t.me/+C6soq7UnH3Q1Zjg0)")
    
#     # Learning Outcomes Wheel
#     st.subheader("Course Learning Outcomes")
#     fig = go.Figure(go.Pie(
#         labels=["Pressure Analysis", "Fluid Properties", "Flow Analysis", 
#                 "System Design", "Problem Solving"],
#         values=[25, 20, 25, 15, 15],
#         hoverinfo='label+percent',
#         textinfo='value',
#         hole=0.4
#     ))
#     st.plotly_chart(fig)
    
#     # Course Timeline
#     st.subheader("Course Timeline")
#     timeline = pd.DataFrame({
#         "Week": ["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14"],
#         "Topic": ["Introduction & Properties", "Fluid Statics", "Midterm Exam", 
#                   "Fluid Dynamics", "Pipeline Systems", "Applications", "Review"]
#     })
#     fig = go.Figure(data=[go.Bar(
#         x=timeline["Week"],
#         y=[1]*len(timeline["Week"]),
#         text=timeline["Topic"],
#         orientation='v'
#     )])
#     fig.update_layout(
#         yaxis=dict(
#             visible=False
#         ),
#         barmode='stack',
#         margin=dict(t=20, b=10, l=10, r=10)
#     )
#     st.plotly_chart(fig)
    
#     # Concept Map
#     st.subheader("Concept Map")
#     nodes = [
#         Node(id="Fluid Mechanics", label="Fluid Mechanics"),
#         Node(id="Statics", label="Fluid Statics"),
#         Node(id="Dynamics", label="Fluid Dynamics"),
#         Node(id="Properties", label="Fluid Properties"),
#         Node(id="Applications", label="Real-World Applications")
#     ]
#     edges = [
#         Edge(source="Fluid Mechanics", target="Statics"),
#         Edge(source="Fluid Mechanics", target="Dynamics"),
#         Edge(source="Fluid Mechanics", target="Properties"),
#         Edge(source="Fluid Mechanics", target="Applications"),
#         Edge(source="Statics", target="Applications"),
#         Edge(source="Dynamics", target="Applications"),
#         Edge(source="Properties", target="Statics"),
#         Edge(source="Properties", target="Dynamics")
#     ]
#     config = Config(width=700, height=500, nodeHighlightBehavior=True, 
#                     highlightColor="#F7A7A6", collapsible=True)
#     agraph(nodes=nodes, edges=edges, config=config)
    
#     # Historical Timeline
#     st.subheader("Historical Development")
#     timeline = pd.DataFrame({
#         "Year": [1628, 1663, 1738, 1755, 1822, 1827, 1883, 1934],
#         "Event": [
#             "Giovanni Borelli studies blood circulation",
#             "Edme Mariotte discovers what would be called Boyle's Law",
#             "Daniel Bernoulli publishes 'Hydrodynamica'",
#             "Leonhard Euler formulates fluid equations",
#             "Claude Navier and George Stokes develop Navier-Stokes equations",
#             "Jean-Baptiste Poiseuille studies flow in tubes",
#             "Osborne Reynolds discovers turbulence transition",
#             "Theodore von Kármán studies vortex shedding"
#         ]
#     })
#     fig = go.Figure(data=[go.Bar(
#         x=timeline["Year"],
#         y=[1]*len(timeline["Year"]),
#         text=timeline["Event"],
#         orientation='v'
#     )])
#     fig.update_layout(
#         yaxis=dict(
#             visible=False
#         ),
#         barmode='stack',
#         margin=dict(t=20, b=10, l=10, r=10)
#     )
#     st.plotly_chart(fig)

# def show_properties():
#     st.header("Properties of Fluids")
    
#     # Properties of Matter Comparison
#     st.subheader("Comparison of Gases and Liquids")
#     comparison_data = {
#         "Property": ["Molecular Bonding", "Volume Constancy", "Moldability", "Free Surface", "Compressibility", "Effect of Temperature on Density", "Effect of Temperature on Viscosity"],
#         "Gases": ["Weak bonds, molecules are far apart", "Volume changes with the container", "Takes the shape of the container", "No free surface", "High compressibility", "Density decreases with temperature increase", "Viscosity increases with temperature increase"],
#         "Liquids": ["Strong bonds, molecules are close together", "Maintains volume regardless of container", "Takes the shape of the container", "Has a free surface", "Low compressibility", "Density decreases with temperature increase", "Viscosity decreases with temperature increase"]
#     }
#     comparison_df = pd.DataFrame(comparison_data)
#     st.dataframe(comparison_df.style.set_properties(**{'text-align': 'left'}))
    
#     # Units of Measurement
#     st.subheader("Units of Measurement")
#     units_data = {
#         "Quantity": ["Length", "Time", "Mass", "Temperature", "Force"],
#         "SI Units": ["Meter (m)", "Second (s)", "Kilogram (kg)", "Kelvin (K)", "Newton (N)"],
#         "British Units": ["Feet/Inches (ft/in)", "Second (s)", "Slug", "Fahrenheit (F)", "Pound (lb)"]
#     }
#     units_df = pd.DataFrame(units_data)
#     st.dataframe(units_df.style.set_properties(**{'text-align': 'left'}))
    
#     # Definitions
#     st.subheader("Key Properties")
#     st.write("""
#     - **Density (ρ):** Mass per unit volume [kg/m³]
#     - **Specific Weight (γ):** Weight per unit volume [N/m³]
#     - **Specific Gravity (SG):** Ratio to water's density
#     - **Viscosity:** Resistance to flow
#     """)
#     st.plotly_chart(go.Figure(data=[go.Surface(z=np.random.rand(10,10))], 
#                               layout=go.Layout(title="3D Property Visualization")))
    
#     # Property Calculator
#     st.subheader("Property Calculator")
#     prop_choice = st.selectbox("Select Calculation", [
#         "Density from Mass/Volume",
#         "Specific Weight from Density",
#         "Specific Gravity"
#     ])
    
#     if prop_choice == "Density from Mass/Volume":
#         mass = st.number_input("Mass (kg)", min_value=0.0)
#         vol = st.number_input("Volume (m³)", min_value=0.0)
#         if st.button("Calculate"):
#             density = mass / vol if vol != 0 else 0
#             st.success(f"Density: {density:.2f} kg/m³")
            
#     elif prop_choice == "Specific Gravity":
#         liquid_weight = st.number_input("Liquid Weight (N)", min_value=0.0)
#         volume = st.number_input("Volume (m³)", min_value=0.0)
#         if st.button("Calculate SG"):
#             specific_weight = liquid_weight / volume if volume != 0 else 0
#             sg = specific_weight / 9810  
#             st.success(f"Specific Gravity: {sg:.2f}")

#     # Practice Problems
#     st.subheader("Example Problem Solver")
#     example = st.selectbox("Choose Example", ["Example 1", "Example 2", "Example 3"])
    
#     if example == "Example 1":
#         st.write("A liquid has a weight of 4,905 N and is contained in a container with a volume of 0.315 m³.")
#         if st.button("Solve Example 1"):
#             γ = 4905 / 0.315
#             ρ = γ / 9.81
#             sg = γ / 9810
#             st.write(f"Specific Weight (γ): {γ:.2f} N/m³")
#             st.write(f"Density (ρ): {ρ:.2f} kg/m³")
#             st.write(f"Specific Gravity: {sg:.2f}")
    
#     elif example == "Example 2":
#         st.write("Given that the specific weight of water is 9.81 kN/m³ and the specific gravity of mercury is 13.56.")
#         if st.button("Solve Example 2"):
#             ρ_water = 1000  # kg/m³
#             γ_mercury = 9.81 * 13.56  # kN/m³
#             ρ_mercury = γ_mercury / 9.81 * 1000  # kg/m³
#             st.write(f"Density of Water (ρ_water): {ρ_water} kg/m³")
#             st.write(f"Specific Weight of Mercury (γ_mercury): {γ_mercury} kN/m³")
#             st.write(f"Density of Mercury (ρ_mercury): {ρ_mercury:.2f} kg/m³")
    
#     elif example == "Example 3":
#         st.write("An empty container has a mass of 550 g. When filled with water, its total mass is 8.5 kg, and when filled with another fluid of unknown density, its total mass is 12.25 kg.")
#         if st.button("Solve Example 3"):
#             mass_container = 0.550  # kg
#             mass_water = 8.5 - mass_container  # kg
#             volume_container = mass_water / 1000  # m³ (since density of water is 1000 kg/m³)
#             mass_fluid = 12.25 - mass_container  # kg
#             ρ_fluid = mass_fluid / volume_container  # kg/m³
#             γ_fluid = ρ_fluid * 9.81  # N/m³
#             sg_fluid = γ_fluid / 9810  # specific gravity relative to water
#             mass_100cm3_fluid = ρ_fluid * 0.0001  # mass of 100 cm³ (0.0001 m³)
#             st.write(f"Volume of the fluid in the container: {volume_container:.4f} m³")
#             st.write(f"Density of the fluid: {ρ_fluid:.2f} kg/m³")
#             st.write(f"Specific Weight of the fluid: {γ_fluid:.2f} N/m³")
#             st.write(f"Specific Gravity of the fluid: {sg_fluid:.2f}")
#             st.write(f"Mass of 100 cm³ of the fluid: {mass_100cm3_fluid:.2f} kg")
    
#     # Fluid Comparison Tool
#     st.subheader("Fluid Property Comparison")
#     fluids = pd.DataFrame({
#         "Fluid": ["Water", "Mercury", "Oil", "Air"],
#         "Density (kg/m³)": [1000, 13595, 870, 1.225],
#         "Viscosity (Pa·s)": [0.001, 0.00155, 0.08, 0.000018],
#         "Specific Heat (J/kg·K)": [4181, 139, 2000, 1005]
#     })
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=fluids["Fluid"],
#         y=fluids["Density (kg/m³)"],
#         name='Density',
#         marker_color='indianred'
#     ))
#     fig.add_trace(go.Bar(
#         x=fluids["Fluid"],
#         y=fluids["Viscosity (Pa·s)"]*1e6,
#         name='Viscosity (µPa·s)',
#         marker_color='lightsalmon'
#     ))
#     st.plotly_chart(fig)
    
#     # Temperature-Viscosity Relationship
#     st.subheader("Temperature-Viscosity Relationship")
#     temp = np.linspace(0, 100, 20)
#     viscosity_water = 0.001*(1 - 0.018*temp)
#     viscosity_oil = 0.8 - 0.0075*temp
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=temp,
#         y=viscosity_water*1e6,
#         mode='lines+markers',
#         name='Water',
#         line=dict(color='blue')
#     ))
#     fig.add_trace(go.Scatter(
#         x=temp,
#         y=viscosity_oil*1e6,
#         mode='lines+markers',
#         name='Oil',
#         line=dict(color='orange')
#     ))
#     fig.update_layout(
#         yaxis_title="Viscosity (µPa·s)",
#         xaxis_title="Temperature (°C)"
#     )
#     st.plotly_chart(fig)
    
#     # Specific Gravity Slider
#     st.subheader("Specific Gravity Visualizer")
#     sg = st.slider("Specific Gravity", 0.1, 10.0, 1.0)
    
#     fig = go.Figure(go.Bar(
#         x=["Your Fluid", "Water"],
#         y=[sg, 1.0],
#         marker_color=['#3498db', '#2ecc71']
#     ))
#     fig.update_layout(
#         yaxis_title="Density Relative to Water",
#         title=f"Specific Gravity: {sg:.2f}"
#     )
#     st.plotly_chart(fig)
    
#     # 3D Fluid Property Explorer
#     st.subheader("3D Fluid Property Explorer")
#     temperature = st.slider("Temperature (°C)", 0, 100, 25)
#     pressure = st.slider("Pressure (atm)", 1, 10, 1)
    
#     x = np.linspace(0, 100, 50)
#     y = np.linspace(1, 10, 50)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sin(X/10) + np.cos(Y) + 2  
    
#     fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
#     fig.update_layout(
#         title="Density (kg/m³)",
#         scene=dict(
#             xaxis_title="Temperature (°C)",
#             yaxis_title="Pressure (atm)",
#             zaxis_title="Density (kg/m³)"
#         )
#     )
#     st.plotly_chart(fig)

# def show_statics():
#     st.header("Fluid Statics")
    
#     # Pressure Applications
#     st.subheader("Pressure Applications")
#     pressure_applications = ["Pascal’s Law", "Pressure Variation", "Plane Areas", "Curved Areas"]
#     for application in pressure_applications:
#         st.checkbox(application)
    
#     tab1, tab2, tab3 = st.tabs(["Pascal's Law", "Pressure Variation", "Forces on Surfaces"])
    
#     with tab1:
#         st.subheader("Pascal's Law Demonstration")
#         force = st.number_input("Input Force (N)", min_value=0.0)
#         area = st.number_input("Area (m²)", min_value=0.0)
#         if st.button("Calculate Pressure"):
#             pressure = force / area if area != 0 else 0
#             st.write(f"Resultant Pressure: {pressure:.2f} Pa")
    
#     with tab2:
#         st.subheader("Hydrostatic Pressure Calculator")
#         depth = st.number_input("Depth (m)", min_value=0.0)
#         density = st.number_input("Fluid Density (kg/m³)", value=1000.0)
#         if st.button("Calculate Pressure at Depth"):
#             pressure = density * 9.81 * depth
#             st.write(f"Hydrstatic Pressure: {pressure:.2f} Pa")
    
#     with tab3:
#         st.subheader("Buoyancy Calculator")
#         object_volume = st.number_input("Object Volume (m³)", 0.01, 10.0, 0.1)
#         fluid_density = st.slider("Fluid Density (kg/m³)", 500, 15000, 1000)
        
#         buoyant_force = fluid_density * 9.81 * object_volume
        
#         fig = go.Figure(go.Indicator(
#             mode="number",
#             value=buoyant_force,
#             number={'prefix': "Buoyant Force: ", 'suffix': " N"},
#             title={"text": "Buoyancy Calculation"}
#         ))
#         st.plotly_chart(fig)
    
#     # Hydrostatic Pressure Gauges
#     st.subheader("Hydrostatic Pressure Gauges")
#     depth = st.slider("Depth (m)", 0.0, 20.0, 5.0)
#     pressure = 1000 * 9.81 * depth
    
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=pressure,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Hydrostatic Pressure"},
#         gauge={
#             'axis': {'range': [None, 200000], 'visible': True},
#             'bar': {'color': "lightgreen"},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray"}
#     ))
#     st.plotly_chart(fig)
    
#     # Manometer Simulator
#     st.subheader("Manometer Simulator")
#     pressure = st.slider("Applied Pressure (kPa)", -100, 100, 20)
#     fluid = st.selectbox("Manometer Fluid", ["Water", "Mercury", "Oil"])
    
#     density = 1000 if fluid == "Water" else 13595 if fluid == "Mercury" else 870
#     height = pressure * 1000 / (density * 9.81)
    
#     fig = go.Figure()
#     fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=height,
#                   fillcolor="lightblue", line=dict(color="blue"))
#     fig.add_annotation(x=0.5, y=height/2,
#                       text=f"{fluid} Column",
#                       showarrow=False)
#     fig.update_layout(
#         height=400,
#         yaxis=dict(title="Height (m)", range=[0, max(1, height*1.2)]),
#         xaxis=dict(showgrid=False, zeroline=False, visible=False),
#         plot_bgcolor="white"
#     )
#     st.plotly_chart(fig)
    
#     # Pressure Distribution Map
#     st.subheader("Pressure Distribution on Submerged Surface")
#     width = st.slider("Plate Width (m)", 1.0, 10.0, 2.0)
#     height = st.slider("Plate Height (m)", 1.0, 10.0, 3.0)
#     depth = st.slider("Depth of Submersion (m)", 1.0, 20.0, 5.0)
    
#     y = np.linspace(0, height, 50)
#     pressure = 1000 * 9.81 * (depth + y)
    
#     fig = go.Figure(data=go.Heatmap(
#         z=np.tile(pressure, (20, 1)),
#         colorscale='Viridis',
#         colorbar=dict(title="Pressure (Pa)")
#     ))
    
#     fig.update_layout(
#         xaxis_title="Width (m)",
#         yaxis_title="Height (m)",
#         title="Pressure Distribution on Submerged Plate"
#     )
#     st.plotly_chart(fig)

# def show_dynamics():
#     st.header("Fluid Dynamics Interactive Tools")
    
#     tab1, tab2, tab3 = st.tabs(["Moody Diagram", "Bernoulli", "Continuity"])
    
#     with tab1:
#         st.subheader("Interactive Moody Diagram")
#         col1, col2 = st.columns(2)
#         with col1:
#             Re = st.slider("Reynolds Number (Re)", 1e3, 1e7, 5e5, step=1000.0, key='moddy_re')
#             roughness = st.select_slider("Relative Roughness (ε/D)", 
#                                        options=[0.0001, 0.001, 0.01, 0.05], key='moddy_roughness')
#         plot_moody(Re, roughness)
#         st.markdown("""
#         **Understanding the Diagram:**
#         - Blue line: Laminar flow regime (Re < 2000)
#         - Red line: Turbulent flow regime (Re > 4000)
#         - Green dot: Current selected parameters
#         """)
    
#     with tab2:
#         st.subheader("Bernoulli Equation Simulator")
#         diameter_bernoulli = st.slider("Pipe Diameter (m)", 0.1, 2.0, 0.5, key='bernoulli_diameter')
#         velocity = st.slider("Initial Velocity (m/s)", 0.1, 10.0, 2.0, key='bernoulli_velocity')
#         elevation = st.slider("Elevation Change (m)", -10.0, 10.0, 0.0, key='bernoulli_elevation')
#         plot_bernoulli(diameter_bernoulli, velocity, elevation)
    
#     with tab3:
#         st.subheader("Flow Measurement Devices")
#         device_type = st.radio("Select Device", ["Orifice", "Venturi", "Rotameter"], key='device_type')
        
#         if device_type == "Orifice":
#             beta = st.slider("Beta Ratio (d/D)", 0.2, 0.8, 0.5, key='orifice_beta')
#             Reynolds = 1000 * 2 * 1 / 0.001  
#         elif device_type == "Venturi":
#             beta = st.slider("Beta Ratio (d/D)", 0.2, 0.8, 0.5, key='venturi_beta')
#             Reynolds = 1000 * 3 * 1 / 0.001  
#         else:
#             beta = st.slider("Float Position", 0.1, 1.0, 0.5, key='rotameter_float')
#             Reynolds = 1000 * 0.5 * 1 / 0.001  
    
#         fig = go.Figure()
#         if device_type == "Orifice":
#             fig.add_shape(type="circle", x0=-beta, y0=-beta, x1=beta, y1=beta, 
#                          line=dict(color="red"), xref="x", yref="y")
#             fig.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1, 
#                          line=dict(color="blue"), xref="x", yref="y")
#         elif device_type == "Venturi":
#             x = np.linspace(-1, 1, 100)
#             y_upper = np.where(x < 0, 1, beta)
#             y_lower = np.where(x < 0, -1, -beta)
#             fig.add_trace(go.Scatter(x=x, y=y_upper, fill=None, mode='lines', line_color='red'))
#             fig.add_trace(go.Scatter(x=x, y=y_lower, fill='tonexty', mode='lines', line_color='red'))
#         else:
#             fig.add_shape(type="path", 
#                          path="M 0,0 Q 0.5,1 1,0.5 L 1,-0.5 Q 0.5,-1 0,0 Z",
#                          fillcolor="rgba(255,0,0,0.5)", line=dict(color="red"))
        
#         fig.update_layout(
#             width=500,
#             height=500,
#             xaxis_range=[-1.2, 1.2],
#             yaxis_range=[-1.2, 1.2],
#             xaxis_visible=False,
#             yaxis_visible=False
#         )
#         st.plotly_chart(fig)
    
#     # Momentum Equation
#     st.subheader("Momentum Equation")
#     st.write("The momentum equation relates the sum of forces acting on a fluid to the change in momentum of the fluid.")
    
#     # Flow Regime Indicator
#     st.subheader("Flow Regime Indicator")
#     Re_regime = st.slider("Reynolds Number", 100, 10000000, 500000, key='regime_re')
    
#     if Re_regime < 2000:
#         regime = "Laminar"
#         color = "lightgreen"
#     elif Re_regime < 4000:
#         regime = "Transitional"
#         color = "gold"
#     else:
#         regime = "Turbulent"
#         color = "red"
        
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=Re_regime,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': f"Flow Regime: {regime}"},
#         gauge={
#             'shape': "bullet",
#             'axis': {'range': [None, 10000000], 'visible': True},
#             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': Re_regime},
#             'bgcolor': "white",
#             'steps': [
#                 {'range': [0, 2000], 'color': "lightgreen"},
#                 {'range': [2000, 4000], 'color': "gold"},
#                 {'range': [4000, 10000000], 'color': "red"}]
#         }
#     ))
#     st.plotly_chart(fig)
    
#     # Velocity Profile Visualizer
#     st.subheader("Velocity Profile Visualizer")
#     diameter_velocity = st.slider("Pipe Diameter for Velocity Profile (m)", 0.01, 1.0, 0.1, key='velocity_diameter')
#     velocity_center = st.slider("Center Velocity (m/s)", 0.1, 10.0, 2.0, key='center_velocity')
#     viscosity = st.slider("Fluid Viscosity (Pa·s)", 0.0001, 0.1, 0.001, key='fluid_viscosity')
    
#     r = np.linspace(0, diameter_velocity/2, 50)
#     velocity = velocity_center * (1 - (r/(diameter_velocity/2))**2)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=r,
#         y=velocity,
#         fill='tozeroy',
#         name='Velocity Profile'
#     ))
#     fig.update_layout(
#         xaxis_title="Radius (m)",
#         yaxis_title="Velocity (m/s)",
#         title="Parabolic Velocity Profile (Laminar Flow)"
#     )
#     st.plotly_chart(fig)
    
#     # Energy Grade Line Plotter
#     st.subheader("Energy Grade Line Visualization")
#     head = st.slider("Total Head (m)", 10, 100, 50, key='total_head')
#     friction_factor = st.slider("Friction Factor", 0.001, 0.1, 0.02, key='friction_factor')
#     pipe_length = st.slider("Pipe Length (m)", 10, 1000, 200, key='pipe_length')
#     diameter_eql = st.slider("Pipe Diameter for EGL (m)", 0.1, 2.0, 0.5, key='eql_diameter')
    
#     x = np.linspace(0, pipe_length, 20)
#     EGL = head - friction_factor * (x / diameter_eql)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=EGL,
#         mode='lines',
#         name='Energy Grade Line',
#         line=dict(color='blue', width=3)
#     ))
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=EGL - 1,
#         mode='lines',
#         name='Hydraulic Grade Line',
#         line=dict(color='green', width=3, dash='dot')
#     ))
#     fig.update_layout(
#         yaxis_title="Head (m)",
#         xaxis_title="Pipe Length (m)",
#         title="Energy Grade Line and Hydraulic Grade Line"
#     )
#     st.plotly_chart(fig)


# def show_applications():
#     st.header("Engineering Applications")
    
#     # Application Topics
#     st.subheader("Application Topics")
#     application_topics = ["Pipe Networks", "Maritime Constructions", "Hydraulic Structures", "Natural Flows & Hydrology", "Aerodynamics", "Renewable Energy"]
#     for topic in application_topics:
#         st.checkbox(topic)
    
#     # Buoyancy
#     st.subheader("Buoyancy")
#     st.write("Buoyancy is the upward force exerted by a fluid on an submerged object.")
    
#     tab1, tab2 = st.tabs(["Water Tower Design", "Pipeline Optimization"])
    
#     with tab1:
#         st.subheader("Municipal Water System Design")
#         pressure = st.slider("Required Pressure at Ground Level (kPa)", 
#                            100, 500, 300)
#         density = st.selectbox("Fluid Density (kg/m³)", 
#                              [1000, 13600], format_func=lambda x: "Water" if x==1000 else "Mercury")
#         water_tower_design(pressure*1000, density)
#         st.markdown("""
#         **Design Considerations:**
#         - Minimum height to maintain required pressure
#         - Material strength calculations
#         - Safety factors (typically 3-5x)
#         """)
    
#     with tab2:
#         st.subheader("Pipeline Network Optimization")
#         flow_rate = st.number_input("Required Flow Rate (m³/s)", 0.1, 10.0, 1.0)
#         diameter = st.slider("Pipe Diameter (m)", 0.1, 2.0, 0.5)
        
#         velocity = flow_rate / (np.pi * (diameter/2)**2)
#         reynolds = velocity * diameter * 1000 / (1e-3)  
    
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Flow Velocity", f"{velocity:.2f} m/s")
#         with col2:
#             st.metric("Reynolds Number", f"{reynolds:.2e}")
        
#         st.markdown("""
#         **Optimization Guidelines:**
#         - Maintain Re < 2000 for laminar flow (energy efficient)
#         - Typical water supply velocity: 1-3 m/s
#         """)
    
#     # Pipeline Network Simulator
#     st.subheader("Pipeline Network Simulator")
#     nodes = ["Reservoir", "Pump", "Valve", "Consumer"]
#     edges = [
#         ("Reservoir", "Pump", 0.5),
#         ("Pump", "Valve", 0.3),
#         ("Valve", "Consumer", 0.4)
#     ]
    
#     G = nx.DiGraph()
#     for node in nodes:
#         G.add_node(node)
#     for edge in edges:
#         G.add_edge(edge[0], edge[1], weight=edge[2])
        
#     pos = nx.spring_layout(G)
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)
    
#     node_x = []
#     node_y = []
#     node_text = []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node)
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines'))
    
#     fig.add_trace(go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers+text',
#         text=node_text,
#         textposition="top center",
#         hoverinfo='text',
#         textfont_size=12,
#         marker=dict(
#             color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'],
#             size=30,
#             line_width=2)))
    
#     fig.update_layout(
#         showlegend=False,
#         hovermode='closest',
#         margin=dict(b=20,l=5,r=5,t=40),
#         annotations=[dict(
#             showarrow=False,
#             xref="paper", yref="paper",
#             x=0.005, y=-0.002 )]
#     )
#     st.plotly_chart(fig)
    

#     # Water Distribution System
#     st.subheader("Water Distribution System")

#     # Input controls
#     flow_rate = st.slider("Total Flow Rate (m³/s)", 0.01, 1.0, 0.1)
#     num_homes = st.slider("Number of Homes", 10, 1000, 100)
#     avg_flow = flow_rate / num_homes

#     # Display calculated metric
#     st.metric("Average Flow per Home", f"{avg_flow:.4f} m³/s")

#     # Create distribution visualization
#     fig = go.Figure()

#     # Define coordinates (simulated for visualization)
#     x_coords = [1, 2, 3, 4]
#     y_coords = [5, 4, 3, 2]
#     labels = [
#         f"Reservoir: {flow_rate} m³/s",
#         "Pump Station",
#         "Distribution Node",
#         f"Homes: {num_homes} units"
#     ]
#     sizes = [30, 20, 15, 10]
#     colors = ['blue', 'red', 'orange', 'green']

#     # Add flow lines
#     fig.add_trace(go.Scatter(
#         x=x_coords,
#         y=y_coords,
#         mode='lines',
#         line=dict(color='rgba(0,0,0,0.3)', width=2)
#     ))

#     # Add markers and labels
#     for i in range(len(x_coords)):
#         fig.add_trace(go.Scatter(
#             x=[x_coords[i]],
#             y=[y_coords[i]],
#             mode='markers+text',
#             marker=dict(size=sizes[i], color=colors[i]),
#             text=labels[i],
#             textposition='bottom center',
#             showlegend=False
#         ))

#     # Update layout
#     fig.update_layout(
#         title="Water Distribution Network",
#         xaxis_title="Relative Position",
#         yaxis_title="Relative Position",
#         margin=dict(l=40, r=40, t=40, b=40),
#         plot_bgcolor='rgb(230, 230,230)',
#         xaxis=dict(showgrid=False, zeroline=False),
#         yaxis=dict(showgrid=False, zeroline=False)
#     )

#     # Display visualization
#     st.plotly_chart(fig)
    
#     # Hydraulic System Animation
#     st.subheader("Hydraulic Lift System")
#     force1 = st.slider("Input Force (N)", 100, 10000, 500)
#     area1 = st.slider("Small Piston Area (m²)", 0.01, 0.1, 0.02)
#     area2 = st.slider("Large Piston Area (m²)", 0.1, 1.0, 0.2)
    
#     force2 = force1 * (area2 / area1)
    
#     st.write(f"Output Force: {force2:.2f} N")
    
#     fig = go.Figure()
#     fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=force1/1000,
#                   fillcolor="lightblue", line=dict(color="blue"))
#     fig.add_shape(type="rect", x0=2, y0=0, x1=3, y1=force2/1000,
#                   fillcolor="lightgreen", line=dict(color="green"))
#     fig.add_annotation(x=0.5, y=force1/2000,
#                       text=f"Input<br>{force1} N",
#                       showarrow=False)
#     fig.add_annotation(x=2.5, y=force2/2000,
#                       text=f"Output<br>{force2:.2f} N",
#                       showarrow=False)
#     fig.update_layout(
#         yaxis=dict(range=[0, max(force1, force2)/1000 * 1.2], visible=False),
#         xaxis=dict(visible=False),
#         height=300,
#         margin=dict(l=20, r=20, t=20, b=20)
#     )
#     st.plotly_chart(fig)
    
#     # Spillway Flow Visualization
#     st.subheader("Spillway Flow Visualization")
#     spillway_type = st.radio("Spillway Type", ["Broad-crested", "Sharp-crested", "Ogee"])
#     flow_rate = st.slider("Flow Rate (m³/s)", 1.0, 50.0, 10.0)
    
#     x = np.linspace(-5, 5, 100)
#     if spillway_type == "Broad-crested":
#         y = np.where(np.abs(x) < 2, 0, np.exp(-(x**2)/5))
#     elif spillway_type == "Sharp-crested":
#         y = np.where(np.abs(x) < 1, 0, np.exp(-(x**2)/5))
#     else:
#         y = np.where(x < 0, np.exp(-(x**2)/5), np.exp(-((x-2)**2)/5))
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y,
#         fill='tozeroy',
#         name='Water Surface',
#         fillcolor='lightblue'
#     ))
    
#     fig.add_shape(type="rect", x0=-2, y0=-0.5, x1=2, y1=0,
#                  line=dict(color="black"), fillcolor="gray")
    
#     fig.update_layout(
#         xaxis_title="Width (m)",
#         yaxis_title="Height (m)",
#         title=f"{spillway_type} Spillway"
#     )
#     st.plotly_chart(fig)

# def show_resources():
#     st.header("Additional Resources")
    
#     # Unit Conversion Calculator Enhancements
#     st.subheader("Unit Conversion Calculator")
#     quantity = st.selectbox("Quantity", ["Pressure", "Flow Rate", "Length", "Area", "Volume"])
    
#     if quantity == "Pressure":
#         units = ["Pa", "kPa", "bar", "psi", "atm"]
#     elif quantity == "Flow Rate":
#         units = ["m³/s", "L/s", "m³/h", "ft³/s", "gpm"]
#     elif quantity == "Length":
#         units = ["m", "km", "mm", "ft", "in"]
#     elif quantity == "Area":
#         units = ["m²", "km²", "mm²", "ft²", "in²"]
#     else:
#         units = ["m³", "L", "gal", "ft³", "in³"]
    
#     col1, col2 = st.columns(2)
#     with col1:
#         value = st.number_input("Value", 1.0)
#         from_unit = st.selectbox("From Unit", units)
    
#     with col2:
#         to_unit = st.selectbox("To Unit", units)
#         convert = st.button("Convert")
    
#     if convert:
#         conversion_factors = {
#             "Pressure": {
#                 ("Pa", "kPa"): 0.001, ("Pa", "bar"): 0.00001, ("Pa", "psi"): 0.000145, ("Pa", "atm"): 0.0000098692,
#                 ("kPa", "Pa"): 1000, ("kPa", "bar"): 0.01, ("kPa", "psi"): 0.145, ("kPa", "atm"): 0.0098692,
#                 ("bar", "Pa"): 100000, ("bar", "kPa"): 1000, ("bar", "psi"): 14.5038, ("bar", "atm"): 0.98692,
#                 ("psi", "Pa"): 6894.76, ("psi", "kPa"): 6.89476, ("psi", "bar"): 0.0689476, ("psi", "atm"): 0.068046,
#                 ("atm", "Pa"): 101325, ("atm", "kPa"): 101.325, ("atm", "bar"): 1.01325, ("atm", "psi"): 14.6959
#             },
#             "Flow Rate": {
#                 ("m³/s", "L/s"): 1000, ("m³/s", "m³/h"): 3600, ("m³/s", "ft³/s"): 35.3147, ("m³/s", "gpm"): 15850.3,
#                 ("L/s", "m³/s"): 0.001, ("L/s", "m³/h"): 3.6, ("L/s", "ft³/s"): 0.035315, ("L/s", "gpm"): 15.8503,
#                 ("m³/h", "m³/s"): 1/3600, ("m³/h", "L/s"): 0.27778, ("m³/h", "ft³/s"): 0.0098096, ("m³/h", "gpm"): 4.40287,
#                 ("ft³/s", "m³/s"): 0.0283168, ("ft³/s", "L/s"): 28.3168, ("ft³/s", "m³/h"): 100, ("ft³/s", "gpm"): 448.929,
#                 ("gpm", "m³/s"): 0.00006309, ("gpm", "L/s"): 0.06309, ("gpm", "m³/h"): 0.227125, ("gpm", "ft³/s"): 0.0022282
#             },
#             "Length": {
#                 ("m", "km"): 0.001, ("m", "mm"): 1000, ("m", "ft"): 3.28084, ("m", "in"): 39.3701,
#                 ("km", "m"): 1000, ("km", "mm"): 1e6, ("km", "ft"): 3280.84, ("km", "in"): 39370.1,
#                 ("mm", "m"): 0.001, ("mm", "km"): 1e-6, ("mm", "ft"): 0.00328084, ("mm", "in"): 0.0393701,
#                 ("ft", "m"): 0.3048, ("ft", "km"): 0.0003048, ("ft", "mm"): 304.8, ("ft", "in"): 12,
#                 ("in", "m"): 0.0254, ("in", "km"): 0.0000254, ("in", "mm"): 25.4, ("in", "ft"): 0.0833333
#             },
#             "Area": {
#                 ("m²", "km²"): 1e-6, ("m²", "mm²"): 1e6, ("m²", "ft²"): 10.7639, ("m²", "in²"): 1550.003,
#                 ("km²", "m²"): 1e6, ("km²", "mm²"): 1e12, ("km²", "ft²"): 10763910.417, ("km²", "in²"): 1550003100.001,
#                 ("mm²", "m²"): 1e-6, ("mm²", "km²"): 1e-12, ("mm²", "ft²"): 0.0000107639, ("mm²", "in²"): 0.001550003,
#                 ("ft²", "m²"): 0.092903, ("ft²", "km²"): 9.2903e-8, ("ft²", "mm²"): 92903, ("ft²", "in²"): 144,
#                 ("in²", "m²"): 0.00064516, ("in²", "km²"): 6.4516e-10, ("in²", "mm²"): 645.16, ("in²", "ft²"): 0.00694444
#             },
#             "Volume": {
#                 ("m³", "L"): 1000, ("m³", "gal"): 264.172, ("m³", "ft³"): 35.3147, ("m³", "in³"): 61023.7,
#                 ("L", "m³"): 0.001, ("L", "gal"): 0.264172, ("L", "ft³"): 0.0353147, ("L", "in³"): 61.0237,
#                 ("gal", "m³"): 0.00378541, ("gal", "L"): 3.78541, ("gal", "ft³"): 0.133681, ("gal", "in³"): 231,
#                 ("ft³", "m³"): 0.0283168, ("ft³", "L"): 28.3168, ("ft³", "gal"): 7.48052, ("ft³", "in³"): 1728,
#                 ("in³", "m³"): 1.63871e-5, ("in³", "L"): 0.0163871, ("in³", "gal"): 0.004329, ("in³", "ft³"): 0.000578704
#             }
#         }
        
#         try:
#             result = value * conversion_factors[quantity][(from_unit, to_unit)]
#             st.write(f"Converted Value: {result:.4f} {to_unit}")
#         except:
#             st.write("Conversion not available")
    
#     # Progress Tracker
#     st.subheader("Progress Tracker")
#     st.write("Completed Modules:")
#     for module, status in st.session_state.progress.items():
#         st.checkbox(module.replace("_", " ").title(), value=status, disabled=True)
    
#     # Learning Resources
#     st.subheader("Learning Resources")
#     st.write("Download lecture notes and tutorials from our Telegram channel:")
#     st.markdown("[Join Telegram Group](https://t.me/+C6soq7UnH3Q1Zjg0)")
    
#     # Formula Reference Carousel
#     st.subheader("Key Formulas")
#     formulas = [
#         "Bernoulli's Equation: \( \frac{P}{\rho g} + \frac{v^2}{2g} + z = constant \)",
#         "Continuity Equation: \( \rho_1 A_1 v_1 = \rho_2 A_2 v_2 \)",
#         "Reynolds Number: \( Re = \frac{\rho v L}{\mu} \)",
#         "Hydrostatic Pressure: \( P = \rho g h \)",
#         "Darcy-Weisbach Equation: \( h_f = f \frac{L}{D} \frac{v^2}{2g} \)"
#     ]
#     formula = st.selectbox("Select Formula", formulas)
#     st.markdown(f"**Selected Formula:** {formula}")
    
#     # Progress Thermometer
#     st.subheader("Course Progress")
#     progress = sum(st.session_state.progress.values()) / len(st.session_state.progress)
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=progress*100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Course Completion"},
#         gauge={
#             'axis': {'range': [None, 100], 'visible': True},
#             'bar': {'color': "green"},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 25], 'color': "lightgray"},
#                 {'range': [25, 50], 'color': "lightyellow"},
#                 {'range': [50, 75], 'color': "lightgreen"},
#                 {'range': [75, 100], 'color': "lightblue"}]
#         }
#     ))
#     st.plotly_chart(fig)
    
#     # Glossary Flashcards
#     st.subheader("Glossary Flashcards")
#     terms = {
#         "Buoyancy": "The upward force exerted by a fluid on an submerged object",
#         "Viscosity": "A measure of a fluid's resistance to flow",
#         "Pressure": "Force per unit area applied to a surface",
#         "Turbulence": "Chaotic fluid flow characterized by vortices and mixing",
#         "Laminar Flow": "Smooth, regular fluid flow with parallel layers"
#     }
    
#     term = st.selectbox("Select Term", list(terms.keys()))
#     st.write(f"**Definition:** {terms[term]}")
    
#     # Research Tool Navigator
#     st.subheader("Research Tools")
#     tools = pd.DataFrame({
#         "Tool": ["Computational Fluid Dynamics (CFD)", "Particle Image Velocimetry (PIV)", 
#                  "Laser Doppler Anemometry (LDA)", "Hot-Wire Anemometry", "Pressure Sensors"],
#         "Description": [
#             "Numerical analysis and simulation of fluid flows",
#             "Optical measurement technique for fluid velocity",
#             "Laser-based velocity measurement method",
#             "Temperature-based velocity measurement",
#             "Direct pressure measurement devices"
#         ],
#         "Application": ["Complex flow simulations", "Experimental flow visualization", 
#                       "Precision velocity measurements", "Turbulent flow analysis", 
#                       "Pressure distribution studies"]
#     })
    
#     st.dataframe(tools.style.set_properties(**{'text-align': 'left'}))
    
#     # Units' Prefixes
#     st.subheader("Units' Prefixes")
#     prefixes_data = {
#         "Prefix": ["Giga", "Mega", "Kilo", "Hecto", "Deca", "Deci", "Centi", "Milli", "Micro", "Nano"],
#         "Symbol": ["G", "M", "k", "h", "da", "d", "c", "m", "μ", "n"],
#         "Factor": ["10^9", "10^6", "10^3", "10^2", "10^1", "10^-1", "10^-2", "10^-3", "10^-6", "10^-9"]
#     }
#     prefixes_df = pd.DataFrame(prefixes_data)
#     st.dataframe(prefixes_df.style.set_properties(**{'text-align': 'left'}))

# # Main Application
# def main():
#     init_session_state()
#     st.sidebar.title("Navigation")
#     sections = {
#         "Course Introduction": show_intro,
#         "Properties of Fluids": show_properties,
#         "Fluid Statics": show_statics,
#         "Fluid Dynamics": show_dynamics,
#         "Real-World Applications": show_applications,
#         "Resources & Tools": show_resources
#     }
    
#     selection = st.sidebar.radio("Select Section", list(sections.keys()))
#     sections[selection]()
    
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("**Progress**")
#     for module in st.session_state.progress:
#         st.session_state.progress[module] = st.sidebar.checkbox(
#             module.replace("_", " ").title(),
#             value=st.session_state.progress[module]
#         )

# if __name__ == "__main__":
#     main()

