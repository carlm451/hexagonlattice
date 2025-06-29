import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, Optional, List
import warnings
from joblib import Parallel, delayed
from collections import deque



class OptimizedHexLattice:
    """
    Highly optimized hexagonal lattice generator and visualizer.
    Designed to handle large lattices efficiently while maintaining compatibility.
    """
    
    def __init__(self, a: float = 1.0):
        """Initialize with lattice spacing parameter."""
        self.a = a
        self.hexagon_side_length = a / np.sqrt(3)
        # Pre-compute hexagon angles for efficiency
        self.hex_angles = np.linspace(-np.pi / 6, 2 * np.pi - np.pi / 6, 7)[:-1]
        self.cos_angles = np.cos(self.hex_angles)
        self.sin_angles = np.sin(self.hex_angles)
    
    def generate_lattice_fast(self, dim1_points: int, dim2_points: int) -> np.ndarray:
        """
        Ultra-fast lattice generation using pure NumPy vectorization.
        Returns structured array for memory efficiency.
        """
        # Use meshgrid with optimized indexing
        x_indices = np.arange(dim1_points, dtype=np.int32)
        y_indices = np.arange(dim2_points, dtype=np.int32)
        x_steps, y_steps = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Flatten once and reuse
        x_steps_flat = x_steps.ravel()
        y_steps_flat = y_steps.ravel()
        
        # Vectorized coordinate calculation
        px = self.a * (x_steps_flat - 0.5 * y_steps_flat)
        py = self.a * (np.sqrt(3) * 0.5) * y_steps_flat
        
        # Return structured array for memory efficiency
        dtype = [('x', 'f8'), ('y', 'f8'), ('x_step', 'i4'), ('y_step', 'i4')]
        result = np.empty(len(px), dtype=dtype)
        result['x'] = px
        result['y'] = py
        result['x_step'] = x_steps_flat
        result['y_step'] = y_steps_flat
        
        return result
    
    def to_dataframe(self, lattice_data: np.ndarray) -> pd.DataFrame:
        """Convert structured array to DataFrame for compatibility."""
        labels = [f"({x},{y})" for x, y in zip(lattice_data['x_step'], lattice_data['y_step'])]
        
        return pd.DataFrame({
            'x': lattice_data['x'],
            'y': lattice_data['y'],
            'label': labels,
            'x_step': lattice_data['x_step'],
            'y_step': lattice_data['y_step']
        })
    
    def create_hexagon_mesh(self, lattice_data) -> Tuple[np.ndarray, np.ndarray]:
        """Create hexagon vertices efficiently."""
        if isinstance(lattice_data, pd.DataFrame):
            centers_x = lattice_data['x'].values
            centers_y = lattice_data['y'].values
        else:  # structured array
            centers_x = lattice_data['x']
            centers_y = lattice_data['y']
        
        centers_x = centers_x[:, np.newaxis]
        centers_y = centers_y[:, np.newaxis]
        
        vertices_x = centers_x + self.hexagon_side_length * self.cos_angles
        vertices_y = centers_y + self.hexagon_side_length * self.sin_angles
        
        return vertices_x, vertices_y

# Global instance for backward compatibility
_global_lattice = OptimizedHexLattice()

def generate_hexagonal_lattice(a: float = 1.0, dim1_points: int = 10, dim2_points: int = 10) -> pd.DataFrame:
    """
    Backward compatible function that returns DataFrame.
    Optimized internally but maintains original interface.
    """
    global _global_lattice
    if _global_lattice.a != a:
        _global_lattice = OptimizedHexLattice(a)
    
    # Generate using fast method
    lattice_data = _global_lattice.generate_lattice_fast(dim1_points, dim2_points)
    
    # Convert to DataFrame for compatibility
    return _global_lattice.to_dataframe(lattice_data)

def visualize_lattice(df: pd.DataFrame, a: float, dim1_points: int) -> go.Figure:
    """
    Optimized 2D visualization maintaining original interface.
    """
    global _global_lattice
    if _global_lattice.a != a:
        _global_lattice = OptimizedHexLattice(a)
    
    n_points = len(df)
    
    # Adaptive settings based on size
    show_labels = n_points <= 400
    
    fig = go.Figure()
    
    # Add lattice points
    if show_labels:
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers+text',
            text=df['label'],
            textposition="top center",
            marker=dict(size=max(3, min(8, 100 // np.sqrt(n_points))), color='blue'),
            name='Lattice Points',
            hovertemplate='<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(size=max(2, min(6, 80 // np.sqrt(n_points))), color='blue'),
            name='Lattice Points',
            hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
        ))
    
    # Add hexagons efficiently
    if n_points <= 2500:  # Only for reasonable sizes
        vertices_x, vertices_y = _global_lattice.create_hexagon_mesh(df)
        
        hex_lines_x = []
        hex_lines_y = []
        
        for i in range(len(df)):
            # Close hexagon
            hex_x = np.append(vertices_x[i], vertices_x[i, 0])
            hex_y = np.append(vertices_y[i], vertices_y[i, 0])
            
            hex_lines_x.extend(hex_x.tolist())
            hex_lines_y.extend(hex_y.tolist())
            hex_lines_x.append(None)
            hex_lines_y.append(None)
        
        fig.add_trace(go.Scatter(
            x=hex_lines_x,
            y=hex_lines_y,
            mode='lines',
            line=dict(color='red', width=max(0.5, min(2, 50 // np.sqrt(n_points)))),
            name='Hexagons',
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Optimized layout
    dim2_points = len(df) // dim1_points
    fig.update_layout(
        title=f"Hexagonal Lattice ({dim1_points}×{dim2_points})",
        xaxis_title="Cartesian X Coordinate",
        yaxis_title="Cartesian Y Coordinate",
        hovermode="closest",
        showlegend=n_points < 500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        font=dict(size=max(8, min(12, 150 // np.sqrt(n_points))))
    )
    
    return fig

def visualize_lattice_3d(df: pd.DataFrame, a: float, dim1_points: int,
                        wall_height: float = 1.0,
                        show_lattice_points: bool = True,
                        show_walls: bool = True) -> go.Figure:
    """
    Optimized 3D visualization maintaining original interface.
    """
    global _global_lattice
    if _global_lattice.a != a:
        _global_lattice = OptimizedHexLattice(a)
    
    n_points = len(df)
    dim2_points = n_points // dim1_points
    
    # Auto-determine detail level
    if n_points <= 100:
        detail_level = 'high'
        line_width = 3
        marker_size = 6
    elif n_points <= 400:
        detail_level = 'medium'
        line_width = 2
        marker_size = 4
    else:
        detail_level = 'low'
        line_width = 1
        marker_size = 2
        show_arrows = False  # Disable arrows for large lattices
        show_lattice_points = True  # Always show points for reference
    
    fig = go.Figure()
    
    # Add lattice points
    if show_lattice_points:
        if detail_level == 'high' and n_points <= 200:
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=[wall_height/2] * n_points,
                mode='markers+text',
                text=df['label'],
                textposition="top center",
                marker=dict(size=marker_size, color='blue'),
                name='Lattice Points',
                hovertemplate='<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=[wall_height/2] * n_points,
                mode='markers',
                marker=dict(size=marker_size, color='blue'),
                name='Lattice Points',
                hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
            ))
    
    # Add 3D hexagon structures
    if show_walls:
        _add_hexagon_walls_mesh_3d(fig, df, _global_lattice, wall_height)
    
    if detail_level == 'high':
        _add_full_wireframes_3d(fig, df, _global_lattice, wall_height, line_width)
    elif detail_level == 'medium':
        _add_medium_wireframes_3d(fig, df, _global_lattice, wall_height, line_width)
    else:  # low detail
        _add_simple_outlines_3d(fig, df, _global_lattice, wall_height, line_width)

    # Add ground reference
    if n_points < 1000:
        _add_ground_reference(fig, df, a)
    
    fig.update_layout(
        title=f"3D Hexagonal Lattice ({dim1_points}×{dim2_points}) - {detail_level.title()} Detail",
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Z Coordinate (Height)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        showlegend=n_points < 500,
        font=dict(size=max(8, min(12, 150 // np.sqrt(n_points))))
    )
    
    return fig

def _calculate_wall_geometry_for_hexagon(hex_data, lattice, wall_height):
    """Calculates the vertices and faces for a single hexagon's walls. Pure function for parallelization."""
    center_x, center_y = hex_data['x'], hex_data['y']
    
    hex_corners_x = center_x + lattice.hexagon_side_length * lattice.cos_angles
    hex_corners_y = center_y + lattice.hexagon_side_length * lattice.sin_angles
    
    # Vertices for the 6 walls (12 points total: 6 bottom, 6 top)
    x_coords = np.concatenate([hex_corners_x, hex_corners_x])
    y_coords = np.concatenate([hex_corners_y, hex_corners_y])
    z_coords = np.concatenate([np.zeros(6), np.full(6, wall_height)])
    
    # Faces for the 6 walls (12 triangles total)
    i_faces, j_faces, k_faces = [], [], []
    for v in range(6):
        v_next = (v + 1) % 6
        p_bl, p_br = v, v_next
        p_tl, p_tr = v + 6, v_next + 6
        
        # First triangle
        i_faces.append(p_bl)
        j_faces.append(p_br)
        k_faces.append(p_tr)
        
        # Second triangle
        i_faces.append(p_bl)
        j_faces.append(p_tr)
        k_faces.append(p_tl)
        
    return x_coords, y_coords, z_coords, i_faces, j_faces, k_faces

def _add_hexagon_walls_mesh_3d(fig, df, lattice, wall_height):
    """Adds vertical walls for all hexagons using a single, parallelized, multi-color mesh."""
    if df.empty:
        return

    # 1. Generate the color map for the entire lattice
    n_points = df['x_step'].max() + 1
    m_points = df['y_step'].max() + 1
    color_map = _generate_aperiodic_color_map(n_points, m_points)
    palette = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFA1']

    # 2. Use joblib to calculate wall geometry for all hexagons in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_wall_geometry_for_hexagon)(row, lattice, wall_height) for _, row in df.iterrows()
    )

    # 3. Unpack and aggregate the results into a single mesh definition
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    all_face_colors = []
    vertex_offset = 0

    for i, (x, y, z, i_f, j_f, k_f) in enumerate(results):
        # Append vertices
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
        
        # Get color for this hexagon
        hex_data = df.iloc[i]
        q, r = hex_data['x_step'], hex_data['y_step']
        color_index = color_map.get((q, r), 0)
        color = palette[color_index]
        
        # There are 12 triangles (faces) per hexagon. Each needs a color.
        all_face_colors.extend([color] * 12)
        
        # Offset the face indices for the global mesh and append
        all_i.extend([idx + vertex_offset for idx in i_f])
        all_j.extend([idx + vertex_offset for idx in j_f])
        all_k.extend([idx + vertex_offset for idx in k_f])
        
        vertex_offset += 12 # Each hexagon adds 12 vertices

    # 4. Add the single, efficient mesh trace containing all colored walls
    fig.add_trace(go.Mesh3d(
        x=all_x, y=all_y, z=all_z,
        i=all_i, j=all_j, k=all_k,
        facecolor=all_face_colors,
        opacity=0.6,
        showlegend=False,
        hoverinfo='skip'
    ))

# Helper functions for 3D visualization
def _add_full_wireframes_3d(fig, df, lattice, wall_height, line_width):
    """Add full 3D wireframes."""
    vertices_x, vertices_y = lattice.create_hexagon_mesh(df)
    
    wire_x, wire_y, wire_z = [], [], []
    
    for i in range(len(df)):
        hex_x, hex_y = vertices_x[i], vertices_y[i]
        
        # Vertical edges
        for j in range(6):
            wire_x.extend([hex_x[j], hex_x[j], None])
            wire_y.extend([hex_y[j], hex_y[j], None])
            wire_z.extend([0, wall_height, None])
        
        # Horizontal edges (bottom and top)
        for z_level in [0, wall_height]:
            hex_x_closed = np.append(hex_x, hex_x[0])
            hex_y_closed = np.append(hex_y, hex_y[0])
            wire_x.extend(hex_x_closed.tolist() + [None])
            wire_y.extend(hex_y_closed.tolist() + [None])
            wire_z.extend([z_level] * 7 + [None])
    
    fig.add_trace(go.Scatter3d(
        x=wire_x, y=wire_y, z=wire_z,
        mode='lines',
        line=dict(color='red', width=line_width),
        name='Hexagon Walls',
        showlegend=False,
        hoverinfo='skip'
    ))

def _add_medium_wireframes_3d(fig, df, lattice, wall_height, line_width):
    """Add medium detail wireframes (top and some verticals)."""
    vertices_x, vertices_y = lattice.create_hexagon_mesh(df)
    
    wire_x, wire_y, wire_z = [], [], []
    
    for i in range(len(df)):
        hex_x, hex_y = vertices_x[i], vertices_y[i]
        
        # Every other vertical edge to reduce complexity
        for j in range(0, 6, 2):
            wire_x.extend([hex_x[j], hex_x[j], None])
            wire_y.extend([hex_y[j], hex_y[j], None])
            wire_z.extend([0, wall_height, None])
        
        # Top edge only
        hex_x_closed = np.append(hex_x, hex_x[0])
        hex_y_closed = np.append(hex_y, hex_y[0])
        wire_x.extend(hex_x_closed.tolist() + [None])
        wire_y.extend(hex_y_closed.tolist() + [None])
        wire_z.extend([wall_height] * 7 + [None])
    
    fig.add_trace(go.Scatter3d(
        x=wire_x, y=wire_y, z=wire_z,
        mode='lines',
        line=dict(color='red', width=line_width),
        name='Hexagon Outlines',
        showlegend=False,
        hoverinfo='skip'
    ))

def _add_simple_outlines_3d(fig, df, lattice, wall_height, line_width):
    """Add simple top outlines only."""
    vertices_x, vertices_y = lattice.create_hexagon_mesh(df)
    
    outline_x, outline_y, outline_z = [], [], []
    
    for i in range(len(df)):
        hex_x = np.append(vertices_x[i], vertices_x[i, 0])
        hex_y = np.append(vertices_y[i], vertices_y[i, 0])
        
        outline_x.extend(hex_x.tolist() + [None])
        outline_y.extend(hex_y.tolist() + [None])
        outline_z.extend([wall_height] * 7 + [None])
    
    fig.add_trace(go.Scatter3d(
        x=outline_x, y=outline_y, z=outline_z,
        mode='lines',
        line=dict(color='red', width=line_width),
        name='Hexagon Outlines',
        showlegend=False,
        hoverinfo='skip'
    ))

def _add_ground_reference(fig, df, a):
    """Add ground reference."""
    if len(df) == 0:
        return
        
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    
    # Adaptive resolution
    resolution = max(5, min(15, int(40 / np.sqrt(len(df)))))
    
    ground_x, ground_y = np.meshgrid(
        np.linspace(x_min - a/2, x_max + a/2, resolution),
        np.linspace(y_min - a/2, y_max + a/2, resolution)
    )
    
    fig.add_trace(go.Scatter3d(
        x=ground_x.flatten(),
        y=ground_y.flatten(),
        z=np.zeros(ground_x.size),
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.3),
        name='Ground',
        showlegend=False,
        hoverinfo='skip'
    ))

# Performance monitoring functions
def get_performance_info(dim1_points: int, dim2_points: int) -> dict:
    """Get performance information for given lattice size."""
    total_points = dim1_points * dim2_points
    
    if total_points <= 100:
        performance_level = "Excellent"
        recommendation = "Full detail rendering with all features enabled"
    elif total_points <= 400:
        performance_level = "Good"
        recommendation = "Medium detail rendering, some features may be disabled"
    elif total_points <= 1000:
        performance_level = "Fair"
        recommendation = "Reduced detail rendering, labels and arrows disabled"
    else:
        performance_level = "Challenging"
        recommendation = "Minimal detail rendering, may be slow"
    
    return {
        'total_points': total_points,
        'performance_level': performance_level,
        'recommendation': recommendation
    }

def _generate_aperiodic_color_map(n_points, m_points):
    """Generates a color map using aperiodic inflation rules."""
    num_colors = 6
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    color_map = {}

    seed = (n_points // 2, m_points // 2)
    color_map[seed] = 0

    queue = deque([seed])
    visited = {seed}

    while queue:
        q, r = queue.popleft()
        current_color = color_map.get((q, r), 0)
        next_colors = [(current_color + 1) % num_colors, (current_color + 2) % num_colors]

        for i, (dq, dr) in enumerate(directions):
            nq, nr = q + dq, r + dr
            if 0 <= nq < n_points and 0 <= nr < m_points and (nq, nr) not in visited:
                color_map[(nq, nr)] = next_colors[i % 2]
                visited.add((nq, nr))
                queue.append((nq, nr))
    
    return color_map

if __name__ == "__main__":
    # Test compatibility
    print("Testing backward compatibility...")
    
    # Test original interface
    df = generate_hexagonal_lattice(a=1.0, dim1_points=5, dim2_points=5)
    print(f"Generated DataFrame with {len(df)} points")
    
    # Test 2D visualization
    fig_2d = visualize_lattice(df, 1.0, 5)
    print("2D visualization created successfully")
    
    # Test 3D visualizations
    fig_3d = visualize_lattice_3d(df, 1.0, 5)
    fig_3d_simple = visualize_lattice_3d_simple(df, 1.0, 5)
    print("3D visualizations created successfully")
    
    print("All compatibility tests passed!")