import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

def generate_hexagonal_lattice(a=1.0, dim1_points=10, dim2_points=10):
    """
    Generates Cartesian coordinates for a hexagonal lattice based on the given description.

    Args:
        a (float): The lattice spacing size.
        dim1_points (int): Number of points along the first principal direction (x-axis).
        dim2_points (int): Number of points along the second principal direction (120-degree).

    Returns:
        pandas.DataFrame: A DataFrame with 'x', 'y' Cartesian coordinates and 'label' for each point.
    """
    x_coords = []
    y_coords = []
    labels = []

    # Primitive translation vectors:
    # v1 = (a, 0) along the x-axis
    # v2 = (a * cos(120°), a * sin(120°)) rotated 120 degrees counterclockwise from x-axis
    # cos(120°) = -0.5
    # sin(120°) = sqrt(3)/2

    for y_steps in range(dim2_points):
        for x_steps in range(dim1_points):
            # Calculate Cartesian coordinates for the point (x_steps, y_steps)
            # P = x_steps * v1 + y_steps * v2
            px = a * x_steps + a * (-0.5) * y_steps
            py = a * (np.sqrt(3)/2) * y_steps

            x_coords.append(px)
            y_coords.append(py)
            labels.append(f"({x_steps},{y_steps})")

    return pd.DataFrame({'x': x_coords, 'y': y_coords, 'label': labels})

def visualize_lattice(df, a, dim1_points):
    """
    Visualizes the hexagonal lattice using Plotly and returns the figure object (2D version).

    Args:
        df (pandas.DataFrame): DataFrame containing 'x', 'y' coordinates and 'label'.
        a (float): The lattice spacing size.
        dim1_points (int): Number of points along the first principal direction (x-axis).

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    # Add lattice points
    fig.add_trace(go.Scatter(x=df['x'].tolist(), y=df['y'].tolist(), mode='markers+text', text=df['label'],
                             textposition="top center",
                             marker=dict(size=8, color='blue'),
                             name='Lattice Points'))
    fig.update_layout(
        xaxis_title="Cartesian X Coordinate",
        yaxis_title="Cartesian Y Coordinate",
        hovermode="closest",
        showlegend=False,
        width=900,
        height=800,
        # Ensure aspect ratio is respected for hexagonal shape
        yaxis=dict(scaleanchor="x", scaleratio=1),
        font=dict(size=10)
    )
    
    hexagon_side_length = a / np.sqrt(3)
    angles = np.linspace(-np.pi / 6, 2 * np.pi - np.pi / 6, 7)[:-1] # 6 angles for 6 vertices, closing the shape, rotated 30 degrees clockwise

    for index, row in df.iterrows():
        center_x = row['x']
        center_y = row['y']

        hex_x = center_x + hexagon_side_length * np.cos(angles)
        hex_y = center_y + hexagon_side_length * np.sin(angles)

        fig.add_trace(go.Scatter(x=np.append(hex_x, hex_x[0]).tolist(), y=np.append(hex_y, hex_y[0]).tolist(),
                                 mode='lines',
                                 line=dict(color='red', width=2),
                                 name=f'Hexagon at {row["label"]}'))

    for index, row in df.iterrows():
        x_steps = int(row['label'].split(',')[0].strip('('))
        y_steps = int(row['label'].split(',')[1].strip(')'))

        # Draw arrow in positive x direction
        if x_steps < dim1_points - 1:
            start_x = row['x']
            start_y = row['y']
            
            # The next point in the x-direction is (x_steps + 1, y_steps)
            end_x = a * (x_steps + 1) + a * (-0.5) * y_steps
            end_y = a * (np.sqrt(3)/2) * y_steps

            fig.add_annotation(
                x=end_x,
                y=end_y,
                ax=start_x,
                ay=start_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="green"
            )

    # Update title to reflect the added hexagons and arrows
    fig.update_layout(title="Hexagonal Lattice Visualization with Hexagons and X-direction Arrows")
    return fig
    
def visualize_lattice_3d_simple(df, a, dim1_points, wall_height=1.0, show_lattice_points=True, show_arrows=False):
    """
    Alternative 3D visualization using simple cylinder walls (more reliable rendering).
    """
    fig = go.Figure()
    
    hexagon_side_length = a / np.sqrt(3)
    
    # Add cylindrical walls for each hexagon edge
    for index, row in df.iterrows():
        center_x = row['x']
        center_y = row['y']
        
        # Get hexagon vertices
        hex_x, hex_y = create_hexagon_vertices(center_x, center_y, hexagon_side_length)
        
        # Create walls as cylinders along each edge
        for i in range(len(hex_x)):
            next_i = (i + 1) % len(hex_x)
            
            # Calculate midpoint and direction for wall segment
            mid_x = (hex_x[i] + hex_x[next_i]) / 2
            mid_y = (hex_y[i] + hex_y[next_i]) / 2
            
            # Create a thick line (wall) between vertices
            fig.add_trace(go.Scatter3d(
                x=[hex_x[i], hex_x[next_i]],
                y=[hex_y[i], hex_y[next_i]],
                z=[wall_height/2, wall_height/2],
                mode='lines',
                line=dict(color='red', width=15),  # Very thick line to simulate wall
                name=f'Wall {index}-{i}',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add vertical pillars at corners
            fig.add_trace(go.Scatter3d(
                x=[hex_x[i], hex_x[i]],
                y=[hex_y[i], hex_y[i]], 
                z=[0, wall_height],
                mode='lines',
                line=dict(color='darkred', width=8),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add ground reference
    if len(df) > 0:
        x_min, x_max = df['x'].min() - a, df['x'].max() + a
        y_min, y_max = df['y'].min() - a, df['y'].max() + a
        
        # Ground plane as scatter points
        ground_x, ground_y = np.meshgrid(
            np.linspace(x_min, x_max, 20),
            np.linspace(y_min, y_max, 20)
        )
        fig.add_trace(go.Scatter3d(
            x=ground_x.flatten(),
            y=ground_y.flatten(),
            z=np.zeros_like(ground_x.flatten()),
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add lattice points
    if show_lattice_points:
        fig.add_trace(go.Scatter3d(
            x=df['x'].tolist(),
            y=df['y'].tolist(),
            z=[wall_height/2] * len(df),
            mode='markers+text',
            text=df['label'],
            textposition="top center",
            marker=dict(size=8, color='blue'),
            name='Lattice Points'
        ))
    
    # Add arrows
    if show_arrows:
        for index, row in df.iterrows():
            x_steps = int(row['label'].split(',')[0].strip('('))
            y_steps = int(row['label'].split(',')[1].strip(')'))
            
            if x_steps < dim1_points - 1:
                start_x = row['x']
                start_y = row['y']
                arrow_z = wall_height / 2
                
                end_x = a * (x_steps + 1) + a * (-0.5) * y_steps
                end_y = a * (np.sqrt(3)/2) * y_steps
                
                fig.add_trace(go.Scatter3d(
                    x=[start_x, end_x],
                    y=[start_y, end_y],
                    z=[arrow_z, arrow_z],
                    mode='lines',
                    line=dict(color='green', width=8),
                    name=f'Arrow {row["label"]}',
                    showlegend=False
                ))
    
    # Update layout
    fig.update_layout(
        title="3D Hexagonal Lattice with Wall Structure",
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate", 
            zaxis_title="Z Coordinate (Height)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=900,
        height=800,
        showlegend=False,
        font=dict(size=10)
    )
    
    return fig

def create_hexagon_vertices(center_x, center_y, hexagon_side_length):
    """
    Create the vertices of a hexagon given its center and side length.
    
    Args:
        center_x (float): X coordinate of hexagon center
        center_y (float): Y coordinate of hexagon center
        hexagon_side_length (float): Side length of the hexagon
    
    Returns:
        tuple: (hex_x, hex_y) arrays of hexagon vertex coordinates
    """
    angles = np.linspace(-np.pi / 6, 2 * np.pi - np.pi / 6, 7)[:-1]  # 6 vertices
    hex_x = center_x + hexagon_side_length * np.cos(angles)
    hex_y = center_y + hexagon_side_length * np.sin(angles)
    return hex_x, hex_y

def create_hexagon_wall_mesh(hex_x, hex_y, wall_height=1.0, base_z=0.0):
    """
    Create a 3D mesh for hexagonal walls by extruding the 2D hexagon.
    
    Args:
        hex_x (array): X coordinates of hexagon vertices
        hex_y (array): Y coordinates of hexagon vertices
        wall_height (float): Height of the walls
        base_z (float): Z coordinate of the base
    
    Returns:
        tuple: (vertices, faces) for the 3D mesh
    """
    n_vertices = len(hex_x)
    
    # Ensure we have valid vertices
    if n_vertices < 3:
        return None, None
    
    # Create vertices: bottom hexagon + top hexagon
    vertices = []
    
    # Bottom vertices
    for i in range(n_vertices):
        vertices.append([hex_x[i], hex_y[i], base_z])
    
    # Top vertices
    for i in range(n_vertices):
        vertices.append([hex_x[i], hex_y[i], base_z + wall_height])
    
    vertices = np.array(vertices)
    
    # Create faces for the walls (no bottom or top faces, just walls)
    faces = []
    
    # Wall faces (rectangles split into triangles)
    for i in range(n_vertices):
        next_i = (i + 1) % n_vertices
        
        # Bottom triangle of wall face
        faces.append([i, next_i, next_i + n_vertices])
        # Top triangle of wall face
        faces.append([i, next_i + n_vertices, i + n_vertices])
    
    return vertices, faces

def visualize_lattice_3d(df, a, dim1_points, wall_height=1.0, show_lattice_points=True, show_arrows=False):
    """
    Visualizes the hexagonal lattice as a 3D structure with extruded hexagonal walls.
    Fixed version that handles null data issues.

    Args:
        df (pandas.DataFrame): DataFrame containing 'x', 'y' coordinates and 'label'.
        a (float): The lattice spacing size.
        dim1_points (int): Number of points along the first principal direction (x-axis).
        wall_height (float): Height of the hexagonal walls.
        show_lattice_points (bool): Whether to show the original lattice points.
        show_arrows (bool): Whether to show directional arrows.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()
    
    # Input validation
    if df is None or len(df) == 0:
        print("Error: DataFrame is empty or None")
        return fig
    
    hexagon_side_length = a / np.sqrt(3)
    
    # Debug: Print some information
    print(f"Processing {len(df)} hexagons with side length {hexagon_side_length}")
    print(f"Wall height: {wall_height}")
    
    # Add wireframe edges instead of mesh (more reliable)
    for index, row in df.iterrows():
        try:
            center_x = row['x']
            center_y = row['y']
            
            # Get hexagon vertices
            hex_x, hex_y = create_hexagon_vertices(center_x, center_y, hexagon_side_length)
            
            # Validate vertices
            if len(hex_x) == 0 or len(hex_y) == 0:
                print(f"Warning: Invalid vertices for hexagon {index}")
                continue
            
            # Add wireframe edges for better visibility and reliability
            for i in range(len(hex_x)):
                next_i = (i + 1) % len(hex_x)
                
                # Vertical edges
                fig.add_trace(go.Scatter3d(
                    x=[hex_x[i], hex_x[i]],
                    y=[hex_y[i], hex_y[i]],
                    z=[0, wall_height],
                    mode='lines',
                    line=dict(color='darkred', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Bottom edges
                fig.add_trace(go.Scatter3d(
                    x=[hex_x[i], hex_x[next_i]],
                    y=[hex_y[i], hex_y[next_i]],
                    z=[0, 0],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Top edges
                fig.add_trace(go.Scatter3d(
                    x=[hex_x[i], hex_x[next_i]],
                    y=[hex_y[i], hex_y[next_i]],
                    z=[wall_height, wall_height],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
        except Exception as e:
            print(f"Error processing hexagon {index}: {e}")
            continue
    
    # Add a simple ground plane for reference using Scatter3d
    if len(df) > 0:
        try:
            x_min, x_max = df['x'].min() - a, df['x'].max() + a
            y_min, y_max = df['y'].min() - a, df['y'].max() + a
            
            # Create a simple grid for ground reference
            ground_x, ground_y = np.meshgrid(
                np.linspace(x_min, x_max, 10),
                np.linspace(y_min, y_max, 10)
            )
            
            fig.add_trace(go.Scatter3d(
                x=ground_x.flatten(),
                y=ground_y.flatten(),
                z=np.zeros_like(ground_x.flatten()),
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='Ground',
                showlegend=False,
                hoverinfo='skip'
            ))
        except Exception as e:
            print(f"Error creating ground plane: {e}")
    
    # Optionally add lattice points as markers
    if show_lattice_points:
        try:
            fig.add_trace(go.Scatter3d(
                x=df['x'].tolist(),
                y=df['y'].tolist(),
                z=[wall_height/2] * len(df),  # Place points at mid-height
                mode='markers+text',
                text=df['label'].tolist(),
                textposition="top center",
                marker=dict(size=6, color='blue'),
                name='Lattice Points'
            ))
        except Exception as e:
            print(f"Error adding lattice points: {e}")
    
    # Optionally add directional arrows (simplified for 3D)
    if show_arrows:
        try:
            for index, row in df.iterrows():
                try:
                    label_parts = row['label'].strip('()').split(',')
                    x_steps = int(label_parts[0])
                    y_steps = int(label_parts[1])
                    
                    # Draw arrow in positive x direction
                    if x_steps < dim1_points - 1:
                        start_x = row['x']
                        start_y = row['y']
                        arrow_z = wall_height / 2
                        
                        # The next point in the x-direction is (x_steps + 1, y_steps)
                        end_x = a * (x_steps + 1) + a * (-0.5) * y_steps
                        end_y = a * (np.sqrt(3)/2) * y_steps
                        
                        # Add arrow as a line
                        fig.add_trace(go.Scatter3d(
                            x=[start_x, end_x],
                            y=[start_y, end_y],
                            z=[arrow_z, arrow_z],
                            mode='lines',
                            line=dict(color='green', width=6),
                            name=f'Arrow {row["label"]}',
                            showlegend=False
                        ))
                except Exception as e:
                    print(f"Error processing arrow for row {index}: {e}")
                    continue
        except Exception as e:
            print(f"Error adding arrows: {e}")
    
    # Update layout for 3D visualization
    fig.update_layout(
        title="3D Hexagonal Lattice with Wall Structure (Wireframe)",
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Z Coordinate (Height)",
            aspectmode='data',  # Maintain aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Nice viewing angle
            )
        ),
        width=900,
        height=800,
        showlegend=False,
        font=dict(size=10)
    )
    
    return fig

# Example usage and test function
def test_lattice_visualization():
    """
    Test function to demonstrate the lattice visualization
    """
    try:
        # Generate a small lattice for testing
        df = generate_hexagonal_lattice(a=1.0, dim1_points=3, dim2_points=3)
        print(f"Generated lattice with {len(df)} points")
        print(df.head())
        
        # Test 2D visualization
        fig_2d = visualize_lattice(df, a=1.0, dim1_points=3)
        print("2D visualization created successfully")
        
        # Test 3D visualization (simple version)
        fig_3d_simple = visualize_lattice_3d_simple(df, a=1.0, dim1_points=3, wall_height=1.0)
        print("3D simple visualization created successfully")
        
        # Test 3D visualization (fixed version)
        fig_3d = visualize_lattice_3d(df, a=1.0, dim1_points=3, wall_height=1.0, 
                                      show_lattice_points=True, show_arrows=True)
        print("3D visualization created successfully")
        
        return fig_2d, fig_3d_simple, fig_3d
        
    except Exception as e:
        print(f"Error in test function: {e}")
        return None, None, None

if __name__ == "__main__":
    test_lattice_visualization()