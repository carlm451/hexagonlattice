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
    Visualizes the hexagonal lattice using Plotly and returns the figure object.

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
