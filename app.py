from flask import Flask, render_template, request
import plotly
import plotly.graph_objects as go
import json
from hexagonal_lattice_generator import generate_hexagonal_lattice, visualize_lattice, visualize_lattice_3d, visualize_lattice_3d_simple

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_lattice', methods=['POST'])
def generate_lattice_data():
    n_points = int(request.form['n_points'])
    m_points = int(request.form['m_points'])
    
    # Get visualization mode (2D or 3D)
    viz_mode = request.form.get('viz_mode', '2d')
    
    # Get 3D specific parameters
    wall_height = float(request.form.get('wall_height', 1.0))
    show_lattice_points = request.form.get('show_lattice_points', 'true').lower() == 'true'
    show_arrows = request.form.get('show_arrows', 'false').lower() == 'true'

    if n_points > 25 or m_points > 25:
        return {'error': 'N and M values cannot exceed 25.'}, 400

    a_val = 1.0  # Keep lattice spacing as 1.0 for now

    lattice_df = generate_hexagonal_lattice(a=a_val, dim1_points=n_points, dim2_points=m_points)
    
    # Choose visualization based on mode
    if viz_mode == '3d':
        # Try the mesh-based version first, fallback to simple version
        try:
            fig = visualize_lattice_3d(
                lattice_df, 
                a_val, 
                n_points, 
                wall_height=wall_height,
                show_lattice_points=show_lattice_points,
                show_arrows=show_arrows
            )
        except Exception as e:
            print(f"Mesh visualization failed: {e}, using simple version")
            fig = visualize_lattice_3d_simple(
                lattice_df, 
                a_val, 
                n_points, 
                wall_height=wall_height,
                show_lattice_points=show_lattice_points,
                show_arrows=show_arrows
            )
    else:
        fig = visualize_lattice(lattice_df, a_val, n_points)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

if __name__ == '__main__':
    app.run(debug=True)