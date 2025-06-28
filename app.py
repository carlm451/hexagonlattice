from flask import Flask, render_template, request
import plotly
import plotly.graph_objects as go
import json
from hexagonal_lattice_generator import generate_hexagonal_lattice, visualize_lattice

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_lattice', methods=['POST'])
def generate_lattice_data():
    n_points = int(request.form['n_points'])
    m_points = int(request.form['m_points'])

    if n_points > 25 or m_points > 25:
        return {'error': 'N and M values cannot exceed 25.'}, 400

    a_val = 1.0 # Keep lattice spacing as 1.0 for now

    lattice_df = generate_hexagonal_lattice(a=a_val, dim1_points=n_points, dim2_points=m_points)
    fig = visualize_lattice(lattice_df, a_val, n_points)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

if __name__ == '__main__':
    app.run(debug=True)