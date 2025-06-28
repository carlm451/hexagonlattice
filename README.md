# Hexagonal Lattice Visualizer

This is a web application that generates and visualizes a 3D hexagonal lattice based on user-defined dimensions. It features extruded hexagonal cells and connections, presented with a cyberpunk-themed interface.

## Features

-   **Interactive 3D Visualization**: Displays a hexagonal lattice in a 3D environment using Plotly.
-   **Customizable Dimensions**: Users can specify the number of points (N and M) along the two principal lattice directions via input fields.
-   **Extruded Geometry**: Hexagon walls and connections between lattice points are rendered as extruded 3D rectangular prisms, giving a honeycomb-like appearance.
-   **Directional Coloring**: Connections along different principal directions are distinguished by color (blue, orange, purple).
-   **Cyberpunk Theme**: The web interface features a dark, futuristic, and neon-accented design.
-   **Input Validation**: Limits N and M inputs to a maximum of 25 for performance.

## Technologies Used

-   **Backend**: Python 3, Flask
-   **Frontend**: HTML, CSS, JavaScript, Plotly.js (for interactive graphing)
-   **Libraries**: `numpy`, `pandas`, `plotly`

## Setup and Run

Follow these steps to set up and run the application locally:

1.  **Navigate to the Project Directory**:
    ```bash
    cd /home/carlbrady/gemini/library
    ```

2.  **Create a Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    ```bash
    source venv/bin/activate
    ```

4.  **Install Dependencies**:
    Install the required Python libraries using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Flask Application**:
    ```bash
    python app.py
    ```

6.  **Access the Application**:
    Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

-   Enter desired integer values for N (Dim 1 Points) and M (Dim 2 Points) in the input fields.
-   Click the "Generate Lattice" button to update the 3D visualization.
-   You can interact with the 3D plot by dragging to rotate, scrolling to zoom, and right-clicking to pan.

## File Structure

-   `app.py`: The main Flask application file, handling web routes and data processing.
-   `hexagonal_lattice_generator.py`: Contains the core logic for generating lattice coordinates and creating the Plotly 3D figure.
-   `requirements.txt`: Lists all Python dependencies.
-   `templates/index.html`: The HTML template for the web interface, including input fields and the Plotly graph container.
-   `venv/`: The virtual environment directory (created during setup).
