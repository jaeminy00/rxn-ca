from plotly.graph_objs import Figure
import numpy as np
import plotly.graph_objects as go
from plotly.io import from_json

def get_plotted_data(fig: Figure) -> dict:
    """Get the plotted data from a Plotly Figure and return it as a dictionary to be serialized"""
    
    data_dict = {}
    for i, data in enumerate(fig.data):
        data_dict[data.name] = {
            "x": list(data['x']),
            "y": list(data['y']),
        }
    return {'data': data_dict, 'figure_json': fig.to_json()}

def get_plot_from_json_data(json_data: dict) -> Figure:
    """Get a Plotly Figure from a dictionary of plotted data to be deserialized"""
    if json_data.get("figure_json"):
        return from_json(json_data["figure_json"])
    else:
        try:
            data_dict = json_data["data"]
        except KeyError:
            data_dict = json_data
            
        fig = Figure()
        for name, data in data_dict.items():
            fig.add_trace(go.Scatter(x=data["x"], y=data["y"], name=name))
        return fig