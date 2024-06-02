import os
import json

import torch
import plotly

from camformat3d import NeRFCamera
from camformat3d.visualization.viz import plot_cameras

from pytorch3d.renderer import join_cameras_as_batch

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
                <body>{plotly_html}</body></html>"""

def main():
    
    height = 800
    width = 800
    
    with open("./data/lego/transforms_train.json", 'r') as f:
        camera_info = json.load(f)

    nerf_camera_list = []
    pytorch3d_camera_list = []
    
    camera_angle_x = camera_info["camera_angle_x"]
    
    frames = camera_info["frames"]
    for idx in range(len(frames)):
        camera = NeRFCamera(
            camera_angle_x=camera_angle_x,
            c2w=torch.Tensor(frames[idx]['transform_matrix'])[:4, :4],
            height=height,
            width=width,
        )
        nerf_camera_list.append(camera)
        pytorch3d_camera_list.append(camera.convert_to_pytorch3d())
    
    pytorch3d_camera_batch = join_cameras_as_batch(pytorch3d_camera_list)
    
    # Visualize
    os.makedirs("results", exist_ok=True)
    
    fig = plot_cameras(pytorch3d_camera_batch)
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    
    with open("results/nerf_result.html", "w") as f:
        s = HTML_TEMPLATE.format(plotly_html=html_plot)
        f.write(s)
    

if __name__ == "__main__":
    main()