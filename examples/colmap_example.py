import os
import plotly

from pytorch3d.renderer import join_cameras_as_batch

from camformat3d.utils import read_cameras_from_sparse
from camformat3d import ColmapCamera
from camformat3d.visualization.viz import plot_cameras

HTML_TEMPLATE = """<html><head><meta charset="utf-8"/></head>
                <body>{plotly_html}</body></html>"""
                
def main():
    camera_info = read_cameras_from_sparse("./data/garden/sparse/0/", ".bin")
    
    colmap_camera_list = []
    pytorch3d_camera_list = []
    
    for idx in range(len(camera_info)):
        camera = ColmapCamera(
            intrinsics=camera_info[idx]["intrinsics"],
            extrinsics=camera_info[idx]["w2c"],
            height=camera_info[idx]["H"],
            width=camera_info[idx]["W"],
        )
        colmap_camera_list.append(camera)
        pytorch3d_camera_list.append(camera.convert_to_pytorch3d())
    
    pytorch3d_camera_batch = join_cameras_as_batch(pytorch3d_camera_list)
    
    # Visualize
    os.makedirs("results", exist_ok=True)
    
    fig = plot_cameras(pytorch3d_camera_batch)
    html_plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn")
    
    with open("results/colmap_result.html", "w") as f:
        s = HTML_TEMPLATE.format(plotly_html=html_plot)
        f.write(s)


if __name__ == "__main__":
    main()
