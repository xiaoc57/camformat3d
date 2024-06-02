import matplotlib
import matplotlib.pyplot as plt

from pytorch3d.vis.plotly_vis import plot_scene

def plot_cameras(cameras):

    fig = plot_scene(
        {
            "scene": {
                idx: camera for idx, camera in enumerate(cameras)
            }
        },
        camera_scale=0.3,
    )
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")
    
    num_cameras = len(cameras)
    for idx in range(num_cameras):
        # mapping to 0 - 1
        normalized_idx = idx / (num_cameras - 1)
        # convert to hex
        hex_color = matplotlib.colors.to_hex(cmap(normalized_idx))
        fig.data[0].line.color = hex_color
    
    return fig