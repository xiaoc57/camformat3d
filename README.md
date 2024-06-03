# Camera Format 3D (camformat3d)
This Python library provides functionalities to convert various types of camera representations into the format used by PyTorch3D, and vice versa. It simplifies the integration of different camera models with PyTorch3D, enabling seamless transitions and compatibility in 3D rendering and vision tasks.

<img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/pytorch3dlogo.png" width="900"/>

## Installation

For detailed instructions refer to [INSTALL.md](INSTALL.md).

## License

camformat3d is released under the [MIT License](LICENSE).

## How to use?

```python
convert_to_pytorch3d # convert a camera to pytorch3d
convert_from_pytorch3d # pytorch3d convert to other types
```

For more usage examples, please refer to the `examples` directory in this repository. These examples demonstrate how to apply the library in various scenarios and with different camera types.

## Supported Camera Types

- [x] NeRF
- [x] Colmap

