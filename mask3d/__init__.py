import os
from pathlib import Path
from typing import Any
import warnings
import hydra
from omegaconf import DictConfig
import torch

from .utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)   
from .datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_200, 
    SCANNET_COLOR_MAP_200
)
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import wget

warnings.filterwarnings("ignore")

URLS = {
    "checkpoints/scannet200/scannet200_benchmark.ckpt": "https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt",
}

def replace_path(path: str) -> str:
    """ Replace the path with the correct path for the current system """
    cwd = os.getcwd()
    parent_folder = Path(__file__).parent.parent # the folder before mask3d
    rel_path = str(parent_folder.relative_to(cwd))
    out_path = os.path.join(cwd, rel_path, path)
    return out_path

def replace_target_hydra_recursively(cfg: DictConfig, rel_path: str) -> DictConfig:
    cfg_copy = cfg.copy()
    for k, v in cfg_copy.items():
        if isinstance(v, DictConfig):
            cfg_copy[k] = replace_target_hydra_recursively(v, rel_path)
        if k == "_target_":
            # print(f"Replacing {k} {v} with {rel_path}.{v}")
            cfg_copy[k] = f"{rel_path}.{v}"
    return cfg_copy

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        model: DictConfig = cfg.model
        # it is possible model looks like this
        # {'_target_': 'mask3d.models.Mask3D', 'hidden_dim': 128, 
        # 'dim_feedforward': 1024, 'num_queries': 150, 'num_heads': 8, 
        # 'num_decoders': 3, 'dropout': 0.0, 'pre_norm': False, 
        # 'use_level_embed': False, 'normalize_pos_enc': True, 
        # 'positional_encoding_type': 'fourier', 'gauss_scale': 1.0, 
        # 'hlevels': [0, 1, 2, 3], 'non_parametric_queries': True, 
        # 'random_query_both': False, 'random_normal': False, 
        # 'random_queries': False, 'use_np_features': False, 
        # 'sample_sizes': [200, 800, 3200, 12800, 51200], 
        # 'max_sample_size': False, 'shared_decoder': True, 
        # 'num_classes': '${general.num_targets}', 
        # 'train_on_segments': '${general.train_on_segments}', 
        # 'scatter_type': 'mean', 'voxel_size': '${data.voxel_size}', 
        # 'config': {'backbone': {'_target_': 'mask3d.models.Res16UNet34C', 
        # 'config': {'dialations': [1, 1, 1, 1], 'conv1_kernel_size': 5, 
        # 'bn_momentum': 0.02}, 'in_channels': '${data.in_channels}', 
        # 'out_channels': '${data.num_labels}', 'out_fpn': True}}}

        # and we need to instantiate the model differently because 
        # we are assuming the current repo is a submodule, not a package
        # so we need to import the model from the submodule
        rel_path = str(Path(replace_path(".")).relative_to(Path.cwd())).replace("/", ".")
        model = replace_target_hydra_recursively(model, rel_path)
        self.model = hydra.utils.instantiate(model)

    def forward(self, x: torch.Tensor, raw_coordinates: torch.Tensor) -> dict:
        return self.model(x, raw_coordinates=raw_coordinates)

def download_model(checkpoint_path: str = "checkpoints/scannet200/scannet200_benchmark.ckpt") -> None:
    out_path = replace_path(checkpoint_path)
    if not os.path.exists(out_path) and checkpoint_path not in URLS:
        raise ValueError(f"Checkpoint not found: {checkpoint_path} and no URL provided in {URLS}")
    elif not os.path.exists(out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        current_url = URLS[checkpoint_path]
        print(f"Downloading model {current_url} to {filename}")
        filename = wget.download(current_url, out=str(Path(out_path).parent))
    else:
        filename = out_path
    return filename

def get_model(checkpoint_path: str = "checkpoints/scannet200/scannet200_benchmark.ckpt") -> InstanceSegmentation:
    # Initialize the directory with config files
    checkpoint_path = download_model(checkpoint_path)
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    dataset_name = checkpoint_path.split('/')[-1].split('_')[0]
    if dataset_name == 'scannet200':
        cfg.general.num_targets = 201
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 200
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
    if dataset_name == 'scannet':
        cfg.general.num_targets = 19
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 20
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file: str) -> o3d.geometry.TriangleMesh:
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def prepare_data(mesh: o3d.geometry.TriangleMesh, device: torch.device) -> tuple[
    ME.SparseTensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    
    return data, points, colors, features, unique_map, inverse_map


def map_output_to_pointcloud(mesh: o3d.geometry.TriangleMesh, 
                             outputs: dict[str, torch.Tensor],
                             inverse_map: np.ndarray,
                             label_space: str='scannet200',
                             confidence_threshold: float=0.9) -> np.ndarray:
    
    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map])  # mapping the mask back to the original point cloud
    
    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    labels_mapped = np.zeros((len(mesh.vertices), 1))

    for i, (l, c, m) in enumerate(
        sorted(zip(labels, confidences, masks_binary), reverse=False)):
        
        if label_space == 'scannet200':
            label_offset = 2
            if l == 0:
                l = -1 + label_offset
            else:
                l = int(l) + label_offset
                        
        labels_mapped[m == 1] = l
        
    return labels_mapped

def save_colorized_mesh(
        _mesh: o3d.geometry.TriangleMesh,
        labels_mapped: np.ndarray, 
        output_file: str, 
        colormap: str='scannet') -> o3d.geometry.TriangleMesh:
    # colorize mesh
    mesh = _mesh.__copy__()
    colors = np.zeros((len(mesh.vertices), 3))
    for li in np.unique(labels_mapped):
        if colormap == 'scannet':
            raise ValueError('Not implemented yet')
        elif colormap == 'scannet200':
            v_li = VALID_CLASS_IDS_200[int(li)]
            colors[(labels_mapped == li)[:, 0], :] = SCANNET_COLOR_MAP_200[v_li]
        else:
            raise ValueError('Unknown colormap - not supported')
    
    colors = colors / 255.
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_file, mesh)
    return mesh

if __name__ == '__main__':
    model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load input data
    pointcloud_file = 'data/pcl.ply'
    mesh = load_mesh(pointcloud_file)
    
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        
    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)
    
    # save colorized mesh
    save_colorized_mesh(mesh, labels, 'data/pcl_labelled.ply', colormap='scannet200')
    