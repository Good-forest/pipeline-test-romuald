import mlstac
import sen2sr
import torch
import cubo
import rasterio
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path('data/test')
img_path = 'tmp'

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    lat=39.49152740347753,
    lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    edge_size=300,
    resolution=10
)

# Prepare the data to be used in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)
X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Load the model
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)


# Apply model
superX = sen2sr.predict_large(
    model=model,
    X=X, # The input tensor
    overlap=32, # The overlap between the patches
).squeeze(0).numpy()
print(superX.shape)


processed_path = PROCESSED_DIR / f"sr_{img_path}.tif"
meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'count': superX.shape[0],
    'height': superX.shape[1],
    'width': superX.shape[2],
}
processed_path.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(processed_path, 'w', **meta) as dst:
    dst.write(superX.astype(np.float32))

with rasterio.open(processed_path, 'w', **meta) as dst:
    dst.write(superX.astype(np.float32))

# comparison_path = COMPARISON_DIR / f"comp_{img_path.stem}.png"

