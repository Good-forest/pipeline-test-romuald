import mlstac
import torch
from pathlib import Path
import rasterio
import numpy as np
import cubo
import skimage



PROCESSED_DIR = Path('data/test')
img_path = 'tmp'

i = 0
def save_file(arr, block_info=None):
    """ Save file to foo-x-y.tif, where x and y are block locations """
    print(arr)
    global i
    i += 1
    filename = PROCESSED_DIR / f"foo-{i}.tif"
    skimage.io.imsave(filename, arr)
    return arr


# Download the model
# mlstac.download(
#   file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
#   output_dir="model/SEN2SRLite",
# )

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlstac.load("model/SEN2SRLite").compiled_model(device=device)
model = model.to(device)

# Create a Sentinel-2 L2A data cube for a specific location and date range
da = cubo.create(
    # lat=39.49152740347753,
    # lon=-0.4308725142800361,
    collection="sentinel-2-l2a",
    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    # start_date="2023-01-01",
    # end_date="2023-12-31",

    lat=47.848151988493385,
    lon=13.379491178028564,
   # bands=["B02","B03","B04"],
    start_date="2020-01-01",
    end_date="2021-01-01",
    edge_size=128,
    resolution=10,
    # query={"eo:cloud_cover": {"lt": 40}}
)

facet = (da.sel(band=["B04","B03","B02"])/2000).clip(0,1).plot.imshow(col="time",col_wrap = 5)
# show facet

facet.fig.savefig("test.png")

# Prepare the data to be used in the model, eelect just one sample 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_s2_numpy = (da[11].compute().to_numpy() / 10_000).astype("float32")
X = torch.from_numpy(original_s2_numpy).float().to(device)

# Apply model
superX = model(X[None]).squeeze(0).numpy()
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

