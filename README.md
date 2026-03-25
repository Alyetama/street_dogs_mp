# Street Dogs Mapillary

This script fetches image metadata and downloads images from the Mapillary API within a specified geographic bounding box. It specifically filters the data to locate and download images that Mapillary's computer vision has tagged with `animal--ground-animal`.

The script uses multi-threading for speed and implements a robust checkpointing system so you don't lose progress if a large download gets interrupted.

## Prerequisites

1.  **Python 3.x**
2.  **Mapillary API Key:** You need a developer token from Mapillary. Create a `.env` file in the same directory as the script and add your key:
    ```env
    MLY_KEY=your_mapillary_access_token_here
    ```

## Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the script, open `mp_api.py` and modify the following constants at the top of the file to suit your needs:

* **Coordinates (`WEST, SOUTH, EAST, NORTH`):** These define the geographic bounding box (in longitude and latitude) that the script will scan. 
* **`REGION_NAME`:** A string used to name your output files and directories (e.g., `'valdivia', 'south_america', etc.`).
* **`MAX_WORKERS`:** This dictates the number of concurrent threads used for making API requests and downloading images. 
    * *Default is 50.* * Increase this for faster downloads if you have a strong network connection. 
    * Decrease this (e.g., to 10 or 20) if you are encountering `429 Too Many Requests` errors or if your network is struggling.

## Usage

Once your `.env` file is set up and your coordinates are configured, run the script via the command line:

```bash
python mp_api.py
```

## Outputs

After a successful run, the script will generate the following in a directory named `<REGION_NAME>`:
* `topology_checkpoint_<REGION_NAME>.json`: A cached map of image IDs to sequence IDs.
* `metadata_checkpoint_<REGION_NAME>.jsonl`: Cached metadata for all found images.
* `all_data_<REGION_NAME>.csv`: A CSV containing the metadata for *every* image found in the bounding box.
* `ground_animals_<REGION_NAME>.csv`: A filtered CSV containing only metadata for images featuring ground animals.
* `ground_animals_<REGION_NAME>.json`: GeoJSON-like features of the ground animal detections.
* `ground_animals_<REGION_NAME>/`: A folder containing the actual downloaded `.jpg` images of the animals.
```

