"""
Fetch ArcGIS Wayback imagery tiles for N spatial targets sampled inside a GeoJSON polygon,
filtered by cloud cover percentage and Shannon entropy.

python /home/susanket/esri-vlm/preprocess/fetch-images.py --geojson /data/susanket/data/global/world.geojson --grid-shp /data/susanket/data/global/demo-sampling/grids.shp -n 1000000 --out /data/susanket/vlm/images/image-06.parquet --mapping-out /data/susanket/vlm/map.parquet

Data Preprocessing Ratios:
- 60% Single Images: Fixed zoom 18, year 2026 (release 26334)
- 40% Multi-Group (Multi-Scale & Multi-Temporal):
    * Multi-Scale Zoom distribution: z14 (10%), z15 (20%), z16 (25%), z17 (20%), z18 (25%)
    * Multi-Temporal: Minimum of 2 distinct release IDs per location from:
        - 3026 (2014)
        - 239 (2018)
        - 45134 (2022)
        - 26334 (2026)

Filtering Rules:
1. Cloud Cover: Discard images with > 5% cloud cover.
2. Entropy Survival:
   - Entropy < 4.0: Discard (0% survival)
   - 4.0 <= Entropy <= 6.0: 25% survival chance
   - Entropy > 6.0: Always keep (100% survival)

Output:
Saves saved image records into chunked parquet files (e.g. image-00.parquet, image-01.parquet).
"""

import argparse
import asyncio
import io
import math
import os
import random
import time
import uuid
from dataclasses import dataclass

os.environ['SHAPE_RESTORE_SHX'] = "YES"

import aiohttp
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from shapely.geometry import Point

# Release ID to Year Mapping
WAYBACK_RELEASES = {
    3026: 2014,
    239: 2018,
    45134: 2022,
    26334: 2026,
}
RELEASE_IDS = list(WAYBACK_RELEASES.keys())

TILE_URL_TMPL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer/tile/{release_id}/{zoom}/{y}/{x}"
)


# --------------------------------------------------------------------------
# Image Analysis Functions
# --------------------------------------------------------------------------
def compute_image_entropy(img: Image.Image) -> float:
    """Calculates Shannon entropy for a PIL Image (0 to 8 bits)."""
    try:
        gray_img = img.convert("L")
        histogram = gray_img.histogram()
        total_pixels = sum(histogram)

        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)
        return entropy
    except Exception:
        return 0.0


def compute_cloud_cover(img: Image.Image, brightness_thresh=200, sat_thresh=35) -> float:
    """Calculates Cloud Percentage (%) based on RGB brightness and saturation."""
    try:
        rgb_img = img.convert("RGB")
        img_np = np.array(rgb_img, dtype=np.float32)

        R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        luminance = (R + G + B) / 3.0

        max_c = np.maximum(R, np.maximum(G, B))
        min_c = np.minimum(R, np.minimum(G, B))
        saturation = np.where(max_c == 0, 0, (max_c - min_c) / max_c) * 255.0

        cloud_mask = (luminance >= brightness_thresh) & (saturation <= sat_thresh)
        cloud_pct = float((np.sum(cloud_mask) / cloud_mask.size) * 100.0)
        return cloud_pct
    except Exception:
        return 100.0  # Treat corrupt/unreadable images as full cloud to drop them


def evaluate_image_survival(image_bytes: bytes) -> tuple[bool, float, float]:
    """Applies Cloud and Entropy filters to determine if an image should survive."""
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # 1. Cloud Cover Check (> 5% discarded)
        cloud_pct = compute_cloud_cover(img)
        if cloud_pct > 5.0:
            return False, cloud_pct, 0.0

        # 2. Shannon Entropy Check
        entropy = compute_image_entropy(img)

        if entropy < 4.0:
            return False, cloud_pct, entropy  # 0% survival
        elif 4.0 <= entropy <= 6.0:
            survives = random.random() < 0.25  # 25% survival chance
            return survives, cloud_pct, entropy
        else:
            return True, cloud_pct, entropy  # > 6.0 always survives

    except Exception:
        return False, 100.0, 0.0


# --------------------------------------------------------------------------
# Sampling Points Uniformly per Grid Cell
# --------------------------------------------------------------------------
def sample_points_in_geojson_grid(geojson_path: str, grid_path: str, total_n_points: int) -> np.ndarray:
    print(f"Loading GeoJSON boundary from {geojson_path}...")
    poly_gdf = gpd.read_file(geojson_path)
    if poly_gdf.crs is not None and poly_gdf.crs.to_epsg() != 4326:
        poly_gdf = poly_gdf.to_crs(epsg=4326)
    region_poly = poly_gdf.geometry.unary_union

    print(f"Loading Grid Shapefile from {grid_path}...")
    grid_gdf = gpd.read_file(grid_path)
    if grid_gdf.crs is not None and grid_gdf.crs.to_epsg() != 4326:
        grid_gdf = grid_gdf.to_crs(epsg=4326)

    print("Computing grid cell intersections...")
    valid_cells = grid_gdf[grid_gdf.geometry.intersects(region_poly)].copy()
    num_cells = len(valid_cells)

    if num_cells == 0:
        raise ValueError("No grid cells from the shapefile intersect with the provided GeoJSON polygon!")

    pts_per_cell = math.ceil(total_n_points / num_cells)
    print(f"Intersecting grid cells: {num_cells:,}")
    print(f"Targeting ~{pts_per_cell:,} samples per grid cell to reach ~{total_n_points:,} total points...")

    all_sampled_coords = []

    for idx, cell in enumerate(valid_cells.geometry):
        intersection = cell.intersection(region_poly)
        if intersection.is_empty:
            continue

        min_x, min_y, max_x, max_y = intersection.bounds
        cell_coords = []
        batch_size = max(pts_per_cell * 3, 1000)

        while len(cell_coords) < pts_per_cell:
            rand_lons = np.random.uniform(min_x, max_x, size=batch_size)
            rand_lats = np.random.uniform(min_y, max_y, size=batch_size)

            pts = [Point(x, y) for x, y in zip(rand_lons, rand_lats)]
            valid_pts = [p for p in pts if intersection.contains(p)]

            for p in valid_pts:
                cell_coords.append((p.x, p.y))
                if len(cell_coords) == pts_per_cell:
                    break

        all_sampled_coords.extend(cell_coords)

        if len(all_sampled_coords) >= total_n_points:
            all_sampled_coords = all_sampled_coords[:total_n_points]
            break

        if (idx + 1) % max(1, num_cells // 10) == 0:
            print(f"  Processed {idx + 1}/{num_cells} grid cells ({len(all_sampled_coords):,} points sampled)...")

    sampled_array = np.array(all_sampled_coords, dtype=np.float64)
    print(f"Successfully sampled {len(sampled_array):,} spatially balanced points across {num_cells:,} grid cells.")
    return sampled_array


def sample_points_in_geojson_simple(geojson_path: str, n_points: int) -> np.ndarray:
    print(f"Loading GeoJSON from {geojson_path}...")
    gdf = gpd.read_file(geojson_path)

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    polygon = gdf.geometry.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds

    sampled_coords = []
    print(f"Sampling {n_points:,} random spatial centroids within polygon...")

    batch_size = max(n_points * 2, 10000)
    while len(sampled_coords) < n_points:
        rand_lons = np.random.uniform(min_x, max_x, size=batch_size)
        rand_lats = np.random.uniform(min_y, max_y, size=batch_size)

        pts = [Point(x, y) for x, y in zip(rand_lons, rand_lats)]
        valid_pts = [p for p in pts if polygon.contains(p)]

        for p in valid_pts:
            sampled_coords.append((p.x, p.y))
            if len(sampled_coords) == n_points:
                break

    return np.array(sampled_coords, dtype=np.float64)


# --------------------------------------------------------------------------
# Preprocessing Dataset Pipeline
# --------------------------------------------------------------------------
def generate_tile_fetch_requests(coords: np.ndarray):
    n_total = len(coords)
    n_multi = int(n_total * 0.60)   # 60% of the total imagery is mutli-resolution
    n_single = n_total - n_multi

    print(f"Dataset Preprocessing Plan:\n"
          f"  - Total centroids: {n_total:,}\n"
          f"  - Single image targets (60%): {n_single:,}\n"
          f"  - Multi-group targets (40%): {n_multi:,}")

    indices = np.random.permutation(n_total)
    single_indices = indices[:n_single]
    multi_indices = indices[n_single:]

    zoom_levels = [17, 18, 19, 20]#[14, 15, 16, 17, 18]
    zoom_probs = [0.1, 0.4, 0.4, 0.1]#[0.10, 0.20, 0.25, 0.20, 0.25]

    requests = []

    # 1. Single Images
    for idx in single_indices:
        lon, lat = coords[idx]
        gid = str(uuid.uuid4())
        z = 19
        rel_id = random.choice([26334, 22252, 13192, 51127, 58924])  # 2026/2025 release
        year = 2025 # WAYBACK_RELEASES[rel_id]

        x, y = latlon_to_tile(lat, lon, z)
        requests.append({
            "group_id": gid,
            "is_multi_group": False,
            "centroid_lon": lon,
            "centroid_lat": lat,
            "zoom": z,
            "release_id": rel_id,
            "year": year,
            "x": x,
            "y": y,
            "file_name": f"{rel_id}_{z}_{x}_{y}"
        })

    # 2. Multi-Scale & Multi-Temporal
    for idx in multi_indices:
        lon, lat = coords[idx]
        gid = str(uuid.uuid4())

        num_releases = np.random.randint(2, len(RELEASE_IDS) + 1)
        selected_releases = np.random.choice(RELEASE_IDS, size=num_releases, replace=False)

        num_zooms = np.random.randint(1, 4)
        selected_zooms = np.random.choice(zoom_levels, size=num_zooms, p=zoom_probs, replace=False)

        for z in selected_zooms:
            x, y = latlon_to_tile(lat, lon, z)
            for rel_id in selected_releases:
                year = WAYBACK_RELEASES[rel_id]
                requests.append({
                    "group_id": gid,
                    "is_multi_group": True,
                    "centroid_lon": lon,
                    "centroid_lat": lat,
                    "zoom": int(z),
                    "release_id": int(rel_id),
                    "year": int(year),
                    "x": int(x),
                    "y": int(y),
                    "file_name": f"{rel_id}_{z}_{x}_{y}"
                })

    print(f"Generated {len(requests):,} total tile requests from {n_total:,} centroids.")
    return requests


# --------------------------------------------------------------------------
# Tile Math & Tools
# --------------------------------------------------------------------------
def latlon_to_tile(lat: float, lon: float, zoom: int):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_center_lonlat(x: int, y: int, zoom: int):
    n = 2.0 ** zoom
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


# --------------------------------------------------------------------------
# Fetch Worker
# --------------------------------------------------------------------------
@dataclass
class FetchResult:
    key: str
    file_name: str
    content: bytes | None


async def fetch_one(session, sem, release_id, zoom, x, y, max_retries=4, timeout=15):
    url = TILE_URL_TMPL.format(release_id=release_id, zoom=zoom, y=y, x=x)
    file_name = f"{release_id}_{zoom}_{x}_{y}"
    key = f"{release_id}/{zoom}/{x}/{y}"

    async with sem:
        backoff = 1.0
        for _ in range(max_retries):
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return FetchResult(key, file_name, content)
                    elif resp.status == 404:
                        return FetchResult(key, file_name, None)
                    elif resp.status in (429, 503):
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    else:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
        return FetchResult(key, file_name, None)


async def fetch_batch(session, sem, batch):
    tasks = [
        fetch_one(session, sem, r["release_id"], r["zoom"], r["x"], r["y"])
        for r in batch
    ]
    return await asyncio.gather(*tasks)


def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def get_chunk_filename(base_path: str, chunk_idx: int) -> str:
    """Formats file path to 'image-00.parquet', 'image-01.parquet', etc.

    using the directory of the base_path argument.
    """
    dirname = os.path.dirname(base_path)
    filename = f"image-{chunk_idx:02d}.parquet"
    return os.path.join(dirname, filename) if dirname else filename


# --------------------------------------------------------------------------
# Main Execution Driver
# --------------------------------------------------------------------------
async def run(args):
    # 1. Sample points (using grid shapefile if provided)
    if args.grid_shp:
        coords = sample_points_in_geojson_grid(args.geojson, args.grid_shp, args.num_points)
    else:
        coords = sample_points_in_geojson_simple(args.geojson, args.num_points)

    # 2. Build multi-scale/multi-temporal preprocessing requests
    all_requests = generate_tile_fetch_requests(coords)

    mapping_table = pa.table({
        "group_id": [r["group_id"] for r in all_requests],
        "is_multi_group": [r["is_multi_group"] for r in all_requests],
        "centroid_lat": [r["centroid_lat"] for r in all_requests],
        "centroid_lon": [r["centroid_lon"] for r in all_requests],
        "zoom": [r["zoom"] for r in all_requests],
        "release_id": [r["release_id"] for r in all_requests],
        "year": [r["year"] for r in all_requests],
        "file_name": [r["file_name"] for r in all_requests],
    })
    pq.write_table(mapping_table, args.mapping_out)
    print(f"Wrote centroid-to-tile mapping -> {args.mapping_out}")

    # 3. Deduplicate tile fetching targets
    unique_tiles = {}
    for r in all_requests:
        key = (r["release_id"], r["zoom"], r["x"], r["y"])
        if key not in unique_tiles:
            unique_tiles[key] = r

    unique_list = list(unique_tiles.values())
    print(f"{len(unique_list):,} unique physical tiles to fetch out of {len(all_requests):,} total target requests.")

    # 4. Checkpoint & Resume setup
    ckpt_path = args.out + ".ckpt"
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming: {len(done):,} tiles already fetched, skipping them.")

    todo = [r for r in unique_list if f"{r['release_id']}/{r['zoom']}/{r['x']}/{r['y']}" not in done]
    print(f"{len(todo):,} tiles remaining to download.")

    schema = pa.schema([
        ("group_id", pa.string()),
        ("is_multi_group", pa.bool_()),
        ("zoom", pa.int32()),
        ("release_id", pa.int32()),
        ("year", pa.int32()),
        ("tile_lat", pa.float64()),
        ("tile_lon", pa.float64()),
        ("file_name", pa.string()),
        ("image_bytes", pa.binary()),
        ("cloud_pct", pa.float32()),
        ("entropy", pa.float32()),
    ])

    # Dynamic Parquet Chunking Setup
    chunk_idx = 0
    records_in_current_chunk = 0
    current_chunk_path = get_chunk_filename(args.out, chunk_idx)
    
    # Handle resuming into current chunk or rolling over
    while os.path.exists(current_chunk_path):
        meta = pq.read_metadata(current_chunk_path)
        if meta.num_rows < args.chunk_size:
            records_in_current_chunk = meta.num_rows
            break
        chunk_idx += 1
        current_chunk_path = get_chunk_filename(args.out, chunk_idx)

    writer = pq.ParquetWriter(current_chunk_path, schema)
    print(f"Writing dataset chunks to: {current_chunk_path} (Starting at {records_in_current_chunk:,} records)")

    # 5. Async Fetch Loop
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    sem = asyncio.Semaphore(args.concurrency)

    n_fetched = 0
    n_saved = 0
    n_discarded = 0
    n_missing = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            results = await fetch_batch(session, sem, batch)

            res_dict = {r.key: r for r in results}

            group_id_col, multi_col, zoom_col, rel_col, year_col = [], [], [], [], []
            lat_col, lon_col, name_col, bytes_col = [], [], [], []
            cloud_col, entropy_col = [], []

            with open(ckpt_path, "a") as ckpt_f:
                for req in batch:
                    key = f"{req['release_id']}/{req['zoom']}/{req['x']}/{req['y']}"
                    res = res_dict.get(key)

                    if res is None or res.content is None:
                        n_missing += 1
                        ckpt_f.write(key + "\n")
                        continue

                    n_fetched += 1

                    # Apply Cloud and Entropy Filtering
                    survives, cloud_pct, entropy = evaluate_image_survival(res.content)

                    if not survives:
                        n_discarded += 1
                        ckpt_f.write(key + "\n")
                        continue

                    lon, lat = tile_center_lonlat(req["x"], req["y"], req["zoom"])

                    group_id_col.append(req["group_id"])
                    multi_col.append(req["is_multi_group"])
                    zoom_col.append(req["zoom"])
                    rel_col.append(req["release_id"])
                    year_col.append(req["year"])
                    lat_col.append(lat)
                    lon_col.append(lon)
                    name_col.append(req["file_name"])
                    bytes_col.append(res.content)
                    cloud_col.append(cloud_pct)
                    entropy_col.append(entropy)

                    ckpt_f.write(key + "\n")
                    n_saved += 1
                    records_in_current_chunk += 1

                    # Check chunk size limit (1 Million items)
                    if records_in_current_chunk >= args.chunk_size:
                        # Write current accumulator batch before switching files
                        if name_col:
                            batch_table = pa.table({
                                "group_id": group_id_col,
                                "is_multi_group": multi_col,
                                "zoom": zoom_col,
                                "release_id": rel_col,
                                "year": year_col,
                                "tile_lat": lat_col,
                                "tile_lon": lon_col,
                                "file_name": name_col,
                                "image_bytes": bytes_col,
                                "cloud_pct": cloud_col,
                                "entropy": entropy_col,
                            }, schema=schema)
                            writer.write_table(batch_table)
                            # Reset column lists
                            group_id_col, multi_col, zoom_col, rel_col, year_col = [], [], [], [], []
                            lat_col, lon_col, name_col, bytes_col = [], [], [], []
                            cloud_col, entropy_col = [], []

                        # Close current parquet chunk and open next
                        writer.close()
                        chunk_idx += 1
                        current_chunk_path = get_chunk_filename(args.out, chunk_idx)
                        writer = pq.ParquetWriter(current_chunk_path, schema)
                        records_in_current_chunk = 0
                        print(f"\n[Chunk Limit Reached] Rolled over to: {current_chunk_path}")

            # Write remaining buffered batch to current chunk
            if name_col:
                batch_table = pa.table({
                    "group_id": group_id_col,
                    "is_multi_group": multi_col,
                    "zoom": zoom_col,
                    "release_id": rel_col,
                    "year": year_col,
                    "tile_lat": lat_col,
                    "tile_lon": lon_col,
                    "file_name": name_col,
                    "image_bytes": bytes_col,
                    "cloud_pct": cloud_col,
                    "entropy": entropy_col,
                }, schema=schema)
                writer.write_table(batch_table)

            elapsed = time.time() - t0
            rate = n_fetched / elapsed if elapsed > 0 else 0
            print(f"  [{i + len(batch):,}/{len(todo):,}] fetched={n_fetched:,} "
                  f"saved={n_saved:,} discarded={n_discarded:,} missing={n_missing:,} "
                  f"rate={rate:.1f} tiles/s", end="\r")

    writer.close()
    print()

    print(f"Done. {n_saved:,} total tiles saved across {chunk_idx + 1} chunk file(s), "
          f"{n_discarded:,} discarded by filters, {n_missing:,} missing/404.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", required=True, help="Path to input GeoJSON polygon file")
    p.add_argument("--grid-shp", help="Path to shapefile containing global grid polygons")
    p.add_argument("--num-points", "-n", type=int, default=10000, help="Number of points to sample across the region")
    p.add_argument("--out", required=True, help="Base path for tile images (e.g., ./data/tiles.parquet)")
    p.add_argument("--mapping-out", required=True, help="Output parquet mapping every centroid request to its file_name")
    p.add_argument("--chunk-size", type=int, default=1000000, help="Maximum number of saved images per Parquet chunk file")
    p.add_argument("--concurrency", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=50000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))