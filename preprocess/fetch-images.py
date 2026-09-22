"""
Fetch ArcGIS Wayback imagery tiles for N spatial targets sampled inside a GeoJSON polygon,
balanced evenly across grid cells from an input Grid Shapefile.

Data Preprocessing Ratios:
- 60% Single Images: Fixed zoom 18, year 2026 (release 26334)
- 40% Multi-Group (Multi-Scale & Multi-Temporal):
    * Multi-Scale Zoom distribution: z14 (10%), z15 (20%), z16 (25%), z17 (20%), z18 (25%)
    * Multi-Temporal: Minimum of 2 distinct release IDs per location from:
        - 3026 (2014)
        - 239 (2018)
        - 45134 (2022)
        - 26334 (2026)

Usage:
    python fetch_wayback_tiles.py \
        --geojson world_poly.geojson \
        --grid-shp world_grid.shp \
        --num-points 10000 \
        --out tiles_dataset.parquet \
        --mapping-out centroid_tile_map.parquet \
        --concurrency 150 \
        --batch-size 50000
"""

import argparse
import asyncio
import math
import os
import time
import uuid
from dataclasses import dataclass

os.environ['SHAPE_RESTORE_SHX']="YES"

import aiohttp
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
# Sampling Points Uniformly per Grid Cell
# --------------------------------------------------------------------------
def sample_points_in_geojson_grid(geojson_path: str, grid_path: str, total_n_points: int) -> np.ndarray:
    """
    Reads a GeoJSON file and a Grid Shapefile. Intersects the target region with the grid cells,
    and uniformly samples roughly equal numbers of random (lon, lat) points from each grid cell.
    """
    print(f"Loading GeoJSON boundary from {geojson_path}...")
    poly_gdf = gpd.read_file(geojson_path)
    if poly_gdf.crs is not None and poly_gdf.crs.to_epsg() != 4326:
        poly_gdf = poly_gdf.to_crs(epsg=4326)
    region_poly = poly_gdf.geometry.unary_union

    print(f"Loading Grid Shapefile from {grid_path}...")
    grid_gdf = gpd.read_file(grid_path)
    if grid_gdf.crs is not None and grid_gdf.crs.to_epsg() != 4326:
        grid_gdf = grid_gdf.to_crs(epsg=4326)

    # Find grid cells that actually intersect with our target region polygon
    print("Computing grid cell intersections...")
    valid_cells = grid_gdf[grid_gdf.geometry.intersects(region_poly)].copy()
    num_cells = len(valid_cells)

    if num_cells == 0:
        raise ValueError("No grid cells from the shapefile intersect with the provided GeoJSON polygon!")

    # Calculate target samples per grid cell
    pts_per_cell = math.ceil(total_n_points / num_cells)
    print(f"Intersecting grid cells: {num_cells:,}")
    print(f"Targeting ~{pts_per_cell:,} samples per grid cell to reach ~{total_n_points:,} total points...")

    all_sampled_coords = []

    for idx, cell in enumerate(valid_cells.geometry):
        # The region inside this specific cell
        intersection = cell.intersection(region_poly)
        if intersection.is_empty:
            continue

        min_x, min_y, max_x, max_y = intersection.bounds
        cell_coords = []
        batch_size = max(pts_per_cell * 3, 1000)

        # Rejection sampling inside the cell's bounding box
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
        
        # Trim if we exceed total_n_points
        if len(all_sampled_coords) >= total_n_points:
            all_sampled_coords = all_sampled_coords[:total_n_points]
            break

        if (idx + 1) % max(1, num_cells // 10) == 0:
            print(f"  Processed {idx + 1}/{num_cells} grid cells ({len(all_sampled_coords):,} points sampled)...")

    sampled_array = np.array(all_sampled_coords, dtype=np.float64)
    print(f"Successfully sampled {len(sampled_array):,} spatially balanced points across {num_cells:,} grid cells.")
    return sampled_array


def sample_points_in_geojson_simple(geojson_path: str, n_points: int) -> np.ndarray:
    """Fallback uniform sampling without a grid shapefile."""
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
    """
    Splits input coordinates into 60% single and 40% multi-scale/multi-temporal requests.
    """
    n_total = len(coords)
    n_multi = int(n_total * 0.40)
    n_single = n_total - n_multi

    print(f"Dataset Preprocessing Plan:\n"
          f"  - Total centroids: {n_total:,}\n"
          f"  - Single image targets (60%): {n_single:,}\n"
          f"  - Multi-group targets (40%): {n_multi:,}")

    indices = np.random.permutation(n_total)
    single_indices = indices[:n_single]
    multi_indices = indices[n_single:]

    zoom_levels = [14, 15, 16, 17, 18]
    zoom_probs = [0.10, 0.20, 0.25, 0.20, 0.25]

    requests = []

    # 1. Single Images
    for idx in single_indices:
        lon, lat = coords[idx]
        gid = str(uuid.uuid4())
        z = 18
        rel_id = 26334  # 2026 release
        year = WAYBACK_RELEASES[rel_id]

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


def _merge_parquet(main_path, part_path):
    main_table = pq.read_table(main_path)
    part_table = pq.read_table(part_path)
    combined = pa.concat_tables([main_table, part_table])
    tmp_out = main_path + ".merged"
    pq.write_table(combined, tmp_out)
    os.replace(tmp_out, main_path)
    os.remove(part_path)


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
    ])

    write_mode_new = not os.path.exists(args.out)
    writer = pq.ParquetWriter(args.out, schema) if write_mode_new else None
    if writer is None:
        part_path = args.out + ".resume_part"
        writer = pq.ParquetWriter(part_path, schema)
    else:
        part_path = None

    # 5. Async Fetch Loop
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    sem = asyncio.Semaphore(args.concurrency)

    n_fetched = 0
    n_missing = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            results = await fetch_batch(session, sem, batch)

            res_dict = {r.key: r for r in results}

            group_id_col, multi_col, zoom_col, rel_col, year_col = [], [], [], [], []
            lat_col, lon_col, name_col, bytes_col = [], [], [], []

            with open(ckpt_path, "a") as ckpt_f:
                for req in batch:
                    key = f"{req['release_id']}/{req['zoom']}/{req['x']}/{req['y']}"
                    res = res_dict.get(key)

                    if res is None or res.content is None:
                        n_missing += 1
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

                    ckpt_f.write(key + "\n")
                    n_fetched += 1

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
                }, schema=schema)
                writer.write_table(batch_table)

            elapsed = time.time() - t0
            rate = n_fetched / elapsed if elapsed > 0 else 0
            print(f"  [{i + len(batch):,}/{len(todo):,}] fetched={n_fetched:,} "
                  f"missing={n_missing:,} rate={rate:.1f} tiles/s", end="\r")

    writer.close()
    print()

    if part_path is not None:
        _merge_parquet(args.out, part_path)

    print(f"Done. {n_fetched:,} tiles saved, {n_missing:,} missing/404 -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", required=True, help="Path to input GeoJSON polygon file")
    p.add_argument("--grid-shp", help="Path to shapefile containing global grid polygons")
    p.add_argument("--num-points", "-n", type=int, default=10000, help="Number of points to sample across the region")
    p.add_argument("--out", required=True, help="Output parquet path for tile images")
    p.add_argument("--mapping-out", required=True, help="Output parquet mapping every centroid request to its file_name")
    p.add_argument("--concurrency", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=50000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))