import csv
import requests
import zipfile
import pandas as pd
import io
import re
from collections import defaultdict
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: READ STATION LIST AND SELECT SCHLESWIG-HOLSTEIN
# ============================================================

# Read station metadata from CSV file
stations = []

with open("stations.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # Read as dictionary for easy column access
    for row in reader:
        # Pad station IDs with leading zeros to ensure 5-digit format
        row["Stations_id"] = row["Stations_id"].zfill(5)
        stations.append(row)

# Extract station IDs for Schleswig-Holstein region
schleswig_holstein_station_ids = [
    row["Stations_id"]
    for row in stations
    if row["Bundesland"] == "Schleswig-Holstein"
]

# Define a specific subset of station IDs to keep (filtered list)
ids_to_keep = {'00788', '01200', '01379', '01963', '02115', '02303', '02429', '02564', '02907',
               '02961', '03032', '03086', '03897', '04039', '04393', '04466', '04919',
               '05078', '05516', '05877', '05930', '06163'}

# Filter to only keep stations in the predefined list
schleswig_holstein_station_ids = [
    i for i in schleswig_holstein_station_ids
    if i in ids_to_keep
]

print(f"Number of Schleswig-Holstein stations: {len(schleswig_holstein_station_ids)}")

# Convert station list to DataFrame for easier manipulation
stations_df = pd.DataFrame(stations)

# Keep only selected Schleswig-Holstein stations
stations_df = stations_df[
    stations_df["Stations_id"].isin(schleswig_holstein_station_ids)
]

# Convert coordinate columns to numeric type, coercing errors to NaN
stations_df["geoBreite"] = pd.to_numeric(
    stations_df["geoBreite"], errors="coerce")
stations_df["geoLaenge"] = pd.to_numeric(
    stations_df["geoLaenge"], errors="coerce")
stations_df["Stationshoehe"] = pd.to_numeric(
    stations_df["Stationshoehe"], errors="coerce")

# Drop rows with missing coordinate data
stations_df = stations_df.dropna(subset=["geoBreite", "geoLaenge"])

# Create a GeoDataFrame with point geometries for mapping
stations_geometry = [Point(lon, lat) for lon, lat in zip(
    stations_df["geoLaenge"], stations_df["geoBreite"])]
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=stations_geometry,
    crs="EPSG:4326"  # WGS84 coordinate system (latitude/longitude)
)

# Load NUTS-2 regional boundaries shapefile
nuts2 = gpd.read_file("map_de/nuts/NUTS250_N2.shp")

# Select the Schleswig-Holstein region using its NUTS code
nuts2_sh = nuts2[nuts2["NUTS_CODE"] == "DEF0"]

# Define a Lambert projection suitable for Germany (UTM zone 32N in meters)
lambert_crs = "EPSG:25832"

# Convert both station points and regional boundaries to Lambert coordinates
stations_gdf_lambert = stations_gdf.to_crs(lambert_crs)
nuts2_sh_lambert = nuts2_sh.to_crs(lambert_crs)

# Create a high-quality plot with specified dimensions and resolution
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Plot regional boundaries (outline only, no fill)
nuts2_sh_lambert.boundary.plot(
    ax=ax,
    color="gray",  # Boundary line color
    edgecolor="black",
    linewidth=0.8,
    alpha=0.5
)

# Plot weather stations as points
stations_gdf_lambert.plot(
    ax=ax,
    markersize=80,  # Increased size for better visibility
    color="#324AB2",
    edgecolor="k",
    linewidth=0.8,
    alpha=0.9,
    label="Weather Stations"
)

# Add station ID labels next to each point
for idx, row in stations_gdf_lambert.iterrows():
    ax.annotate(
        text=row["Stations_id"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(5, 5),  # Offset from point in pixels
        textcoords="offset points",
        fontsize=14,
        color="#C71585",
        fontweight='bold'
    )

# Set axis labels and title
ax.set_xlabel("Easting (m, EPSG:25832)", fontsize=12)
ax.set_ylabel("Northing (m, EPSG:25832)", fontsize=12)
ax.set_title(
    "DWD Wind Stations in Schleswig-Holstein", 
    fontsize=25,
    fontweight='bold'
)

# Add a semi-transparent grid for better orientation
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)

# Add a legend to identify station points
#ax.legend(loc='upper right', fontsize=18)

# Maintain equal aspect ratio to prevent map distortion
ax.set_aspect('equal')

# Remove axis ticks while keeping labels
ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)

# Adjust layout with padding to prevent clipping
plt.tight_layout(pad=2.0)
# Turn off axis lines for cleaner appearance
ax.set_axis_off()

# Save the map as a high-quality PDF file
fig.savefig(
    'map.pdf',
    dpi=300,
    bbox_inches='tight',  # Ensure all elements fit within saved area
    format='pdf',
    facecolor='white',  # White background
    edgecolor='none'
)

# ============================================================
# STEP 2: LIST ALL ZIP FILES ON DWD SERVER
# ============================================================

# Define base URL for DWD hourly wind data
BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/wind/historical/"
)

# Download and parse HTML page to find all ZIP file links
html = requests.get(BASE_URL).text
# Use regex to extract ZIP filenames (historical wind data)
zip_files = re.findall(r'href="(stundenwerte_FF_.*?_hist\.zip)"', html)

print(f"Found {len(zip_files)} ZIP files.")


# ============================================================
# STEP 3: MAP ZIP FILES TO STATION IDS
# ============================================================

# Create a dictionary mapping station IDs to their associated ZIP files
station_zip_map = defaultdict(list)

for z in zip_files:
    # Extract station ID from filename (third underscore-separated element)
    station_id = z.split("_")[2]
    station_zip_map[station_id].append(z)


# ============================================================
# STEP 4: DOWNLOAD, READ & EXTRACT MAX WIND SPEED (FF) DATA
# ============================================================

all_data = []

# Process each selected station
for station_id in schleswig_holstein_station_ids:
    print(f"\nProcessing station {station_id}...")

    if station_id not in station_zip_map:
        print("  No FF data available.")
        continue

    # Download and process each ZIP file for the current station
    for zip_name in station_zip_map[station_id]:
        zip_url = BASE_URL + zip_name

        try:
            response = requests.get(zip_url)
            response.raise_for_status()  # Check for HTTP errors

            # Open ZIP archive from memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the data file inside the ZIP
                txt_files = [
                    f for f in z.namelist()
                    if f.startswith("produkt_ff_stunde") and f.endswith(".txt")
                ]

                # Read each data file (usually only one per ZIP)
                for txt in txt_files:
                    with z.open(txt) as f:
                        df = pd.read_csv(
                            f,
                            sep=";",
                            encoding="ISO-8859-1",
                            na_values="-999"  # DWD missing value code
                        )

                        # Keep only timestamp and wind speed columns
                        df = df[["MESS_DATUM", "   F"]]

                        # Rename wind speed column for clarity
                        df = df.rename(columns={"   F": "FF"})

                        # Remove rows with missing or negative wind speeds
                        df = df[df["FF"].notna()]
                        df = df[df["FF"] >= 0]

                        # Add station ID column
                        df["Stations_id"] = station_id

                        all_data.append(df)

        except Exception as e:
            print(f"  Error reading {zip_name}: {e}")


# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

# Combine all station data into a single DataFrame
wind_ff_df = pd.concat(all_data, ignore_index=True)

# Preview the data
print(wind_ff_df.head())

# Reshape to matrix format: rows=timestamps, columns=stations
wind_ff_matrix = wind_ff_df.pivot_table(
    index="MESS_DATUM",
    columns="Stations_id",
    values="FF",
    aggfunc="first"   # No aggregation needed (already hourly values)
)

# Convert timestamp strings to datetime objects
wind_ff_matrix.index = pd.to_datetime(
    wind_ff_matrix.index.astype(str),
    format="%Y%m%d%H"
)

# Filter for specific time period (September 2013 to March 2018)
wind_ff_2008_2024 = wind_ff_matrix.loc[
    (wind_ff_matrix.index >= "2013-09-01 00:00") &
    (wind_ff_matrix.index <= "2018-03-31 23:00")
]

# Save the final matrix to CSV
wind_ff_2008_2024.to_csv(
    "hourly_ff_schleswig_holstein_matrix.csv"
)