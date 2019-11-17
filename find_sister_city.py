#!/usr/bin/python
# find_sister_city.py

# flake8: noqa

import os
import geocoder
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import rasterio as rio
from rasterio.plot import plotting_extent
import rasterstats as rs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import earthpy.plot as ep



###################################################################
# TODO: 
    # add option to return analog city within a certain country, or to exclude
    # from same country/region

    # add humidity data, at least

    # switch to 0.5 deg. global data

    # dump all cities' extracted vals to a CSV that I can just read in


###################################################################


# get target city
target = input("\nWhat city's climate sister cities are you looking for?")
print("\nNow geocoding your location...\n")
g = geocoder.osm(target)
assert g.status == 'OK', ("Open Street Map geocoder was not successful! "
                          "Please try again.")
target_lat, target_lon = g.latlng
target_pt = Point(target_lon, target_lat)

# read in the cities
print("\nNow reading a database of world cities...\n")
world_cities = gpd.read_file('./data/World_Cities.shp')
cities = world_cities[['CITY_NAME', 'CNTRY_NAME', 'geometry']]

# add in the target city
target_row = {'CITY_NAME': [target],
              'CNTRY_NAME': ["?"],
           'geometry': [target_pt]}
target_gdf = gpd.GeoDataFrame(pd.DataFrame(target_row),
                              crs=cities.crs,
                              geometry='geometry')
cities = cities.append(target_gdf)


# read in climate data

# (code yanked from top response at
# https://gis.stackexchange.com/questions/223910/using-rasterio-or-gdal-to-
#stack-multiple-bands-without-using-subprocess-commands
"""
file_list =  ['./data/wc2-5/' + f for f in os.listdir(
                                        './data/wc2-5/') if f.endswith('bil')]

with rasterio.open(file_list[0]) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count = len(file_list))

# Read each layer and write it to stack
with rasterio.open('./data/wc2-5/stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(file_list, start=1):
        with rasterio.open(layer) as src1:
            dst.write_band(id, src1.read(1))
"""

print("\nNow reading a database of world climate...\n")
with rio.open('./data/wc2-5/stack.tif') as clim_src:
    clim = clim_src.read(masked=True)
    clim_meta = clim_src.profile

# extract raster values at points
bioclim_cols = []
for n in range(clim.shape[0]):
    city_clim = rs.point_query(cities,
                               clim[n, :, :],
                               affine=clim_meta['transform'],
                               nodata=-9999)
    bioclim_cols.append('bio%i' % (n + 1))
    cities.loc[:, 'bio%i' % (n + 1)] = city_clim

# drop NANs
cities = cities.dropna()

# standardize the climate columns
bioclim_data = cities.loc[:, bioclim_cols].values
bioclim_data = StandardScaler().fit_transform(bioclim_data)

# run climate PCA, add as columns to cities
print(("\nNow doing some data science (running PCA, "
       "then finding your city's nearest neighbors)...\n"))
pca = PCA(n_components=19)
pcs = pca.fit_transform(bioclim_data)

# find nearest neighbor of last row (i.e. target city)
kdt = KDTree(pcs[:-1,], leaf_size=30, metric='euclidean')
analog_idxs = kdt.query(pcs[-1].reshape((1, pcs.shape[1])), k=50)[1].ravel()
analogs = cities.iloc[analog_idxs][['CITY_NAME', 'CNTRY_NAME']]
print("Here are your top 50 climate sister cities:\n")
print(analogs)


# plot
fig, ax = plt.subplots(figsize=(10, 10))
ep.plot_bands(clim[0, :, :],
              extent=plotting_extent(clim_src),
              cmap='Greys',
              title='%s and its climate sister cities' % target,
              scale=False,
              ax=ax)
cities.iloc[analog_idxs, :].plot(ax=ax,
                                 marker='o',
                                 markersize=100,
                                 cmap='YlGn_r')
cities[cities.CITY_NAME == target].plot(ax=ax,
                                        marker='*',
                                        markersize=100,
                                        color='purple')
ax.set_axis_off()
plt.show()

