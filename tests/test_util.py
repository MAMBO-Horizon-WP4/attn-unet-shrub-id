import geopandas as gpd
from shrubnet.utils import raster_to_geodataframe


def test_raster_to_gdf(mask_path):
    gdf = raster_to_geodataframe(mask_path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf.geometry)
    assert len(gdf.total_bounds)
