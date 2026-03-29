## Short description of the different variables

Target variablei:

- wildfires_25yrs : Target variable is the wildfire probability
  expressed as the ratio between the number of occurred wildfires
  divided by the number of years (here 25 years)

Potentially predicitve features:

- NDVI_mean_march and NDVI_mean_aug: Mean Normalized Difference
  vegetation Index (NDVI) for a day in March and a day in August. The
  data was obtained from the Land Monitoring Service of Copernicus. In
  detail, the NDVI values from Copernicus were downloaded for always the
  same day in the years 2000, 2010, 2019, and then averaged across the
  three years.
- mean_temp, mean_precipitation and max_temp_aug: All three variables
  are obtained from PRISM (https://prism.oregonstate.edu/recent/), where
  datasets on annual average temperatures and precipitation with a
  resolution of 4 x 4 km are freely available. In detail, the three
  variables represent the averaged temperature/precipitation between the
  years 2000 to 2024. As August is the warmest month in Southern
  California, also the mean maximum temperature of August calculated
  from the years 2000, 2010, and 2020 is provided.
- LULC_2019: The averaged land use and land cover (LULC) between 2000
  and 2025 could not be calculated , thus as a proxy, for the temporal
  avg. LULC, the LULC for the year 2019 is provided. The data was taken
  from the National Land Cover Database (NLCD)
- slope and aspect: The inclination (slope) and orientation (aspect) of
  a terrain are derived from a Digital Terrain Model (DTM), namely the
  GMTED2010 from the U.S. Geological Survey and Center, 2010, with a
  spatial resolution of 1 km
- soil_silt: the silt fraction of the soil might pose some predictive
  power when it comes to wildfire risk. The raster data was retrieved
  from soilgrids.org
- 1_road_dist: Describes the distance to the next closest road