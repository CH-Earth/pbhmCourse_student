import geospatialtools.terrain_tools as terrain_tools
import time
import pandas as pd
import scipy.stats
import datetime
import geopandas as gpd
import netCDF4 as nc
import rasterio
import pickle
import os
import numpy as np
import copy
from pathlib import Path

#Let's define the network database
def assemble_river_network_and_height_bands(self,tda,dh):
    self.channel_threshold = tda #m2
    self.calculate_drainage_area()
    self.delineate_river_network()
    self.delineate_basins()
    self.calculate_reach_properties()
    self.calculate_height_above_nearest_drainage()
    self.basins[self.hand == -9999] = -9999
    self.dh = dh #meters
    self.discretize_hand()
    self.channels_vector = self.channels_vector.set_crs("EPSG:4326")
    self.channels_vector_reproj = self.channels_vector.to_crs("+proj=laea +lon_0=-120.234375 +lat_0=49.5878293 +datum=WGS84 +units=m +no_defs")
    self.reach_length = []
    self.stream_length = 0
    self.reach_lat = []
    self.reach_lon = []
    self.ngru = np.max(np.unique(self.basins))
    self.nhru = np.max(np.unique(self.hbands))
    for geom in self.channels_vector_reproj:
     self.stream_length += geom.length
     self.reach_length.append(geom.length)
    for geom in self.channels_vector:
     self.reach_lon.append(geom.centroid.x)
     self.reach_lat.append(geom.centroid.y)
     self.reach_length = np.array(self.reach_length)
    self.reach_lat = np.array(self.reach_lat)
    self.reach_lon = np.array(self.reach_lon)
    self.reach_slope = self.db_channels['slope'][0:self.ngru]

    return self

def polygonize_gru_and_hru_maps(bow,sdir):

 #Output GRU map to file
 ifile = 'data/dem.tif'
 ofile = '%s/grus.tif' % sdir
 src = rasterio.open(ifile)
 fp = rasterio.open(ofile,'w',**src.profile)
 fp.write(bow.basins.astype(np.float32),1)
 fp.close()

 #Polygonize the GRU raster
 os.system('rm -f %s/grus.s*' % sdir)
 os.system('rm -f %s/grus.p*' % sdir)
 os.system('rm -f %s/grus.d*' % sdir)
 os.system('gdal_polygonize.py -q -8 %s/grus.tif %s/grus.shp grus ID' % (sdir,sdir))

 #Update GRU shapefile with elevation data
 gru_shp = gpd.read_file('%s/grus.shp' % sdir)
 gru_shp = gru_shp.dissolve(by='ID')
 for i, row in gru_shp.iterrows():
  gru = gru_shp.iloc[i-1]
  m = bow.basins == i
  if np.sum(m) == 0:continue
  area = bow.dx**2*np.sum(m) #m2 (fixed dx bad assumption...)
  gru_shp.loc[i,'GRU_area'] = area
  gru_shp.loc[i,'GRU_elev'] = np.nanmean(bow.dem[m])
 gru_shp.to_file("%s/grus.shp" % sdir)

 #Output HRU map to file
 ifile = 'data/dem.tif'
 ofile = '%s/hrus.tif' % sdir
 src = rasterio.open(ifile)
 fp = rasterio.open(ofile,'w',**src.profile)
 fp.write(bow.hbands.astype(np.float32),1)
 fp.close()

 #Polygonize the HRU raster
 os.system('rm -f %s/hrus.s*' % sdir)
 os.system('rm -f %s/hrus.p*' % sdir)
 os.system('rm -f %s/hrus.d*' % sdir)
 os.system('gdal_polygonize.py -q -8 %s/hrus.tif %s/hrus.shp hrus ID' % (sdir,sdir))

 #Add mean elevation and area to shapefile
 hru_shp = gpd.read_file('%s/hrus.shp' % sdir)
 hru_shp = hru_shp.dissolve(by='ID')
 for i, row in hru_shp.iterrows():
  hru = hru_shp.iloc[i-1]
  m = bow.hbands == i
  if np.sum(m) == 0:continue  
  area = bow.dx**2*np.sum(m) #m2 (fixed dx bad assumption...)
  hru_shp.loc[i,'HRU_area'] = area
  hru_shp.loc[i,'HRU_elev_m'] = np.nanmean(bow.dem[m])

 #Add the rest of missing info to the hrus shapefile
 gru_shp = gpd.read_file('%s/grus.shp' % sdir)
 gru_shp = gru_shp.dissolve(by='ID')
 #hru_shp = gpd.read_file('%s/hrus.shp' % sdir)
 #hru_shp = hru_shp.dissolve(by='ID')
 hru_shp['ID_source'] = -9999 #GRU ID
 hru_shp['ele_source'] = -9999 #GRU elevation
 hru_shp['lat_source'] = -9999 #centroid longitude of GRU
 hru_shp['lon_source'] = -9999 #centroid latitude of GRU
 hru_shp['ID_target'] = 0 #HRU ID
 hru_shp['weight'] = 1 #weight
 hru_shp['lat_target'] = -9999 #HRU latitude
 hru_shp['lon_target'] = -9999 #HRU longitude
 for i, row in hru_shp.iterrows():
    hru_shp.loc[i,'ID_target'] = i #HRU ID
    hru_shp.loc[i,'weight'] = 1 #weight
    hru_shp.loc[i,'lat_target'] = hru_shp.iloc[i-1].geometry.centroid.y #HRU latitude
    hru_shp.loc[i,'lon_target'] = hru_shp.iloc[i-1].geometry.centroid.x #HRU longitude
    #Retrieve GRU information
    m = bow.hbands == i
    j = np.unique(bow.basins[m])[0]
    hru_shp.loc[i,'ID_source'] = j # GRU ID
    hru_shp.loc[i,'lat_source'] = gru_shp.iloc[j-1].geometry.centroid.y  # GRU lat
    hru_shp.loc[i,'lon_source'] = gru_shp.iloc[j-1].geometry.centroid.x # GRU lon
    hru_shp.loc[i,'ele_source'] = gru_shp.iloc[j-1]['GRU_elev'] # GRU lon

 #Extract the soil type and veg type
 soiltype = rasterio.open('data/usda_soil_classes_from_soilgrids_v2.tif').read(1)
 #Gap fill soil type
 soiltype[soiltype == 0] = scipy.stats.mode(soiltype[soiltype != 0])[0]
 lctype = rasterio.open('data/igbp_land_use_from_modis_v2.tif').read(1)
 lctype[lctype == 0] = scipy.stats.mode(lctype[lctype != 0])[0]
 for i, row in hru_shp.iterrows():
  m = bow.hbands == i
  hru_shp.loc[i,'soil_type'] = scipy.stats.mode(soiltype[m])[0]
  veg_type = scipy.stats.mode(lctype[m])[0]
  if veg_type == 17:veg_type = 16
  hru_shp.loc[i,'veg_type'] = veg_type
 hru_shp.to_file("%s/hrus.shp" % sdir)

 return

def prepare_hru_forcing(bow,sdir):

 #Create a sample grid using the mask
 file = 'data/bow_20001001-20010930_rc_cmpd.nc' #hardcoded
 mask_file = '%s/grus.tif' % sdir
 fp = nc.Dataset(file)
 dx = 0.05 #hardcoded to match forcing resolution
 dy = 0.05 #hardcoded to match forcing resolution
 minlat = fp['lat'][0] - dy/2
 maxlat = fp['lat'][-1] + dy/2
 minlon = fp['lon'][0] - dx/2
 maxlon = fp['lon'][-1] + dx/2
 fp.close()
 file_coarse = '%s/meteo_coarse.tif' % sdir
 os.system('gdalwarp -q -overwrite -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s' % (dx,dy,minlon,minlat,maxlon,maxlat,mask_file,file_coarse))

 #Define the coarse and fine scale mapping
 maskij = rasterio.open(file_coarse).read(1)
 for i in np.arange(maskij.shape[0]):
  maskij[i,:] = np.arange(i*maskij.shape[1],(i+1)*maskij.shape[1])
 profile = rasterio.open(file_coarse).profile
 fp = rasterio.open('%s/meteo_coarse.tif' % sdir,'w',**profile)
 fp.write(np.flipud(maskij),1)
 fp.close()

 #Downscale mapping to HRU raster spatial resolution
 fp = rasterio.open('%s/grus.tif' % sdir,'r')
 dx = fp.res[0]
 dy = fp.res[1]
 minx = fp.bounds.left
 maxx = fp.bounds.right
 miny = fp.bounds.bottom
 maxy = fp.bounds.top
 #Regrid and downscale
 file_in = file_coarse
 file_out = '%s/meteo_fine.tif' % sdir
 os.system('gdalwarp -q -overwrite -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f %s %s' % (dx,dy,minx,miny,maxx,maxy,file_in,file_out))

 #Define the mapping directory
 mapping_info = {}

 #Read in the coarse and fine mapping
 file_coarse = '%s/meteo_coarse.tif' % sdir
 file_fine = '%s/meteo_fine.tif' % sdir
 file_dem = 'data/dem.tif'
 mask_coarse = rasterio.open(file_coarse).read(1)
 mask_fine = rasterio.open(file_fine).read(1)
 dem = rasterio.open(file_dem).read(1) # reading the DEM
 nlat = mask_coarse.shape[0]
 nlon = mask_coarse.shape[1]
    
 #Compute the mapping for each hru (hband)
 nhru = np.max(bow.hbands)
 for hru in np.arange(1,nhru+1):
  m = bow.hbands == hru # the ID of hru
  icells = np.unique(mask_fine[m][mask_fine[m] != -9999.0].astype(np.int)) # the location of forcing
  counts = np.bincount(mask_fine[m][mask_fine[m] != -9999.0].astype(np.int))
  coords,pcts,dem_coarse = [],[],[]
  dem_fine = np.mean(dem[m]) # average DEM of each hru
  for icell in icells: # loop oever the grid cells that intersect with the data
   ilat = int(np.floor(icell/mask_coarse.shape[1]))
   jlat = icell - ilat*mask_coarse.shape[1]
   pct = float(counts[icell])/float(np.sum(counts))
   coords.append([ilat,jlat])
   pcts.append(pct)
   dem_coarse.append(np.mean(dem[(mask_fine == icell) & (dem != -9999)])) # DEM value of forcing grid
  dem_coarse = np.array(dem_coarse) # pass the dem of forcing
  pcts = np.array(pcts)
  coords = list(np.array(coords).T)
  mapping_info[hru] = {'pcts':pcts,'coords':coords,'dem_coarse':dem_coarse,'dem_fine':dem_fine}
 
 #Read in all the meteorological data into memory (this is the easiest way for the assigment but you don't have to read it all of it in)
 vars = ['pptrate','airtemp','SWRadAtm','LWRadAtm','spechum','airpres','windspd']
 fpm = nc.Dataset(file)
 wrf_meteo = {}
 wrf_meteo['pptrate'] = (fpm['PREC_ACC_NC'][:] + fpm['PREC_ACC_C'][:])/3600.0
 wrf_meteo['airtemp'] = fpm['T2'][:] #K
 wrf_meteo['spechum'] = fpm['Q2'][:] #K
 wrf_meteo['airpres'] = fpm['P'][:] #Pa
 wrf_meteo['windspd'] = (fpm['U10'][:]**2 + fpm['V10'][:]**2)**0.5 #m/s
 wrf_meteo['SWRadAtm'] = fpm['SWDNB'][:] #W/m2
 wrf_meteo['LWRadAtm'] = fpm['LWDNB'][:] #W/m2
    
 #Create container for the data
 nt = 8760 #hard coded for one full water year
 meteorology = {}
 for var in vars:
  meteorology[var] = np.zeros((nt,nhru))

 #Downscale the meteorology
 wrf_meteo_ds = Downscale_Meteorology(wrf_meteo,mapping_info)

 #Extract data
 flag_downscale = True
 #flag_downscale = False
 for var in vars:
  for hru in mapping_info:
   pcts = mapping_info[hru]['pcts']
   if flag_downscale == False:
    coords = mapping_info[hru]['coords']
    #forcing_dem = mapping_info[hru]['dem_coarse']
    #hru_dem = mapping_info[hru]['dem_fine']
    coords[0][coords[0] >= wrf_meteo[var].shape[1]] = wrf_meteo[var].shape[1] - 1
    coords[1][coords[1] >= wrf_meteo[var].shape[2]] = wrf_meteo[var].shape[2] - 1
    tmp = wrf_meteo[var][:nt,coords[0],coords[1]]
   else:
    tmp = wrf_meteo_ds[hru][var]
   #if var is 'airtemp':
   # tmp = pcts*tmp - np.matlib.repmat( (hru_ele - forcing_ele) * 6.5*10^-3 , nt, 1)
   #else:
   # tmp = pcts*tmp
   #meteorology[var][:,hru-1] = np.sum(tmp,axis=1)
   meteorology[var][:,hru-1] = np.sum(pcts*tmp,axis=1)

 #Create datetime array
 dates = []
 date = datetime.datetime(2000,10,1,0) #hard coded for assignment
 while date <= datetime.datetime(2001,9,30,23):
  dates.append(date)
  date = date + datetime.timedelta(hours=1)
 dates = np.array(dates)

 #Read in hru shp for lat/lon info
 hru_shp = gpd.read_file('%s/hrus.shp' % sdir)

 #Create the forcing file
 file = '%s/forcing.nc' % sdir
 fp = nc.Dataset(file,'w')
 fp.createDimension('hru',nhru)
 fp.createDimension('time',nt)
 #define time step
 ptr = fp.createVariable('data_step','f8',fill_value=-999.)
 ptr[:] = 3600.0
 ptr.setncattr('long_name','data step length in seconds')
 ptr.setncattr('unit','s')
 #define time
 ptr = fp.createVariable('time','i4',('time',))
 ptr.units = 'hours since %4d-10-01' % dates[0].year
 ptr.calendar = 'standard'
 ptr.standard_name = 'time'
 ptr.axis = 'T'
 ptr.long_name = 'time'
 ptr[:] = nc.date2num(dates,units=ptr.units,calendar=ptr.calendar)
 #define lat
 ptr = fp.createVariable('lat','f8',('hru',))
 ptr.long_name = 'latitude'
 ptr.units = 'degrees_north'
 ptr.standard_name = 'latitude'
 ptr[:] = list(hru_shp.lat_target)[:]
 #define lon
 ptr = fp.createVariable('lon','f8',('hru',))
 ptr.long_name = 'longitude'
 ptr.units = 'degrees_east'
 ptr.standard_name = 'longitude'
 ptr[:] = list(hru_shp.lon_target)[:]
 #define hru id
 ptr = fp.createVariable('hruId','i4',('hru',))
 ptr.long_name = 'subbasin ID'
 ptr.units = '1'
 ptr[:] = range(1,nhru+1)
 #define pptrate
 ptr = fp.createVariable('pptrate','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'Precipitation rate'
 ptr.units = 'kg m**-2 s**-1'
 ptr[:] = meteorology['pptrate'][:,:]
 #define airtemp
 ptr = fp.createVariable('airtemp','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'air temperature at the measurement height'
 ptr.units = 'K'
 ptr[:] = meteorology['airtemp'][:,:]
 #define SWRadAtm
 ptr = fp.createVariable('SWRadAtm','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'downward shortwave radiation at the upper boundary'
 ptr.units = 'W m**-2'
 ptr[:] = meteorology['SWRadAtm'][:,:]
 #define LWRadAtm
 ptr = fp.createVariable('LWRadAtm','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'downward longwave radiation at the upper boundary'
 ptr.units = 'W m**-2'
 ptr[:] = meteorology['LWRadAtm'][:,:]
 #define spechum
 ptr = fp.createVariable('spechum','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'specific humidity at the measurement heigh'
 ptr.units = 'g g**-1'
 ptr[:] = meteorology['spechum'][:,:]
 #define pptrate
 ptr = fp.createVariable('airpres','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'air pressure at the measurement height'
 ptr.units = 'Pa'
 ptr[:] = meteorology['airpres'][:,:]
 #define pptrate
 ptr = fp.createVariable('windspd','f4',('time','hru'),fill_value=-9999)
 ptr.long_name = 'wind speed at the measurement height'
 ptr.units = 'm s**-1'
 ptr[:] = meteorology['windspd'][:,:]
 fp.close()

 return

def Downscale_Meteorology(db_data,mapping_info):
 
 #Iterate per hru
 db_org = {}
 db_ds = {}
 for hru in mapping_info:
  db_org[hru] = {}
  db_ds[hru] = {}
  #Collect the data
  for var in db_data:
   pcts = mapping_info[hru]['pcts']
   coords = mapping_info[hru]['coords']
   coords[0][coords[0] >= db_data[var].shape[1]] = db_data[var].shape[1] - 1
   coords[1][coords[1] >= db_data[var].shape[2]] = db_data[var].shape[2] - 1
   db_org[hru][var] = db_data[var][:,coords[0],coords[1]]
  df = mapping_info[hru]['dem_fine']
  dc = mapping_info[hru]['dem_coarse']
  #A.Downscale temperature
  dT = -6.0*10**-3*(df - dc)
  db_ds[hru]['airtemp'] = dT[np.newaxis,:] + db_org[hru]['airtemp']
  #db_ds[hru]['tair'] = db_org[hru]['tair'][:]
  #B.Downscale longwave
  #0.Compute radiative temperature 
  sigma = 5.67*10**-8
  emis = 1.0
  trad = (db_org[hru]['LWRadAtm']/sigma/emis)**0.25
  #1.Apply lapse rate to trad
  trad = dT[np.newaxis,:] + trad
  #2.Compute longwave with new radiative tempearture
  db_ds[hru]['LWRadAtm'] = emis*sigma*trad**4
  db_ds[hru]['LWRadAtm'] = db_org[hru]['LWRadAtm'][:] #OVERRIDE
  #C.Downscale pressure
  psurf = db_org[hru]['airpres'][:]*np.exp(-10**-3*(df-dc)/7.2)
  db_ds[hru]['airpres'] = psurf[:]
  db_ds[hru]['airpres'] = db_org[hru]['airpres'][:] #OVERRIDE
  #D.Downscale specific humidity
  #db_ds[hru]['spfh'] = db_org[hru]['spfh'][:]
  #Convert to vapor pressure
  e = db_org[hru]['airpres'][:]*db_org[hru]['spechum'][:]/0.622 #Pa
  esat = 1000*saturated_vapor_pressure(db_org[hru]['airtemp'][:] - 273.15) #Pa
  rh = e/esat
  esat = 1000*saturated_vapor_pressure(db_ds[hru]['airtemp'][:] - 273.15) #Pa
  e = rh*esat
  q = 0.622*e/db_ds[hru]['airpres']
  db_ds[hru]['spechum'] = q[:]
  db_ds[hru]['spechum'] = db_org[hru]['spechum'] #OVERRIDE
  #E.Downscale shortwave radiation
  db_ds[hru]['SWRadAtm'] = db_org[hru]['SWRadAtm'][:]
  #F.Downscale wind speed
  db_ds[hru]['windspd'] = db_org[hru]['windspd'][:]
  #G.Downscale precipitation
  db_ds[hru]['pptrate'] = db_org[hru]['pptrate'][:]

 return db_ds

def saturated_vapor_pressure(T):
    es = 0.6112*np.exp(17.67*T/(T + 243.5))
    return es

def prepare_routing_files(bow,sdir):

 ofile = '%s/network_topology.nc' % sdir
 gru_shp = gpd.read_file('%s/grus.shp' % sdir)
 ngru = np.array(list(gru_shp.GRU_area)).size
 os.system('rm -f %s' % ofile)
 fp = nc.Dataset(ofile,'w',format="NETCDF4")
 fp.createDimension('n',ngru)
 var = fp.createVariable('basin_area','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','basin_area')
 var.setncattr('unit','m*2')
 var[:] = list(gru_shp.GRU_area)
 var = fp.createVariable('length','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','length of the segment')
 var.setncattr('unit','m')
 var[:] = list(bow.reach_length)[0:-1]
 var = fp.createVariable('slope','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','slope of the segment')
 var.setncattr('unit','-')
 var[:] = bow.db_channels['slope'][0:ngru]
 var = fp.createVariable('lon','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','lon')
 var.setncattr('unit','-')
 var[:] = list(bow.reach_lon)[0:-1]
 var = fp.createVariable('lat','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','lat')
 var.setncattr('unit','-')
 var[:] = list(bow.reach_lat)[0:-1]
 var = fp.createVariable('hruid','i8',('n',),fill_value = -1)
 var.setncattr('long_name','hru id')
 var.setncattr('unit','-')
 var[:] = range(1,ngru+1)
 var = fp.createVariable('seg_id','i8',('n',),fill_value = -1)
 var.setncattr('long_name','seg id')
 var.setncattr('unit','-')
 var[:] = range(1,ngru+1)
 var = fp.createVariable('seg_hr_id','i8',('n',),fill_value = -1)
 var.setncattr('long_name','seg hr id')
 var.setncattr('unit','-')
 var[:] = range(1,ngru+1)
 var = fp.createVariable('tosegment','i8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','downstream segment')
 var.setncattr('unit','-')
 var[1:] = bow.topology[1:ngru]+1
 var = fp.createVariable('width','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','width')
 var.setncattr('unit','m')
 var[:] = 50 #meters
 var = fp.createVariable('manning','f8',('n',),fill_value = -9999.99)
 var.setncattr('long_name','manning')
 var.setncattr('unit','-')
 var[:] = 0.03
 fp.close()

 return

def prepare_summa_attribute_file(sdir):

 path_base = Path(sdir)
 # Specify where the shapefile can be found
 path_shp = Path(sdir)
 name_shp = 'hrus.shp'
 # Specify where the new file needs to go
 file_att = 'attributes.nc'
 # Specify where the forcing file is. We need this to ensure hru order is the same as in the forcing
 str_for = '%s/forcing.nc' % sdir
 # Create filename strings
 str_shp = str(path_shp / name_shp)
 str_att = str(path_base / file_att)
 # Open the shapefile
 shp = gpd.read_file(str_shp)
 # Open the source files and create the new .nc file
 with nc.Dataset(str_att, "w", format="NETCDF4") as att, nc.Dataset(str_for, "r") as forc:
    
    # Extract a bunch of stuff we'll need
    # ------------------------------------------------------------------------------------------
  
    # Find the number of HRU's we'll be dealing with
    num_gru = len(np.unique(shp['ID_source']))
    num_hru = len(shp)
    #print('Generating attributes for ' + str(num_gru) + "GRU's and " + str(num_hru) + ' HRU''s.')

    # Make the .nc file
    # ------------------------------------------------------------------------------------------
    
    # === Some general attributes
    att.setncattr('Author', "Created by W. Knoben")
    att.setncattr('History','Created ' + time.ctime(time.time()))
    att.setncattr('Purpose','Create attributes.nc file for SUMMA runs across North America on the MERIT Hydro basins')

    # === Define the dimensions 
    att.createDimension('hru',num_hru)
    att.createDimension('gru',num_gru) 
    
    # === Define the variables
    var = 'hruId'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index of hydrological response unit (HRU)')
    
    var = 'gruId'
    att.createVariable(var, 'i4', 'gru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index of grouped response unit (GRU)')
    
    var = 'hru2gruId'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index of GRU to which the HRU belongs')
    
    var = 'downHRUindex'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index of downslope HRU (0 = basin outlet)')
    
    var = 'longitude'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'Decimal degree east')
    att[var].setncattr('long_name', 'Longitude of HRU''s centroid')
    
    var = 'latitude'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'Decimal degree north')
    att[var].setncattr('long_name', 'Latitude of HRU''s centroid')
    
    var = 'elevation'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'm')
    att[var].setncattr('long_name', 'Elevation of HRU''s centroid')
    
    var = 'HRUarea'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'm^2')
    att[var].setncattr('long_name', 'Area of HRU')
    
    var = 'tan_slope'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'm m-1')
    att[var].setncattr('long_name', 'Average tangent slope of HRU')
    
    var = 'contourLength'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'm')
    att[var].setncattr('long_name', 'Contour length of HRU')
    
    var = 'slopeTypeIndex'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index defining slope')
    
    var = 'soilTypeIndex'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index defining soil type')
    
    var = 'vegTypeIndex'
    att.createVariable(var, 'i4', 'hru', fill_value = False)
    att[var].setncattr('units', '-')
    att[var].setncattr('long_name', 'Index defining vegetation type')
    
    var = 'mHeight'
    att.createVariable(var, 'f8', 'hru', fill_value = False)
    att[var].setncattr('units', 'm')
    att[var].setncattr('long_name', 'Measurement height above bare ground')
    
    # === Progress ===
    #print('Completed initialization of .nc file.')
    progress = 0
    
    # === Loop over the GRU ID's and fill
    for idx in range(0,num_gru):
        att['gruId'][idx] = pd.unique(shp['ID_source'])[idx] # note that pandas' unique() preserves ordering, whereas numpy's unique() also sorts > we do NOT want these sorted without our knowledge
    
    # === Loop over all shapes in the shapefile and assign attributes ===
    for idx in range(0,num_hru):
        
        # Show a progress report
        #print(str(progress) + ' out of ' + str(num_hru) + ' HRUs completed.')
        
        # We need to follow the order of HRUs in the forcing file. 
        # Find where in the shapefile the current forcing-HRU can be found
        idx_in_shape, = np.where(shp['ID_target'].values.astype('int') == forc['hruId'][idx])[0]
        
        # Fill values
        att['hruId'][idx] = shp.iloc[idx_in_shape]['ID_target']
        att['hru2gruId'][idx] = shp.iloc[idx_in_shape]['ID_source']
        att['longitude'][idx] = shp.iloc[idx_in_shape]['geometry'].centroid.x
        att['latitude'][idx] = shp.iloc[idx_in_shape]['geometry'].centroid.y
        att['HRUarea'][idx] = shp.loc[idx_in_shape]['HRU_area']
        att['elevation'][idx] = shp.loc[idx_in_shape]['HRU_elev_m']
        att['soilTypeIndex'][idx] = shp.loc[idx_in_shape]['soil_type']
        att['vegTypeIndex'][idx] = shp.loc[idx_in_shape]['veg_type']
        
        # Downslope HRU index
        tmp_gru_full = shp[shp['ID_source'] == shp.iloc[idx_in_shape]['ID_source']].copy() # store the full GRU in a temp variable
        tmp_sort_elev = np.sort(tmp_gru_full['HRU_elev_m']) # sort the elevations in this GRU
        tmp_idx_of_current_hru_in_sort = np.where(tmp_sort_elev == shp.iloc[idx_in_shape]['HRU_elev_m']) # find where the current HRU's elevation is in the sorted elevations
        if tmp_idx_of_current_hru_in_sort[0] == 0: # HRU with lowest sorted elevation; i.e. outlet
            tmp_downslope_hru_id = 0 # outlet HRU ID = 0
        else: # not outlet so must have a downslope HRU
            tmp_idx_of_downslope_hru_in_sort = tmp_idx_of_current_hru_in_sort[0]-1 # index of downslope HRU elevation in sorted elevation 
            tmp_elev_of_downslope_hru = tmp_sort_elev[tmp_idx_of_downslope_hru_in_sort] # actual downslope HRU elevation in sorted array
            tmp_idx_of_downslope_hru_in_shp = np.where(tmp_gru_full['HRU_elev_m'] == tmp_elev_of_downslope_hru[0]) # index in the full GRU of the downslope HRU found by matching elevations
            tmp_downslope_hru_id = int(tmp_gru_full.iloc[tmp_idx_of_downslope_hru_in_shp]['ID_target']) # downslope HRU ID of current HRU
        att['downHRUindex'][idx] = tmp_downslope_hru_id
        #print('Downslope HRU index for HRU ' + str(shp.iloc[idx_in_shape]['ID_target']) + ' is ' + str(tmp_downslope_hru_id))
        
        # Placeholders
        #att['soilTypeIndex'][idx] = -999
        #att['vegTypeIndex'][idx] = -999
                
        # Constants
        att['tan_slope'][idx] = 0.1
        att['contourLength'][idx] = 30
        att['slopeTypeIndex'][idx] = 1
        att['mHeight'][idx] = 3
        
        # Increment the counter
        progress += 1
        
 return

def create_and_fill_nc_var(nc, newVarName, newVarDim, newVarType, newVarVals, fillVal):

    ncvar = nc.createVariable(newVarName, newVarType, (newVarDim, 'hru',),fill_value=fillVal)
    ncvar[:] = newVarVals   # store data in netcdf file

    return

def prepare_summa_coldstate_file(sdir):

 path_base = Path(sdir)
 # Specify where the forcing file can be found, which has the HRU IDs we need
 path_for = Path(sdir)
 file_for = 'forcing.nc'
 # Define a location for the new file
 file_cs = 'coldState.nc'
 # Create filename strings
 str_for = str(path_for / file_for)
 str_cs = str(path_base / file_cs)
 # Find the number of HRUs
 with nc.Dataset(str_for, "r") as forc:
    num_hru = len(forc['hruId'])
 # Specify the dimensions
 midSoil = 8
 midToto = 8
 ifcToto = midToto+1
 scalarv = 1
 # Specify variables fill values
 dt = 3600 # [s] per time step. ERA5 ime step is hourly, so 60*60
 layerDepth = np.asarray([0.025, 0.075, 0.15, 0.25, 0.5, 0.5, 1, 1.5])
 layerHeight = np.asarray([0, 0.025, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 4])
 # Create the empty trial params file
 with nc.Dataset(str_cs, "w", format="NETCDF4") as cs, nc.Dataset(str_for, "r") as forc:

    # === Some general attributes
    cs.setncattr('Author', "Created by W. Knoben")
    cs.setncattr('History','Created ' + time.ctime(time.time()))
    cs.setncattr('Purpose','Create a cold state .nc file for SUMMA runs for the Bow river above Banff')

    # === Define the dimensions
    cs.createDimension('hru',num_hru)
    cs.createDimension('midSoil',midSoil)
    cs.createDimension('midToto',midToto)
    cs.createDimension('ifcToto',ifcToto)
    cs.createDimension('scalarv',scalarv)

    # === Variables ===
    var = 'hruId'
    cs.createVariable(var, 'i4', 'hru', fill_value = False)
    cs[var].setncattr('units', '-')
    cs[var].setncattr('long_name', 'Index of hydrological response unit (HRU)')
    cs[var][:] = forc['hruId'][:]

    # time step size
    fillWithThis = np.full((1,num_hru), dt)
    create_and_fill_nc_var(cs, 'dt_init', 'scalarv', 'f8', fillWithThis, False)

    # Number of layers
    fillWithThis = np.full((1,num_hru), midSoil, dtype='int')
    create_and_fill_nc_var(cs, 'nSoil', 'scalarv', 'i4', fillWithThis, False)

    var = 'nSnow'
    fillWithThis = np.zeros((1,num_hru), dtype='int')
    create_and_fill_nc_var(cs, 'nSnow', 'scalarv', 'i4', fillWithThis, False)

    # STATES: scalarCanopyIce, scalarCanopyLiq, scalarSnowDepth, scalarSWE, scalarSfcMeltPond
    fillWithThis = np.zeros((1,num_hru)) # fill value for all these variables
    create_and_fill_nc_var(cs, 'scalarCanopyIce', 'scalarv', 'f8', fillWithThis, False)
    create_and_fill_nc_var(cs, 'scalarCanopyLiq', 'scalarv', 'f8', fillWithThis, False)
    create_and_fill_nc_var(cs, 'scalarSnowDepth', 'scalarv', 'f8', fillWithThis, False)
    create_and_fill_nc_var(cs, 'scalarSWE', 'scalarv', 'f8', fillWithThis, False)
    create_and_fill_nc_var(cs, 'scalarSfcMeltPond', 'scalarv', 'f8', fillWithThis, False)

    # STATES: scalarAquiferStorage
    fillWithThis = np.full((1,num_hru), 1.0)
    create_and_fill_nc_var(cs, 'scalarAquiferStorage', 'scalarv', 'f8', fillWithThis, False)

    # scalarSnowAlbedo
    fillWithThis = np.zeros((1,num_hru))
    create_and_fill_nc_var(cs, 'scalarSnowAlbedo', 'scalarv', 'f8', fillWithThis, False)

    # TEMPERATURES: scalarCanairTemp, scalarCanopyTemp
    fillWithThis = np.full((1,num_hru), 283.16)
    create_and_fill_nc_var(cs, 'scalarCanairTemp', 'scalarv', 'f8', fillWithThis, False)
    create_and_fill_nc_var(cs, 'scalarCanopyTemp', 'scalarv', 'f8', fillWithThis, False)

    # LAYER: iLayerHeight
    fillWithThis = np.full((num_hru,ifcToto), layerHeight).transpose()
    create_and_fill_nc_var(cs, 'iLayerHeight', 'ifcToto', 'f8', fillWithThis, False)

    # LAYER: mLayerDepth
    fillWithThis = np.full((num_hru,midToto), layerDepth).transpose()
    create_and_fill_nc_var(cs, 'mLayerDepth', 'midToto', 'f8', fillWithThis, False)

    # LAYER: mLayerTemp
    fillWithThis = np.full((midToto,num_hru), 283.16)
    create_and_fill_nc_var(cs, 'mLayerTemp', 'midToto', 'f8', fillWithThis, False)

    # LAYER: mLayerVolFracIce
    fillWithThis = np.full((midToto,num_hru), 0.0)
    create_and_fill_nc_var(cs, 'mLayerVolFracIce', 'midToto', 'f8', fillWithThis, False)

    # LAYER: mLayerVolFracLiq
    fillWithThis = np.full((midToto,num_hru), 0.2)
    create_and_fill_nc_var(cs, 'mLayerVolFracLiq', 'midToto', 'f8', fillWithThis, False)

    # LAYER: mLayerMatricHead
    fillWithThis = np.full((midSoil,num_hru), -1.0)
    create_and_fill_nc_var(cs, 'mLayerMatricHead', 'midSoil', 'f8', fillWithThis, False)

 return

def prepare_summa_trialparms_file(sdir):

 path_base = Path(sdir)
 # Specify where the forcing file can be found, which has the HRU IDs we need
 path_for = Path(sdir)
 file_for = 'forcing.nc'
 # Specify where the attribute file can be found, which has the HRU IDs we need
 file_att = 'attributes.nc'
 # Define a location for the new file
 file_tp = 'trialParams.nc'
 # Create filename strings
 str_for = str(path_for / file_for)
 str_att = str(path_base / file_att)
 str_tp = str(path_base / file_tp)
 # Find the number of HRUs
 with nc.Dataset(str_att, "r") as att:
    num_hru = len(att['hruId'])
    num_gru = len(att['gruId'])
 # Find the number of HRUs
 with nc.Dataset(str_att, "r") as att:
    num_hru = len(att['hruId'])
    num_gru = len(att['gruId'])
 # Create the empty trial params file
 with nc.Dataset(str_tp, "w", format="NETCDF4") as tp, nc.Dataset(str_att, "r") as att:
    
    # === Some general attributes
    tp.setncattr('Author', "Created by W. Knoben")
    tp.setncattr('History','Created ' + time.ctime(time.time()))
    tp.setncattr('Purpose','Create empty trialParams.nc file for SUMMA runs for Bow river')
    
    # === Define the dimensions 
    tp.createDimension('hru',num_hru)
    tp.createDimension('gru',num_gru) # these are the same as HRU's in this set up    
    
    # === Variables ===
    var = 'hruId'
    tp.createVariable(var, 'i4', 'hru', fill_value = False)
    tp[var].setncattr('units', '-')
    tp[var].setncattr('long_name', 'Index of hydrological response unit (HRU)')
    tp[var][:] = att['hruId'][:]
    
    var = 'gruId'
    tp.createVariable(var, 'i4', 'gru', fill_value = False)
    tp[var].setncattr('units', '-')
    tp[var].setncattr('long_name', 'Index of grouped response unit (GRU)')
    tp[var][:] = att['gruId'][:]
 #with nc.Dataset(str_tp, "r") as tp:
 #   print(tp['hruId'][:])
 #   print(tp['gruId'][:])

 return
 
def prepare_summa_remainder_files(sdir):

 #Forcing file list
 file = '%s/forcingFileList.txt' % sdir
 fp = open(file,'w')
 fp.write('forcing.nc\n')
 fp.close()
 #Summa files
 os.system('cp data/summa_files/* %s/.' % sdir)
 #Update filemanager file
 os.system('cp %s/fileManager.txt %s/tmp.txt' % (sdir,sdir))
 fp0 = open('%s/fileManager.txt' % sdir,'w')
 fp1 = open('%s/tmp.txt' % sdir,'r')
 for line in fp1:
     if '$SETTINGS_PATH' in line:line = line.replace('$SETTINGS_PATH','%s/' % sdir)
     if '$FORCING_PATH' in line:line = line.replace('$FORCING_PATH','%s/' % sdir)
     if '$OUTPUT_PATH' in line:line = line.replace('$OUTPUT_PATH','%s/'% sdir)
     fp0.write(line)
 fp0.close()
 fp1.close()

 return

def prepare_mizuroute_remainder_files(sdir):

 #MizuRoute files
 os.system('cp data/mizuroute_files/* %s/.' % sdir)
 #Update filemanager file
 os.system('cp %s/mizuroute.control %s/tmp.txt' % (sdir,sdir))
 fp0 = open('%s/mizuroute.control' % sdir,'w')
 fp1 = open('%s/tmp.txt' % sdir,'r')
 for line in fp1:
     if '$ANCILLARY_PATH' in line:line = line.replace('$ANCILLARY_PATH','%s' % sdir)
     if '$INPUT_PATH' in line:line = line.replace('$INPUT_PATH','%s' % sdir)
     if '$OUTPUT_PATH' in line:line = line.replace('$OUTPUT_PATH','%s'% sdir)
     fp0.write(line)
 fp0.close()
 fp1.close()

 return

def assemble_summa_configuration(tda,dh):
    
 if ((np.log10(tda) < 7) | (dh < 50)):
   print(r'To avoid memory and storage constraints, tda must be higher than 7 and dh must be higher than 100. Please adjust the parameters accordingly and rerun the script to assemble the summa configuration')
   return

 file = 'data/dem.tif'
 #Set up the modeling directory
 sdir = 'summa_sim/tda%.2f_dh%.2f' % (np.log10(tda),dh)
 os.system('rm -rf %s' % sdir)
 os.system('mkdir -p %s' % sdir)

 #Open and do some basic terrain analysis on the data
 print('1) Performing basic terrain analysis')
 bow = terrain_tools.terrain_analysis(file)

 #Assemble the river network and height bands
 print('2) Assembling the river network and height bands')
 bow = assemble_river_network_and_height_bands(bow,tda,dh)
 #pickle.dump(bow,open('test.pck','wb'))
 #bow = pickle.load(open('test.pck','rb'))

 #Polygonize gru and hru maps
 print('3) Polygonizing GRU and HRU maps')
 polygonize_gru_and_hru_maps(bow,sdir)

 #Prepare HRU forcing
 print('4) Preparing HRU forcing')
 prepare_hru_forcing(bow,sdir)

 #Prepare mizuRoute files
 print('5) Preparing routing files')
 prepare_routing_files(bow,sdir)

 #Prepare SUMMA attributes file
 print('6) Preparing SUMMA attribute file')
 prepare_summa_attribute_file(sdir)

 #Prepare SUMMA cold state file
 print('6) Preparing SUMMA cold state file')
 prepare_summa_coldstate_file(sdir)

 #Prepare SUMMA trialParams file
 print('7) Preparing SUMMA trialParms file')
 prepare_summa_trialparms_file(sdir)

 #Prepare remainder of SUMMA files
 print('8) Preparing remainder SUMMA files')
 prepare_summa_remainder_files(sdir)

 #Prepare remainder of mizuRoute files
 print('9) Preparing remainder mizuRoute files')
 prepare_mizuroute_remainder_files(sdir)

 #Save some info to the instance
 bow.sdir = sdir

 return bow



