#!/bin/bash

# get the module
module load nco

# define new directories
gruPath=../output/gru
hruPath=../output/hru
mkdir -p $gruPath
mkdir -p $hruPath

# define gru variables
gru1=gru
gru2=gruId
gru3=averageRoutedRunoff_mean

# loop through files
for oldFile in ../output/camels_nldas_G*.nc ; do

 # get the new file names
 fileName="$(basename $oldFile)"  # split out the file name
 gruFile=${gruPath}/$fileName     # new file name
 hruFile=${hruPath}/$fileName     # new file name
 echo $oldFile

 # extract the hru and gru variables - this is necessary because `ncrcat` 
 # concatenates along the record dimension. In SUMMA outputs, `time` is the
 # record (unlimited) dimension by default. This means that for gru variables
 # we need to make the 'gru' dimension the record dimension and for `hru` 
 # variables, `hru` needs to be the record dimension. Here we separate `gru`
 # and `hru` variables into two separate files.
 ncks -O    -C -v${gru1},${gru2},${gru3} $oldFile $gruFile
 ncks -O -x -C -v${gru1},${gru2},${gru3} $oldFile $hruFile

 # reorder dimensions so that the gru/hru is the unlimited dimension, instead of time
 ncpdq -O -a gru,time $gruFile $gruFile
 ncpdq -O -a hru,time $hruFile $hruFile

done  # looping through files

# concatenate the files
echo Concatenating files
combineFile=camels_nldas_day.nc
ncrcat -O ${gruPath}/*G* ${gruPath}/${combineFile}
ncrcat -O ${hruPath}/*G* ${hruPath}/${combineFile}

# perturb dimensions - make time the unlimited dimension again
echo Perturb dimensions
ncpdq -O -a time,gru ${gruPath}/${combineFile} ${gruPath}/${combineFile}
ncpdq -O -a time,hru ${hruPath}/${combineFile} ${hruPath}/${combineFile}

# combine gru and hru files
echo Combining files
cp ${hruPath}/${combineFile} ../output/camels_nldas_full.nc
ncks -A ${gruPath}/${combineFile} ../output/camels_nldas_full.nc

# get the subset for offline testing
ncks -O -F -d time,1,100 ../output/camels_nldas_full.nc ../output/camels_nldas_subset.nc
