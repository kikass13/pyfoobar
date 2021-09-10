
import os
import sys
import shutil

from PIL import Image

import osmFetcher.rule_utils as rule_utils
import osmFetcher.osmium_utils as osmium_utils
import osmFetcher.type_utils as type_utils
import osmFetcher.mapnik_utils as mapnik_utils


OUTPUT_PATH = "outputs"
def createOutputDir():
	if(not os.path.isdir(OUTPUT_PATH)):
		os.mkdir(OUTPUT_PATH)
	else:
		shutil.rmtree(OUTPUT_PATH)


def main(rulePath, osmPath):
	print("============================")
	cfg = rule_utils.RuleConfiguration(rulePath)

	### start osm parsing
	h = osmium_utils.StsGrabHandler(cfg)
	h.apply_file(osmPath)

	### debug map features
	#print(h.features["Building"])
	#print(h.features["Street"])

	### create output dir
	createOutputDir()

	###create shapefiles 
	type_utils.ShapeFiler.createShapefiles(OUTPUT_PATH, h, cfg)

	### create map using mapnik
	outImagePath, bounds = mapnik_utils.createMap(OUTPUT_PATH, h, cfg)
	print(bounds)
	### bounds > WEST, SOUTH, EAST, NORTH
	img = Image.open(outImagePath)
	width, height = img.size
	pixels = img.load()

	### DEBUG STUFF
	#
	import ImageDraw
	
	ORIGINX = 60243796
	ORIGINY = 508306145
	# xPerc = (GOALX - LEFT) / (RIGHT - LEFT)
	xPerc = (ORIGINX - bounds[0] ) / (bounds[2] - bounds[0])
	# yPerc = (GOALX - LEFT) / (RIGHT - LEFT)
	yPerc = (ORIGINY - bounds[1] ) / (bounds[3] - bounds[1])
	print("xperc: %s" % xPerc)
	print("yperc: %s" % yPerc)
	xPixels = xPerc * width
	yPixels = (1- yPerc) * height
	print("xpixel: %s" % xPixels)
	print("ypixel: %s" % yPixels)
	draw = ImageDraw.Draw(img)
	draw.ellipse((xPixels-10, yPixels-10, xPixels+10, yPixels+10), outline ='blue')
	draw.ellipse((xPixels-11, yPixels-11, xPixels+11, yPixels+11), outline ='yellow')
	draw.ellipse((xPixels-12, yPixels-12, xPixels+12, yPixels+12), outline ='blue')

	img.show()
	print("============================")

	#import pyproj
	#myProj = pyproj.Proj("+proj=utm +zone=23K, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
	#lat = 50.8306145
	#lon = 6.0243796
	#heading = 0.0  
	#UTMx, UTMy = myProj(lon, lat)
	#print(UTMx, UTMy)
	print("============================")

	#return 

	import utm
	# #### do map origin
	# lat = 50.8306145
	# lon = 6.0243796
	# myUtmEasting, myUtmNorthing, myUtmZoneNumber, myUtmZoneLetter = utm.from_latlon(lat, lon)
	# print("Utm Easting: %s" % myUtmEasting)
	# print("Utm Northing: %s" % myUtmNorthing)
	# print("Utm Zone: %s%s" % (myUtmZoneNumber, myUtmZoneLetter))
	
	# print("*******")

	# ### do random measured pose form site setups
	# ###
	# ##  position: [42.9279999999999972715158946812, -29.4869999999999983231191436062, 1.0869999999999999662492200514]
	# ##  rotation: [0.00200003400086702426072249316746, -0.00400006800173404852144498633493, -0.585009945253604501935740245244, 0.811013787351578585571587609593]
	# ##  latitude: 50.8303652250935869005843414925
	# ##  longitude: 6.02500517796630319367068295833
	# lat = 50.8303652250935869005843414925
	# lon = 6.02500517796630319367068295833
	# randomUtmEasting, randomUtmNorthing, randomUtmZoneNumber, randomUtmZoneLetter = utm.from_latlon(lat, lon)
	# print("Utm Easting: %s" % randomUtmEasting)
	# print("Utm Northing: %s" % randomUtmNorthing)
	# print("Utm Zone: %s%s" % (randomUtmZoneNumber, myUtmZoneLetter))
	# print("*******")
	# eastingMap = randomUtmEasting - myUtmEasting
	# northingMap = randomUtmNorthing - myUtmNorthing
	# print("NewEasting: %s" % eastingMap)
	# print("NewNorthing: %s" % northingMap)
	# print("********")
	# print("GROUNDTRUTH: 42.9279999999999972715158946812, -29.4869999999999983231191436062")
	# print("GROUNDTRUTHDIFF\n >> EAST: %s\n >> NORTH: %s" % (eastingMap - 42.9279999999999972715158946812, northingMap - -29.4869999999999983231191436062))


	lat1, lon1, heading1 = 48.7863202, 8.9196414, 0.0
	east1, north1, zoneid1, zonesec1 = utm.from_latlon(lat1, lon1)

	lat2, lon2, heading2 = 48.7863868, 8.9197895, 250.0
	east2, north2, zoneid2, zonesec2 = utm.from_latlon(lat2, lon2)

	eastingMap = east2 - east1
	northingMap = north2 - north1

	import math
	#The formula for calculating grid convergence (sometimes called meridian convergence) 
	#for spherical UTM projections was given (very incorrectly until just now) at How to Calculate North?
	#In case that is not clear
	#y = arctan [tan (s - s0) * sin p]
	#where
	#y is grid convergence,
	#s0 is longitude of UTM zone's central meridian, see https://gisgeography.com/central-meridian/
	#p, s are latitude, longitude of point in question
	utmoffset = math.atan(math.tan(math.radians(lon1-9.0)) * math.radians(math.sin(lat1)))
	print("UtmOffset: %s rad" % "{:.15f}".format(utmoffset))
	print("NewEasting: %s m" % eastingMap)
	print("NewNorthing: %s m" % northingMap)
	print("NewHeading: %s deg" % 0.0)


	print("============================")
#############################################################################################################

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python %s <.nikrulez-File> <.osm-File>" %sys.argv[0])
		sys.exit(-1)
	else:
		ruleFilePath = sys.argv[1]
		osmFilePath = sys.argv[2]

		if(ruleFilePath and osmFilePath):
			main(ruleFilePath, osmFilePath)


#############################################################################################################