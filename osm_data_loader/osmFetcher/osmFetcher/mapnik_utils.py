import os
import sys
import re

from mapnik import *
### https://github.com/mapnik/mapnik/wiki/GettingStartedInPython

def lookupConfigurableValueString(s, dict):
	searchRegex = "\$\((.*?)\)"
	keys = re.findall(searchRegex, s) # Search for all brackets (and store their content)
	if(keys):
		for k in keys:
			default = None
			if("=" in k):
				k, default = k.split("=")
			val = dict.get(k, default)
			if(val):
				##replace the keys with their values
				s = re.sub(searchRegex, val, s)
	return s


def setRandomPropertyStuf(ref, k, v, lookupDict={}):
	val = None
	### does the render tag value need to make a lookup into the feature tags?
	### if yes, get the feature tag name used for the renderer attribute value
	valueStr = str(v)
	if(lookupDict):
		valueStr = lookupConfigurableValueString(valueStr, lookupDict)
	### do normal stuff
	try:
		val = eval(valueStr)
		setattr(ref, k, val)
	except Exception as e:
		try:
			val = v
			setattr(ref, k, val)
		except Exception as e:
			print(e)
			return False
	return val

def renderStuff(m, path, handler, cfg):
	ds = Shapefile(file=path)
	fs = ds.featureset()
	f = fs.next()
	tags = f.attributes

	featureName = str(tags["name"])
	featureGroup = tags["group"]
	featureIndex = tags["index"]
	layerName = "l_%s" % featureName
	styleName = "s_%s" % featureName
	############################################################
	### GRAB FEATURE DATA
	if(not featureName in handler.features):
		print("No feature found for '%s'" % featureName)
		return
	featureInfo = handler.features[featureName][featureIndex]
	############################################################
	### GRAB RENDER DATA
	if(not featureName in cfg.renderings):
		print("No rendering found for '%s'" % featureName)
		return

	renderInfo = cfg.renderings[featureName]
	draw = renderInfo.tags.pop("draw")
	############################################################
	symbolizers = []
	if(draw == "line"):
		symbolizers.append(LineSymbolizer())
	elif(draw == "marker"):
		symbolizers.append(MarkersSymbolizer())
	elif(draw == "fill"):
		symbolizers.append(PolygonSymbolizer())
	elif(draw == "building"):
		symbolizers.append(BuildingSymbolizer())
	elif(draw == "poly"):
		symbolizers.append(LineSymbolizer())
		symbolizers.append(PolygonSymbolizer())
	else:
		raise Exception("Could not create mapnik.Symbolizer with draw flag of '%s'" % draw_)
	############################################################
	### SET SYMBOLIZERS AND ATTRIBUTES
	for sym in symbolizers:
		for k,v, in renderInfo.tags.items():
			val = setRandomPropertyStuf(sym, k,v, featureInfo.tags)
			print(" -- [%s] %s: %s" %(type(val),k,val))
	############################################################
	print("Creating layer '%s' ..." % layerName)

	layer = Layer(layerName) 
	s = Style() # style object to hold rules
	r = Rule() # rule object to hold symbolizers
	############################################################
	for sym in symbolizers:
		r.symbols.append(sym)  # add the symbolizer to the rule object
	s.rules.append(r) # now add the rule to the style and we're done
	m.append_style(styleName, s) # Styles are given names only as they are applied to the map
	layer.datasource = ds
	layer.styles.append(styleName)
	m.layers.append(layer)



def createMap(path, handler, cfg):

	imageSizeX = cfg.properties.pop("width", 2000)
	imageSizeY = cfg.properties.pop("height", 2000)
	### use mapnick to create an image
	m = Map(imageSizeX,imageSizeY)
	srsString = cfg.properties.pop("srs", "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
	#m.srs = "+init=epsg:3857" # web mercator: epsg:3857 , mercator(wsg84): epsg:4326
	m.srs = srsString
	# set background color  and other random properties
	for k,v  in cfg.properties.items():
		val = setRandomPropertyStuf(m, k,v)
		print(" -- Propery > [%s] %s: %s" %(type(val),k,val))

	for subdir, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".shp"):
				filename = os.path.splitext(file)[0]
				filepath = os.path.join(subdir, file)
				print("Processing '%s'" % file)

				renderStuff(m, filepath, handler, cfg)
			else:
				continue

	#m.aspect_fix_mode = mapnik.aspect_fix_mode.ADJUST_BBOX_HEIGHT
	#extent = mapnik.Box2d(50.88663, 6.92449, 50.88283, 6.91823)
	#m.zoom_to_box(envelope)
	#m.zoom(0.8)
	m.zoom_all()
	bounds = m.envelope() ###WEST, SOUTH, EAST, NORTH image coordinates (object boundaries/locations, not pixels)
	# Write the data to a png image called world.png in the current directory
	outputPath = os.path.join(path, "map.png")
	#print("Rendering '%s' ..." % outputPath)
	render_to_file(m, outputPath, "png")

	return outputPath, bounds


