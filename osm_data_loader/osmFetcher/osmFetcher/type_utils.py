import os
import sys

import shapefile
### export Shapefile?
### https://pypi.org/project/pyshp/#writing-shapefiles

import osmium
#https://docs.osmcode.org/pyosmium/latest/ref_osm.html
#https://github.com/osmcode/pyosmium/

import osmium_utils



		
class Rendering(object):
	def __init__(self, tagDict):
		super(Rendering, self).__init__()
		self.tags = tagDict
		self.draw = tagDict.get("draw", "")
	@staticmethod
	def create(rrule):
		if(rrule):
			i = Rendering(rrule.getTags())
		else:
			i = Rendering.createDefault()
		return i
	@staticmethod
	def createDefault():
		i = Rendering({"draw": "line", "stroke": "black", "stroke_width": 3.0, "stroke_opacity": 1.0})
		return i

	def writeFieldData(self, writer):
		writer.field("name")
		writer.field("group")
		writer.field("index", "N")
		pass
	def writeFeatureShape(self, writer, feature):
		###prepare shape stuff
		if(self.draw == "fill"):
			data = [[[l.x,l.y] for l in feature.locations]]
			if(data):
				writer.poly(data)
		elif(self.draw == "marker"):
			l = next(iter(feature.locations))
			writer.point(l.x,l.y)
		elif(self.draw == "line"):
			data = [[[l.x,l.y] for l in feature.locations]]
			if(data):
				writer.line(data)
		else:
			#print(Exception("Unknown 'draw' flag in renderer:\n%s" % self.__dict__))
			#return False
			data = [[[l.x,l.y] for l in feature.locations]]
			writer.poly(data)
		##write fields
		fields = [feature.getName(), feature.getGroup(), feature.getIndex()]
		#writer.record("")
		writer.record(*fields)
		###
		return True



class Feature(object):
	def __init__(self, index, name, group, locations, tagDict):
		super(Feature, self).__init__()
		self.tags = tagDict
		self.locations = locations
		self.name = name
		self.index = index
		self.group = group
	def isValid(self):
		if(not self.locations):
			return False
		for l in self.locations:
			if(not l):
				return False
		return True
	def getName(self):
		return self.name
	def getIndex(self):
		return self.index
	def getGroup(self):
		return self.group
	def __repr__(self):
		str = ""
		for k,v in self.__dict__.items():
			str +=  "%s: %s, " % (k,v)
		return "%s{\n%s\n}\n" % (self.__class__.__name__, str)
	@staticmethod
	def create(index, name, group, obj):
		d, nodeLocations = Feature.grabData(obj)
		i = Feature(index, name, group, nodeLocations, d)
		if(not i.isValid()):
			return None
		return i
	@staticmethod
	def grabData(obj):
		d = {}
		temp = []
		nodeLocations = []
		for t in obj.tags:
				d[t.k] = t.v
		
		if isinstance(obj, osmium.osm._osm.Way):
			temp = obj.nodes
		elif(isinstance(obj, osmium.osm._osm.Node)):
			temp = [obj]
		elif(isinstance(obj, osmium.osm._osm.Area)):
			for ring in obj.outer_rings():
				for node in ring:
					temp.append(node)

		for n in temp:
			### repair broken node locations if possible
			convertedL = osmium_utils.repairBrokenNodeLocation(n)
			if(convertedL):
				nodeLocations.append(convertedL)
			else:
				nodeLocations = []
				break

		return d, nodeLocations


class ShapeFiler(object):
	@staticmethod
	def createShapefiles(path, handler, cfg):
		### go through all features and find a rendering rule for them
		### after that, create a shapefile for each feature group
		for k,features in handler.features.items():
			rrule = cfg.findRenderingRule(k)
			r = Rendering.create(rrule)
			print("Generating shapefile for '%s' [count: %s]" % (k, len(features)))
			filename = os.path.join(path, k)
			with shapefile.Writer(filename) as writer:
				r.writeFieldData(writer)
				for f in features:
					success = r.writeFeatureShape(writer, f)
					if(not success):
						print("Nothing generated for '%s'" % k)