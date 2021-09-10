import os
import sys

import osmium
#https://docs.osmcode.org/pyosmium/latest/ref_osm.html
#https://github.com/osmcode/pyosmium/


import type_utils



brokenNodeLocations = {}
def repairBrokenNodeLocation(node):
	global brokenNodeLocations
	convertedL = None
	nid = "%s" % node
	if(node.location.valid()):
		convertedL = Location.createFromLocation(node.location)
	else:
		if(nid in brokenNodeLocations):
			convertedL = Location.createFromLocation(brokenNodeLocations[nid])
			print("Repaired broken node id + location '%s': %s" % (nid,convertedL))
	return convertedL

class Location():
	def __init__(self, x, y):
		self.x = x
		self.y = y
	@staticmethod
	def create(x,y):
		l = Location(x, y)
		return l
	@staticmethod
	def createFromLocation(loc):
		l = Location(loc.x, loc.y)
		return l
	def __repr__(self):
		return "[%s/%s]" % (self.x, self.y)


class StsGrabHandler(osmium.SimpleHandler):
	def __init__(self, rulesCfg):
		super(StsGrabHandler, self).__init__()
		self.cfg = rulesCfg
		self.features = {}

	def node(self, n):
		if(n.id < 0):
			print("ID < 0 found: %s" % n)
			global brokenNodeLocations
			str = "%s" % n.id
			brokenNodeLocations[str] = n.location
		self.gatherFeatureStuff("node", n)

	def way(self, w):
		self.gatherFeatureStuff("way", w)
	
	def relation(self, r):
		pass
		#+print("REL: %s" % r)

	def area(self, a):
		self.gatherFeatureStuff("area", a)

	def gatherFeatureStuff(self, group, obj):
		frule = self.cfg.findFeatureRule(obj.tags, group=group)
		if(frule):
			###create Feature
			index = 0
			try:
				index = len(self.features[frule.getName()])
			except Exception as e:
				pass
			f = type_utils.Feature.create(index, frule.getName(), frule.getGroup(), obj)
			if(f):
				### save feature
				if(not f.getName() in self.features):
					self.features[f.getName()] = []
					print("Extracting features for '%s'" % f.getName())
				self.features[f.getName()].append(f)