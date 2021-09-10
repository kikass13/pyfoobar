import os
import sys

import yaml
import collections





class RuleConfiguration(object):
	def __init__(self, path):
		super(RuleConfiguration, self).__init__()
		self.path = path
		self.features = collections.OrderedDict()
		self.renderings = collections.OrderedDict()
		self.properties = collections.OrderedDict()
		self.init()

	def init(self):
		with open(self.path) as f:
			self.d_ = yaml.load(f)
		self.parseFeatures()
		self.parseProperties()
		self.parseRenderings()
		
	def parseFeatures(self):
		for group, l in self.d_["features"].items():
			if(not l):
				continue
			for f in l:
				feature = None
				try:
					feature = FeatureRule(group, f)
				except Exception as e:
					print(e)
				if(feature):
					self.features[group, feature.getName()] = feature
	def parseProperties(self):
		for k, v in self.d_["properties"].items():
			self.properties[k] = v
	def parseRenderings(self):
		if(self.d_["renderings"]):
			for r in self.d_["renderings"]:
				rendering = None
				try:
					rendering = RenderingRule(r)
				except Exception as e:
					raise
				if(rendering):
					self.renderings[rendering.getName()] = rendering
	def findFeatureRule(self, osm_obj, group=None):
		success = False
		for k,f in self.features.items():
			### check if the group is correct
			if(group):
				if(not k[0] == group):
					continue
			try:
				success = f.isTrue(osm_obj)
			except Exception as e:
				continue
			if(success):
				return f
		return None
	def findRenderingRule(self, name):
		rrule = self.renderings.get(name, None)
		return rrule


class RenderingRule(object):
	def __init__(self, entry):
		super(RenderingRule, self).__init__()
		self.tags = {}
		self.init(entry)

	def init(self, entry):
		self.name = entry["target"]
		d = entry["define"]
		self.tags.update(d)
	def getName(self):
		return self.name
	def getTags(self):
		return self.tags
class FeatureRule(object):
	def __init__(self, group, entry):
		super(FeatureRule, self).__init__()
		self.group = group
		self.init(entry)
	def init(self, entry):
		self.name = next(iter(entry))
		self.evaluation = entry[self.name]

	def isTrue(self, f):
		### f is an instance (same as defined inside the evaulationStr of the rulez file)
		result = False
		val = eval(self.evaluation)
		if(val):
			result = True
		return result
	def getName(self):
		return self.name
	def getGroup(self):
		return self.group
