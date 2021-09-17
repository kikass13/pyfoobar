#!usr/bin/python

#imports
from enum import Enum     # for enum34, or the stdlib version
import os
import sys
from distutils.dir_util import copy_tree
import shutil
from shutil import copyfile
import json
import glob

import cantools #https://pypi.python.org/pypi/cantools/5.2.0




def isSpecRead(spec):
	isRead = spec[2] == "r"
	return isRead
def isSpecWrite(spec):
	isWrite = spec[2] == "w"
	return isWrite

def grabRuleset(ruleset):
	name = ruleset["id"]
	ns = ruleset["ns"]
	typ = ruleset["type"]
	trigger = ruleset["publish_trigger"]
	rules = ruleset["rules"]
	return name, ns, typ, trigger, rules

def grabRule(rule):
	target = rule["in"]
	action = rule["set"]
	return target, action


dir_base_c = "base_c"
#c_filename = "can_interface_node.cpp"
#h_filename = "can_interface_node.h"
cmake_filename = "CMakeLists.txt"
xml_filename = "package.xml"
file_ending = ".dbc"





singlequerys = [
	"<>AUTO_INSERT:PACKAGE_NAME<>",

	"<>AUTO_INSERT:MESSAGE_NAME<>",
	"<>AUTO_INSERT:MESSAGE_ID<>",
	"<>AUTO_INSERT:MESSAGE_TYPE<>",

	"<>AUTO_INSERT:SIGNAL_TYPE<>",
	"<>AUTO_INSERT:SIGNAL_NAME<>",
	"<>AUTO_INSERT:SIGNAL_STARTBIT<>",
	"<>AUTO_INSERT:SIGNAL_LENGTH<>",
	"<>AUTO_INSERT:SIGNAL_ISBIGENDIAN<>",
	"<>AUTO_INSERT:SIGNAL_ISSIGNED<>",
	"<>AUTO_INSERT:SIGNAL_FACTOR<>",
	"<>AUTO_INSERT:SIGNAL_OFFSET<>",
	"<>AUTO_INSERT:PACKAGE_NS<>",
	"<>AUTO_INSERT:MESSAGE_DLC<>",
	"<>AUTO_INSERT:MESSAGE_COUNT<>",
]

multiquerys = [
	"//<>AUTO_INSERT:MESSAGE_DECODE<>",     #this is used for decoding messageparams and signals from binary
	"//<>AUTO_INSERT:MESSAGE_DECODE_ALL<>",     #this is used for decoding messageparams and signals from binary
	"//<>AUTO_INSERT:MESSAGE_PUBLISHERS<>",  #this is used for inserting multiple publishers
	"//<>AUTO_INSERT:MESSAGE_ENCODE<>",  # this is used for decoding messageparams and signals from binary
	"//<>AUTO_INSERT:MESSAGE_SUBSCRIBER<>",  # this is used for inserting multiple publishers
	"//<>AUTO_INSERT:MESSAGE_CALLBACK_DECLARATIONS<>",

	#cmake thingys
	"<>AUTO_INSERT:MESSAGE_FILES<>",
	"<>AUTO_INSERT:RULSET_NAMESPACES<>",

	#ruleset thingys
	"//<>AUTO_INSERT:RULESET_HEADERS<>",            #this is used for inserting multiple header includes
	"//<>AUTO_INSERT:MESSAGE_RULESET_PUBLISHER_DECLARATIONS<>",  #this is used for inserting multiple publishers
	"//<>AUTO_INSERT:MESSAGE_RULESET_PUBLISHER_DEFINITIONS<>",

	"//<>AUTO_INSERT:MESSAGE_RULESET_VARIABLES<>",
    "//<>AUTO_INSERT:MESSAGE_RULESET_CONVERSION<>",
    "//<>AUTO_INSERT:MESSAGE_RULESET_PUBLISH<>",

	#package xml thingys
	"<>AUTO_INSERT:RULESET_DEPENDS<>",

	### generalized stuff
	"//<>AUTO_INSERT:DO_PER_MESSAGE<>"
]

outputdir = "outputs"
backupdir = "outputs/.backup"
packagepath = ""
msgspath = ""
srcpath = ""
includepath = ""

#ros msg data types
#http://wiki.ros.org/msg

def createOutputDir():
	# create output directory
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
def createBackupDir():
	# create backup directory
	if not os.path.exists(backupdir):
		os.makedirs(backupdir)


def createDirs(packagename):
	global msgspath
	global packagepath

	createOutputDir()

	packagepath = outputdir + "/" + packagename
	# create package directory
	if not os.path.exists(packagepath):
		os.makedirs(packagepath)
	else:
		### backup old package with that name 
		createBackupDir()
		copy_tree(packagepath, backupdir)
		### remove older package with that name
		shutil.rmtree(packagepath, ignore_errors=False, onerror=None)
		os.makedirs(packagepath)

	#create msg dir
	#msgspath = os.path.join(outputdir, packagename, "msg")
	#if not os.path.exists(msgspath):
	#	os.makedirs(msgspath)

	# create src and include dir
	srcpath = os.path.join(outputdir, packagename, "src")
	if not os.path.exists(srcpath):
		os.makedirs(srcpath)

	### create <packagename> dir inside of "include" because ROS demands it 
	includepath = os.path.join(outputdir, packagename, "include", packagename)
	if not os.path.exists(includepath):
		os.makedirs(includepath)


def findAppropriateDataType(s):
	"This function finds a suitable ROS data type (msg) for a can signal"
	type = "float64"
	#insert usefull algotihm here
	#
	#
	return type



def createMsgFiles(messageList, msgpath):
	"This function creates ros message files from the dbc messages and its containing signals"
	for msg in messageList:
		filename = msg.name
		str = ""
		### add header for timestamp into message definition
		str += "Header header\n"
		### add rest of the signals from .dbc file to message definition
		for s in msg.signals:
			datatype = findAppropriateDataType(s)
			str = str + datatype + " " + s.name + "\n"
		#print(str)
		#create the message files
		with open(os.path.join(msgspath, "%s.msg"%filename), "w") as myfile:
			myfile.write(str)


def copyBaseC(genType, packagename):
	myBase = os.path.join(dir_base_c, genType)

	#copy src and include
	copy_tree(os.path.join(myBase, "src"), os.path.join(packagepath, "src"))
	copy_tree(os.path.join(myBase, "include"), os.path.join(packagepath, "include", packagename))
	#tmp = os.path.join(myBase, "msg")
	#if os.path.exists(tmp):
	#	copy_tree(tmp, os.path.join(packagepath, "msg"))
	#copy package.xml and cmakeList
	#copyfile(os.path.join(myBase, "package.xml"),  os.path.join(packagepath, "package.xml"))
	copyfile(os.path.join(myBase, "CMakeLists.txt"),  os.path.join(packagepath, "CMakeLists.txt"))
	copyfile(os.path.join(myBase, "README.md"),  os.path.join(packagepath, "README.md"))


def copyConfigs(paths, pkg):
	tempDir = os.path.join(outputdir, pkg, "can_gen_configs") 
	if not os.path.exists(tempDir):
		os.makedirs(tempDir)
	for p in paths:
		copyfile(p, os.path.join(tempDir, os.path.basename(p)))

def replaceQueryStringWithData(strline, olddata, newdata):
	#print("FOUND " + strline + "[" + olddata + "]")+
	strline = strline.replace(olddata, newdata)
	return strline


def findCodeBlock(content, start_string, end_string = None):
	if not end_string:
		end_string = start_string
	i = None
	j = None

	for line_num, line in enumerate(content[0:]):
		if not i and line.strip() == start_string:
			i = line_num
		elif i and line.strip() == end_string:
			j = line_num+1
			break
	return i, j


def getCodeBlock(text_data, start_line, end_line):
	code_block = text_data[start_line + 1: end_line - 1]
	text_data[start_line: end_line] = ""
	return code_block


def replaceLineDict(text_data, replacement_dict):
	out_text = text_data ### buffer
	for key_word, replce_word in replacement_dict.items():
		if text_data.find(key_word) != -1:
			out_text = out_text.replace(key_word, replce_word)
	return out_text
def replaceBlockDict(text_data, replacement_dict):
	### get list of string
	### for each line check for available key word from dict and replace with its value
	out_text = []
	for line in text_data:
		buffer_line = line
		for key_word, replce_word in replacement_dict.items():
			if buffer_line.find(key_word) != -1:
				buffer_line = buffer_line.replace(key_word, replce_word)
		out_text.append(buffer_line)
	return out_text


def checkLineReadWriteDefinition(line):
	readDef = True
	writeDef = True
	rIndex = line.find("<R>")
	if (rIndex != -1):
		writeDef = False
	wIndex = line.find("<W>")
	if (wIndex != -1):
		readDef = False
	line = line.replace("<R>", "")
	line = line.replace("<W>", "")
	return readDef, writeDef, line

def encodeMsgMultiQuery(content, packagename, messageList, specifications):
	output_strings = []
	start_line, end_line = findCodeBlock(content, "//<>AUTO_INSERT:MESSAGE_ENCODE<>")
	if start_line and end_line:
		### create a buffer of the source code

		callback_function_block = getCodeBlock(content, start_line, end_line)

		### create a copy of that buffer for each message in our messageList
		n_callback_function_block = [callback_function_block for n in range(0, len(messageList))]

		for msg, spec in zip(messageList, specifications): #for s in msg.signals:
			isWrite = isSpecWrite(spec)
			if not isWrite:
				continue

			replacement_dict = {
				"<>AUTO_INSERT:PACKAGE_NS<>": str(spec[1]),
				"<>AUTO_INSERT:MESSAGE_NAME<>": str(msg.name),
				"<>AUTO_INSERT:PACKAGE_NAME<>": packagename,
				"<>AUTO_INSERT:MESSAGE_ID<>": str(msg.frame_id),
				"<>AUTO_INSERT:MESSAGE_DLC<>": str(msg.length),
			}
			
			msg_callback_block = callback_function_block[:]
			signal_start_line, signal_end_line = findCodeBlock(msg_callback_block, "//<>AUTO_INSERT:SIGNAL_ENCODE<>")
			signal_code_block = getCodeBlock(msg_callback_block,signal_start_line,signal_end_line)

			msg_callback_block = replaceBlockDict(msg_callback_block, replacement_dict)
			# msg_callback_block = ''.join(msg_callback_block)
			running_index = signal_start_line
			for signal in msg.signals:
				signal_code = signal_code_block[:]
				signal_replace_dict = {
					"<>AUTO_INSERT:SIGNAL_STARTBIT<>": str(signal.start),
					"<>AUTO_INSERT:SIGNAL_LENGTH<>": str(signal.length),
					"<>AUTO_INSERT:SIGNAL_NAME<>": str(signal.name)
				}
				signal_code = replaceBlockDict(signal_code, signal_replace_dict)
				code_size = len(signal_code)
				signal_code = ''.join(signal_code)
				msg_callback_block.insert(running_index, signal_code)
				running_index += 1
				# seq = (msg_callback_block, signal_code)
			msg_callback_block = ''.join(msg_callback_block)
			output_strings.append(msg_callback_block)

	output_string = ''.join(output_strings)
	# print(output_string)
	return output_string


def decodeMsgMultiQuery(content, packagename, messageList, specifications, rulesets, skipReadMessages=True):
	output_strings = []

	start_line, end_line = findCodeBlock(content, "//<>AUTO_INSERT:MESSAGE_DECODE<>")
	if not end_line:
		start_line, end_line = findCodeBlock(content, "//<>AUTO_INSERT:MESSAGE_DECODE_ALL<>")
	if start_line and end_line:
		### create a buffer of the source code

		callback_function_block = getCodeBlock(content, start_line, end_line)

		### create a copy of that buffer for each message in our messageList
		n_callback_function_block = [callback_function_block for n in range(0, len(messageList))]

		for msg, spec in zip(messageList, specifications): #for s in msg.signals:
			isRead = isSpecRead(spec)
			if skipReadMessages and not isRead:
				continue

			replacement_dict = {
				"<>AUTO_INSERT:PACKAGE_NS<>": str(spec[1]),
				"<>AUTO_INSERT:MESSAGE_NAME<>": str(msg.name),
				"<>AUTO_INSERT:PACKAGE_NAME<>": packagename,
				"<>AUTO_INSERT:MESSAGE_ID<>": str(msg.frame_id),
				"<>AUTO_INSERT:MESSAGE_DLC<>": str(msg.length),
			}

			#################################################
			### extra special ruleset stuff:
			conversionStr = "// Conversion aggregations are specified here ..."
			publishStr = "// Conversion publishers are specified here ..."
			conversionList = [conversionStr]
			publishList = [publishStr]
			### grab all rules and code some c variable handling
			for set in rulesets:
				name, ns, typ, trigger, rules = grabRuleset(set)
				### decide if we want to convert stuff inside of this message
				for r in rules:
					target, action = grabRule(r)
					if(msg.name == target ):
						conversionList.append("\t\t\t %s.%s ;" % ("%s_var"%name , action))

				### decide if we want to publish inside of this message
				if(msg.name == trigger):
					publishList.append("\t\t\t pub_%s_%s.publish(%s) ;" % (spec[1], name, "%s_var"%name))
			
			# ### set up a dictionary for all those data to replace the code
			conversionStr = "\n".join(conversionList) + "\n"
			publishStr = "\n".join(publishList) + "\n"
			replacement_dict_ruleset = {
			 	"//<>AUTO_INSERT:MESSAGE_RULESET_MESSAGE_PUBLISH<>" : publishStr,
			 	"//<>AUTO_INSERT:MESSAGE_RULESET_MESSAGE_CONVERSION<>": conversionStr,
			}
			### update our replacement dict with the RULESET stuff
			replacement_dict.update(replacement_dict_ruleset)
	
			
			#################################################

			msg_callback_block = callback_function_block[:]
			signal_start_line, signal_end_line = findCodeBlock(msg_callback_block, "//<>AUTO_INSERT:SIGNAL_DECODE<>")
			signal_code_block = getCodeBlock(msg_callback_block,signal_start_line,signal_end_line)

			msg_callback_block = replaceBlockDict(msg_callback_block, replacement_dict)
			# msg_callback_block = ''.join(msg_callback_block)
			running_index = signal_start_line
			for signal in msg.signals:
				signal_code = signal_code_block[:]
				signlaType = None
				if signal.length > 0 and signal.length <= 8:
					signalType = "uint8_t"
				elif signal.length > 8 and signal.length <= 16:
					signalType = "uint16_t"
				elif signal.length > 16 and signal.length <= 32:
					signalType = "uint32_t"
				else:
					signalType = "uint64_t"
				
				signal_replace_dict = {
					"<>AUTO_INSERT:SIGNAL_OFFSET<>": str(signal.offset),
					"<>AUTO_INSERT:SIGNAL_LENGTH<>": str(signal.length),
					"<>AUTO_INSERT:SIGNAL_NAME<>": str(signal.name),
					"<>AUTO_INSERT:SIGNAL_STARTBIT<>": str(signal.start),
					"<>AUTO_INSERT:SIGNAL_ISBIGENDIAN<>": "true" if signal.byte_order == "big_endian" else "false",
					"<>AUTO_INSERT:SIGNAL_ISSIGNED<>": "true" if signal.is_signed == True else "false",
					"<>AUTO_INSERT:SIGNAL_FACTOR<>": str(signal.scale),
					"<>AUTO_INSERT:SIGNAL_TYPE<>": signalType,
					"<>AUTO_INSERT:SIGNAL_PARENT_MESSAGE<>": msg.name,
				}
				signal_code = replaceBlockDict(signal_code, signal_replace_dict)
				code_size = len(signal_code)
				signal_code = ''.join(signal_code)
				msg_callback_block.insert(running_index, signal_code)
				running_index += 1
				# seq = (msg_callback_block, signal_code)
			msg_callback_block = ''.join(msg_callback_block)
			output_strings.append(msg_callback_block)

	output_string = ''.join(output_strings)
	return output_string


def autoFill(files, packagename, messageList, msg_specifics, rulesets):
	for file in files:
		with open(file) as f:
			content = f.readlines()
		# check for multiquerys in file
		for i, line in enumerate(content):
			for querystr in multiquerys:
				charindex = line.find(querystr)
				if (charindex != -1):
					if (querystr == "//<>AUTO_INSERT:RULESET_HEADERS<>"):
						# insert headers into the h file
						headerquery = line[charindex + len(querystr):]
						# remove headerquery, so we dont find it again and again and again
						content[i] = ""
						# add include headers
						for set in rulesets:
							name, ns, typ, trigger, rules = grabRuleset(set)
							str = replaceQueryStringWithData(headerquery, "<>AUTO_INSERT:PACKAGE_NAME<>", ns)
							str = replaceQueryStringWithData(str, "<>AUTO_INSERT:MESSAGE_NAME<>", typ)
							content.insert(i, str)


					elif (querystr == "//<>AUTO_INSERT:MESSAGE_RULESET_PUBLISHER_DECLARATIONS<>"):  # MULTICHECK PUBLISHERS
						headerquery = line[charindex + len(querystr):]
						# remove headerquery, so we dont find it again and again and again
						content[i] = ""
						# add include headers
						for msg, spec in zip(messageList, msg_specifics):
							### only write this publisher query if this specific message is defined with a read ("r") flag
							isRead = isSpecRead(spec)
							if(isRead):
								for set in rulesets:
									name, ns, typ, trigger, rules = grabRuleset(set)
									if(msg.name == trigger):
										str = replaceQueryStringWithData(headerquery, "<>AUTO_INSERT:MESSAGE_NAME<>", name)
										str = replaceQueryStringWithData(str, "<>AUTO_INSERT:PACKAGE_NS<>", spec[1])
										content.insert(i, str)

					elif (querystr == "//<>AUTO_INSERT:DO_PER_MESSAGE<>"):
						query = line[charindex + len(querystr):]
						### check read or write in next symbols
						readDef, writeDef, query = checkLineReadWriteDefinition(query)
						# remove headerquery, so we dont find it again and again and again
						content[i] = ""
						for msg, spec in zip(messageList, msg_specifics):
							isWrite = isSpecWrite(spec)
							isRead = isSpecRead(spec)
							if (writeDef != isWrite) and (readDef != isRead):
								#print("defw: %s, defr: %s \n w: %s, r: %s" % (writeDef, readDef, isWrite, isRead))
								continue

							replacement_dict = {
							"<>AUTO_INSERT:PACKAGE_NS<>": spec[1],
							"<>AUTO_INSERT:MESSAGE_NAME<>": msg.name,
							"<>AUTO_INSERT:PACKAGE_NAME<>": packagename,
							}
							### only write this publisher query if this specific message is defined with a read ("r") flag
							block = replaceLineDict(query, replacement_dict)
							content.insert(i, block)
					###############################
					### New implementation
					if (querystr == "//<>AUTO_INSERT:MESSAGE_DECODE<>"):
						string = decodeMsgMultiQuery(content, packagename, messageList, msg_specifics, rulesets)
						content.insert(i, string)
					if (querystr == "//<>AUTO_INSERT:MESSAGE_DECODE_ALL<>"):
						string = decodeMsgMultiQuery(content, packagename, messageList, msg_specifics, rulesets, skipReadMessages=False)
						content.insert(i, string)
					if (querystr == "//<>AUTO_INSERT:MESSAGE_ENCODE<>"):  # MULTICHECK MESSAGE ENCODE
						string = encodeMsgMultiQuery(content, packagename, messageList, msg_specifics, rulesets)
						content.insert(i, string)
					################################
		# check for singlequerys in file
		for i, line in enumerate(content):
			# singlequerys last
			for querystr in singlequerys:
				charindex = line.find(querystr)
				if (charindex != -1):
					if (querystr == singlequerys[0]):  # SINGLECHECK PACKAGE_NAME
						content[i] = replaceQueryStringWithData(line, querystr, packagename)
					elif (querystr == "<>AUTO_INSERT:MESSAGE_COUNT<>"):  # SINGLECHECK PACKAGE_NAME
						content[i] = replaceQueryStringWithData(line, querystr, str(len(messageList)))
					else:
						pass

		# write out file
		f = open(file, 'w')
		for line in content:
			f.write("%s" % line)

def autoFillC(packagename, messageList, msg_specifics, rulesets):
	sourceFilePath = os.path.join("./", packagepath, "src")
	for file in glob.glob("%s/*.cpp" % sourceFilePath):
		with open(file) as f:
			content = f.readlines()

		decoding_msg_toggle = False
		decoding_msg_str = []

		decodedMessages = {}

		# check for multiquerys in file
		for i, line in enumerate(content):

			# read multiple next lines until stopquery if decoding_togglre is active
			if decoding_msg_toggle == True:
				decoding_msg_str.append(line)
				content[i] = ""

			for querystr in multiquerys:
				charindex = line.find(querystr)
				if (charindex != -1):

					if (querystr == "//<>AUTO_INSERT:MESSAGE_PUBLISHERS<>"):  # MULTICHECK MESSAGE PUBLISHERS  //<>AUTO_INSERT:MESSAGE_PUBLISHERS<>
						publisherquery = line[charindex + len(querystr):]
						content[i] = ""
						for msg, spec in zip(messageList, msg_specifics):
							### only write this publisher query if this specific message is defined with a read ("r") flag
							isRead = isSpecRead(spec)
							if(isRead):
								str = replaceQueryStringWithData(publisherquery, "<>AUTO_INSERT:MESSAGE_NAME<>", msg.name)
								str = replaceQueryStringWithData(str, "<>AUTO_INSERT:PACKAGE_NS<>", spec[1])
								content.insert(i, str)

					elif(querystr == "//<>AUTO_INSERT:MESSAGE_RULESET_PUBLISHER_DEFINITIONS<>"):
						publisherquery = line[charindex + len(querystr):]
						content[i] = ""
						for msg, spec in zip(messageList, msg_specifics):
							### only write this publisher query if this specific message is defined with a read ("r") flag
							isRead = isSpecRead(spec)
							if(isRead):
								for set in rulesets:
									name, ns, typ, trigger, rules = grabRuleset(set)
									if(msg.name == trigger):
										str = replaceQueryStringWithData(publisherquery, "<>AUTO_INSERT:MESSAGE_NAME<>", name)
										str = replaceQueryStringWithData(str, "<>AUTO_INSERT:PACKAGE_NS<>", spec[1])
										str = replaceQueryStringWithData(str, "<>AUTO_INSERT:RULE_NS<>", ns)
										str = replaceQueryStringWithData(str, "<>AUTO_INSERT:RULE_TYPE<>", typ)
										content.insert(i, str)

					elif(querystr == "//<>AUTO_INSERT:MESSAGE_RULESET_VARIABLES<>"):
						content[i] = ""
						for set in rulesets:
							name, ns, typ, trigger, rules = grabRuleset(set)
							str = "%s::%s %s_var ;\n" % (ns, typ, name)
							content.insert(i, str)


					elif (querystr == "//<>AUTO_INSERT:MESSAGE_SUBSCRIBER<>"):  # MULTICHECK MESSAGE PUBLISHERS  //<>AUTO_INSERT:MESSAGE_PUBLISHERS<>
						subscriberquery = line[charindex + len(querystr):]
						content[i] = ""
						for msg, spec in zip(messageList, msg_specifics):
							### only write this publisher query if this specific message is defined with a read ("r") flag
							isWrite = isSpecWrite(spec)
							if(isWrite):
								str = replaceQueryStringWithData(subscriberquery, "<>AUTO_INSERT:MESSAGE_NAME<>", msg.name)
								str = replaceQueryStringWithData(str, "<>AUTO_INSERT:PACKAGE_NS<>", spec[1])
								content.insert(i, str)
					elif (querystr == "//<>AUTO_INSERT:DO_PER_MESSAGE<>"):
						query = line[charindex + len(querystr):]
						### check read or write in next symbols
						readDef, writeDef, query = checkLineReadWriteDefinition(query)
						# remove headerquery, so we dont find it again and again and again
						content[i] = ""
						index = 0
						for msg, spec in zip(messageList, msg_specifics):
							isWrite = isSpecWrite(spec)
							isRead = isSpecRead(spec)
							if (writeDef != isWrite) and (readDef != isRead):
								#print("defw: %s, defr: %s \n w: %s, r: %s" % (writeDef, readDef, isWrite, isRead))
								continue

							replacement_dict = {
							"<>AUTO_INSERT:INDEX<>": "%s"%index,
							"<>AUTO_INSERT:PACKAGE_NS<>": spec[1],
							"<>AUTO_INSERT:MESSAGE_NAME<>": msg.name,
							"<>AUTO_INSERT:PACKAGE_NAME<>": packagename,
							"<>AUTO_INSERT:MESSAGE_ID<>": "%s"% msg._frame_id,
							}
							### only write this publisher query if this specific message is defined with a read ("r") flag
							block = replaceLineDict(query, replacement_dict)
							content.insert(i, block)
							index+=1
					###############################
					### New implementation
					if (querystr == "//<>AUTO_INSERT:MESSAGE_DECODE<>"):
						string = decodeMsgMultiQuery(content, packagename, messageList, msg_specifics, rulesets)
						content.insert(i, string)

					if (querystr == "//<>AUTO_INSERT:MESSAGE_ENCODE<>"):  # MULTICHECK MESSAGE ENCODE
						string = encodeMsgMultiQuery(content, packagename, messageList, msg_specifics)
						content.insert(i, string)
					################################

		#check for singlequerys in file
		for i, line in enumerate(content):
			#singlequerys last
			for querystr in singlequerys:
				charindex = line.find(querystr)
				if(charindex != -1):
					if(querystr == singlequerys[0]):    # SINGLECHECK PACKAGE_NAME
						content[i] = replaceQueryStringWithData(line, querystr, packagename)
					elif (querystr == singlequerys[1]):   # SINGLECHECK MESSAGE_NAME
						content[i] = replaceQueryStringWithData(line, querystr, packagename)
					elif (querystr == "<>AUTO_INSERT:MESSAGE_COUNT<>"):  # SINGLECHECK PACKAGE_NAME
						readDef, writeDef, query = checkLineReadWriteDefinition(line)
						count = 0
						if(readDef):
							for rm, spec in zip(messageList, msg_specifics):
								if(readDef == isSpecRead(spec)):
									count +=1 
						content[i] = replaceQueryStringWithData(query, querystr, "%s"%count )
					else:
						pass

		#if we find //<>AUTO_INSERT:MESSAGE_DECODE<>, replace it with all contents from decodedMessages
		index = 0
		for i, line in enumerate(content):
			if (line.find("//<>AUTO_INSERT:MESSAGE_DECODE<>") != -1):
				print("inserting decoded messages and signals to line %s" % i)
				content[i] = ""
				index = i
		#we know where we have to put decodedMessages, so do it
		for key, value in decodedMessages.items():
			content.insert(index, "// Message Decoded: " + key.__str__() + "\n")
			index = index + 1
			for decoded_line in value:
				content.insert(index, decoded_line)
				index = index + 1

		#write out file
		f = open(file, 'w')
		for line in content:
			f.write("%s" % line)


def autoFillH(packagename, messageList, msg_specifics, rulesets):
	headerFilePath = os.path.join("./", packagepath, "include", packagename)
	files = glob.glob("%s/*.hpp" % headerFilePath)
	autoFill(files, packagename, messageList, msg_specifics, rulesets)
	


def autoFillCmake(packagename, messageList, msg_specifics, rulesets):
	file = packagepath + "/" + cmake_filename;
	with open(file) as f:
		content = f.readlines()

	# check for multiquerys in file
	for i, line in enumerate(content):

		for querystr in multiquerys:
			charindex = line.find(querystr)
			if (charindex != -1):
				if (querystr == "<>AUTO_INSERT:MESSAGE_FILES<>"): 
					headerquery = line[charindex + len(querystr):]
					# remove headerquery, so we dont find it again and again and again
					content[i] = ""
					# add include headers
					for msg in messageList:
						str = replaceQueryStringWithData(headerquery, singlequerys[1], msg.name)
						content.insert(i, str)

				elif (querystr == "<>AUTO_INSERT:RULSET_NAMESPACES<>"):  
					content[i] = ""
					namespaceList = []
					for set in rulesets:
						name, ns, typ, trigger, rules = grabRuleset(set)
						if(ns not in namespaceList):
							namespaceList.append(ns)
					for x in namespaceList:
						content.insert(i, "   %s" % ns)

	# check for singlequerys in file
	for i, line in enumerate(content):
		# singlequerys last
		for querystr in singlequerys:
			charindex = line.find(querystr)
			if (charindex != -1):
				if (querystr == singlequerys[0]):  # SINGLECHECK PACKAGE_NAME
					content[i] = replaceQueryStringWithData(line, querystr, packagename)

	# write out file
	f = open(file, 'w')
	for line in content:
		f.write("%s" % line)


def autoFillXml(packagename, messageList, msg_specifics, rulesets):
	file = packagepath + "/" + xml_filename;
	with open(file) as f:
		content = f.readlines()

	# check for singlequerys in file
	for i, line in enumerate(content):

		for querystr in multiquerys:
			charindex = line.find(querystr)
			if (charindex != -1):
				if (querystr == "<>AUTO_INSERT:RULESET_DEPENDS<>"):
					str = ""
					namespaceList = []
					for set in rulesets:
						name, ns, typ, trigger, rules = grabRuleset(set)
						if(ns not in namespaceList):
							namespaceList.append(ns)
					for x in namespaceList:
						str += "<depend>%s</depend>\n" % x
					content[i] = replaceQueryStringWithData(line, querystr, str)

		# singlequerys last
		for querystr in singlequerys:
			charindex = line.find(querystr)
			if (charindex != -1):
				if (querystr == singlequerys[0]):  # SINGLECHECK PACKAGE_NAME
					content[i] = replaceQueryStringWithData(line, querystr, packagename)
					
	# write out file
	f = open(file, 'w')
	for line in content:
		f.write("%s" % line)

def autoFillReadme(packagename, messageList, msg_specifics, rulesets):
	filePath = os.path.join("./", packagepath)
	files = [os.path.join(filePath, "README.md")]
	autoFill(files, packagename, messageList, msg_specifics, rulesets)

def tryToRepairBrokenDb(db):
	#repair duplicate signal names
	from collections import Counter  # Counter counts the number of occurrences of each item
	for msg in db.messages:
		myList = []
		for s in msg.signals:
			myList.append(s.name)
		counts = Counter(myList)  # so we have: {'name':3, 'state':1, 'city':1, 'zip':2}
		for s, num in counts.items():
			if num > 1:  # ignore strings that only appear once
				print("Duplicate signals found: %s" % (s))
				# get reference to signal with that specific name
				# change the name of that signal to the suffixed one
				suffix = 0
				for i, signal in enumerate(msg.signals):
					#get signal reference
					signalReference = msg.signals[i]
					if signalReference.name == s:  # s is the string name of the signal here
						#THIS WILL NOT WORK, BECAUSE THIS PROPERTY HAS NO SETTER IN CANTOOLS VERSION 12.0
						#signalReference.name = s + str(suffix)
						#SO WE HAVE TO HACK IT VIA THE __dict__ of our signalReference
						# NOT WORKING #setattr(signalReference, "name", s + str(suffix))
						signalReference.__dict__["_name"] = s + str(suffix)
						print("Added suffix [%s] for signal: %s" % (str(suffix), signalReference.name))
						suffix = suffix + 1



FILE_ENDING_CONFIGURATION = "cfg"
FILE_ENDING_CONFIGURATION_DATABASE = "db"
FILE_ENDING_CONFIGURATION_SPECIFICS = "specs"
FILE_ENDING_CONFIGURATION_CONVERSION_RULESETS = "rules"

def createConfigurationFiles(pkg, db, nsList, sourceFiles):
	pre = outputdir
	configFileName = "%s/%s.%s" % (pre, pkg, FILE_ENDING_CONFIGURATION)
	#dbFileName = "%s/%s.%s" % (pre, pkg, FILE_ENDING_CONFIGURATION_DATABASE)
	dbFileNames = sourceFiles
	specificsFileName = "%s/%s.%s" % (pre, pkg, FILE_ENDING_CONFIGURATION_SPECIFICS)
	rulesetsFileName = "%s/%s.%s" % (pre, pkg, FILE_ENDING_CONFIGURATION_CONVERSION_RULESETS)

	# write out specifics file
	with open(specificsFileName, 'w') as file:
		specifics = [(msg.name, ns, "r") for msg, ns in zip(db.messages, nsList)]
		j = json.dumps(specifics, indent=4, sort_keys=True)
		file.write(j)
	# write out configuration file
	with open(configFileName, 'w') as file:
		dataDict = {"packagename" : pkg, "database" : dbFileNames, "specifics" : specificsFileName, "rulesets": rulesetsFileName}
		j = json.dumps(dataDict, indent=4)
		file.write(j)
	# write out conversions file
	with open(rulesetsFileName, 'w') as file:
		j = json.dumps([], indent=4)
		file.write(j)

def openConfiguration(configFilePath):
	dataDict = json.load(open(configFilePath, 'r'))
	packagename = dataDict.get("packagename", None)
	databaseFiles = dataDict.get("database", None)
	specificsFile = dataDict.get("specifics", None)
	rulesetsFile = dataDict.get("rulesets", None)

	paths = [specificsFile, rulesetsFile]

	if(packagename and databaseFiles and specificsFile and rulesetsFile):
		# reinterpretate that data
		database, msg_namespaces = parseData(databaseFiles)
		specifics = json.load(open(specificsFile, 'r'))
		rulesets = json.load(open(rulesetsFile, 'r'))

	return paths, packagename, database, specifics, rulesets


class Mode(Enum):
	NONE = 0
	CREATE_CONFIGURATION= 1
	GENERATE_ROS = 2


def parseData(files):
	dbList = {}
	for file in files:
		filename, file_extension = os.path.splitext(file)
		if(file_extension == file_ending):
			db = cantools.db.load_file(file)
			dbList[filename] = db

	# combine all db object in one database
	total_database = cantools.database.Database()
	# list of name space for msgs
	msg_namespaces = []
	for ns, database in dbList.items():
		total_database.messages.extend(database.messages)
		### the namespace could contain .dbc paths (dbc/myRandomDbcFile.dbc)
		### so we want to cut the prefix stuff away > os.path.basename does that for paths
		ns = [os.path.basename(ns) for num_msg in database.messages]
		msg_namespaces.extend(ns)

	return total_database, msg_namespaces


def main():

	mode = Mode.NONE
	configurationFile = None

	try:
		command = sys.argv[1]
		if(command == "configure"):
			packagename = sys.argv[2] # User defined Package name
			files = sys.argv[3:] # list of .dbc files or a .config file
			mode = Mode.CREATE_CONFIGURATION
		elif(command == "generate"):
			genType = sys.argv[2]
			configurationFile = sys.argv[3] # .config file
			mode = Mode.GENERATE_ROS
		else:
			print("Unknown command %s" % command)
			sys.exit(1)
	except:
		print("Please give .dbc or .config files as input parameters")
		sys.exit(1)


	# check our mode and then do your thing
	if(mode == Mode.CREATE_CONFIGURATION):
		# prepare all file objects
		total_database, msg_namespaces = parseData(files)

		#try to repair broken signals (arrays use same signal names for example)
		print("#Reparing ...")
		tryToRepairBrokenDb(total_database)
		print("#Creating Configuration ...")
		createOutputDir()
		createConfigurationFiles(packagename, total_database, msg_namespaces, files)

	elif(mode == Mode.GENERATE_ROS):
		if(configurationFile):
			#load configuration files and grab necessary information
			paths, packagename, database, msg_specifics, rulesets = openConfiguration(configurationFile)

			#create ros package (dirs and files)
			print("#Creating relevant files from base ...")
			createDirs(packagename)
			#createMsgFiles(database.messages, msgspath)
			
			copyBaseC(genType, packagename)
			copyConfigs(paths, packagename)

			#try to replace all dummys in c code with actual data
			print("#Fill C ...")
			autoFillC(packagename, database.messages, msg_specifics, rulesets)
			# try to replace all dummys in include code with actual data
			print("#Fill H ...")
			autoFillH(packagename, database.messages, msg_specifics, rulesets)
			# try replace all dummys in cmakefile
			print("#Fill Cmake ...")
			autoFillCmake(packagename, database.messages, msg_specifics, rulesets)
			#try replacing dummy values in package.xml
			#print("#Fill Xml ...")
			#autoFillXml(packagename, database.messages, msg_specifics, rulesets)
			#try replacing dummy values in Readme.md
			print("#Fill Readme ...")
			autoFillReadme(packagename, database.messages, msg_specifics, rulesets)

	print("#Done ...")
	return



if __name__ == "__main__":
	main()
