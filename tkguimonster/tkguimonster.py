#from Tkinter import *
#from mttkinter import mtTkinter as tk
from mttkinter import mtTkinter as mtk

import tkFileDialog

import os

################################################################################
################ D E F I N E     G U I     F I E L D S  ########################
################################################################################


GLOBAL_BUTTON_COLOR="light gray"
def setGlobalButtonColor(color):
	global GLOBAL_BUTTON_COLOR
	GLOBAL_BUTTON_COLOR = color

def chooseButtonColor(configColor):
	global GLOBAL_BUTTON_COLOR
	btnColor = GLOBAL_BUTTON_COLOR
	if(configColor):
		btnColor = configColor
	return btnColor

def convertToColorCode(widget, color):
	if(color[0] != "#"):
		bgCode = widget.winfo_rgb(color)
		color = "#%04x%04x%04x" % bgCode
	return color
def clamp(val, minimum=0, maximum=65535):
	if val < minimum:
		return minimum
	if val > maximum:
		return maximum
	return val
def colorscale(hexstr, scalefactor):
	"""
	Scales a hex string by ``scalefactor``. Returns scaled hex string.

	To darken the color, use a float value between 0 and 1.
	To brighten the color, use a float value greater than 1.

	>>> colorscale("#DF3C3C", .5) >>> #6F1E1E
	"""
	hexstr = hexstr.strip('#')
	if scalefactor < 0 or len(hexstr) != 12: return hexstr
	r, g, b = int(hexstr[:4], 16), int(hexstr[4:8], 16), int(hexstr[8:], 16)
	r = clamp(r * scalefactor)
	g = clamp(g * scalefactor)
	b = clamp(b * scalefactor)
	#print(r,g,b)
	return "#%04x%04x%04x" % (r, g, b)

def findLongestStrInList(list):
	lenList=[]
	for x in list: 
		lenList.append(len(x))
	return max(lenList)
def findLongestStrInLists(lists):
	list = []
	for l in lists: list.extend(l)
	return findLongestStrInList(list)

class View(object):
	def __init__(self, key=""):
		self.key = key
		self.parent = None
		self.views = []
		self.tk=None
		self.configArgs = {}
	def setParent(self, parent):
		self.parent = parent
	def getContext(self):
		return self.parent.getContext()
	def addView(self, view):
		views = []
		if(isinstance(view, View)): self.views.append(view)
		else: self.views.extend(view)
		for v in self.views:
			v.setParent(self)

		
	def __str__(self, depth=0):
		s = " "*depth + "[%s] %s\n" % (type(self), self.key)
		s += ("").join([" "*depth + "%s" % (c.__str__(depth+2)) for c in self.views])
		return s
	#def makeView(self, r=0, c=0):
	#	return r,c
	def makeView(self,r, c, master):
		for v in self.views:
			v.makeView(r,c, master)
		return r,c, master
	def getValue(self):
		return None
	def setValue(self, v):
		pass

	def config(self, **kwargs):
		self.configArgs = kwargs

	def grabConfigArg(self, cfg):
		arg = None
		if(cfg in self.configArgs):
			arg = self.configArgs.get(cfg)
			del self.configArgs[cfg]
		return arg


class Text(View):
	def __init__(self, key, label):
		super(Text, self).__init__(key)
		self.label = label
	def makeView(self, master):
		self.l = mtk.Label(master, text=self.label, **self.configArgs)
		self.l.pack()
		return self.l
	def refresh(self, label):
		self.l.configure(text=str(label))

class ActionButton(View):
	def __init__(self, key, label, actionCallback=None, image=None, height=0, width=0):
		super(ActionButton, self).__init__(key)
		self.label = label
		self.args = {}
		if(width != 0): self.args["width"] = width
		if(height !=0): self.args["height"]=height
		self.actionCallback = actionCallback
		self.state = False
		self.imagePath=image
		self.image=None
		if(self.imagePath):
			try:
				self.image = mtk.PhotoImage(file=self.imagePath, width=width, height=height) 
			except Exception as e:
				print("No button icon: %s" % e)

	def onClick(self):
		self.state = not self.state
		if(self.actionCallback):
			self.actionCallback(self)
	def makeView(self, master):
		bg = master.cget('bg')
		frame = mtk.Frame(master, bg=bg)
		self.buttonbg = convertToColorCode(master, chooseButtonColor(self.grabConfigArg("buttoncolor")))
		self.buttonTk = mtk.Button(	frame, text=self.label, #anchor='w', 
					bg=self.buttonbg, 
					activebackground=colorscale(self.buttonbg, 0.95),
					image=self.image, compound = mtk.TOP,
					command = self.onClick, **self.args)
		self.buttonTk.pack(padx=5, pady=5)
		#frame.pack(fill=mtk.X, expand=True)
		return frame

	def setColor(self, color):
		self.buttonTk.config(bg=color)
	def resetColor(self):
		self.buttonTk.config(bg=self.buttonbg)
	def disable(self):
		self.buttonTk.config(state="disabled")
	def enable(self):
		self.buttonTk.config(state="normal")


class Checkbox(View):
	def __init__(self, key, label, actionCallback=None, selectColor=""):
		super(Checkbox, self).__init__(key)
		self.label = label
		self.actionCallback = actionCallback
		self.state = False
		self.selectColor = selectColor
	def onClick(self):
		self.state = not self.state
		if(self.actionCallback):
			self.actionCallback(self)
	def makeView(self, master):
		bg = master.cget('bg')
		frame = mtk.Frame(master, bg=bg)
		checkbuttonState = mtk.IntVar()
		checkbuttonState.set(int(self.state))
		b = mtk.Checkbutton(frame, variable=checkbuttonState, anchor='w', bg=bg, 
						command = self.onClick, 
						highlightthickness=0, bd=0, selectcolor=self.selectColor)
		b.pack(padx=5, pady=5)
		return frame
	def getValue(self):
		return self.state


class Container(View):
	def __init__(self, key):
		super(Container, self).__init__(key)
	def makeView(self, master):
		f = mtk.Frame(master)
		for v in self.views:
			tk = v.makeView(f)
			tk.pack()
		return f

class Horizontal(Container):
	def __init__(self, key, padding=0, direction="left", directions=[]):
		super(Horizontal, self).__init__(key)
		self.direction = mtk.LEFT
		if(direction == "right"):
			self.direction=mtk.RIGHT
		self.padding = padding
		self.directions = directions

	def makeView(self, master):
		horizontal = mtk.Frame(master, bg=master.cget('bg'))
		for i, v in enumerate(self.views):
			tk = v.makeView(horizontal)
			if len(self.directions) == 0:
				tk.pack(side=self.direction, anchor="w", padx=self.padding)
			else:
				tk.pack(side=self.directions[i], anchor="w", padx=self.padding)
		horizontal.pack()
		self.tk = horizontal

		return horizontal

class Vertical(Container):
	def __init__(self, key, padding=0):
		super(Vertical, self).__init__(key)
		self.padding = padding

	def makeView(self, master):
		vertical = mtk.Frame(master, bg=master.cget('bg'))
		for v in self.views:
			tk = v.makeView(vertical)
			tk.pack(side = mtk.TOP, fill=mtk.X, anchor="w", pady=self.padding)
		self.tk = vertical
		vertical.pack()
		return vertical


class ButtonField(View):
	def __init__(self, key, label, actionCallback=None, height=0, width=0):
		super(ButtonField, self).__init__(key)
		self.label = label
		self.state = False
		self.actionCallback = actionCallback
		self.args = {}
		if(width != 0): self.args["width"] = width
		if(height !=0): self.args["height"]=height

	def onClick(self, mybutton):
		self.state = not self.state
		if(self.actionCallback):
			self.actionCallback(self)

	def makeView(self, master):
		frame = mtk.Frame(master, **self.configArgs)
		frame.pack(side=mtk.TOP, fill=mtk.X, padx=5, pady=5)
		l = mtk.Label(frame, width=15, text=self.label, anchor='w', **self.configArgs)
		l.pack(side=mtk.LEFT)
		self.button = ActionButton("", self.label, actionCallback=self.onClick, **self.args)
		self.button.config(**self.configArgs)
		buttonTk = self.button.makeView(frame)
		buttonTk.pack(side=mtk.RIGHT)
		return frame

	def setColor(self, color):
		self.button.setColor(color)
	def resetColor(self):
		self.button.resetColor()
	def disable(self):
		self.button.disable()
	def enable(self):
		self.button.enable()


class CheckboxField(View):
	def __init__(self, key, label, actionCallback=None, selectColor=""):
		super(CheckboxField, self).__init__(key)
		self.label = label
		self.state = False
		self.actionCallback = actionCallback
		self.selectColor = selectColor

	def onClick(self, mybutton):
		self.state = not self.state
		if(self.actionCallback):
			self.actionCallback(self)

	def makeView(self, master):
		frame = mtk.Frame(master, **self.configArgs)
		frame.pack(side=mtk.TOP, fill=mtk.X, padx=5, pady=5)
		l = mtk.Label(frame, text=self.label, anchor='w', **self.configArgs)
		l.pack(side=mtk.LEFT)
		self.cbx = Checkbox("", self.label, actionCallback=self.onClick, selectColor=self.selectColor)
		self.cbx.config(**self.configArgs)
		tk = self.cbx.makeView(frame)
		tk.pack(side=mtk.RIGHT)
		return frame
	def getValue(self):
		return self.cbx.getValue()

class EntryField(View):
	def __init__(self, key, label, defaultEntry="", enterCallback=None):
		super(EntryField, self).__init__(key)
		self.label = label
		self.defaultEntry = defaultEntry
		self.entryConfig = {}
		self.enterCallback = self.enter
		self.enterCallback=enterCallback
		
	def makeView(self, master):
		frame = mtk.Frame(master, **self.configArgs)
		bg = frame.cget('bg')
		l = mtk.Label(frame, width=15, text=self.label, bg=bg, anchor="w")
		v = mtk.StringVar(frame, value=self.defaultEntry)
		self.ent = mtk.Entry(frame, textvariable=v)
		self.ent.bind("<Return>", self.enter)
		l.pack(side=mtk.LEFT, padx=5)
		self.ent.pack(side=mtk.LEFT, expand=True, fill=mtk.X)
		frame.pack(fill=mtk.X, pady=5, anchor="w")
		return frame
#	def config(self, **kwargs):
#		self.entryConfig = kwargs
	def enter(self, event):
		if(self.enterCallback):
			self.enterCallback(self, self.ent.get())
	def getValue(self):
		return self.ent.get()
	def setValue(self, val):
		self.ent.delete(0,mtk.END)
		self.ent.insert(0,val)

class FileDialogField(View):
	def __init__(self, key, label, defaultEntry="", valueChangedCallback=None):
		super(FileDialogField, self).__init__(key)
		self.label = label
		self.defaultEntry = defaultEntry
		self.valueChangedCallback = valueChangedCallback
	def makeView(self, master):
		frame = mtk.Frame(master, **self.configArgs)
		bg = frame.cget('bg')
		holder = mtk.Frame(frame, bg=bg)
		self.ent = EntryField("", self.label, defaultEntry=self.defaultEntry, enterCallback=self.onEnter)
		self.ent.config(**self.configArgs)
		entTk = self.ent.makeView(holder)
		b = mtk.Button(entTk, text="...", bg=chooseButtonColor(self.grabConfigArg("buttoncolor")), command = self.onClick)
		b.pack(side=mtk.RIGHT, padx=(5,0), pady=1)
		holder.pack(fill=mtk.X, pady=5)
		frame.pack()
		return frame
	def onEnter(self, sender, contentStr):
		if(self.valueChangedCallback):
			self.valueChangedCallback(self, contentStr)
	def onClick(self):
		initialDir = "/"
		if self.defaultEntry:
			initialDir = os.path.dirname(self.defaultEntry)
		
		path = tkFileDialog.askopenfilename(initialdir = initialDir,title = "Select file")
		if isinstance(path, str):
			self.defaultEntry = path
			self.ent.setValue(path)
			if(self.valueChangedCallback):
				self.valueChangedCallback(self, path)
	def getValue(self):
		return self.ent.getValue()


class OptionField(View):
	def __init__(self, key, label, options=[], direction="horizontal", valueChangedCallback=None, boxWidth=None, labelWidth=None, defaultEntry=None):
		super(OptionField, self).__init__(key)
		self.label = label
		self.options = options
		if(not self.options): 	### if list of options is empty, we have to have an option at all times
			self.options = " "
		self.direction = direction
		self.valueChangedCallback = valueChangedCallback
		### preset values in comboBox
		self.defaultEntry = defaultEntry
		if self.defaultEntry == None and options:
			self.defaultEntry = options[0]
		#self.selectedValue = self.defaultEntry

		### configure boxwidth for longest element in optionsList or set to given value
		### or better not do that
		self.boxWidth = boxWidth
		### configure min labelWidth if necessary
		self.labelWidth = len(self.label)
		if(labelWidth): self.labelWidth = labelWidth

	def refresh(self, options, entry=None):
		# Reset var and delete all old options
		import Tkinter as tk
		if entry is None and not options:
			self.variable.set("")
		elif entry is None:
			self.variable.set(options[0])
		else:
			self.variable.set(entry)
		self.optionMenu['menu'].delete(0, 'end')
		# Insert list of new options (tk._setit hooks them up to var)
		for o in options:
			self.optionMenu['menu'].add_command(label=o, command=lambda value=o: (self.variable.set(value), self.valueChanged(value)) )

	### Define gui for checked fields
	def makeView(self, master):
		frame = mtk.Frame(master, **self.configArgs)
		bg = frame.cget('bg')
		bg = convertToColorCode(master, bg)

		l = mtk.Label(frame, text=self.label, width=self.labelWidth, anchor="w", bg=bg)
		self.variable = mtk.StringVar(frame)
		self.variable.set(self.defaultEntry)
		self.optionMenu = mtk.OptionMenu(frame, self.variable, command=self.valueChanged, *self.options)
		self.optionMenu.config(bg=bg, highlightthickness=0,  activebackground=colorscale(bg,0.95))#, activeforeground=colorscale(bg,1.5) )
		if(self.boxWidth): self.optionMenu.config(width=self.boxWidth)

		if(self.direction == "vertical"):
			l.pack(side=mtk.TOP, fill=mtk.X)
			self.optionMenu.pack(side=mtk.BOTTOM, fill=mtk.X, expand=True)
		else:
			l.pack(side=mtk.LEFT, fill=mtk.X)
			self.optionMenu.pack(side=mtk.RIGHT, fill=mtk.X, expand=True)
		
		frame.pack(fill=mtk.X, expand=True)
		return frame

	def valueChanged(self, value):
		if(self.valueChangedCallback):
			self.valueChangedCallback(self, value)

	def getValue(self):
		return self.variable.get()

class GitRepoSelector(View):
	def __init__(self, key, label, repoDict={}, latestRepo = None, latestBranch = None, actionCallback=None, type=0, 
				specialIconClone=None, specialCallbackClone=None):
		super(GitRepoSelector, self).__init__(key)
		self.label = label

		### define repos and branches from repoDict
		self.repoDict = repoDict
		self.latestRepo = latestRepo
		self.latestBranch = latestBranch
		self.actionCallback = actionCallback
		self.chooseList = ["Branch", "Tag"]
		### try loading special things
		self.specialCallbackClone = specialCallbackClone
		self.specialIconClone  = specialIconClone

		self.chooseFieldDefault = "Branch"
		self.branches = []
		self.repositorys = []

		if self.latestRepo and self.latestBranch:
			indexOfLatestRepo = self.repoDict.keys().index(self.latestRepo)

			self.repositorys = self.repoDict.keys()

			for value in self.chooseList:
				if self.latestBranch in self.repoDict.values()[indexOfLatestRepo][value]:
					self.branches = self.repoDict.values()[indexOfLatestRepo][value]
					self.chooseFieldDefault = value
		else:
			self.branches = self.repoDict.values()[0]["Branch"]
			self.repositorys = self.repoDict.keys()

		#for r, bl in repositorys.items():
		#	self.repositorys.append(r)
		#	self.branches.extend(bl)

		self.type=type
		self.REPOLABELSTR = "Repository: "
		self.BRANCHLABELSTR = ""
		self.CHOOSELABELSTR = ""


	def makeView(self, master):
		### grab specific args
		headerbg = self.grabConfigArg("headerbg")
		frame = mtk.Frame(master, **self.configArgs)
		bg = frame.cget('bg')

		frame1 = mtk.Frame(frame, **self.configArgs)
		frame2 = mtk.Frame(frame, **self.configArgs)
		frame3 = mtk.Frame(frame, **self.configArgs)
		frame4 = mtk.Frame(frame, **self.configArgs)
		
		l = Text("", self.label)

		### special configuration for this view
		minLabelWidth = max(len(self.REPOLABELSTR), len(self.BRANCHLABELSTR))
		minBoxWidth = None
		if(self.type==0): minBoxWidth = findLongestStrInLists([self.repositorys, self.branches])
		else: minBoxWidth = 18
		self.repo = OptionField(self.key+"_repo", self.REPOLABELSTR, self.repositorys, direction="horizontal", 
								valueChangedCallback = self.onRepoChanged,
								boxWidth=minBoxWidth, labelWidth=minLabelWidth, defaultEntry=self.latestRepo)
		self.branch = OptionField(self.key+"_branch", self.BRANCHLABELSTR, self.branches, direction="horizontal", 
									valueChangedCallback = self.onBranchChanged,
									boxWidth=minBoxWidth, labelWidth="0", defaultEntry=self.latestBranch)
		self.choose = OptionField(self.key+"_choose", self.CHOOSELABELSTR, self.chooseList, direction="horizontal", 
									valueChangedCallback = self.onChooseChanged,
									boxWidth=10, labelWidth=minLabelWidth, defaultEntry=self.chooseFieldDefault)
		### add special button
		self.specialButtonClone = ActionButton(self.key + "_clone" , "", image=self.specialIconClone, actionCallback=self.onCloneClick, width=18,height=18)
		#self.specialButtonClone = ActionButton("", self.label, actionCallback=self.onCloneClick)
		#buttonTk = self.specialButtonClone.makeView(frame4)
		#buttonbg = convertToColorCode(frame4, chooseButtonColor(self.grabConfigArg("buttoncolor")))
		#self.specialButtonCloneTk = mtk.Button(frame4, compound=mtk.TOP, image=self.specialIconClone, text="",  #anchor='w', 
		#			bg=buttonbg, 
		#			activebackground=colorscale(buttonbg, 0.95),
		#			command = self.onCloneClick)
		#self.specialButtonCloneTk.pack(padx=0, pady=0)

		### configurate these views
		l.config(anchor='w', font='Helvetica 10 bold', bg=headerbg)
		self.repo.config(bg=bg)
		self.branch.config(bg=bg)
		self.choose.config(bg=bg)

		### create those new views for realz
		ltk = l.makeView(frame)
		repotk = self.repo.makeView(frame1)
		choosetk = self.choose.makeView(frame3)
		branchtk = self.branch.makeView(frame2)
		specialClonetk = self.specialButtonClone.makeView(frame4)
		specialClonetk.pack(side=mtk.RIGHT)
		
		ltk.pack(side=mtk.TOP, fill=mtk.X)
		if(self.type==1):
			frame1.pack(side=mtk.LEFT, padx=5, pady=5, fill=mtk.X)
			frame3.pack(side=mtk.LEFT, padx=5, pady=5, fill=mtk.X)
			frame2.pack(side=mtk.LEFT, padx=5, pady=5, fill=mtk.X)
			frame4.pack(side=mtk.LEFT, padx=5, pady=5, fill=mtk.X)
		else:
			frame1.pack(side=mtk.TOP, padx=5, pady=5, anchor="w", fill=mtk.X)
			frame3.pack(side=mtk.TOP, padx=5, pady=5,anchor="w", fill=mtk.X)
			frame2.pack(side=mtk.TOP, padx=5, pady=5,anchor="w", fill=mtk.X)
			frame4.pack(side=mtk.TOP, padx=5, pady=5,anchor="w", fill=mtk.X)
		return frame

	def reloadContents(self):
		self.repositorys = [r for r in self.repoDict.keys()]

	def onRepoChanged(self, sender, newVal):
		#print("repo changed ...")
		#print("setting new options ...")
		branches = self.repoDict[newVal][self.chooseList[0]]
		self.branch.refresh(branches)
		self.choose.refresh(self.chooseList)

	def onChooseChanged(self, sender, newVal):
		#print("repo changed ...")
		#print("setting new options ...")
		repo = self.repo.getValue()
		choose = self.choose.getValue()
		branches = self.repoDict[repo][choose]
		self.branch.refresh(branches)

	def onBranchChanged(self, sender, newVal):
		#print("Branch Changed to %s" % newVal)
		if(self.actionCallback):
			self.actionCallback(self)

	def onCloneClick(self, sender):
		if(self.specialCallbackClone):
			self.specialCallbackClone(sender)

	def getValue(self):
		repo = self.repo.getValue()
		branch = self.branch.getValue()
		return {"repo" : repo, "branch" : branch}


################################################################################
####################### W I N D O W   T H I N G S ##############################
################################################################################


def getWindowDefaults(kwargs):
	kwargs["key"] = "" 				if "key" not in kwargs			else kwargs["key"]
	kwargs["bg"] = "light gray" 	if "bg" not in kwargs			else kwargs["bg"]
	kwargs["position"] = "+100+100"	if "position" not in kwargs 	else kwargs["position"]
	kwargs["size"] = "100x100" 		if "size" not in kwargs 		else kwargs["size"]
	kwargs["title"] = "Window" 		if "title" not in kwargs		else kwargs["title"]
	kwargs["manager"] = None 		if "manager" not in kwargs		else kwargs["manager"]
	return kwargs



class Window(View):
	""" holder class for a window, manages the Tk() object and creates a default layout
		which will be used (vertical layout in this case). It is used as an abstraction layer
		for the user, so that nobody has to manage these references and/or so that the
		user has a consistent idea about what a window is"""
	def __init__(self, **kwargs):
		### set parameter members
		self.__dict__.update(getWindowDefaults(kwargs))
		super(Window, self).__init__(self.key)
		### set advanced behavioral members
		self.ok = True
		self.isClosing = False
		### create first layout
		self.layout = Vertical("layout", padding=5)
		self.friends = []

	def setContext(self, context, icon):
		self.root = context
		self.root.protocol("WM_DELETE_WINDOW", self.close)
		self.root.title(self.key)
		self.root.geometry(self.size + self.position)
		self.root.config(bg=self.bg)
		### binding events of tkinter
		### The size of the widget changed. The new size is provided in the width and height attributes
		### of the event object passed to the callback. On some platforms, it can mean that the location changed. 
		self.root.bind("<Configure>", self.onWindowChange)
		if icon:
			self.root.tk.call('wm', 'iconphoto', self.root._w, mtk.PhotoImage(file=icon))

	def open(self):
		self.layout = self.createViews(self.layout)
		### apply window configs to layout!!!
		self.layout.config(**self.configArgs) 
		self.root.config(**self.configArgs)
		### add main layout to window
		self.addView(self.layout)
		### mainframe parent
		self.tk = self.layout.makeView(self.root)


	def createViews(self, layout):
		return layout


	def update(self):
		### check my closing flag (toggled by the update function closeEvent [same thread])
		if(self.isClosing): 
			self.closeForRealz()
		pass



	### i want to close so badly
	def close(self):
		self.isClosing = True
		self.ok = False

	### overrridable user event callback
	def onClose(self):
		pass

	def closeForRealz(self):
		self.onClose() ### notify user
		self.manager.closeWindow(self)

	def block(self): ### block all other widget events
		self.root.grab_set()
	def unblock(self): ###release all other widget events if necessary
		self.root.grab_release()

	def destroy(self): ### destroy yourself
		self.root.destroy()

	def onWindowChange(self, event):
		for f in self.friends:
			try:
				f.onWindowChange(event)
			except Exception as e:
				pass

	### this view does not return its parent context, it IS the context
	def getContext(self):
		return self.root

	def isOk(self):
		return self.ok
	def getWidth(self):
		return self.root.winfo_width()
	def getHeight(self):
		return self.root.winfo_height()
	def getX(self):
		return self.root.winfo_x()
	def getY(self):
		return self.root.winfo_y() - 55 # 55 to offset window menu bar and shit :D

	def addMenubar(self, barsDict):
		menuBar = mtk.Menu(self.root)
		for key in barsDict.keys():
			menu = mtk.Menu(self.root)
			menuBar.add_cascade(label = key, menu = menu)
			#print barsDict[key].keys()
			#print barsDict[key].values()
			for commandKey, commandValue in barsDict[key].iteritems():
				#print commandKey
				#print commandValue
				menu.add_command(label=commandKey, command = commandValue)
		self.root.config(menu=menuBar)

	def addFriend(self, friend):
		self.friends.append(friend)
	def removeFriend(self, fried):
		self.friends.remove(friend)

class StickyWindow(Window):
	def __init__(self, parent, **kwargs):
		# do something sticky
		self.friend = parent
		### preset some other things for the window attributes
		kwargs["position"] = "+%s+%s" % (self.friend.getX()+self.friend.getWidth(), self.friend.getY())
		super(StickyWindow, self).__init__(**kwargs)
		### tell our parent that we exist
		self.friend.addFriend(self)
		
	### destructor
	def __exit__(self, exc_type, exc_value, traceback):
		self.friend.removeFriend(self)

	#def update(self):
	#	super(StickyWindow, self).update()
	#	self.snapToParent()
	def onWindowChange(self, event):
		self.snapToParent()

	def snapToParent(self):
		### atm snap to the top right of the parent! (:
		try:
			newX = self.friend.getX()+self.friend.getWidth()
			newY = self.friend.getY()
			self.root.geometry('%dx%d+%d+%d' % (self.getWidth(), self.getHeight(), newX, newY))
		except Exception as e:
			self.close()

class WindowManager(object):
	def __init__(self):
		self.windows = []
		self.ok = True
		#self.mutex = Lock()
		self.removeWindows = []
		self.blockingWindows = []
		### create main manager tk context (root)
		self.root = mtk.Tk()
		### make the main context invisible, (it is there, but nobody sees it)
		self.root.withdraw()

	def start(self):
		import time
		while(self.ok):
			time.sleep(0.05)
			### update main context (tk update_idletasks etc)
			self.updateMainContext()
			### update window object views
			self.updatePendingWindows()
			### close all waiting windows if necessary
			self.removePendingWindows()

		print("Window Manager done ...")
		self.root.destroy()
		return
		
	def openWindow(self, window, blocking=False, icon=None):
		### setup window
		newContext = mtk.Toplevel(self.root)
		newContext.deiconify()
		### set new context for window
		window.setContext(newContext, icon)
		### add new window to our list
		self.windows.append(window)
		### check window type
		### blocking windows trigger all windows before them to fuck off :)
		if blocking == True:
			window.block()
			self.blockingWindows.append(window)
		### open the window and let it render its views
		window.open()

	def closeWindow(self, window):
		if window in self.windows:
			self.windows.remove(window)
			self.ok = len(self.windows) != 0
			self.removeWindows.append(window)
			### check and pply if blocking matters for this window
			self.applyBlockingWindowState(window)

	def closeAllWindows(self):
		### Copy windows list. So the list we iterate through is another than the list we remove the windows from
		windows_copy = self.windows[:]
		for window in windows_copy:
			self.windows.remove(window)
			self.removeWindows.append(window)
		self.ok = len(self.windows) != 0
		### close window if others still exist
		### else we are not ok and the manager will close the thread immeadiatley
		### this is only for testing purposes, so that the window closes faster .. 
		#if(self.ok): window.destroy()


	def applyBlockingWindowState(self, window):
		### check if the removed window was blocking ... 
		### if it was blocking, we want to search for other blocking windows in the stack
		### if there is no other, we unblock all windows again
		### else we let the blocking window grab the set by blocking 
		window.unblock()
		if(window in self.blockingWindows):
			self.blockingWindows.remove(window)
			if(len(self.blockingWindows) == 0):
				pass
			else:
				self.blockingWindows[-1].block()


	def updateWindow(self, window):
		#print("Closed Window %s" % window.key)
		window.update()

	def addMenubar(self, window, barsDict):
		window.addMenubar(barsDict)




	### main loop calls
	def updateMainContext(self):
		self.root.update_idletasks()
		self.root.update()

	def updatePendingWindows(self):
		for window in self.windows:
			window.update()
			
	def removePendingWindows(self):
		for window in self.removeWindows:
			window.destroy()
		self.removeWindows = []