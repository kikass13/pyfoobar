
import threading
import time
import os
import json
import sys

import tkguimonster



GIT_CLONE_BUTTON_ICON_PATH = os.path.join(os.path.dirname(__file__), "download.png")


class UserInputWindow(tkguimonster.Window):
	def __init__(self, **kwargs):
		super(UserInputWindow, self).__init__(**kwargs)
		self.selectors = []
		self.stickyWindow = None

	def createViews(self, layout):
		self.root.protocol("WM_DELETE_WINDOW", self.btnExitClicked)
		### config my window
		myColor = "white"
		self.config(bg=myColor)
		tkguimonster.setGlobalButtonColor("lavender")

		### Entryfields for username and path to robot info
		self.userName = tkguimonster.EntryField("field_user", "Your Name: ")

		
		repoDict = {
						"repo1" :  {"Branch" : ["branch1", "branch2"], "Tag" : ["tag1", "tag2"]},
						"repo2" :  {"Branch" : ["foo", "bar"], "Tag" : []},
					}

		for r,c in repoDict.items():
			selector = tkguimonster.GitRepoSelector( "id_%s" % r, "", repoDict=repoDict,
											latestRepo=None, latestBranch=None,
											actionCallback=self.repoSelectCallback,
											type=1,
											specialIconClone=GIT_CLONE_BUTTON_ICON_PATH, specialCallbackClone=self.cloneSingleRepo)
		selector.config(bg="white smoke", headerbg="coral")
		self.selectors.append(selector)

		### do some other stuff
		vertL = tkguimonster.Vertical("", padding=0)
		vertR = tkguimonster.Vertical("", padding=0)

		### Option fields for hosts
		self.masterHost = tkguimonster.OptionField("masterHost", "Master host: ", [], direction="horizontal", boxWidth=12, defaultEntry=None)
		self.deployHost = tkguimonster.OptionField("deployHost", "Deploy host: ", [], direction="horizontal", boxWidth=12, defaultEntry=None)

		### Command fields
		self.commandFields = [
								tkguimonster.ButtonField("run", "run", actionCallback=self.cmdBtnClicked, height=2, width=20),
								tkguimonster.ButtonField("clean", "Clean", actionCallback=self.cmdBtnClicked, height=2, width=20),
							]

		### Checkboxes
		self.cbx1 = tkguimonster.CheckboxField("cbx1_", "i want things hard", selectColor="lavender")

		self.btnOptions = tkguimonster.ActionButton("btn_options", "More options", height=1, width=15, actionCallback=self.btnOptionsClicked)

		### Message field
		vMessages = tkguimonster.Vertical("", padding=0)
		self.messageHeader = tkguimonster.Text("messageHeader", "Messagebox")
		self.message = tkguimonster.Text("message", "")
		self.messageHeader.config(font='Helvetica 10 bold')
		self.message.config(fg="red", font='Helvetica 10 bold')
		vMessages.addView(self.messageHeader)
		vMessages.addView(self.message)

		###config these
		self.userName.config(bg=myColor)
		self.masterHost.config(bg=myColor)
		self.deployHost.config(bg=myColor)
		for v in self.commandFields:
			v.config(bg="white smoke")
		self.cbx1.config(bg=myColor)

		##add these to our window layout
		layout.addView(self.userName)
		layout.addView(self.selectors)

		### change padding from the main layout of this window
		layout.padding=5
		##add these to our window layout
		layout.addView(self.masterHost)
		layout.addView(self.deployHost)
		layout.addView(self.commandFields[5:])
		layout.addView(self.btnOptions)
		layout.addView(vMessages)

		return layout

	def repoSelectCallback(self, sender):
		print("Selected something: %s" % sender)

	def btnOptionsClicked(self, sender):
		### we open a sticky window
		textCommandFields = tkguimonster.Text("text", "Deploy commands")
		textCommandFields.config(bg="coral", font='Helvetica 10 bold')

		textCheckButtons = tkguimonster.Text("text", "Skip some things")
		textCheckButtons.config(bg="coral", font='Helvetica 10 bold')

		fields = [textCommandFields, self.commandFields[:5], textCheckButtons, self.cbx1]
		if self.stickyWindow is None or self.stickyWindow.isOk() == False:
			self.stickyWindow = stickyOptions(self, fields, manager=self.manager, key="More Options", title="stickstoock", size="400x450")
			self.manager.openWindow(self.stickyWindow)


	def btnExitClicked(self):
		# Open a Closing  Window
		exitWindow = ExitWindow(manager=self.manager,  key="Exit Startscript", size="250x200")
		self.manager.openWindow(exitWindow, blocking=True)
		#self.manager.closeAllWindows()


	def cmdBtnClicked(self, sender):
		command = sender.key
		### do stuff threaded with these actions
		t = threading.Thread(name='my_service', target=lambda:(self.disableButtons(sender), self.executeAction(command), self.enableButtons(sender)) )
		t.start()

	def cloneSingleRepo(self, sender):
		self.cmdBtnClicked(sender)

	def executeAction(self, command):
		### Get inputs
		masterHostMachine = self.masterHost.getValue()
		deployHostMachine = self.deployHost.getValue()

		### Check if the date on every host is nearly the same
		if command == 'run':
			self.message.refresh("running ... ")
			print("running action command ...")
			time.sleep(2)
			self.message.refresh("")
		else:
			time.sleep(2)
			print("Some other stuff ... [%s]" % command)


	def disableButtons(self, sender):
		sender.setColor("red")
		sender.disable()
		for field in self.commandFields:
			if field.key is not "suicide":
				try:
					field.disable()
				except:
					pass

	def enableButtons(self, sender):
		sender.resetColor()
		sender.enable()
		for field in self.commandFields:
			try:
				field.resetColor()
				field.enable()
			except:
				pass


class ExitWindow(tkguimonster.Window):
	def __init__(self, **kwargs):
		super(ExitWindow, self).__init__(**kwargs)

	def createViews(self, layout):
		myColor = "white"
		self.config(bg=myColor)

		v = tkguimonster.Vertical("", padding=15)
		text = tkguimonster.Text("text", "Stop Hosts and Driver-Hardware?")
		text.config(bg=myColor, font='Helvetica 10 bold')

		h = tkguimonster.Horizontal("", padding=0)
		answerNo = tkguimonster.ActionButton("no", "No", height=3, width=5, actionCallback=self.answerIsNo)
		answerYes = tkguimonster.ActionButton("yes", "Yes", height=3, width=5, actionCallback=self.answerIsYes)
		h.addView(answerNo)
		h.addView(answerYes)
		v.addView(text)
		v.addView(h)
		layout.addView(v)

		###done layout
		return layout

	def answerIsNo(self, sender):
		self.manager.closeAllWindows()
		self.close()
	def answerIsYes(self, sender):
		self.manager.closeAllWindows()
		self.close()


class stickyOptions(tkguimonster.StickyWindow):
	def __init__(self, parent, fields, **kwargs):
		super(stickyOptions, self).__init__(parent, **kwargs)
		self.fields = fields

	def createViews(self, layout):
		myColor = "white"
		self.config(bg=myColor)

		v = tkguimonster.Vertical("", padding=0)
		for field in self.fields:
			v.addView(field)
		layout.addView(v)

		###done layout
		return layout


def main():
	wm = tkguimonster.WindowManager()
	userInputWindow = UserInputWindow(manager=wm, key="Startscript", size="900x640")
	wm.openWindow(userInputWindow)
	wm.start()

if __name__ == '__main__':
	main()
