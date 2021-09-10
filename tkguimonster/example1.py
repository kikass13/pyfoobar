
import threading
import time
import os
import json
import sys

import tkguimonster


class UserInputWindow(tkguimonster.Window):
	def __init__(self, **kwargs):
		super(UserInputWindow, self).__init__(**kwargs)

	def createViews(self, layout):
		### config my window
		myColor = "white"
		self.config(bg=myColor)
		tkguimonster.setGlobalButtonColor("lavender")
		### Command fields
		self.commandFields = [
		  tkguimonster.ButtonField("run", "Run", actionCallback=self.cmdBtnClicked, height=2, width=20),
		  tkguimonster.ButtonField("watch", "Watch", actionCallback=self.cmdBtnClicked, height=2, width=20),
		  tkguimonster.ButtonField("eat", "Eat", actionCallback=self.cmdBtnClicked, height=2, width=20),
		  tkguimonster.ButtonField("clean", "Clean", actionCallback=self.cmdBtnClicked, height=2, width=20),
		]

		### Message field
		vMessages = tkguimonster.Vertical("", padding=0)
		###config these
		for v in self.commandFields:
			v.config(bg="white smoke")
		### change padding from the main layout of this window
		layout.padding=5
		##add these to our window layout
		layout.addView(self.commandFields)
		return layout

	def cmdBtnClicked(self, sender):
		command = sender.key
		### do stuff threaded with these actions
		t = threading.Thread(name='my_service', target=lambda:(self.disableButtons(sender), self.executeAction(command), self.enableButtons(sender)) )
		t.start()

	def executeAction(self, command):
		print("[%s] ..." % command)
		time.sleep(2)

		if(command == "run"):
			pass
		elif(command == "eat"):
			pass
		### ...

	def disableButtons(self, sender):
		sender.setColor("red")
		sender.disable()
		for field in self.commandFields:
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

def main():
	wm = tkguimonster.WindowManager()
	userInputWindow = UserInputWindow(manager=wm, key="win1", size="300x300")
	wm.openWindow(userInputWindow)
	wm.start()

if __name__ == '__main__':
	main()
