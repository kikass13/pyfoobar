# tkguimonster

## General
This is a window and layout GUI library extension for python tk. Because tk is inherently bad and unusablew for proper scaling and dynamic layouting, I  added an abstraction layer between the tklib and the user code inside a managed (multithreadable) framework. 

The Main process can create a manager instance which handles all the interior and the user code can specify windows and their layout / content. These windows will contain Views, which specify their underlying tk stuff themselfs and provide a nice callback-based abstraction layer.

I added some fancy View-Elements which had to do with Git-repository + branch/tag selection and some sticky windows, because these are allways nice to have. I will eventually do more elements like graphs or tables ... :)


## Dependencies

You'll need the following:

 * python2.7
 * python-pip2
 * mttkinter (via pip)
 
## Simple Example (example1.py):

I want to create a Window called ```win1``` and inside that window i want to have 10 Buttons in a ```Vertical``` layout (top to bottom) . Each button gets it's own id (key) and I want all of those buttons to activate my callback function ```self.btnCallback``` when the button is pressed. Inside that function, I check which button was pressed by evaluating the ```sender``` object and doing the things I want to do for each button.

```
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


```
