#text update lib made for HackNC project, window with simple text updates
from tkinter import *

class myGUI():

        #initialize the class
        def __init__(self):
                self.root = Tk()
                self.v = StringVar()
                self.root.title("Custom Gesture")
                #set text and window properties on the line below
                w = Label(self.root, textvariable=self.v, width = 10, height = 3, font=("Helvetica", 72), bg="Black", fg = "Green")
                self.v.set("testtext")
                w.pack()

        #call this in your other code to update the text
        def setText(self, text):
                self.v.set(text)
                self.root.update()