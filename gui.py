from tkinter import *
from PIL import ImageTk,Image 

root = Tk()
root.configure(bg='#141414')
root.wm_title("Social Distancing Detector")
root.geometry("550x500")
root.resizable(width=True, height=True)

def onclick():
    label2 = Label(root, text='clicked')
    label2.pack()


Label1 = Label(root, text='Social Distancing Detector')
Label1.configure(font=("Avenir", 20),bg='#141414', fg="#FCA311" )
Label1.pack()

video_path = Entry(root, borderwidth=.1)

video_path.pack()

button1 = Button(root, text='Submit', padx=30, pady=5,
                 command=onclick)
button1.configure(font=("Avenir", 15), fg="#141414", highlightbackground='#FFFFFF',relief=GROOVE, borderwidth=.1)
button1.pack()


canvas = Canvas(root, width = 300, height = 300)      

img = Image.open("img.jpg")
img = img.resize((310, 310), Image.ANTIALIAS)  
img = ImageTk.PhotoImage(img)
canvas.create_image(0,0, anchor=NW, image=img) 
canvas.configure(borderwidth=.5)
canvas.pack() 

root.mainloop()
# graph showing social distancing voilation with time
# #141414
# #14213D
# #FCA311
# #E5E5E5
# #FFFFFF
