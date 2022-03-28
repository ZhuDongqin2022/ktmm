from tkinter import *
import Task1

#初始化Tk()
myWindow = Tk()
#设置标题
myWindow.title('Task1')
#设置窗口大小
width = 380
height = 300
#获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
screenwidth = myWindow.winfo_screenwidth()
screenheight = myWindow.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth-width)/2, (screenheight-height)/2)
myWindow.geometry(alignstr)
#设置窗口是否可变长、宽，True：可变，False：不可变
myWindow.resizable(width=False, height=True)


v1 = StringVar()
v2 = StringVar()
v3 = StringVar()
v4 = StringVar()
v5 = StringVar()
v6 = StringVar()

e1 = Entry(myWindow, width=25, textvariable=v1, state="readonly", justify='center').grid(row=0, column=1)
e2 = Entry(myWindow, width=25, textvariable=v2, state="readonly", justify='center').grid(row=1, column=1)
e3 = Entry(myWindow, width=25, textvariable=v3, state="readonly", justify='center').grid(row=2, column=1)
e4 = Entry(myWindow, width=25, textvariable=v4, state="readonly", justify='center').grid(row=3, column=1)
e5 = Entry(myWindow, width=25, textvariable=v5, state="readonly", justify='center').grid(row=4, column=1)
e5 = Entry(myWindow, width=25, textvariable=v6, state="readonly", justify='center').grid(row=5, column=1)

def text(st,v):
    v.set(str(st))

def Tem(t,v):
    v.set(float(t))

def Time(v):
    T = Task1.T_total
    v.set(float(T))

Tem_max, Tem_min = Task1.faces_points()
Button(myWindow, text="Вариант", width=15, fg="blue", command=lambda:text('Вариант4(model2)', v1)).grid(row=0, column=0, sticky=W, padx=10, pady=5)
Button(myWindow, text="Коэффициент", width=15, fg="blue", command=lambda:text('Data.csv', v2)).grid(row=1, column=0, sticky=W, padx=10, pady=5)
Button(myWindow, text="Температур", width=15, fg="blue", command=lambda:text('Tem.csv', v3)).grid(row=2, column=0, sticky=W, padx=10, pady=5)
Button(myWindow, text="T_max", width=15, fg="blue", command=lambda:Tem(Tem_max, v4)).grid(row=3, column=0, sticky=W, padx=10, pady=5)
Button(myWindow, text="T_min", width=15, fg="blue", command=lambda:Tem(Tem_min, v5)).grid(row=4, column=0, sticky=W, padx=10, pady=5)
Button(myWindow, text="Общее время", width=15, fg="blue", command=lambda:Time(v6)).grid(row=5, column=0, sticky=W, padx=10, pady=5)

myWindow.mainloop()


