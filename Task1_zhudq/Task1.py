#Zhu Dongqin
import math
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

global T_total, T_choose
T_total = 100
T_choose = 15 ##选择一个时间对应温度

def read_obj(obj_path):
    with open(obj_path) as file:
        points = []
        faces = []
        part = np.zeros(5)
        count = 0
        k = 0
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[2]), float(strs[3]), float(strs[4])))
            if strs[0] == "f":
                faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
                count = count + 1
            if strs[0] == "g":
                part[k] = count
                k = k + 1
    points = np.array(points)
    faces = np.array(faces)
    return points, faces, part


def triangle_area(x, y, z):
    a = ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2) ** 0.5
    b = ((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2 + (x[2] - z[2]) ** 2) ** 0.5
    c = ((z[0] - y[0]) ** 2 + (z[1] - y[1]) ** 2 + (z[2] - y[2]) ** 2) ** 0.5
    p = (a+b+c)/2
    S = ( p*(p-a)*(p-b)*(p-c))**0.5
    return S


def surface_Si(f, v):
    sum = 0
    for i in range(f.shape[0]):
        #print(f[i, 0], v[f[i, 0] - 1], f[i, 1], v[f[i, 1] - 1], f[i, 2], v[f[i, 2] - 1])
        sum = sum+triangle_area(v[f[i, 0]-1], v[f[i, 1]-1], v[f[i, 2]-1])
    return  sum


def surface_Sij(f1, f2, f3, f4, f5, v):
    ftemp = []
    #f1-f2
    for i in range(f1.shape[0]):
        if (v[f1[i,0]-1,2]==6 and v[f1[i,1]-1,2]==6 and v[f1[i,2]-1,2]==6):
            ftemp.append(f1[i])
    S12 = surface_Si(np.array(ftemp),v)

    ftemp = []
    # f2-f3
    for i in range(f2.shape[0]):
        if (v[f2[i,0]-1,2]==0 and v[f2[i,1]-1,2]==0 and v[f2[i,2]-1,2]==0):
            ftemp.append(f2[i])
    S23 = surface_Si(np.array(ftemp),v)

    ftemp = []
    # f3-f4
    for i in range(f4.shape[0]):
        if (v[f4[i, 0] - 1, 1] == -5 and v[f4[i, 1] - 1, 1] == -5 and v[f4[i, 2] - 1, 1] == -5):
            ftemp.append(f4[i])
    S34 = surface_Si(np.array(ftemp), v)

    ftemp = []
    # f3-f4
    for i in range(f4.shape[0]):
        if (v[f4[i, 0] - 1, 1] == -6 and v[f4[i, 1] - 1, 1] == -6 and v[f4[i, 2] - 1, 1] == -6):
            ftemp.append(f4[i])
    S45 = surface_Si(np.array(ftemp), v)

    return S12, S23, S34, S45


def sys_ode(y, t, ci, kij, e, S,Q_iR_1 ,Q_iR_2):
    C0 = 5.67
    y1, y2, y3, y4, y5 = y[0], y[1], y[2], y[3], y[4]
    t1 = (1 / ci[0]) * (-kij[0] * (y2 - y1) - e[0] * S[0] * C0 * ((y1 / 100) ** 4) + Q_iR_1[0] + Q_iR_2[0] * np.cos(t / 4))
    t2 = (1 / ci[1]) * (-kij[0] * (y2 - y1) - kij[1] * (y3 - y2) - e[1] * S[1] * C0 * ((y2 / 100) ** 4) )
    t3 = (1 / ci[2]) * (-kij[1] * (y3 - y2) - kij[2] * (y4 - y3) - e[2] * S[2] * C0 * ((y3 / 100) ** 4) )
    t4 = (1 / ci[3]) * (-kij[2] * (y4 - y3) - kij[3] * (y5 - y4) - e[3] * S[3] * C0 * ((y4 / 100) ** 4) )
    t5 = (1 / ci[4]) * (-kij[3] * (y5 - y4) - e[4] * S[4] * C0 * ((y5 / 100) ** 4) )
    return [t1, t2, t3, t4, t5]


def fun(y, kij, e, S):
    C0 = 5.67
    y1, y2, y3, y4, y5 = y[0], y[1], y[2], y[3], y[4]
    return [kij[0] * (y2 - y1) + e[0] * S[0] * C0 * ((y1 / 100) ** 4),
     -kij[0] * (y2 - y1) - kij[1] * (y3 - y2) - e[1] * S[1] * C0 * ((y2 / 100) ** 4),
     -kij[1] * (y3 - y2) - kij[2] * (y4 - y3) - e[2] * S[2] * C0 * ((y3 / 100) ** 4),
     -kij[2] * (y4 - y3) - kij[3] * (y5 - y4) - e[3] * S[3] * C0 * ((y4 / 100) ** 4),
     -kij[3] * (y5 - y4) - e[4] * S[4] * C0 * ((y5 / 100) ** 4)]


def picture(sol, t):
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('T')
    plt.plot(t, sol[:, 0], color='b', label=r'$T_1$')
    plt.plot(t, sol[:, 1], color='c', label=r'$T_2$')
    plt.plot(t, sol[:, 2], color='r', label=r'$T_3$')
    plt.plot(t, sol[:, 3], color='m', label=r'$T_4$')
    plt.plot(t, sol[:, 4], color='g', label=r'$T_5$')
    plt.legend(loc='best')
    plt.savefig("Temperature_time.png")
    plt.show()



#---------------------opengl----------------------#

IS_PERSPECTIVE = True  # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE = np.array([0.0, 0.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False  # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

def getposture():
    global EYE, LOOK_A
    dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
        theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0
    return dist, phi, theta


DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）

def painsurface(face, n):
    for i in range(face.shape[0]):
        glBegin(GL_TRIANGLES)
        glColor3f(Tem_color[n-1, 0],Tem_color[n-1, 1],Tem_color[n-1, 2])
        #print(Tem_color[n-1, 0],Tem_color[n-1, 1],Tem_color[n-1, 2])
        glVertex3f(v[face[i, 0] - 1, 0], v[face[i, 0] - 1, 1], v[face[i, 0] - 1, 2])
        glVertex3f(v[face[i, 1] - 1, 0], v[face[i, 1] - 1, 1], v[face[i, 1] - 1, 2])
        glVertex3f(v[face[i, 2] - 1, 0], v[face[i, 2] - 1, 1], v[face[i, 2] - 1, 2])
        glEnd()

def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H

    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])
    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)
    # ---------------------------------------------------------------
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）
    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-10, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(10, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）
    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -10, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, 10, 0.0)  # 设置y轴顶点（y轴正方向）
    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -10)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, 10)  # 设置z轴顶点（z轴正方向）
    glEnd()  # 结束绘制线段

    # ---------------------------------------------------------------
    painsurface(f1, 1)
    painsurface(f2, 2)
    painsurface(f3, 3)
    painsurface(f4, 4)
    painsurface(f5, 5)
    # ---------------------------------------------------------------

    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def reshape(width, height):
    global WIN_W, WIN_H
    WIN_W, WIN_H = width, height
    glutPostRedisplay()


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y
    MOUSE_X, MOUSE_Y = x, y

    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN

    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H
    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y
        PHI += 2 * np.pi * dy / WIN_H
        PHI %= 2 * np.pi
        THETA += 2 * np.pi * dx / WIN_W
    THETA %= 2 * np.pi
    r = DIST * np.cos(PHI)

    EYE[1] = DIST * np.sin(PHI)
    EYE[0] = r * np.sin(THETA)
    EYE[2] = r * np.cos(THETA)
    if 0.5 * np.pi < PHI < 1.5 * np.pi:
        EYE_UP[1] = -1.0
    else:
        EYE_UP[1] = 1.0
        glutPostRedisplay()


def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW
    if key == b'\r':  # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()

    elif key == b'\x08':  # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()

    elif key == b' ':  # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE
        glutPostRedisplay()



def run():
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 300)
    glutCreateWindow(b'model of OpenGL')

    init()  # 初始化画布
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()
    glutMainLoop()  # 进入glut主循环


#hsv-->rgb
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b




#if __name__ == "__main__":
def faces_points():
    objfile = "model2.obj"
    global v, f1, f2, f3, f4, f5, Tem_color
    v, ff, part = read_obj(objfile)
    f1 = ff[0:int(part[1]), :]
    f2 = ff[int(part[1]):int(part[2]), :]
    f4 = ff[int(part[2]):int(part[3]), :]
    f3 = ff[int(part[3]):int(part[4]), :]
    f5 = ff[int(part[4]):, :]
    #f = np.vstack((f1, f2, f3, f4, f5))

    ## Si Sij
    S1 = surface_Si(f1, v)
    S2 = surface_Si(f2, v)
    S3 = surface_Si(f3, v)
    S4 = surface_Si(f4, v)
    S5 = surface_Si(f5, v)

    S12, S23, S34, S45 = surface_Sij(f1, f2, f3, f4, f5, v)
    Sij = [S12, S23, S34, S45]
    print('Sij:',S12, S23, S34, S45)
    S1 = S1 - S12
    S2 = S2 - S12 - S23
    S3 = S3 - S23 - S34
    S4 = S4 - S34 - S45
    S5 = S5 - S45
    S = [S1, S2, S3, S4, S5]
    print('Si:',S)


    ## 导入参数_csv
    Data = pd.read_csv('Data.csv')
    Data.columns = ["c", "lambda", "epsilon", "Q_iR","Q_iR_1","Q_iR_2"]
    ci = np.array(Data["c"])
    e = np.array(Data["epsilon"])
    lam = np.array(Data["lambda"])
    Q_iR_1 = np.array(Data["Q_iR_1"])
    Q_iR_2 = np.array(Data["Q_iR_2"])
    #print(Q_iR[0]) type(str)

    kij = np.zeros(4)
    for i in range(4):
        kij[i] = lam[i] * Sij[i]
    #print(kij)

    ## 计算初值
    #init = np.ndarray([1, 0, 0, 0, 0])
    #result = fsolve(fun, [20, 20, 20, 20, 20], args=(kij, e, S))
    #print(result)

    ## 进行ODE
    x = 5 #每一小段时长
    initz = 22, 20, 22, 20, 20
    #initz = 20, 20, 20, 20, 20
    t = np.linspace(0, x, 11)
    sol = odeint(sys_ode, initz, t, args=(ci, kij, e, S, Q_iR_1,Q_iR_2))
    #print(round(T_total/x))
    for i in range(round(T_total/x)-1):
        t = np.linspace(x * (i + 1), x*(i+2), 11)
        sol1 = odeint(sys_ode, initz, t, args=(ci, kij, e, S,Q_iR_1,Q_iR_2))
        sol = np.vstack((sol, sol1[1:sol1.shape[0], :]))
    t = np.linspace(0, T_total, 10 * round(T_total/x)+1)
    picture(sol, t) #print(t)

    ##写入csv
    Data2 = pd.DataFrame({'t': t, 'T1': sol[:, 0], 'T2': sol[:, 1], 'T3': sol[:, 2], 'T4': sol[:, 3], 'T5': sol[:, 4]})
    Data2.to_csv("Tem.csv", index=False, sep=',')

    ##选一点时间计算温度
    cha = np.abs(t-T_choose)
    #print(t[np.where(cha==np.min(cha))])
    index = np.where(cha==np.min(cha))[0]
    #print('T_Choose:',t[index])
    T1, T2, T3, T4, T5 = sol[index,0], sol[index,1], sol[index,2], sol[index,3],sol[index,4]
    print('Temperature:',T1, T2, T3, T4, T5)
    Tem_index = np.array([T1, T2, T3, T4, T5])
    Tem_min = np.min(Tem_index)
    Tem_max = np.max(Tem_index)
    #Tem_hsv#温度对应hsv值
    Tem_rgb = []#温度对应rgb值
    for i in range(5):
        Tem_hsv =250-(Tem_index[i]-Tem_min)/(Tem_max-Tem_min)*250
        Tem_rgb.append(hsv2rgb(Tem_hsv, 1, 0.8))
    Tem_color = np.array(Tem_rgb)/255
    #print(Tem_color)
    return Tem_max, Tem_min


if __name__ == "__main__":

    #T-t温度时间图
    faces_points()
    ##选一点时间在3D图像上可视化
    run()
