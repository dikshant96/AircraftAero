import numpy as np
from scipy.interpolate import interp1d

def get_airfoilpos(file):
    """
    Function to get airfoil coordinates (assumes Selig format dat file)
    :param file: file of airfdoil coordinate
    :return: numpy array of x and y pos
    """
    f = open(file, 'r')
    f_lines = f.readlines()
    f.close()
    pos = np.zeros((len(f_lines)-1, 2))
    for i, line in enumerate(f_lines):
        if i != 0:
            line = line.split()
            x = float(line[0])
            y = float(line[1])
            pos[i-1, 0] = x
            pos[i-1, 1] = y
        else:
            pass
    return pos

def gen_NACAfoil(n,m,p,t):
    """
    NACA 4 series airfoil generator
    :param n: number of points to define airfoil on
    :param m: 1st number (camber)
    :param p: 2nd number (max camber pos)
    :param t: last two numbers (thickness)
    :return: position same format as selig data
    """
    t = t/100.
    m = m/100.
    p = p/10.
    x = np.linspace(0,1,n)
    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
    #thickness distribution
    yt = (t/0.2)*(a0*x**0.5 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
    yc = np.zeros((n,1))
    theta  = np.zeros((n,1))
    #camber distribution
    for i,ix in enumerate(x):
        if 0 <= ix < p:
            yc[i] = (m/p**2)*(2*p*ix - ix**2)
            dyc = (2*m/p**2)*(p-ix)
            theta[i] = np.arctan(dyc)
        elif p <= ix <= 1:
            yc[i] = (m/(1-p)**2)*(1 - 2*p + 2*p*ix - ix**2)
            dyc = (2*m/(1-p)**2)*(p - ix)
            theta[i] = np.arctan(dyc)
    pos = np.zeros((n*2,2))
    j = n
    for i in range(n):
        pos[i, 0] = x[i] - yt[i] * np.sin(theta[i])
        pos[i, 1] = yc[i] + yt[i] * np.cos(theta[i])
        pos[j, 0] = x[i] + yt[i] * np.sin(theta[i])
        pos[j, 1] = yc[i] - yt[i] * np.cos(theta[i])
        j += 1
    pos = np.delete(pos, n, 0)
    pos[:n, :] = pos[n - 1::-1, :]
    return pos, yc

def get_dist(n,bool):
    """
    Defines cosine discretisation to better capture LE and TE gradients
    than equally spaced x distribution
    :param n: number of points
    :return: x vectore with cosine distribution
    """
    if bool == 'cosine':
        beta = np.linspace(0,np.pi,n)
        x = (1-np.cos(beta))/2
    else:
        x = np.linspace(0,1,n)
    return x

def get_camber(pos,n,bool):
    """
    Function to return mean camber line with cosine discretisation
    from upper and lower pos of profile
    :param pos: position airfoil selig data
    :param n: number of discretisation points
    :return: x, y position of discretised camber line
    """
    index_middle = np.argsort(pos[:, 0] - 0.0)[0]
    xtop = pos[:index_middle + 1, 0]
    ytop = pos[:index_middle + 1, 1]
    sort_top = np.argsort(xtop)
    xtop = xtop[sort_top]
    ytop = ytop[sort_top]
    xbottom = pos[index_middle:, 0]
    ybottom = pos[index_middle:, 1]
    sort_bottom = np.argsort(xbottom)
    xbottom = xbottom[sort_bottom]
    ybottom = ybottom[sort_bottom]
    ftop = interp1d(xtop,ytop)
    fbottom = interp1d(xbottom,ybottom)
    x = get_dist(n,bool)
    camber_y = (ftop(x) + fbottom(x))/2.
    return x, camber_y

def get_colpoints(x,y):
    """
    Function takes camber discretisation points and get collocation points
    :param x: x position of vortex points
    :param y: y position of vortex points
    :return: x, y position of collocation points
    """
    xcol = np.zeros((len(x)-1,1))
    ycol = np.zeros((len(y)-1,1))
    j = 0
    for i in range(1,len(x)-1):
        xcol[j] = x[i-1] + (x[i]-x[i-1])/2.
        ycol[j] = y[i-1] + (y[i]-y[i-1])/2.
        j += 1
    return xcol, ycol

def get_coeffMatrix(panels,n):
    A = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(n - 1):
            dx = panels[i].cpoint.x - panels[j].vpoint.x
            dy = panels[i].cpoint.y - panels[j].vpoint.y
            r2 = dx ** 2 + dy ** 2
            Asmall = np.array([dy, -dx])
            A[i, j] = (1 / (2 * np.pi * r2)) * (Asmall[0] * panels[i].n1 + Asmall[1] * panels[i].n2)
    return A

def get_RHS(U,alfa,panels,n):
    RHS = np.zeros((n - 1, 1))
    for i in range(len(RHS)):
        RHS[i] = -U * np.sin(alfa + panels[i].beta)
    return RHS

def get_tangMatrix(panels,n):
    B = np.zeros((n-1,n-1))
    for i in range(n-1):
        for j in range(n-1):
            dx = panels[i].cpoint.x - panels[j].vpoint.x
            dy = panels[i].cpoint.y - panels[j].vpoint.y
            r2 = dx ** 2 + dy ** 2
            Bsmall = np.array([dy, -dx])
            B[i, j] = (1 / (2 * np.pi * r2)) * (Bsmall[0] * panels[i].t1 + Bsmall[1] * panels[i].t2)
    return B

def post_process(U,alfa,Gamma,panels,n):
    B = get_tangMatrix(panels,n)
    V1 = np.matmul(B,Gamma)
    V2 = np.zeros((n-1,1))
    for i in range(len(V2)):
        V2[i] = U*np.cos(alfa+panels[i].beta)
    ut = V2 + V1
    return ut
