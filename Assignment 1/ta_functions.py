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

def get_cosdist(n):
    """
    Defines cosine discretisation to better capture LE and TE gradients
    than equally spaced x distribution
    :param n: number of points
    :return: x vectore with cosine distribution
    """
    beta = np.linspace(0,np.pi,n)
    x = (1-np.cos(beta))/2
    return x

def get_camber(pos,n):
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
    x = get_cosdist(n)
    camber_y = (ftop(x) + fbottom(x))/2.
    return x, camber_y


