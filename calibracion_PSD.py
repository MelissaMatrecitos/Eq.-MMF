#importar librerias
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.visualization import astropy_mpl_style
#plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy.fft import fft, fftfreq
import math 
from scipy.ndimage import gaussian_filter

#*************************************************************************
#Cargar la imagen
image_file = get_pkg_data_filename('calib-100x-400mW_11.23.11.fits')
image= fits.getdata(image_file, ext=0)

#Guardar el número de pixeles e imágenes
n_images=image.shape[0]
c_max=image.shape[1]
r_max=image.shape[2]
#*************************************************************************

#*************************************************************************
#OBTENCIÓN DE LOS DATOS (PUNTOS DE MÁXIMA INTENSIDAD)

#Ajuste gaussiano
# Our function to fit is going to be a sum of two-dimensional Gaussians
def gaussian(x, y, x0, y0, xalpha, yalpha, A):
	return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

#Ajuste gaussiano
def _gaussian(M, *args):
	x, y = M
	arr = np.zeros(x.shape)
	for i in range(len(args)//5):
	   arr += gaussian(x, y, *args[i*5:i*5+5])
	return arr

# dense output mesh, 20x21 in shape
x=np.linspace(0,r_max-1,r_max)
y=np.linspace(0,c_max-1,c_max)
y_mesh,x_mesh = np.meshgrid(x, y)

# We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
xdata = np.vstack((x_mesh.ravel(), y_mesh.ravel()))

x_max=np.zeros(n_images)
y_max=np.zeros(n_images)
t=np.zeros(n_images)
for k in range (n_images):
	#construir la matrix de pixeles
	matriz_pixeles=np.zeros((c_max,r_max))
	for i in range (0,c_max-1): 
		for j in range (0,r_max-1):
			matriz_pixeles[i,j]=image[k,i,j]
			
	"""
	#Para ver los datos
	# Agregamos los puntos en el plano 3D
	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')
	for i in range (0,c_max-1): 
		for j in range (0,r_max-1):
			plt.plot(i, j, matriz_pixeles[i,j], c='c', marker='.')

	# Mostramos el gráfico
	###plt.savefig('Datos.jpeg', format='jpeg', dpi=300)
	plt.show()
	"""
	
	   
	# Initial guesses to the fit parameters.
	p0 = [0, 0, 1, 1, 2]

	
	# Do the fit, using our custom _gaussian function which understands our
	# flattened (ravelled) ordering of the data points.
	popt, pcov = curve_fit(_gaussian, xdata, matriz_pixeles.ravel(), p0)
	fit = np.zeros(matriz_pixeles.shape)
	for i in range(len(popt)//5):
		fit += gaussian(x_mesh, y_mesh, *popt[i*5:i*5+5])
	x_max[k]=popt[0]
	y_max[k]=popt[1]
	"""
	# Plot the 3D figure of the fitted function and the residuals.
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(x_mesh, y_mesh, fit, cmap='plasma')
	#cset = ax.contourf(x_mesh, y_mesh, matriz_pixeles-fit, zdir='z', offset=-4, cmap='plasma')
	ax.set_zlim(-4,np.max(fit))
	plt.show()
	"""
#*************************************************************************

#*************************************************************************
#GUARDAR DATOS Y GRÁFICARLOS EN BRUTO

#Transfromar escala a micras (depende del objetivo)
x_max=0.0796*x_max
y_max=0.0796*y_max

#Escribir máximo en datos
archivo_x='Datos_x.txt'
archivo_y='Datos_y.txt'
Ajustes_x=open(archivo_x,'a')
Ajustes_y=open(archivo_y,'a')
Ajustes_x.write('Time x_max \n')
Ajustes_y.write('Time x_max \n')

#Escribir datos en archivo
for i in range (n_images):
	t[i]=i*0.0033
	Ajustes_x.write('{:.8f} {:.16f}'.format(t[i],x_max[i]) +'\n')	
	Ajustes_y.write('{:.8f} {:.16f}'.format(t[i],y_max[i]) +'\n')	
Ajustes_x.close()
Ajustes_y.close()


# Gráfica de Posición contra tiempo
plt.plot(t, x_max, '.-', color='royalblue', label='x')
plt.legend(loc='best')
plt.ylabel("Posición ($\mu$m)")
plt.xlabel("Tiempo (s)")
plt.ylim(0.45,0.6)
plt.grid(True)
plt.savefig('xvst.jpeg', format='jpeg', dpi=300)
plt.close()

# Gráfica de Posición contra tiempo
plt.plot(t, y_max, '.-', color='deeppink', label='y')
plt.legend(loc='best')
plt.ylabel("Posición ($\mu$m)")
plt.xlabel("Tiempo (s)")
plt.ylim(0.75,0.9)
plt.grid(True)
plt.savefig('yvst.jpeg', format='jpeg', dpi=300)
plt.close()

#*************************************************************************
#TRANSFORMADA DE FOURIER Y GRÁFICAS EN ESCALA LOG LOG

# Number of sample points
N = n_images
# sample spacing
T = 1.0 / 303
time = np.linspace(0.0, N*T, N, endpoint=False)

xf = fft(x_max)
yf=	fft(y_max)
timef = fftfreq(N, T)[:N//2]

plt.loglog(timef[1:N//2], (2.0/N * np.abs(xf[1:N//2]))**2, color='darkviolet')
plt.ylabel("P ($\mu m^2$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Espectro_potencia x.jpeg', format='jpeg', dpi=300)
plt.close()

plt.loglog(timef[1:N//2], (2.0/N * np.abs(yf[1:N//2]))**2, color='darkviolet')
plt.ylabel("P ($\mu m^2$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Espectro_potencia y.jpeg', format='jpeg', dpi=300)
plt.close()
#*************************************************************************

#*************************************************************************
#FILTRO GAUSIANO Y GRÁFICAS LOG LOG

xf_gf=np.zeros(len(xf))
xf_gf=gaussian_filter(2.0/N * np.abs(xf)**2, sigma=1)

yf_gf=np.zeros(len(yf))
yf_gf=gaussian_filter(2.0/N * np.abs(yf)**2, sigma=1)

plt.loglog(timef[1:N//2], xf_gf[1:N//2], color='darkviolet')
plt.ylabel("P ($\mu m^2$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Espectro_potencia_Filtro x.jpeg', format='jpeg', dpi=300)
plt.close()

plt.loglog(timef[1:N//2], yf_gf[1:N//2], color='darkviolet')
plt.ylabel("P ($\mu m^2$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Espectro_potencia_Filtro y.jpeg', format='jpeg', dpi=300)
plt.close()
#*************************************************************************

#*************************************************************************
#AJUSTE LINEAL DE LOS DATOS EN ESCALA LOG LOG

# REGRESIÓN LINEAL
#Definimos la función
def func(x,m,b):
    return  m*x + b
   
#en log
timef_l=np.zeros(len(timef))
xf_norm_l=np.zeros(len(xf_gf))
yf_norm_l=np.zeros(len(yf_gf))

for i in range (1,len(timef)):
	timef_l[i]=math.log10(timef[i])
	xf_norm_l[i]=math.log10(2.0/N * np.abs(xf[i])**2)
	yf_norm_l[i]=math.log10(2.0/N * np.abs(yf[i])**2)

fitx, covx = curve_fit(func, timef_l[50:N//32], xf_norm_l[50:N//32])
Ifitx=func(timef_l[50:N//2],fitx[0],fitx[1])
fit2x, cov2x = curve_fit(func, timef_l[N//32:N//2], xf_norm_l[N//32:N//2])
Ifit2x=func(timef_l[50:N//2],fit2x[0],fit2x[1])

fity, covy = curve_fit(func, timef_l[50:N//32], yf_norm_l[50:N//32])
Ifity=func(timef_l[50:N//2],fity[0],fity[1])
fit2y, cov2y = curve_fit(func, timef_l[N//32:N//2], yf_norm_l[N//32:N//2])
Ifit2y=func(timef_l[50:N//2],fit2y[0],fit2y[1])

#Intersección de las rectas
fx_0=(fit2x[1]-fitx[1])/(fitx[0]-fit2x[0])
x_0=func(fx_0,fitx[0],fitx[1])

fy_0=(fit2y[1]-fity[1])/(fity[0]-fit2y[0])
y_0=func(fy_0,fity[0],fity[1])

#Sacarlo de la escala log log para obtener el valor para el cálculo de parámetros
fx_i=10**fx_0
x_i=10**x_0

fy_i=10**fy_0
y_i=10**y_0
#*************************************************************************

#*************************************************************************
#CONSTANTE DE LA TRAMPA

eta=8.9*10**(-4)
a=(4.5/2)*10**(-6)
kx=x_i*2*np.pi*6*np.pi*eta*a
ky=y_i*2*np.pi*6*np.pi*eta*a

archivo1='Ajuste.txt'
Ajustes1=open(archivo1,'a')
Ajustes1.write('Para x\n')
Ajustes1.write('m    b\n')
Ajustes1.write('{:.16f}    {:.16f}'.format(fitx[0],fitx[1]) +'\n')	
Ajustes1.write('{:.16f}    {:.16f}'.format(fit2x[0],fit2x[1]) +'\n')
Ajustes1.write('fx_0    xf_0\n')
Ajustes1.write('{:.16f}    {:.16f}'.format(fx_i,x_i) +'\n')
Ajustes1.write('kx\n')
Ajustes1.write('{:.16f}'.format(kx) +'\n')
Ajustes1.write('Para y\n')
Ajustes1.write('m    b\n')
Ajustes1.write('{:.16f}    {:.16f}'.format(fity[0],fity[1]) +'\n')	
Ajustes1.write('{:.16f}    {:.16f}'.format(fity[0],fit2y[1]) +'\n')
Ajustes1.write('fy_0    yf_0\n')
Ajustes1.write('{:.16f}    {:.16f}'.format(fy_i,y_i) +'\n')
Ajustes1.write('ky\n')
Ajustes1.write('{:.16f}'.format(ky) +'\n')
Ajustes1.close()

plt.loglog(timef[50:N//2], xf_gf[50:N//2], color='limegreen')
plt.plot(timef[50:N//2],10**Ifitx,label='Lineal fit \n y={:.8f}x+{:.8f}'.format(fitx[0],fitx[1]))
plt.plot(timef[50:N//2],10**(Ifit2x),label='Lineal fit \n y={:.8f}x+{:.8f}'.format(fit2x[0],fit2x[1]))
plt.legend(loc='best')
plt.ylabel("P ($\mu m^(2)$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Ajustesx.jpeg', format='jpeg', dpi=300)
plt.close()

plt.loglog(timef[50:N//2], yf_gf[50:N//2], color='limegreen')
plt.plot(timef[50:N//2],10**Ifity,label='Lineal fit \n y={:.8f}x+{:.8f}'.format(fity[0],fity[1]))
plt.plot(timef[50:N//2],10**(Ifit2y),label='Lineal fit \n y={:.8f}x+{:.8f}'.format(fit2y[0],fit2y[1]))
plt.legend(loc='best')
plt.ylabel("P ($\mu m^(2)$/Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.grid(True)
plt.savefig('Ajustesy.jpeg', format='jpeg', dpi=300)
plt.close()
#*************************************************************************
