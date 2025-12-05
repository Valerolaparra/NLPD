import numpy as np
from scipy import signal, ndimage

# ==============================================================================
# 1. DATOS EMBEBIDOS (Sustituyendo a los ficheros .mat)
# ==============================================================================

class EmbeddedData:
    """
    Contiene los parámetros que originalmente estaban en:
    - PAR_NLP_2017.mat (Parámetros de normalización)
    - Filters_Laplacian_Pyramid_as_steerable.mat (Filtros para el gradiente)
    
    Nota: Se han inicializado con valores estándar funcionales. 
    """
    def __init__(self):
        # --- Datos de PAR_NLP_2017 ---
        # Exponentes (Valores típicos paper 2016/2017)
        self.exp_g = 2.6
        self.exp_s = 2
        self.exp_f = 0.6000000000000001

        self.F1 = np.array([[0.0025,0.0125,0.02  ,0.0125,0.0025],
        [0.0125,0.0625,0.1   ,0.0625,0.0125],
        [0.02  ,0.1   ,0.16  ,0.1   ,0.02  ],
        [0.0125,0.0625,0.1   ,0.0625,0.0125],
        [0.0025,0.0125,0.02  ,0.0125,0.0025]])

        self.sigmas = [0.17, 4.86]

        self.DN_filts = [
            {'F2': np.array([[0.04,0.04,0.05,0.04,0.04],
            [0.04,0.03,0.04,0.03,0.04],
            [0.05,0.04,0.05,0.04,0.05],
            [0.04,0.03,0.04,0.03,0.04],
            [0.04,0.04,0.05,0.04,0.04]])},
            {'F2': np.array([[0.04,0.04,0.05,0.04,0.04],
            [0.04,0.03,0.04,0.03,0.04],
            [0.05,0.04,0.05,0.04,0.05],
            [0.04,0.03,0.04,0.03,0.04],
            [0.04,0.04,0.05,0.04,0.04]])},
            {'F2': np.array([[0.04,0.04,0.05,0.04,0.04],
            [0.04,0.03,0.04,0.03,0.04],
            [0.05,0.04,0.05,0.04,0.05],
            [0.04,0.03,0.04,0.03,0.04],
            [0.04,0.04,0.05,0.04,0.04]])},
            {'F2': np.array([[0.04,0.04,0.05,0.04,0.04],
            [0.04,0.03,0.04,0.03,0.04],
            [0.05,0.04,0.05,0.04,0.05],
            [0.04,0.03,0.04,0.03,0.04],
            [0.04,0.04,0.05,0.04,0.04]])},
            {'F2': np.array([[0.04,0.04,0.05,0.04,0.04],
            [0.04,0.03,0.04,0.03,0.04],
            [0.05,0.04,0.05,0.04,0.05],
            [0.04,0.03,0.04,0.03,0.04],
            [0.04,0.04,0.05,0.04,0.04]])},
            {'F2': np.array([[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])},
            ]

        # --- Datos de Filters_Laplacian_Pyramid_as_steerable (Solo para Gradiente) ---
        # Estos son filtros orientados. Usamos placeholders de ceros si no se cargan,
        # pero el código de distancia (forward) funcionará igual.
        # --- Datos de Filters_Laplacian_Pyramid_as_steerable ---
        # Filtros para la reconstrucción del gradiente (Steerable Pyramid)
        
        self.L_0 = np.array([[0.0025,0.0125,0.02  ,0.0125,0.0025], 
                             [0.0125,0.0625,0.1   ,0.0625,0.0125], 
                             [0.02  ,0.1   ,0.16  ,0.1   ,0.02  ], 
                             [0.0125,0.0625,0.1   ,0.0625,0.0125], 
                             [0.0025,0.0125,0.02  ,0.0125,0.0025]])

        self.H_11 = np.array([
            [-2.5000e-05,-1.2500e-04,-4.0000e-04,-1.1250e-03,-1.6500e-03,-1.1250e-03,-4.0000e-04,-1.2500e-04,-2.5000e-05], 
            [-1.2500e-04,-6.2500e-04,-2.0000e-03,-5.6250e-03,-8.2500e-03,-5.6250e-03,-2.0000e-03,-6.2500e-04,-1.2500e-04], 
            [-4.0000e-04,-2.0000e-03,-6.4000e-03,-1.8000e-02,-2.6400e-02,-1.8000e-02,-6.4000e-03,-2.0000e-03,-4.0000e-04], 
            [-1.1250e-03,-5.6250e-03,-1.8000e-02,-5.0625e-02,-7.4250e-02,-5.0625e-02,-1.8000e-02,-5.6250e-03,-1.1250e-03], 
            [-1.6500e-03,-8.2500e-03,-2.6400e-02,-7.4250e-02, 8.9110e-01,-7.4250e-02,-2.6400e-02,-8.2500e-03,-1.6500e-03], 
            [-1.1250e-03,-5.6250e-03,-1.8000e-02,-5.0625e-02,-7.4250e-02,-5.0625e-02,-1.8000e-02,-5.6250e-03,-1.1250e-03], 
            [-4.0000e-04,-2.0000e-03,-6.4000e-03,-1.8000e-02,-2.6400e-02,-1.8000e-02,-6.4000e-03,-2.0000e-03,-4.0000e-04], 
            [-1.1250e-04,-6.2500e-04,-2.0000e-03,-5.6250e-03,-8.2500e-03,-5.6250e-03,-2.0000e-03,-6.2500e-04,-1.2500e-04], 
            [-2.5000e-05,-1.2500e-04,-4.0000e-04,-1.1250e-03,-1.6500e-03,-1.1250e-03,-4.0000e-04,-1.2500e-04,-2.5000e-05]
        ])

        self.H_12 = np.array([
            [ 0.0000e+00,-1.2500e-04,-6.2500e-04,-1.1250e-03,-1.2500e-03,-1.1250e-03,-6.2500e-04,-1.2500e-04, 0.0000e+00], 
            [ 0.0000e+00,-6.2500e-04,-3.1250e-03,-5.6250e-03,-6.2500e-03,-5.6250e-03,-3.1250e-03,-6.2500e-04, 0.0000e+00], 
            [ 0.0000e+00,-2.0000e-03,-1.0000e-02,-1.8000e-02,-2.0000e-02,-1.8000e-02,-1.0000e-02,-2.0000e-03, 0.0000e+00], 
            [ 0.0000e+00,-5.6250e-03,-2.8125e-02,-5.0625e-02,-5.6250e-02,-5.0625e-02,-2.8125e-02,-5.6250e-03, 0.0000e+00], 
            [ 0.0000e+00,-8.2500e-03,-4.1250e-02,-7.4250e-02, 9.1750e-01,-7.4250e-02,-4.1250e-02,-8.2500e-03, 0.0000e+00], 
            [ 0.0000e+00,-5.6250e-03,-2.8125e-02,-5.0625e-02,-5.6250e-02,-5.0625e-02,-2.8125e-02,-5.6250e-03, 0.0000e+00], 
            [ 0.0000e+00,-2.0000e-03,-1.0000e-02,-1.8000e-02,-2.0000e-02,-1.8000e-02,-1.0000e-02,-2.0000e-03, 0.0000e+00], 
            [ 0.0000e+00,-6.2500e-04,-3.1250e-03,-5.6250e-03,-6.2500e-03,-5.6250e-03,-3.1250e-03,-6.2500e-04, 0.0000e+00], 
            [ 0.0000e+00,-1.2500e-04,-6.2500e-04,-1.1250e-03,-1.2500e-03,-1.1250e-03,-6.2500e-04,-1.2500e-04, 0.0000e+00]
        ])

        self.H_21 = np.array([
            [ 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], 
            [-1.2500e-04,-6.2500e-04,-2.0000e-03,-5.6250e-03,-8.2500e-03,-5.6250e-03,-2.0000e-03,-6.2500e-04,-1.2500e-04], 
            [-6.2500e-04,-3.1250e-03,-1.0000e-02,-2.8125e-02,-4.1250e-02,-2.8125e-02,-1.0000e-02,-3.1250e-03,-6.2500e-04], 
            [-1.1250e-03,-5.6250e-03,-1.8000e-02,-5.0625e-02,-7.4250e-02,-5.0625e-02,-1.8000e-02,-5.6250e-03,-1.1250e-03], 
            [-1.2500e-03,-6.2500e-03,-2.0000e-02,-5.6250e-02, 9.1750e-01,-5.6250e-02,-2.0000e-02,-6.2500e-03,-1.2500e-03], 
            [-1.1250e-03,-5.6250e-03,-1.8000e-02,-5.0625e-02,-7.4250e-02,-5.0625e-02,-1.8000e-02,-5.6250e-03,-1.1250e-03], 
            [-6.2500e-04,-3.1250e-03,-1.0000e-02,-2.8125e-02,-4.1250e-02,-2.8125e-02,-1.0000e-02,-3.1250e-03,-6.2500e-04], 
            [-1.2500e-04,-6.2500e-04,-2.0000e-03,-5.6250e-03,-8.2500e-03,-5.6250e-03,-2.0000e-03,-6.2500e-04,-1.2500e-04], 
            [ 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
        ])

        self.H_22 = np.array([
            [ 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00], 
            [ 0.0000e+00,-6.2500e-04,-3.1250e-03,-5.6250e-03,-6.2500e-03,-5.6250e-03,-3.1250e-03,-6.2500e-04, 0.0000e+00], 
            [ 0.0000e+00,-3.1250e-03,-1.5625e-02,-2.8125e-02,-3.1250e-02,-2.8125e-02,-1.5625e-02,-3.1250e-03, 0.0000e+00], 
            [ 0.0000e+00,-5.6250e-03,-2.8125e-02,-5.0625e-02,-5.6250e-02,-5.0625e-02,-2.8125e-02,-5.6250e-03, 0.0000e+00], 
            [ 0.0000e+00,-6.2500e-03,-3.1250e-02,-5.6250e-02, 9.3750e-01,-5.6250e-02,-3.1250e-02,-6.2500e-03, 0.0000e+00], 
            [ 0.0000e+00,-5.6250e-03,-2.8125e-02,-5.0625e-02,-5.6250e-02,-5.0625e-02,-2.8125e-02,-5.6250e-03, 0.0000e+00], 
            [ 0.0000e+00,-3.1250e-03,-1.5625e-02,-2.8125e-02,-3.1250e-02,-2.8125e-02,-1.5625e-02,-3.1250e-03, 0.0000e+00], 
            [ 0.0000e+00,-6.2500e-04,-3.1250e-03,-5.6250e-03,-6.2500e-03,-5.6250e-03,-3.1250e-03,-6.2500e-04, 0.0000e+00], 
            [ 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
        ])

    def _gaussian_kernel(self, size, sigma):
        """Helper para generar filtros por defecto si no tenemos los datos binarios"""
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / g.sum()

# Instancia global de los datos
DATA = EmbeddedData()

# ==============================================================================
# 2. FUNCIONES AUXILIARES (laplacian_pyramid_s y helpers)
# ==============================================================================

def downsample_s(Im, filt):
    """
    Aplica filtro paso bajo y diezma.
    Equivalente a downsample_s.m
    """
    # MATLAB imfilter 'symmetric' es scipy 'reflect'
    # Scipy correlate es lo más cercano a imfilter (correlación)
    if Im.ndim == 3:
        Im_out = np.zeros_like(Im)
        for ch in range(Im.shape[2]):
            Im_out[:,:,ch] = ndimage.correlate(Im[:,:,ch], filt, mode='reflect')
    else:
        Im_out = ndimage.correlate(Im, filt, mode='reflect')
    
    # Decimate (quedarse con 1 de cada 2 píxeles)
    return Im_out[0::2, 0::2]

def upsample_s(Im, odd, filt):
    """
    Sube la resolución e interpola.
    Equivalente a upsample_s.m
    """
    # Pad array (replicate borders) [1 1 0] en Matlab -> ((1,1), (1,1)) en numpy
    if Im.ndim == 3:
        pad_width = ((1, 1), (1, 1), (0, 0))
    else:
        pad_width = ((1, 1), (1, 1))
        
    Im_padded = np.pad(Im, pad_width, mode='edge') # 'edge' = 'replicate' en Matlab
    
    r, c = Im_padded.shape[0] * 2, Im_padded.shape[1] * 2
    
    if Im.ndim == 3:
        Im_up = np.zeros((r, c, Im.shape[2]))
        Im_up[0::2, 0::2, :] = 4 * Im_padded
        # Convolve/Correlate
        for ch in range(Im.shape[2]):
            Im_up[:,:,ch] = ndimage.correlate(Im_up[:,:,ch], filt, mode='reflect')
        # Remove border
        return Im_up[2:r - 2 - odd[0], 2:c - 2 - odd[1], :]
    else:
        Im_up = np.zeros((r, c))
        Im_up[0::2, 0::2] = 4 * Im_padded
        Im_up = ndimage.correlate(Im_up, filt, mode='reflect')
        return Im_up[2:r - 2 - odd[0], 2:c - 2 - odd[1]]

def laplacian_pyramid_s(Im, N_lev=None, param_filter=None):
    """
    Construye la pirámide Laplaciana.
    """
    # Filtro por defecto (Burt & Adelson) si no se provee
    if param_filter is None:
        f = np.array([.05, .25, .4, .25, .05])
        param_filter = np.outer(f, f)

    if N_lev is None:
        N_lev = int(np.floor(np.log2(min(Im.shape[:2]))))

    pyr = []
    J = Im.copy()
    
    # Bucle para niveles
    for l in range(N_lev - 1):
        # Downsample
        G = downsample_s(J, param_filter)
        
        # Calcular tamaño 'odd' para reconstrucción exacta
        # odd = 2*size(G) - size(J)
        r_g, c_g = G.shape[:2]
        r_j, c_j = J.shape[:2]
        odd = (2*r_g - r_j, 2*c_g - c_j)
        
        # Laplacian: Original - Upsampled(LowPass)
        L = J - upsample_s(G, odd, param_filter)
        pyr.append(L)
        
        J = G # Continuar con la imagen de baja resolución
        
    pyr.append(J) # El último nivel es el residuo paso bajo
    return pyr

# ==============================================================================
# 3. FUNCIÓN PRINCIPAL (NLPdist_lum)
# ==============================================================================




def NLPdist_lum(Im_X, Im_Xp, params_NLP=None, compute_gradient=False):
    """
    Calcula la distancia NLP (Normalized Laplacian Pyramid) entre dos imágenes.
    
    Args:
        Im_X: Imagen a evaluar (numpy array), en cd/m2
        Im_Xp: Imagen original (referencia) O la estructura NLP ya precalculada), en cd/m2
        params_NLP: (Opcional) Diccionario con parámetros. Si es None, usa DATA.
        compute_gradient: (Bool) Si True, retorna también el gradiente (dfX).
        
    Returns:
        fX: Distancia perceptual (float).
        dfX: (Opcional) Gradiente.
        Im_Xp_NLP: Estructura NLP de la referencia.
    """
    
    # Manejo de entrada 1D -> 2D
    if Im_X.ndim == 1:
        sz = int(np.sqrt(len(Im_X)))
        Im_X = Im_X.reshape((sz, sz))
    
    if not isinstance(Im_Xp, list) and Im_Xp.ndim == 1:
        sz = int(np.sqrt(len(Im_Xp)))
        Im_Xp = Im_Xp.reshape((sz, sz))

    # Cargar parámetros por defecto si no existen
    if params_NLP is None:
        params_NLP = DATA
        # Extraer variables locales para facilitar lectura
        exp_g = params_NLP.exp_g
        exp_s = params_NLP.exp_s
        exp_f = params_NLP.exp_f
        DN_filts = params_NLP.DN_filts
        sigmas_base = params_NLP.sigmas
        F1 = params_NLP.F1
    else:
        # Si se pasa un objeto params externo
        exp_g = params_NLP.get('exp_g', DATA.exp_g)
        exp_s = params_NLP.get('exp_s', DATA.exp_s)
        exp_f = params_NLP.get('exp_f', DATA.exp_f)
        DN_filts = params_NLP.get('DN_filts', DATA.DN_filts)
        sigmas_base = params_NLP.get('sigmas', DATA.sigmas)
        F1 = params_NLP.get('F1', DATA.F1)

    # Determinar número de niveles
    min_dim = min(Im_X.shape[:2])
    N_lev = int(np.floor(np.log2(min_dim))) - 2
    # Ajustar lista de sigmas a la longitud necesaria
    sigmas = [sigmas_base[0]] * (N_lev - 1) + [sigmas_base[-1]]

    # --------------------------------------------------------------------------
    # Transformar Im_Xp (Referencia) al dominio NLP (si no se pasó ya transformado)
    # --------------------------------------------------------------------------
    Im_Xp_NLP = []
    DEN_Xp = []
    
    if isinstance(Im_Xp, list):
        Im_Xp_NLP = Im_Xp
    else:
        # Gamma correction
        Im_Xpg = np.power(Im_Xp, 1.0 / exp_g)
        # Laplacian Pyramid
        Lap_dom_ori = laplacian_pyramid_s(Im_Xpg, N_lev, F1)
        
        Im_Xp_NLP = [None] * N_lev
        DEN_Xp = [None] * N_lev
        
        for N_b in range(N_lev):
            filt = DN_filts[N_b]['F2'] if N_b < len(DN_filts) else DN_filts[-1]['F2']
            
            # Pad array para convolución 'valid'
            LL = (np.array(filt.shape) // 2).astype(int)
            band_abs = np.abs(Lap_dom_ori[N_b])
            # Pad symmetric
            A2 = np.pad(band_abs, ((LL[0], LL[0]), (LL[1], LL[1])), mode='reflect')
            
            # Convolución (Divisive Normalization denominator)
            # Matlab conv2(..., 'valid') es scipy convolve2d(..., 'valid')
            conv_res = signal.convolve2d(A2, filt, mode='valid')
            
            DEN = sigmas[N_b] + conv_res
            Im_Xp_NLP[N_b] = Lap_dom_ori[N_b] / DEN
            DEN_Xp[N_b] = conv_res

    # --------------------------------------------------------------------------
    # Transformar Im_X (Target) al dominio NLP y calcular Distancia
    # --------------------------------------------------------------------------
    Im_Xg = np.power(Im_X, 1.0 / exp_g)
    Lap_dom = laplacian_pyramid_s(Im_Xg, N_lev, F1)
    
    Im_X_NLP = [None] * N_lev
    DEN_X = [None] * N_lev
    dif = [None] * N_lev
    fX_aux = np.zeros(N_lev)
    
    # Variables para gradiente
    dfX_aux2b = np.zeros((Im_X.size, N_lev)) if compute_gradient else None

    for N_b in range(N_lev):
        filt = DN_filts[N_b]['F2'] if N_b < len(DN_filts) else DN_filts[-1]['F2']
        
        LL = (np.array(filt.shape) // 2).astype(int)
        band_abs = np.abs(Lap_dom[N_b])
        A2 = np.pad(band_abs, ((LL[0], LL[0]), (LL[1], LL[1])), mode='reflect')
        
        conv_res = signal.convolve2d(A2, filt, mode='valid')
        DEN = sigmas[N_b] + conv_res
        
        Im_X_NLP[N_b] = Lap_dom[N_b] / DEN
        DEN_X[N_b] = conv_res
        
        # Diferencia
        dif[N_b] = Im_X_NLP[N_b] - Im_Xp_NLP[N_b]
        
        # Métrica por banda
        # mean(abs(diff)^exp_s)^(1/exp_s)
        mean_diff = np.mean(np.abs(dif[N_b]) ** exp_s)
        fX_aux[N_b] = mean_diff ** (1.0 / exp_s)
        
        # ----------------------------------------------------------------------
        # Cálculo del Gradiente (Opcional)
        # ----------------------------------------------------------------------
        if compute_gradient:
            # Termino derivada distancia
            AAA = np.sign(dif[N_b]) * (np.abs(dif[N_b])**(exp_s - 1)) / (DEN**2)
            aux1 = Lap_dom[N_b] * AAA
            
            # Filtro rotado 180 grados (rot90(..., 2))
            P = np.rot90(filt, 2)
            
            # conv2(..., 'same')
            auxb = signal.convolve2d(aux1, P, mode='same')
            d_y_z2 = np.sign(Lap_dom[N_b]) * auxb
            
            d_y_z1 = DEN * AAA
            d_y_z = (d_y_z1 - d_y_z2).astype(np.float32)
            
            # Derivada Piramide Laplaciana (Reconstrucción)
            if N_b < (N_lev - 1): # Índices 0-based, N_lev-1 es el último
                # En Matlab: dif_11 = zeros(size(d_y_z)); ...
                dif_11 = np.zeros_like(d_y_z)
                dif_12 = np.zeros_like(d_y_z)
                dif_21 = np.zeros_like(d_y_z)
                dif_22 = np.zeros_like(d_y_z)
                
                # Upsampling rellenando con ceros (checkers)
                # Matlab: dif_11(1:2:end,1:2:end) = d_y_z(1:2:end,1:2:end);
                dif_11[0::2, 0::2] = d_y_z[0::2, 0::2]
                # Nota: Usamos params_NLP para acceder a los filtros H cargados
                der_11 = ndimage.correlate(dif_11, params_NLP.H_11, mode='reflect')
                
                # Matlab: dif_12(1:2:end,2:2:end) = d_y_z(1:2:end,2:2:end);
                dif_12[0::2, 1::2] = d_y_z[0::2, 1::2]
                der_12 = ndimage.correlate(dif_12, params_NLP.H_12, mode='reflect')
                
                # Matlab: dif_21(2:2:end,1:2:end) = d_y_z(2:2:end,1:2:end);
                dif_21[1::2, 0::2] = d_y_z[1::2, 0::2]
                der_21 = ndimage.correlate(dif_21, params_NLP.H_21, mode='reflect')
                
                # Matlab: dif_22(2:2:end,2:2:end) = d_y_z(2:2:end,2:2:end);
                dif_22[1::2, 1::2] = d_y_z[1::2, 1::2]
                der_22 = ndimage.correlate(dif_22, params_NLP.H_22, mode='reflect')
                
                dif_s = der_11 + der_12 + der_21 + der_22
            else:
                dif_s = d_y_z
            
            # Backpropagate through pyramid levels
            for Nb_2 in range(N_b - 1, -1, -1):
                # Matlab crea d_aux zeros(size(dif{Nb_2}))
                # Necesitamos el tamaño del nivel anterior. 
                target_shape = dif[Nb_2].shape
                d_aux = np.zeros(target_shape)
                
                # Insertar dif_s en los impares (upsample x2 with zeros)
                # Matlab: d_aux(1:2:end,1:2:end) = dif_s;
                # CUIDADO: dif_s debe caber en d_aux.
                rows = min(d_aux.shape[0] // 2 + (d_aux.shape[0] % 2), dif_s.shape[0])
                cols = min(d_aux.shape[1] // 2 + (d_aux.shape[1] % 2), dif_s.shape[1])
                
                d_aux[0:2*rows:2, 0:2*cols:2] = dif_s[0:rows, 0:cols]
                
                # Filtrar con L_0
                dif_s = ndimage.correlate(d_aux, params_NLP.L_0, mode='reflect')

            d_di_xpj = dif_s
            
            # Constante por banda (Cálculo idéntico a Matlab)
            sum_abs = np.sum(np.abs(dif[N_b])**params_NLP.exp_s)
            aaa = np.real(sum_abs**(params_NLP.exp_f/params_NLP.exp_s - 1))
            
            if fX_aux[N_b] == 0:
                aaa = 0
            
            Nc = dif[N_b].size
            konst = aaa * (1.0 / (Nc**(params_NLP.exp_f/params_NLP.exp_s)))
            
            # Acumular en el vector de gradiente
            term = konst * (d_di_xpj.flatten() * ((1.0/params_NLP.exp_g) * Im_X.flatten()**((1.0/params_NLP.exp_g)-1)))
            
            if dfX_aux2b is not None:
                dfX_aux2b[:, N_b] = term
            
    # Agrupar métrica final
    # fX = mean(fX_aux^exp_f)^(1/exp_f)
    fX = np.mean(fX_aux ** exp_f) ** (1.0 / exp_f)

    if compute_gradient:
        if fX == 0:
            dfX = np.zeros(Im_X.size)
        else:
            # Fórmula final del gradiente según Matlab:
            # dfX = fX^(1-params_NLP.exp_f) * mean(dfX_aux2b, 2);
            
            # En Python, axis=1 realiza la media a través de los niveles (columnas)
            mean_grads = np.mean(dfX_aux2b, axis=1)
            dfX = (fX ** (1 - exp_f)) * mean_grads
            
        return fX, dfX, Im_Xp_NLP
    
    return fX


def NLPdist_RGB(Im_X, Im_Xp, params_scr=None, params_NLP=None, compute_gradient=False):
    """
    Calcula la distancia NLP aplicando la métrica independientemente a cada canal RGB.
    
    Proceso:
    1. Separa canales R, G, B.
    2. Convierte cada canal de rango [0-1] a cd/m2 físico.
    3. Calcula NLPdist_lum para cada canal.
    4. Devuelve el promedio de las distancias.
    
    Args:
        Im_X: Imagen RGB de prueba (H, W, 3) en rango [0, 1].
        Im_Xp: Imagen RGB de referencia (H, W, 3) en rango [0, 1].
        params_scr: Parámetros del monitor (LM_out, Lm_out, gc).
        compute_gradient: Si True, devuelve el gradiente RGB.
        
    Returns:
        fX_mean: Distancia media de los 3 canales.
        dfX_rgb: (Opcional) Imagen de gradientes (H, W, 3).
    """
    
    # 1. Parámetros de Pantalla (Según NLP_rendering.m)
    if params_scr is None:
        params_scr = {
            'LM_out': 180.0, # Luminancia máxima
            'Lm_out': 5.0,   # Luminancia mínima
            'gc': 2.2        # Gamma
        }
    
    LM = params_scr.get('LM_out', 180.0)
    Lm = params_scr.get('Lm_out', 5.0)
    gc = params_scr.get('gc', 2.2)
    range_L = LM - Lm

    # Asegurar dimensiones
    if Im_X.ndim != 3 or Im_X.shape[2] != 3:
        raise ValueError("Im_X debe ser (H, W, 3)")
    
    if Im_Xp.ndim != 3 or Im_Xp.shape[2] != 3:
        raise ValueError("Im_Xp debe ser (H, W, 3)")

    dist_channels = []
    grads_channels = []
    
    # 2. Iterar sobre canales (0=R, 1=G, 2=B)
    for c in range(3):
        # A. Extraer canal en rango digital [0-1]
        chan_dig = Im_X[:,:,c]
        chan_ref_dig = Im_Xp[:,:,c]
        
        # B. Convertir a Unidades Físicas (cd/m2)
        # Fórmula: Im = (Im_dg.^gc)*(LM_out-Lm_out)+Lm_out
        chan_phys = (chan_dig ** gc) * range_L + Lm
        chan_ref_phys = (chan_ref_dig ** gc) * range_L + Lm
        
        # C. Calcular Distancia NLP en este canal
        res = NLPdist_lum(chan_phys, chan_ref_phys, params_NLP, compute_gradient)
        
        if compute_gradient:
            fX_c, dfX_phys_flat, _ = res
            dist_channels.append(fX_c)
            
            # D. Regla de la cadena para el gradiente
            # Necesitamos d(Dist)/d(Pixel_Digital)
            # dfX_phys viene aplanado, lo reestructuramos a 2D
            dfX_phys = dfX_phys_flat.reshape(chan_dig.shape)
            
            # Derivada de la conversión física:
            # Phys = Dig^gc * range + min
            # d(Phys)/d(Dig) = range * gc * Dig^(gc-1)
            # Evitamos división por cero con maximum
            safe_dig = np.maximum(chan_dig, 1e-8)
            d_phys_d_dig = range_L * gc * (safe_dig ** (gc - 1.0))
            
            # Chain rule: Grad_Digital = Grad_Physical * Deriv_Conversion
            grad_c = dfX_phys * d_phys_d_dig
            grads_channels.append(grad_c)
            
        else:
            # Si no hay gradiente, res es solo fX (o una tupla según tu implementación exacta)
            # Asumiendo que NLPdist_lum devuelve solo float si compute_gradient=False
            if isinstance(res, tuple):
                dist_channels.append(res[0])
            else:
                dist_channels.append(res)

    # 3. Promediar resultados
    fX_mean = np.mean(dist_channels)
    
    if compute_gradient:
        # Combinar gradientes en imagen RGB
        # Como la función final es Mean(R, G, B) = (R+G+B)/3
        # La derivada es 1/3 * (Grad_R, Grad_G, Grad_B)
        dfX_rgb = np.stack(grads_channels, axis=2) / 3.0
        return fX_mean, dfX_rgb
    
    return fX_mean


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================
if __name__ == "__main__":
    # Crear dos imágenes de prueba aleatorias (o cargar con cv2/PIL)
    img1 = np.random.rand(256, 256).astype(np.float32)
    # img2 es img1 con un poco de ruido
    img2 = img1 + 0.01 * np.random.randn(256, 256).astype(np.float32)
    img2 = np.clip(img2, 0, 1)

    print("Calculando distancia NLP...")
    dist = NLPdist_lum(img2, img1)
    print(f"Distancia Perceptual NLP: {dist:.6f}")
    
    print("\nNota: Para usar los pesos exactos del paper de 2017, edita la clase EmbeddedData")
    print("      con los valores numéricos extraídos de PAR_NLP_2017.mat.")