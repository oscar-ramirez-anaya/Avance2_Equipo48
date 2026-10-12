"""Genera el reporte del Avance 2 en HTML, PDF y DOCX a partir de un solo contenido.

Las figuras se toman de la libreta ejecutada (Avance2.48.ipynb), de modo que
el reporte y la libreta siempre muestran los mismos resultados. Uso, desde la raíz del repositorio:
    python3 scripts/generar_documento_avance2.py
Requiere Google Chrome (PDF), PyMuPDF (pie de página e índice) y python-docx.
"""
import base64
import html
import io
import json
import pathlib
import re
import subprocess

import fitz
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CARPETA = RAIZ
LIBRETA = CARPETA / "Avance2.48.ipynb"
CSS = CARPETA / "assets" / "print.css"
LOGO = CARPETA / "assets" / "logo_tec.png"
HTML = CARPETA / "Avance2.48.html"
PDF = CARPETA / "Avance2.48.pdf"
DOCX = CARPETA / "Avance2.48.docx"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/oscar-ramirez-anaya/Avance2_Equipo48"
FECHA = "11 de octubre de 2026"

AZUL = RGBColor(0x0B, 0x3B, 0x8C)
AZUL_TABLA = "1346BD"


# ------------------------------------------------------------------ figuras de la libreta
def figuras_libreta():
    nb = json.loads(LIBRETA.read_text(encoding="utf-8"))
    figs = {}
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if "data" in o and "image/png" in o["data"]:
                figs[c["id"]] = o["data"]["image/png"]
    return figs


# ------------------------------------------------------------------ contenido
# Bloques: ("h2", id, texto) ("h3", id, texto) ("p", html) ("ul", [html]) ("ol", [html])
#          ("tabla", id, titulo, encabezados, filas) ("fig", id, celda, titulo) ("nota", titulo, html)
B = []
h2 = lambda i, t: B.append(("h2", i, t))
h3 = lambda i, t: B.append(("h3", i, t))
p = lambda t: B.append(("p", t))
ul = lambda xs: B.append(("ul", xs))
ol = lambda xs: B.append(("ol", xs))
tabla = lambda i, t, e, f: B.append(("tabla", i, t, e, f))
fig = lambda i, c, t: B.append(("fig", i, c, t))
nota = lambda t, x: B.append(("nota", t, x))

h2("s1", "1. Introducción")
h3("s11", "1.1 Propósito")
p("Este documento reporta la fase de <b>Preparación de los datos</b> del proyecto, conforme a la metodología CRISP-ML(Q) (Visengeriyeva et al., 2023). Parte de la muestra real de imágenes de anaquel entregada por Bnext y la transforma en un conjunto de variables útiles para el aprendizaje automático. El trabajo tiene tres partes: ingeniería de características (construcción, discretización, codificación, transformación y escalamiento), selección y extracción de características, y conclusiones de la fase. Cada decisión se justifica con el conocimiento del dominio o con una prueba sobre los datos.")
p(f"La libreta ejecutada que respalda cada cifra, <code>Avance2.48.ipynb</code>, está en el repositorio <code>{REPO}</code>.")
h3("s12", "1.2 Datos y manejo de la información")
p("La muestra contiene 100 escenas de exhibidores (refrigeradores y vitrinas) en cinco tipos de toma, más 422 recortes de producto etiquetados con su SKU (35 clases). Los recortes son el único subconjunto con variable objetivo, por lo que en ellos se validan las decisiones de selección con un modelo base. Las escenas se usan para caracterizar la calidad de captura.")
nota("Manejo de datos", "Conforme al acuerdo con Bnext, la libreta se ejecutó dentro del servidor de la organización y las imágenes no salieron de su infraestructura. Este reporte y la libreta publicada contienen solo tablas y gráficas agregadas. Los códigos de SKU se reemplazaron por identificadores anónimos (SKU_01 a SKU_35).")
h3("s13", "1.3 Método")
p("La Tabla 1 relaciona cada técnica aplicada con la sección de la libreta que la implementa.")
tabla("t1", "Técnicas aplicadas y su ubicación en la libreta", ["Etapa", "Técnica", "Sección"], [
    ["Extracción base", "Geometría, exposición, color (HSV, Lab), textura LBP, forma HOG", "2"],
    ["Construcción", "Índice de calidad, KB por megapíxel, nitidez por borde, matiz circular, alertas de captura", "3"],
    ["Discretización", "Cortes de dominio (iluminación, formato de empaque) y cuartiles (nitidez)", "4"],
    ["Codificación", "One-hot, ordinal y LabelEncoder", "5"],
    ["Partición", "Bloques temporales por SKU y StratifiedGroupKFold", "6"],
    ["Transformación", "Box-Cox y Yeo-Johnson", "7"],
    ["Escalamiento", "StandardScaler, MinMaxScaler y RobustScaler; prueba de convergencia", "8, 9"],
    ["Selección", "Umbral de varianza, correlación, ANOVA, chi-cuadrado, información mutua", "10 a 12"],
    ["Extracción", "PCA y análisis factorial (KMO, Bartlett, varimax)", "13, 14"],
    ["Validación", "Comparativa de siete configuraciones con regresión logística", "15"],
])

h2("s2", "2. Datos de partida y extracción base")
p("La Tabla 2 resume la muestra. Los recortes por SKU van de 10 a 24, una razón de desbalance de 2.4 (Figura 1), moderada para esta etapa, por lo que no se aplicó remuestreo. No hay valores faltantes en ninguna variable.")
tabla("t2", "Composición de la muestra", ["Unidad", "Grupo", "Imágenes"], [
    ["Escenas", "Foto original del dispositivo", "20"],
    ["Escenas", "Refrigeradores y vitrinas recortados", "40"],
    ["Escenas", "Mueble aislado (refrigerador y vitrina)", "40"],
    ["Recortes de producto", "35 SKU", "422"],
])
fig("f1", "celda-005", "Recortes por SKU (anonimizado). La línea punteada es la media.")
p("Como un modelo tabular no procesa píxeles, cada imagen se convierte en un vector de variables con significado visual. La Tabla 3 describe las familias y la razón de cada una. En total se obtienen 812 variables por recorte; las escenas reciben solo las 17 globales, porque ahí el objetivo es medir la calidad de captura y no identificar productos.")
tabla("t3", "Familias de variables extraídas de cada recorte", ["Familia", "Variables", "Justificación"], [
    ["Geometría y archivo (5)", "Ancho, alto, área, relación de aspecto, KB", "Formato del empaque y resolución de captura."],
    ["Exposición y calidad (7)", "Brillo, contraste, nitidez (varianza del Laplaciano), densidad de bordes, entropía, fracción sub/sobreexpuesta", "Las fotos se toman en tienda con iluminación y dispositivos variables (Pech-Pacheco et al., 2000)."],
    ["Color global (14)", "Saturación, colorfulness, fracción de blanco, matiz circular, momentos de Lab", "El color del empaque es la señal más directa entre marcas (Hasler y Süsstrunk, 2003)."],
    ["Histogramas HSV (20)", "12 de matiz, 4 de saturación, 4 de valor", "Distribución de color, no solo su media."],
    ["Textura LBP (10)", "Histograma de patrones binarios locales uniformes", "Texto, logotipos y relieves con invariancia a la iluminación (Ojala et al., 2002)."],
    ["Forma HOG (756)", "Histogramas de gradientes orientados en 64×128", "Silueta y distribución de bordes del empaque (Dalal y Triggs, 2005)."],
])

h2("s3", "3. Ingeniería de características")
h3("s31", "3.1 Construcción de nuevas características")
p("Las variables nuevas de la Tabla 4 se derivan del conocimiento del dominio de captura en tienda. Todas se calculan con constantes fijas y no con estadísticos del conjunto, así que no filtran información de prueba a entrenamiento.")
tabla("t4", "Características construidas", ["Variable", "Definición", "Justificación"], [
    ["megapixeles", "área / 10⁶", "Unidad interpretable de resolución."],
    ["kb_por_mp", "KB / megapíxeles", "A igual resolución, un JPEG pesa más si tiene detalle y menos si está borroso o sobrecomprimido."],
    ["nitidez_por_borde", "nitidez / densidad de bordes", "Separa “imagen con mucho contenido” de “imagen enfocada”."],
    ["exposicion_ok", "1 − |brillo − 128| / 128", "Distancia a una exposición centrada."],
    ["indice_calidad", "promedio de exposición, nitidez acotada y contraste acotado", "Resume en una variable si la captura es apta."],
    ["flag_borrosa, flag_oscura, flag_sobreexpuesta", "nitidez < 100; brillo < 60; más de 10 % de píxeles con V > 245", "Alertas operativas de captura."],
    ["hue_x, hue_y", "coseno y seno del matiz medio ponderado por saturación", "El matiz es circular: 0° y 360° son el mismo rojo."],
])
p("El índice de calidad separa los tipos de toma (Figura 2): la foto original (2.07 MP) tiene el índice más alto (0.97) y el mueble aislado de vitrina (0.55 MP) el más bajo (0.83), porque cada recorte sucesivo pierde resolución y contraste. Ninguna imagen resultó borrosa ni oscura. En cambio, 55.9 % de los recortes activan la alerta de sobreexposición por las etiquetas blancas y los reflejos del refrigerador. El umbral sirve para escenas, pero debe recalibrarse por producto antes de usarse como alerta. Lo mismo ocurre con el umbral de nitidez de 100: en recortes pequeños y nítidos la varianza del Laplaciano es mucho mayor (cuartiles de 1 869 a 4 563).")
fig("f2", "celda-011", "Índice de calidad y KB por megapíxel por tipo de toma, e índice de calidad medio por SKU.")
h3("s32", "3.2 Discretización")
p("Se discretizaron tres variables con criterios distintos. El brillo se convirtió en <i>nivel de iluminación</i> con cortes de dominio (0–60–190–255: oscuro, normal, claro), interpretables para el personal de tienda y estables entre muestras. La relación de aspecto se convirtió en <i>formato de empaque</i> (horizontal, cuadrado, vertical, muy vertical), que agrupa botellas, latas, cajas y bolsas. La nitidez se discretizó en cuartiles ajustados solo con entrenamiento, porque su distribución es muy asimétrica y los cuartiles reparten los recortes en grupos iguales. Las tres son ordinales y alimentan la prueba chi-cuadrado (Figura 3).")
fig("f3", "celda-014", "Recortes por formato de empaque y escenas por nivel de iluminación.")
h3("s33", "3.3 Codificación de variables categóricas")
tabla("t5", "Codificación aplicada", ["Variable", "Tipo", "Codificación", "Justificación"], [
    ["tipo_toma", "Nominal, 5 niveles", "One-hot", "No hay orden entre tipos de toma y la expansión es pequeña."],
    ["nivel_iluminacion, formato_empaque, nitidez_q", "Ordinal", "Ordinal", "Preserva el orden natural en una columna."],
    ["sku (objetivo)", "Nominal, 35 clases", "LabelEncoder", "Formato de etiqueta que esperan los clasificadores."],
    ["fuente_captura", "Binaria", "Excluida", "Cada SKU proviene de una sola fuente: la variable revela la clase (fuga) y no existirá en producción."],
])

h2("s4", "4. Normalización")
h3("s41", "4.1 Partición sin fuga de información")
p("Antes de ajustar cualquier transformación se separó el conjunto de prueba. Las fotos de cada SKU se tomaron en ráfagas de pocos segundos, así que dos recortes consecutivos son casi duplicados; el Avance 1 ya identificó los duplicados como el principal riesgo de fuga. Por eso se usó una partición por bloques temporales: el primer 70 % de las capturas de cada producto es entrenamiento (304 recortes) y el 30 % final es prueba (118), con las 35 clases en ambos lados. La validación cruzada usa <code>StratifiedGroupKFold</code> con tres bloques consecutivos por SKU como grupos. Todas las transformaciones se ajustan dentro de un <code>Pipeline</code> solo con entrenamiento.")
h3("s42", "4.2 Transformaciones")
p("Se midió la asimetría de las 31 variables continuas construidas a mano. Los histogramas y HOG se excluyeron porque son proporciones acotadas en [0, 1]. Las 9 con |asimetría| > 1 se transformaron así: Box-Cox (Box y Cox, 1964) para las estrictamente positivas por construcción y Yeo-Johnson (Yeo y Johnson, 2000) para las que admiten cero o negativos. La asimetría absoluta media bajó de 1.41 a 0.21 (Tabla 6, Figura 4). Shapiro-Wilk sigue rechazando la normalidad estricta en la mayoría, lo esperable con 304 observaciones y variables acotadas. El objetivo es quitar las colas que dominarían a los modelos lineales y a PCA, y se cumple.")
tabla("t6", "Transformaciones aplicadas (entrenamiento)", ["Variable", "Método", "λ", "Asimetría antes", "Asimetría después"], [
    ["frac_subexp", "Yeo-Johnson", "−62.43", "1.92", "0.32"],
    ["ancho", "Box-Cox", "−3.02", "1.79", "0.64"],
    ["area_px", "Box-Cox", "−1.34", "1.52", "0.11"],
    ["megapixeles", "Box-Cox", "−1.34", "1.52", "0.11"],
    ["entropia", "Yeo-Johnson", "15.83", "−1.41", "−0.18"],
    ["alto", "Box-Cox", "−2.84", "1.27", "0.34"],
    ["nitidez_por_borde", "Box-Cox", "0.06", "1.24", "0.00"],
    ["frac_sobreexp", "Yeo-Johnson", "−4.65", "1.05", "0.12"],
    ["b_sesgo", "Yeo-Johnson", "0.67", "1.02", "0.08"],
])
fig("f4", "celda-021", "Distribución de cuatro variables antes (arriba) y después (abajo) de la transformación de potencia.")
h3("s43", "4.3 Escalamiento")
p("Las variables tienen escalas muy distintas: el área está en cientos de miles de píxeles y los histogramas en fracciones. Se compararon tres escaladores ajustados con entrenamiento (Figura 5) y se midió su efecto en la convergencia de la regresión logística y en kNN (Tabla 7).")
fig("f5", "celda-024", "Cinco variables de ejemplo sin escalar y con cada escalador.")
tabla("t7", "Efecto del escalamiento (conjunto de prueba)", ["Escalador", "F1 regresión logística", "Iteraciones", "Segundos", "F1 kNN"], [
    ["Sin escalar", "0.002", "16", "2.53", "0.053"],
    ["StandardScaler", "0.973", "12", "1.70", "0.918"],
    ["MinMaxScaler", "0.973", "60", "9.48", "0.925"],
    ["RobustScaler", "0.965", "82", "12.36", "0.964"],
])
p("Sin escalar, la regresión logística se detiene con un modelo que no aprende (F1 por debajo del azar, 1/35 ≈ 0.03) y kNN queda dominado por el tamaño en píxeles. <code>StandardScaler</code> logra la convergencia más rápida con el mejor F1 y es el que suponen PCA y el análisis factorial, por lo que se adopta. <code>MinMaxScaler</code> se reserva para chi-cuadrado, que exige valores no negativos, y para el umbral de varianza. <code>RobustScaler</code> no es necesario porque las colas ya se corrigieron con la transformación de potencia. En las escenas se aplicó la misma cadena; un ANOVA de un factor confirma que las 11 variables de calidad difieren entre tipos de toma (p < 0.001), lo que justifica conservar el tipo de toma como variable codificada.")

h2("s5", "5. Selección y extracción de características")
p("Con 304 recortes de entrenamiento y 823 variables candidatas hay más dimensiones que observaciones. Reducirlas baja el almacenamiento, el tiempo de entrenamiento y el riesgo de sobreajuste, y facilita interpretar qué distingue a los productos. Los filtros se aplicaron en cascada, del más barato al más informativo, y cada decisión se validó con una regresión logística en validación cruzada por bloques.")
h3("s51", "5.1 Umbral de varianza y correlación")
p("Con las variables en [0, 1] (Min-Max), un umbral de varianza de 0.001 eliminó solo 2 variables: las alertas de borrosidad y oscuridad, constantes en esta muestra (Figura 6). El filtro de correlación de Spearman con |ρ| > 0.95 eliminó 305 variables, sobre todo celdas HOG vecinas y variables construidas que repiten información, como el área y los megapíxeles o el brillo y la exposición (Figura 7). Quedaron 516 candidatas.")
fig("f6", "celda-030", "Varianza de cada variable tras Min-Max (escala logarítmica). La línea roja es el umbral.")
fig("f7", "celda-032", "Correlación de Spearman entre las variables construidas a mano (entrenamiento).")
h3("s52", "5.2 ANOVA, chi-cuadrado e información mutua")
p("ANOVA F compara la varianza entre clases contra la varianza dentro de ellas y es apropiada para variables continuas con objetivo categórico. Chi-cuadrado mide dependencia entre frecuencias no negativas y la clase. La información mutua capta relaciones no lineales. Los tres coinciden en que el color domina: el histograma de matiz, los componentes de Lab y el matiz circular encabezan los rankings (Figura 8). Entre las 100 variables con mayor F hay 49 de HOG, así que la forma también aporta. Chi-cuadrado coincide solo parcialmente con ANOVA (ρ = 0.64 entre rankings) y declara significativas 17 variables, porque supone frecuencias: funciona bien en los histogramas y mal en las variables continuas.")
fig("f8", "celda-035", "Quince variables más relevantes según cada criterio, por familia.")
p("Para decidir cuántas conservar se evaluó el desempeño contra el número de variables (Figura 9). Los tres criterios alcanzan su máximo cerca de k = 50; ANOVA logra F1 = 0.955 ± 0.016, por encima de usar las 516 (0.893). Con más variables el modelo empieza a sobreajustar.")
fig("f9", "celda-036", "F1 macro en validación cruzada por bloques contra el número de variables seleccionadas.")
h3("s53", "5.3 Análisis de componentes principales")
p("PCA proyecta las variables estandarizadas en direcciones ortogonales de máxima varianza. Hacen falta 85 componentes para retener el 90 % y 122 para el 95 % (Figura 10): comprime 516 variables a una cuarta parte, pero los componentes no tienen lectura directa.")
fig("f10", "celda-039", "Varianza explicada acumulada por PCA (entrenamiento).")
h3("s54", "5.4 Análisis factorial")
p("El análisis factorial modela las variables como combinación de pocos factores latentes y, con rotación varimax, produce cargas interpretables. Se aplicó a las 23 variables manuales restantes. Los datos son factorizables: Bartlett rechaza la matriz identidad (p < 0.001) y el KMO global es 0.69, aceptable según Kaiser (1974). El criterio de Kaiser retuvo 6 factores (Figura 11): F1 luminosidad y calidad de captura, F2 eje amarillo-azul, F3 intensidad cromática, F4 eje rojo-verde, F5 resolución y peso del archivo, y F6 detalle y textura.")
fig("f11", "celda-041", "Cargas factoriales con rotación varimax.")
h3("s55", "5.5 Comparativa y elección")
tabla("t8", "Configuraciones de preparación evaluadas con regresión logística", ["Configuración", "Variables", "F1 CV", "F1 prueba", "Segundos"], [
    ["1. Todas, potencia sin escalar", "823", "0.002", "0.002", "2.58"],
    ["2. Todas, potencia + estándar", "823", "0.882 ± 0.034", "0.973", "1.53"],
    ["3. + varianza + correlación", "516", "0.893 ± 0.039", "0.964", "1.73"],
    ["<b>4. + ANOVA (k = 50)</b>", "<b>50</b>", "<b>0.955 ± 0.016</b>", "<b>0.976</b>", "<b>0.28</b>"],
    ["5. PCA 95 %", "122", "0.887 ± 0.047", "0.964", "0.43"],
    ["6. Solo manuales", "23", "0.904 ± 0.022", "0.927", "0.17"],
    ["7. Factores del análisis factorial", "6", "0.812 ± 0.044", "0.821", "0.27"],
])
fig("f12", "celda-045", "F1 macro en validación cruzada (barras) y en prueba (puntos) por configuración.")
p("Se elige la configuración 4: transformación de potencia, estandarización, filtros de varianza y correlación, y ANOVA con k = 50. Es la mejor y la más estable en validación cruzada, lo confirma en prueba, usa el 6 % de las variables y entrena 5.5 veces más rápido que el conjunto completo. PCA no mejora la validación y sacrifica interpretación. Los 6 factores pierden exactitud, pero muestran que con muy pocas dimensiones interpretables ya se distingue la mayoría de los productos, así que se usarán para explicar errores. El pipeline ajustado y las tablas de características quedaron guardados en el servidor de Bnext.")

h2("s6", "6. Conclusiones de la fase de Preparación de los datos")
p("En CRISP-ML(Q), la preparación de los datos abarca selección, limpieza, construcción y estandarización, con aseguramiento de calidad en cada paso (Visengeriyeva et al., 2023). El balance de la fase es el siguiente.")
ol([
    "<b>Selección.</b> Se trabajó con la muestra real dentro de la infraestructura de Bnext: 100 escenas y 422 recortes de 35 SKU, con desbalance moderado (2.4) que no requiere remuestreo todavía.",
    "<b>Limpieza y calidad.</b> No hay faltantes y las capturas son nítidas y bien iluminadas. La alerta de sobreexposición necesita recalibrarse por producto. Se descartó <i>fuente_captura</i> por fuga de información.",
    "<b>Construcción.</b> Las imágenes se convirtieron en 823 variables con significado visual, más variables de dominio (índice de calidad, KB por megapíxel, matiz circular, alertas), discretizaciones y codificaciones justificadas.",
    "<b>Estandarización.</b> La transformación de potencia redujo la asimetría media de 1.41 a 0.21, y la estandarización hizo converger a la regresión logística en 12 iteraciones y llevó a kNN de 0.05 a 0.92 de F1.",
    "<b>Reducción.</b> Los filtros eliminaron 307 variables redundantes y ANOVA dejó 50 (6 % del original), con mejor validación cruzada (0.955 contra 0.882) y entrenamiento 5.5 veces más rápido. PCA y el análisis factorial quedaron documentados como alternativas.",
    "<b>Aseguramiento de calidad.</b> Partición por bloques temporales sin fuga entre ráfagas, transformaciones ajustadas solo con entrenamiento dentro de un pipeline, semilla fija y artefactos guardados en el servidor.",
])
p("<b>Riesgos que pasan a la fase de modelado.</b> La muestra es pequeña y cada SKU proviene de una sola sesión de captura, por lo que las métricas obtenidas miden la separabilidad de los productos y no la generalización a otras tiendas, fechas o dispositivos. El Avance 1 trabajó con datos sintéticos; este avance es el primero sobre datos reales y confirma que el color es la señal dominante. Las características manuales son una línea base interpretable: la detección en anaquel completo requerirá detectores de objetos con aprendizaje profundo, y las variables de calidad y el pipeline de este avance se reutilizarán para filtrar capturas y explicar errores.")
p("<b>Siguiente fase.</b> Entrenar modelos de clasificación y detección sobre la partición definida, compararlos contra la línea base de la configuración 4 y solicitar a Bnext más sesiones de captura por SKU para validar la generalización.")

BIBLIO = [
    "Box, G. E. P., y Cox, D. R. (1964). An analysis of transformations. <i>Journal of the Royal Statistical Society: Series B (Methodological), 26</i>(2), 211–252. https://doi.org/10.1111/j.2517-6161.1964.tb00553.x",
    "Dalal, N., y Triggs, B. (2005). Histograms of oriented gradients for human detection. En <i>2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition</i> (Vol. 1, pp. 886–893). IEEE. https://doi.org/10.1109/CVPR.2005.177",
    "Galli, S. (2022). <i>Python feature engineering cookbook</i> (2a ed.). Packt Publishing.",
    "Hasler, D., y Süsstrunk, S. (2003). Measuring colourfulness in natural images. En <i>Human Vision and Electronic Imaging VIII</i> (Vol. 5007, pp. 87–95). SPIE. https://doi.org/10.1117/12.477378",
    "Kaiser, H. F. (1974). An index of factorial simplicity. <i>Psychometrika, 39</i>(1), 31–36. https://doi.org/10.1007/BF02291575",
    "Ojala, T., Pietikäinen, M., y Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence, 24</i>(7), 971–987. https://doi.org/10.1109/TPAMI.2002.1017623",
    "Pech-Pacheco, J. L., Cristóbal, G., Chamorro-Martínez, J., y Fernández-Valdivia, J. (2000). Diatom autofocusing in brightfield microscopy: A comparative study. En <i>Proceedings 15th International Conference on Pattern Recognition</i> (Vol. 3, pp. 314–317). IEEE. https://doi.org/10.1109/ICPR.2000.903548",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
    "Visengeriyeva, L., Kammer, A., Bär, I., Kniesz, A., y Plöd, M. (2023). <i>CRISP-ML(Q). The ML lifecycle process</i>. MLOps. INNOQ. https://ml-ops.org/content/crisp-ml",
    "Yeo, I.-K., y Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. <i>Biometrika, 87</i>(4), 954–959. https://doi.org/10.1093/biomet/87.4.954",
]

INTEGRANTES = [("Jesús Soto Medrano", "A01794746"), ("Germán Eduardo Rodríguez Vázquez", "A01796325"),
               ("Oscar Alberto Ramírez Anaya", "A01795438")]
TITULO = "Planograma: IA para detección de productos y ubicación"
SUBTITULO = "Construcción, normalización, selección y extracción de características sobre la muestra de imágenes de anaquel"


# ------------------------------------------------------------------ HTML
def generar_html(figs, paginas):
    def pag(ref):
        return paginas.get(ref, "")
    tablas = [b for b in B if b[0] == "tabla"]
    figuras = [b for b in B if b[0] == "fig"]
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}
    num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    logo = base64.b64encode(LOGO.read_bytes()).decode()
    o = ['<!DOCTYPE html>\n<html lang="es">\n<head>\n<meta charset="utf-8">',
         "<title>Avance 2. Ingeniería de características - Equipo 48</title>",
         '<link rel="stylesheet" href="assets/print.css">',
         "<style>.logo-tec{width:62mm;margin:0 auto 4mm auto;display:block}"
         "figure img{max-width:100%;max-height:150mm}"
         "code{font-family:Menlo,Consolas,monospace;font-size:9pt;hyphens:none}</style>",
         "</head>\n<body>",
         '<section class="portada">\n<div class="portada-encabezado">',
         f'<img class="logo-tec" alt="Tecnológico de Monterrey" src="data:image/png;base64,{logo}">',
         '<p class="programa">Maestría en Inteligencia Artificial Aplicada</p>',
         '<p class="curso">TC5035.10 — Proyecto Integrador (Grupo 10) · Trimestre septiembre – diciembre 2026</p>',
         '<hr class="regla">\n</div>\n<div>',
         '<p class="etiqueta">Avance 2 · Ingeniería de características</p>',
         f'<h1 class="titulo">{TITULO}</h1>', f'<p class="subtitulo">{SUBTITULO}</p>',
         '</div>\n<div>\n<p class="equipo">Equipo 48</p>\n<ul class="integrantes">']
    o += [f'<li><span>{n}</span><span class="matricula">{m}</span></li>' for n, m in INTEGRANTES]
    o += ["</ul>\n</div>", '<div class="portada-pie">',
          "<p><strong>Profesora:</strong> Dra. Grettel Barceló Alonso</p>",
          "<p><strong>Asesor:</strong> Dr. Raúl Valente Ramírez</p>",
          "<p><strong>Sponsor:</strong> Dr. Juan Arturo Nolazco Flores, Director del Hub de Ciencia de Datos, Tecnológico de Monterrey</p>",
          "<p><strong>Organización:</strong> Bnext, Monterrey, Nuevo León</p>",
          f'<p style="margin-top:4mm">{FECHA}</p>', "</div>\n</section>"]
    o.append('<h2>Índice</h2>\n<ul class="indice">')
    for b in B:
        if b[0] in ("h2", "h3"):
            cls = ' class="sub"' if b[0] == "h3" else ""
            o.append(f'<li{cls}><span>{b[2]}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append(f'<li><span>Bibliografía</span><span class="relleno"></span><span class="pag">{pag("biblio")}</span></li></ul>')
    o.append('<h3 class="lista-indice-titulo">Índice de tablas</h3>\n<ul class="indice">')
    for b in tablas:
        o.append(f'<li class="sub"><span>Tabla {num_t[b[1]]}. {b[2]}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append('</ul>\n<h3 class="lista-indice-titulo">Índice de figuras</h3>\n<ul class="indice">')
    for b in figuras:
        corto = b[3].split(". ")[0].rstrip(".")
        o.append(f'<li class="sub"><span>Figura {num_f[b[1]]}. {corto}</span><span class="relleno"></span><span class="pag">{pag(b[1])}</span></li>')
    o.append("</ul>")
    primero = True
    for b in B:
        t = b[0]
        if t == "h2":
            salto = ' class="salto"' if primero else ""
            o.append(f'<h2{salto} id="{b[1]}">{b[2]}</h2>'); primero = False
        elif t == "h3":
            o.append(f'<h3 id="{b[1]}">{b[2]}</h3>')
        elif t == "p":
            o.append(f"<p>{b[1]}</p>")
        elif t in ("ul", "ol"):
            o.append(f"<{t}>" + "".join(f"<li>{x}</li>" for x in b[1]) + f"</{t}>")
        elif t == "nota":
            o.append(f'<div class="nota"><p class="titulo-nota">{b[1]}</p><p>{b[2]}</p></div>')
        elif t == "tabla":
            _, i, tit, enc, filas = b
            o.append(f'<table id="{i}"><caption><b>Tabla {num_t[i]}.</b> {tit}</caption><thead><tr>'
                     + "".join(f"<th>{e}</th>" for e in enc) + "</tr></thead><tbody>"
                     + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
                     + "</tbody></table>")
        elif t == "fig":
            _, i, celda, tit = b
            o.append(f'<figure id="{i}"><img alt="Figura {num_f[i]}" src="data:image/png;base64,{figs[celda]}">'
                     f"<figcaption><b>Figura {num_f[i]}.</b> {tit} Elaboración propia con datos de Bnext.</figcaption></figure>")
    o.append('<h2 id="biblio">Bibliografía</h2>\n<div class="biblio">')
    o += [f"<p>{x}</p>" for x in BIBLIO]
    o.append("</div>\n</body>\n</html>\n")
    HTML.write_text("\n".join(o), encoding="utf-8")


def imprimir_pdf():
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)


def localizar_paginas():
    """Página (1-based) donde aparece cada encabezado, tabla y figura, después del índice."""
    doc = fitz.open(PDF)
    textos = [pg.get_text() for pg in doc]
    paginas = {}
    tablas = [b for b in B if b[0] == "tabla"]
    figuras = [b for b in B if b[0] == "fig"]
    # El índice repite los títulos: la búsqueda empieza después de su última entrada
    ultima = f"Figura {len(figuras)}."
    desde = next(n for n in range(2, len(textos) + 1) if ultima in textos[n - 1]) + 1
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}
    num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    def buscar(clave):
        for n in range(desde, len(textos) + 1):
            if clave in textos[n - 1]:
                return n
        return ""
    for b in B:
        if b[0] in ("h2", "h3"):
            paginas[b[1]] = buscar(b[2])
        elif b[0] == "tabla":
            paginas[b[1]] = buscar(f"Tabla {num_t[b[1]]}.")
        elif b[0] == "fig":
            paginas[b[1]] = buscar(f"Figura {num_f[b[1]]}.")
    paginas["biblio"] = buscar("Bibliografía")
    return paginas


def pie_pdf():
    doc = fitz.open(PDF)
    total = doc.page_count
    for i, pg in enumerate(doc, start=1):
        if i == 1:
            continue
        w, h = pg.rect.width, pg.rect.height
        y = h - 42.5
        pg.draw_line(fitz.Point(56.7, y - 9), fitz.Point(w - 56.7, y - 9), color=(0.79, 0.81, 0.84), width=0.5)
        pg.insert_text(fitz.Point(56.7, y + 2), "TC5035.10 Proyecto Integrador  ·  Avance 2  ·  Equipo 48",
                       fontname="helv", fontsize=7.5, color=(0.42, 0.42, 0.42))
        der = f"Página {i} de {total}"
        pg.insert_text(fitz.Point(w - 56.7 - fitz.get_text_length(der, fontname="helv", fontsize=7.5), y + 2),
                       der, fontname="helv", fontsize=7.5, color=(0.42, 0.42, 0.42))
    doc.set_metadata({"title": "Avance 2. Ingeniería de características — Equipo 48", "author": "Equipo 48",
                      "subject": "TC5035.10 Proyecto Integrador — Avance 2",
                      "keywords": "planograma, ingenieria de caracteristicas, seleccion, PCA, analisis factorial, CRISP-ML(Q)"})
    doc.saveIncr()
    return total


# ------------------------------------------------------------------ DOCX
def runs_desde_html(par, texto, tam=None, color=None, fuente=None):
    """Convierte el marcado mínimo (<b>, <i>, <code>) a runs de Word."""
    estado = {"b": False, "i": False, "code": False}
    for trozo in re.split(r"(</?(?:b|i|code)>)", texto):
        m = re.fullmatch(r"<(/?)(b|i|code)>", trozo)
        if m:
            estado[m.group(2)] = m.group(1) == ""
            continue
        if not trozo:
            continue
        r = par.add_run(html.unescape(trozo))
        r.bold = estado["b"] or None
        r.italic = estado["i"] or None
        if estado["code"]:
            r.font.name = "Consolas"; r.font.size = Pt(9)
        elif fuente:
            r.font.name = fuente
        if tam and not estado["code"]:
            r.font.size = Pt(tam)
        if color:
            r.font.color.rgb = color


def sombrear(celda, hexcolor):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def campo(par, instruccion):
    for tipo, texto in (("begin", None), (None, instruccion), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        if tipo:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tipo); r._r.append(fc)
        elif texto == instruccion:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instruccion; r._r.append(it)
        else:
            r.text = texto


def generar_docx(figs, paginas):
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.4)
    normal = d.styles["Normal"]
    normal.font.name = "Georgia"; normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.15
    for nombre, tam in (("Heading 1", 15), ("Heading 2", 12)):
        st = d.styles[nombre]
        st.font.name = "Arial"; st.font.size = Pt(tam); st.font.color.rgb = AZUL; st.font.bold = True
        rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), "Arial")
    def centrado(texto, tam=None, color=None, fuente=None, negrita=False, antes=0):
        par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(antes)
        runs_desde_html(par, f"<b>{texto}</b>" if negrita else texto, tam, color, fuente)
        return par
    # portada
    par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(LOGO), width=Cm(6.2))
    centrado("Maestría en Inteligencia Artificial Aplicada", 11, RGBColor(0x44, 0x44, 0x44), antes=30)
    centrado("TC5035.10 — Proyecto Integrador (Grupo 10) · Trimestre septiembre – diciembre 2026", 10, RGBColor(0x44, 0x44, 0x44))
    centrado("AVANCE 2 · INGENIERÍA DE CARACTERÍSTICAS", 8.5, RGBColor(0x66, 0x66, 0x66), "Arial", antes=40)
    centrado(TITULO, 20, AZUL, "Arial", negrita=True, antes=6)
    centrado(f"<i>{SUBTITULO}</i>", 10.5, RGBColor(0x55, 0x55, 0x55))
    centrado("Equipo 48", 13, None, "Arial", negrita=True, antes=40)
    t = d.add_table(rows=0, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for n, m in INTEGRANTES:
        fila = t.add_row().cells
        fila[0].text = n; fila[1].text = m
        fila[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    centrado("<b>Profesora:</b> Dra. Grettel Barceló Alonso", 9.5, antes=50)
    centrado("<b>Asesor:</b> Dr. Raúl Valente Ramírez", 9.5)
    centrado("<b>Sponsor:</b> Dr. Juan Arturo Nolazco Flores, Director del Hub de Ciencia de Datos, Tecnológico de Monterrey", 9.5)
    centrado("<b>Organización:</b> Bnext, Monterrey, Nuevo León", 9.5)
    centrado(FECHA, 9.5, RGBColor(0x66, 0x66, 0x66), antes=8)
    # índice (campo TOC; Word lo actualiza al abrir)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    par = d.add_paragraph(); runs_desde_html(par, "Índice", 15, AZUL, "Arial")
    campo(d.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')
    ajustes = d.settings.element
    uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); ajustes.append(uf)
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    # cuerpo
    tablas = [b for b in B if b[0] == "tabla"]; figuras = [b for b in B if b[0] == "fig"]
    num_t = {b[1]: i + 1 for i, b in enumerate(tablas)}; num_f = {b[1]: i + 1 for i, b in enumerate(figuras)}
    for b in B:
        tp = b[0]
        if tp == "h2":
            d.add_heading(b[2], level=1)
        elif tp == "h3":
            d.add_heading(b[2], level=2)
        elif tp == "p":
            par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; runs_desde_html(par, b[1])
        elif tp in ("ul", "ol"):
            for x in b[1]:
                par = d.add_paragraph(style="List Number" if tp == "ol" else "List Bullet"); runs_desde_html(par, x)
        elif tp == "nota":
            tn = d.add_table(rows=1, cols=1); c = tn.rows[0].cells[0]; sombrear(c, "F3F6FB")
            runs_desde_html(c.paragraphs[0], f"<b>{b[1]}</b>", 9.5, AZUL, "Arial")
            runs_desde_html(c.add_paragraph(), b[2], 9.5)
            d.add_paragraph()
        elif tp == "tabla":
            _, i, tit, enc, filas = b
            par = d.add_paragraph(); par.paragraph_format.keep_with_next = True
            runs_desde_html(par, f"<b>Tabla {num_t[i]}.</b> {tit}", 9, AZUL, "Arial")
            tb = d.add_table(rows=1, cols=len(enc)); tb.style = "Table Grid"
            for j, e in enumerate(enc):
                c = tb.rows[0].cells[j]; sombrear(c, AZUL_TABLA)
                runs_desde_html(c.paragraphs[0], f"<b>{e}</b>", 9, RGBColor(0xFF, 0xFF, 0xFF), "Arial")
            for f in filas:
                cs = tb.add_row().cells
                for j, v in enumerate(f):
                    runs_desde_html(cs[j].paragraphs[0], v, 9)
            d.add_paragraph()
        elif tp == "fig":
            _, i, celda, tit = b
            par = d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.keep_with_next = True
            par.add_run().add_picture(io.BytesIO(base64.b64decode(figs[celda])), width=Cm(16))
            par = d.add_paragraph()
            runs_desde_html(par, f"<b>Figura {num_f[i]}.</b> {tit} Elaboración propia con datos de Bnext.", 9, None, "Arial")
    d.add_heading("Bibliografía", level=1)
    for x in BIBLIO:
        par = d.add_paragraph(); par.paragraph_format.left_indent = Cm(1); par.paragraph_format.first_line_indent = Cm(-1)
        runs_desde_html(par, x, 10)
    # pie con número de página
    sec.different_first_page_header_footer = True
    pie = sec.footer.paragraphs[0]
    runs_desde_html(pie, "TC5035.10 Proyecto Integrador  ·  Avance 2  ·  Equipo 48        Página ", 7.5, RGBColor(0x6B, 0x6B, 0x6B), "Arial")
    campo(pie, "PAGE")
    d.core_properties.author = "Equipo 48"
    d.core_properties.title = "Avance 2. Ingeniería de características — Equipo 48"
    d.save(DOCX)


if __name__ == "__main__":
    figs = figuras_libreta()
    generar_html(figs, {})
    imprimir_pdf()
    paginas = localizar_paginas()
    generar_html(figs, paginas)
    imprimir_pdf()
    assert localizar_paginas() == paginas, "La paginación cambió al insertar los números del índice"
    total = pie_pdf()
    generar_docx(figs, paginas)
    print(f"{PDF.name}: {total} páginas | {HTML.name} | {DOCX.name}")
