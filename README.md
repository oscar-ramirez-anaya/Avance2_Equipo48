<div style="text-align: center;">

# Avance 2 | Ingeniería de características

### Construcción, normalización, selección y extracción de características sobre imágenes de anaquel

**Tecnológico de Monterrey** · Maestría en Inteligencia Artificial Aplicada
**TC5035.10 — Proyecto Integrador (Grupo 10)**

**Equipo 48** · **Modalidad:** En equipo · **Fecha:** Octubre 2026

| Integrante | Matrícula |
|---|---|
| Oscar Alberto Ramírez Anaya | A01795438 |
| Germán Eduardo Rodríguez Vázquez | A01796325 |
| Jesús Soto Medrano | A01794746 |

**Profesora:** Dra. Grettel Barceló Alonso · **Asesor:** Dr. Raúl Valente Ramírez
**Sponsor:** Dr. Juan Arturo Nolazco Flores · **Organización:** Bnext

</div>

---

## Descripción

Fase de *Preparación de los datos* de CRISP-ML(Q). A partir de la muestra real de imágenes de anaquel
proporcionada por Bnext (100 escenas de refrigeradores y vitrinas y 422 recortes de producto de 35
SKU), se construye un conjunto de variables útil para el aprendizaje automático, se normaliza y se
reduce con métodos de filtrado y extracción. Cada decisión se justifica con el conocimiento del dominio
o con una prueba sobre los datos.

> **Manejo de datos.** La libreta se ejecutó dentro del servidor de Bnext conforme a
> [`POLITICA_DATOS.md`](POLITICA_DATOS.md). Las imágenes, las tablas de características y el pipeline
> ajustado permanecen en esa infraestructura. Este repositorio contiene solo código, gráficas y tablas
> agregadas; los SKU están anonimizados (`SKU_01` … `SKU_35`).

## Contenido

| Archivo | Descripción |
|---|---|
| **`Avance2.48.ipynb`** | Libreta ejecutada de principio a fin, en secuencia, con salidas y gráficos |
| `Avance2.48.pdf` | Reporte del avance |
| `Avance2.48.docx` | Reporte en formato editable |
| `Avance2.48.html` | Fuente del PDF |
| `scripts/construir_avance2.py` | Genera la libreta (código e interpretaciones en `avance2_textos.json`) |
| `scripts/generar_documento_avance2.py` | Genera HTML, PDF y DOCX con las figuras de la libreta ejecutada |
| `scripts/verificar_publicacion.py`, `.githooks/` | Verificación previa a cada commit según la política de datos |
| `POLITICA_DATOS.md` | Acuerdo de manejo de datos con Bnext |

## El notebook — descargar o visualizar

- **Visualizar en GitHub:** [`Avance2.48.ipynb`](./Avance2.48.ipynb)
- **Visor enriquecido (nbviewer):** https://nbviewer.org/github/oscar-ramirez-anaya/Avance2_Equipo48/blob/main/Avance2.48.ipynb
- **Descargar (raw):** https://raw.githubusercontent.com/oscar-ramirez-anaya/Avance2_Equipo48/main/Avance2.48.ipynb

## Secciones del notebook

**A. Ingeniería de características**
1. Datos de partida: inventario y distribución por SKU.
2. Extracción base: geometría, exposición y calidad, color (HSV, Lab), textura LBP y forma HOG.
3. Construcción de variables de dominio: índice de calidad, KB por megapíxel, nitidez por borde,
   matiz circular y alertas de captura.
4. Discretización: cortes de dominio (iluminación, formato de empaque) y cuartiles (nitidez).
5. Codificación: one-hot, ordinal y LabelEncoder; exclusión de una variable con fuga.
6. Partición por bloques temporales dentro de cada SKU, sin fuga entre ráfagas de captura.
7. Transformaciones Box-Cox y Yeo-Johnson.
8. Escalamiento: StandardScaler, MinMaxScaler y RobustScaler, con prueba de convergencia.
9. Aplicación a las escenas.

**B. Selección y extracción de características**

10. Umbral de varianza. 11. Correlación. 12. ANOVA, chi-cuadrado e información mutua.
13. PCA. 14. Análisis factorial (KMO, Bartlett, varimax). 15. Comparativa de configuraciones.
16. Pipeline final.

**C. Conclusiones de la fase de Preparación de los datos (CRISP-ML(Q)).**

## Resultados principales

| Configuración | Variables | F1 CV | F1 prueba | Segundos |
|---|---:|---:|---:|---:|
| Todas, potencia sin escalar | 823 | 0.002 | 0.002 | 2.58 |
| Todas, potencia + estándar | 823 | 0.882 | 0.973 | 1.53 |
| + varianza + correlación | 516 | 0.893 | 0.964 | 1.73 |
| **+ ANOVA (k = 50)** | **50** | **0.955** | **0.976** | **0.28** |
| PCA 95 % | 122 | 0.887 | 0.964 | 0.43 |
| Solo variables manuales | 23 | 0.904 | 0.927 | 0.17 |
| Factores del análisis factorial | 6 | 0.812 | 0.821 | 0.27 |

- La transformación de potencia redujo la asimetría absoluta media de 1.41 a 0.21.
- Sin escalar, el modelo no aprende; con StandardScaler la regresión logística converge en 12
  iteraciones y kNN pasa de 0.05 a 0.92 de F1.
- El color domina la relevancia (histograma de matiz, Lab, matiz circular) y la forma (HOG) aporta la
  mitad de las 100 variables más relevantes.
- Se elige ANOVA con k = 50: 6 % de las variables, mejor validación cruzada y entrenamiento 5.5 veces
  más rápido.
- El análisis factorial resume las variables manuales en 6 factores interpretables: luminosidad y
  calidad, eje amarillo-azul, intensidad cromática, eje rojo-verde, resolución y textura.

## Reproducir

```bash
python3 scripts/construir_avance2.py
# en el servidor de Bnext:
export PLANOGRAMA_DATA_DIR=<muestra> PLANOGRAMA_OUT_DIR=<salidas>
jupyter nbconvert --to notebook --execute --inplace Avance2.48.ipynb
# de vuelta en el repositorio:
python3 scripts/generar_documento_avance2.py
```

Para activar la verificación automática en cada commit: `git config core.hooksPath .githooks`.
