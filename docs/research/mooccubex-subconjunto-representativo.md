---
title: "Subconjuntos representativos de MOOCCubeX"
subtitle: "Revisión metodológica y protocolo de calibración para EDuRec"
author: "Proyecto EDuRec"
date: "Octubre 2026"
lang: es
---

## Resumen

MOOCCubeX contiene 4.216 cursos, 230.263 vídeos, 358.265 ejercicios, 637.572 conceptos y más de 296 millones de registros de 3.330.294 estudiantes (Yu et al., 2021). Entrenar y ablacionar un recomendador sobre el conjunto completo es costoso, pero reducir datos puede cambiar las conclusiones. Esta revisión narrativa sintetiza la evidencia sobre submuestreo en recomendación y evaluación offline, y propone un protocolo para calibrar subconjuntos. La evidencia muestra que (i) muestrear candidatos o ítems de evaluación invalida comparaciones, (ii) el filtrado por umbrales redefine la población en lugar de reducirla neutralmente, y (iii) los coresets son prometedores pero su evidencia en recomendación es reciente y limitada. En EDuRec el coste dominante es el grafo de conceptos, no las interacciones. Se recomienda muestrear usuarios —no ítems—, mantener el test intacto, usar el sampler hash actual como línea base y validar la fidelidad del ranking con correlación de Spearman, error de efecto y cobertura.

**Palabras clave:** MOOCCubeX; submuestreo; coresets; evaluación offline; recomendación; MOOC.

## Puntos clave

- "Representativo" no es una propiedad del dato: es una propiedad relativa a la inferencia que se quiera hacer. Para un recomendador, la inferencia relevante es la comparación entre modelos.
- La evidencia es contundente en un punto: muestrear el conjunto de candidatos de evaluación rompe las comparaciones, incluso en expectativa (Krichene & Rendle, 2020).
- En EDuRec, reducir usuarios no reduce la tabla de conceptos ni las activaciones HGT; son dos cuellos de botella distintos y requieren palancas distintas.
- La escalera recomendada es: hash uniforme actual → estratificado por actividad y cobertura → muestreo de subgrafos para el grafo → coresets solo como experimento.
- Ningún subconjunto debería aceptarse sin una tarjeta de fidelidad: tamaño, cobertura, correlación de rankings, error de efecto y potencia.

# 1. Introducción

MOOCCubeX se construyó para soportar aprendizaje adaptativo a escala: cursos, vídeos, ejercicios, un grafo fino de conceptos y registros de comportamiento de más de tres millones de estudiantes (Yu et al., 2021). EDuRec lo usa para recomendación de cursos con un modelo que combina historial secuencial y un grafo de conocimiento. En la configuración de referencia del repositorio, el conjunto procesado tiene 40.250 usuarios y 2.865 cursos, y el comentario de `configs/small/kg_rnn-mooccubex.yaml` identifica el cuello de botella real: la tabla de atributos del grafo (215.000 conceptos × `emb_dim`) y las activaciones HGT, no los usuarios ni las interacciones.

Esa distinción es el punto de partida. "Usar menos datos" puede significar tres cosas diferentes:

1. **Menos filas de interacción** (recortar historiales o muestrear eventos).
2. **Menos usuarios** (muestrear la población).
3. **Un grafo más pequeño** (menos conceptos, menos cursos, menos aristas).

Cada una reduce un coste distinto y tiene un riesgo distinto. Confundirlas lleva a elegir la palanca equivocada: muestrear usuarios acelera la carga y el entrenamiento por lotes, pero no reduce la tabla de 215.000 conceptos; muestrear cursos sí reduce el grafo, pero cambia el catálogo y, con él, la tarea de ranking.

La pregunta aplicada es, entonces: ¿qué estrategia produce un subconjunto que preserve las conclusiones de evaluación —orden de los modelos y magnitud del efecto— a una fracción del coste? La literatura ha estudiado el problema en dos frentes separados: la **evaluación** (métricas muestreadas, muestreo de candidatos, elección de split) y la **optimización** (muestreo de subgrafos, coresets, poda de datos). No existe, hasta donde alcanza esta revisión, un protocolo específico para MOOC ni para MOOCCubeX.

Este informe hace cuatro cosas: (a) operacionaliza "representativo" en tres nociones medibles; (b) sintetiza la evidencia sobre las familias de métodos; (c) propone un protocolo de calibración reproducible; y (d) traduce todo a una recomendación concreta para EDuRec. No ejecuta el benchmark sobre los datos: entrega el diseño y los criterios para ejecutarlo.

# 2. Qué significa "representativo"

## 2.1 Tres nociones que no son intercambiables

| Noción | Definición | Métricas típicas | Riesgo principal |
|---|---|---|---|
| **Distribucional** | Las marginales de S se parecen a las de D (actividad, popularidad, longitud de historial) | KS, divergencia Jensen-Shannon, Gini | Igualar marginales no garantiza preservar el ranking de modelos |
| **Cobertura** | S retiene los conceptos, campos, prerrequisitos y la cola larga relevantes | % conceptos cubiertos, entropía de campos, retención del cuartil inferior de popularidad | Sobre-representar ítems raros y distorsionar la métrica |
| **Fidelidad de conclusiones** | S reproduce el orden y el tamaño del efecto de los modelos evaluados en D | Spearman/Kendall, error relativo, error de efecto, potencia | Es la más cara de medir y la única que responde la pregunta |

La práctica habitual en MOOC se queda en la primera o, como mucho, en la segunda. La evidencia de evaluación muestra que eso no basta.

## 2.2 Definición operativa propuesta

Sea D el dataset completo, S un subconjunto, M un conjunto de modelos y m una métrica (por ejemplo, NDCG@10). Se dice que S es **(ε, δ, γ)-fiel** a D para (M, m) si:

- (a) para todo par A, B ∈ M con |m_D(A) − m_D(B)| > ε, se cumple sign(m_S(A) − m_S(B)) = sign(m_D(A) − m_D(B));
- (b) |m_S(X) − m_D(X)| ≤ δ · m_D(X) para todo X ∈ M;
- (c) la cobertura de conceptos y campos de S es al menos γ.

ε captura las diferencias que importan; δ captura el sesgo absoluto tolerable; γ captura la cobertura del contenido. Es una herramienta de diseño, no un estándar de la literatura: los umbrales se fijan antes de mirar los resultados y se justifican por el uso previsto.

## 2.3 Por qué la similitud distribucional no es suficiente

Tres resultados empíricos separan la representatividad distribucional de la fidelidad de conclusiones.

Primero, las métricas muestreadas son inconsistentes: no preservan afirmaciones relativas como "A es mejor que B", ni siquiera en expectativa, y cuanto menor es el muestreo, menos se distinguen los modelos (Krichene & Rendle, 2020). Segundo, reducir el conjunto de ítems objetivo contradice en muchos casos el resultado obtenido con objetivos grandes, y los autores dan una explicación fundamentada de por qué (Cañamares & Castells, 2020). Tercero, la estrategia de partición train/test puede cambiar el ranking de los recomendadores (Meng et al., 2020). Un subconjunto puede parecerse mucho a D en sus marginales y, aun así, invertir el orden de dos modelos.

La implicación es directa: la representatividad hay que **medirla en el espacio de las conclusiones**, no solo en el espacio de las distribuciones.

# 3. Método de la revisión

Revisión narrativa documentada, no sistemática. La búsqueda se hizo en OpenAlex (metadatos y abstracts), ACM Digital Library, arXiv y páginas de editoriales, con bolt-on de referencias. Términos principales: `("dataset subset" OR coreset OR "data pruning" OR sampling) AND (recommender OR recommendation)`, `("offline evaluation") AND (sampling OR split)`, `MOOCCubeX`, `MOOC course recommendation`. Criterios de inclusión: trabajo revisado por pares o preprint con metodología explícita, relevancia directa para submuestreo o evaluación de recomendadores, y publicación entre 2014 y 2026 (con fundamentos de 2017 en adelante). Se excluyeron trabajos puramente de visión no transferibles y recomendadores basados en LLM. El resultado son 25 fuentes.

Limitaciones del método: un solo revisor, sin doble cribado; sesgo de selección hacia literatura en inglés; sin meta-análisis; y una parte de la evidencia procede de visión por computador, no de recomendación. Estas limitaciones se retoman en la sección 7.

# 4. Evidencia sobre las familias de submuestreo

## 4.1 Filtrado estructural: k-core y umbrales

Es la práctica más extendida. Consiste en eliminar usuarios o ítems con menos de k interacciones, iterativamente, hasta que todos los nodos supervivientes cumplen el umbral. NGCF, por ejemplo, declara explícitamente un ajuste **10-core** en sus experimentos (Wang et al., 2019b). El propio paper de MOOCCubeX aplica filtros al construir su tarea de recomendación: descarta registros de visualización por debajo de un umbral de duración y restringe la ventana temporal, quedándose con 33.650 estudiantes (Yu et al., 2021). EDuRec ya hace algo análogo: excluye usuarios con más de 100 enrollments y exige al menos 5 conexiones por usuario y curso (`edurec/settings.py`).

Qué reduce: filas y nodos, de forma barata y reproducible. Qué riesgo tiene: **redefine la población**. Un 10-core elimina precisamente a los usuarios fríos y a los cursos de cola larga; después no se puede afirmar nada sobre cold-start ni sobre cobertura. La evidencia reciente lo confirma: los umbrales de filtrado afectan la validez convergente de la evaluación, y no hay una configuración uniformemente mejor; la validez depende del dataset y de los objetivos densos de evaluación (Parajuli et al., 2026).

Veredicto: útil como limpieza y como línea base, pero no es un método de submuestreo "representativo". Es un cambio de población, y debe reportarse como tal.

## 4.2 Muestreo estratificado y por clusters

La teoría de muestreo dice que estratificar por variables auxiliares reduce la varianza frente al muestreo aleatorio simple, siempre que los estratos estén correlacionados con la métrica. En este dominio, los estratos naturales son el decil de actividad del usuario, el campo o área del curso, y la popularidad del curso. El muestreo por clusters —elegir cursos completos y conservar todos sus enrollments— preserva la integridad de cada curso y evita crear ítems artificialmente fríos.

La evidencia indirecta a favor es la de muestreo de ítems objetivo: si se reduce el conjunto de ítems, las comparaciones se invierten en muchos casos (Cañamares & Castells, 2020). De ahí se sigue una regla prudente: **muestrear usuarios completos antes que interacciones o ítems**, porque el historial de cada usuario queda intacto y el catálogo sigue completo.

No hay, en cambio, un estudio que compare muestreo estratificado contra hash uniforme en MOOC. Es una propuesta razonada, no un hallazgo. Debe validarse con el protocolo de la sección 5.4.

## 4.3 Muestreo de subgrafos para GNN

Aquí el objetivo no es reducir el dataset, sino el coste por época. GraphSAGE muestrea vecinos por capa (Hamilton et al., 2017); Cluster-GCN construye lotes a partir de subgrafos densos obtenidos por clustering (Chiang et al., 2019); GraphSAINT muestrea el grafo de entrenamiento para construir minibatches completos (Zeng et al., 2020); HGT propone HGSampling para grafos heterogéneos a escala web (Hu et al., 2020). KGAT, el pariente cercano del modelo de EDuRec, explota relaciones de alto orden en el grafo de conocimiento (Wang et al., 2019a).

Qué reduce: memoria y tiempo por época, no el tamaño del dataset. La evaluación puede seguir siendo sobre el grafo completo. Qué riesgo tiene: varianza de muestreo, sesgo hacia comunidades densas en Cluster-GCN, y acoplamiento entre sampler y arquitectura. Una alternativa radical es cambiar el modelo por uno de coste lineal, como LTGNN (Zhang et al., 2024), en lugar de muestrear. En recomendación, LightGCN mostró que una propagación lineal simple compite con GNN más pesadas (He et al., 2020), lo que sugiere que parte del coste del grafo puede reducirse sin muestrear.

Para EDuRec esta es la palanca más directa si el problema es OOM o lentitud: el modelo ya usa capas HGT (`gnn_layers: 2`, `gnn_heads: 4`), y HGSampling es el sampler natural para ese caso.

## 4.4 Coresets, poda de datos y destilación

Un coreset es un subconjunto ponderado que aproxima el objetivo de optimización del conjunto completo (Feldman, 2020). En aprendizaje profundo, la poda de datos puntúa ejemplos por su importancia —norma del gradiente (GraNd) o norma L2 del error (EL2N)— y descarta fracciones grandes sin perder precisión (Paul et al., 2021). Con una métrica de poda suficientemente buena, la poda puede batir la ley de potencias que relaciona error y tamaño de datos (Sorscher et al., 2022).

Pero hay dos cautelas importantes. Primera: en regímenes de compresión alta (≤30% de retención), la poda aleatoria es un baseline fuerte y supera a la mayoría de los métodos basados en puntuaciones; existen resultados de "no free lunch" para la poda por scores (Ayed & Hayou, 2023). Segunda: la evidencia específica de recomendación es reciente y escasa. El trabajo de Ju et al. (2024) investiga sistemáticamente coresets para recomendadores y encuentra que pueden preservar propiedades del dataset original y ser competitivos frente al entrenamiento completo, pero es una excepción en un panorama dominado por visión. La destilación de datasets —sintetizar un conjunto pequeño— también cubre interacciones usuario-ítem en su revisión (Sachdeva & McAuley, 2023; Lei & Tao, 2023), pero su madurez en recomendación es baja.

Veredicto: prometedor, no de primera línea. Requiere una referencia completa para calibrar, es caro y puede sesgar hacia la cola popular. Nivel 4 de la escalera, no nivel 1.

## 4.5 Muestreo en la evaluación: el punto más delicado

Las métricas muestreadas son inconsistentes y no preservan afirmaciones relativas, ni siquiera en expectativa (Krichene & Rendle, 2020). El análisis posterior muestra una relación aproximadamente lineal entre evaluación muestreada y global para Recall@K, y propone estimadores y muestreo adaptativo para corregir el sesgo (Li et al., 2023). Pero la conclusión práctica no cambia: **si se comparan modelos, no se muestrea el conjunto de candidatos**. Y si se hace por coste, hay que corregir y reportarlo.

El problema no es aislado: los fallos de evaluación se copian de un paper a otro sin cuestionarlos (Hidasi & Czapp, 2023), y la falta de reproducibilidad agrava la comparación (Dacrema et al., 2021).

## 4.6 Splits y leakage

La elección del split cambia el ranking de modelos (Meng et al., 2020). Por tanto, el orden de operaciones importa: primero se fija el split temporal, después se muestrea dentro de train, y el test se mantiene intacto —o se muestrea por usuario, conservando todos sus positivos. Muestrear interacciones después del split puede dejar usuarios sin positivos en test, y truncar historiales cambia la distribución de entrada. Cualquiera de las dos cosas produce métricas que no son comparables con la referencia.

# 5. Aplicación a MOOCCubeX y EDuRec

## 5.1 Estructura: el eje correcto es el usuario

MOOCCubeX tiene 4.216 cursos y 3,33 millones de estudiantes (Yu et al., 2021). Es una proporción extrema: muchos usuarios, pocos ítems. En la configuración de EDuRec quedan 2.865 cursos y 40.250 usuarios. Eso tiene dos consecuencias.

Primera: **muestrear ítems es peligroso**. El catálogo es pequeño y es el ancla del grafo de conocimiento; reducirlo cambia la tarea de ranking y hace incomparables los resultados con el conjunto completo. Segunda: **muestrear usuarios es natural**. Se conserva el catálogo, se conservan los historiales completos y la métrica sigue estimando el rendimiento sobre la población de usuarios, con varianza conocida.

La tarea del paper de MOOCCubeX es recomendación secuencial de vídeo, no de cursos (Yu et al., 2021). EDuRec formula una tarea distinta —predecir enrollments—, así que los filtros del paper no son directamente transplantables; sirven como referencia de práctica, no como receta.

## 5.2 Qué hace ya el sampler actual

El README describe `--max-interactions`: selecciona usuarios en orden hash estable hasta alcanzar un presupuesto de interacciones y elimina del catálogo estático a los usuarios no seleccionados. Es un muestreo uniforme de usuarios sin reemplazo, determinista para un presupuesto dado. Sus propiedades:

- **Ventaja:** insesgado para estimandos a nivel de usuario, reproducible y ya implementado.
- **Límite 1:** no estratifica; con presupuestos pequeños puede infrarrepresentar cursos raros y conceptos poco frecuentes.
- **Límite 2:** al ser determinista, no permite estimar varianza entre muestras. Para el protocolo de fidelidad conviene variar la sal del hash y generar varias réplicas.

Es una línea base perfectamente válida. El problema no es el sampler, es que hoy no se mide si el subconjunto preserva las conclusiones.

## 5.3 Escalera de métodos para EDuRec

| Nivel | Método | Reduce | Cuándo | Riesgo |
|---|---|---|---|---|
| **N0** | Hash uniforme actual | Usuarios e interacciones | Desarrollo; línea base | Sin estratificar |
| **N1** | Estratificado por actividad y campo | Usuarios, con cobertura | Benchmark de ablaciones | Sobre-representa la cola |
| **N2** | Cluster por curso | Cursos, usuarios y grafo | Inferencia por curso | Cambia el catálogo |
| **N3** | Subgraph sampling + `emb_dim` | Memoria y tiempo | OOM o lentitud | Varianza del sampler |
| **N4** | Coreset / poda | Filas de entrenamiento | Eficiencia de datos | Caro; evidencia escasa |

La lectura correcta es que **N3 y N1 no compiten**: N3 ataca el coste del grafo, N1 ataca el tamaño de la población. Si el problema es que el modelo no cabe en memoria, empezar por N3 (bajar `emb_dim` de 256 a 64 ya reduce la tabla de conceptos ~4×, según el comentario de `configs/small/`). Si el problema es que cada época tarda demasiado o que las ablaciones son caras, N1.

## 5.4 Protocolo de calibración de fidelidad

| Paso | Acción | Salida |
|---|---|---|
| 0 | Congelar la referencia: dataset procesado completo, split temporal, zoo de modelos (`KGSeq` y `SASRecText`), métricas (NDCG@10, Recall@20), 3–5 semillas | Línea base reproducible |
| 1 | Perfilar la estructura: distribución de actividad, popularidad de cursos, cobertura de conceptos y campos, k-core | Tabla de estructura |
| 2 | Generar candidatos: N0, N1 y, si aplica, N2, a fracciones 5%, 10%, 25%, 50% y 100% | Conjuntos etiquetados |
| 3 | Entrenar cada modelo × subconjunto × semilla, con el mismo preprocesamiento y split | Matriz de métricas |
| 4 | Calcular fidelidad: Spearman/Kendall del ranking, error relativo máximo, error de efecto, cobertura, KS/JSD, potencia | Informe de fidelidad |
| 5 | Aplicar la regla de decisión y elegir el tamaño mínimo que cumple | Subconjunto aceptado |
| 6 | Emitir la tarjeta del subconjunto | Artefacto auditable |

**Criterios de aceptación sugeridos** (puntos de partida, no estándares):

| Criterio | Umbral | Por qué |
|---|---|---|
| Spearman del ranking de modelos | ≥ 0,90 | Preserva el orden |
| Error relativo máximo | ≤ 5% | Evita inflar o deflactar métricas |
| Error de efecto en el par top-2 | ≤ 10% | Preserva la conclusión "A > B" |
| Cobertura de conceptos | ≥ 90% | Evita perder la señal del KG |
| Cobertura de campos | ≥ 90% | Evita sesgo disciplinar |
| Potencia (MDE) | ≤ brecha observada top-2 | Evita concluir sobre ruido |

El último criterio es el que suele faltar. La lógica de diseño de tamaño de conjunto de temas de Sakai (2016), basada en potencia y anchura de intervalos de confianza, se adapta aquí a usuarios: estimar la varianza con un piloto y elegir n para detectar la brecha observada con potencia 0,8. Reportar solo p-valores no basta; hay que reportar tamaños de efecto e intervalos (Sakai, 2014).

**Regla de decisión:** elegir el tamaño más pequeño que cumpla todos los criterios en las tres semillas, con intervalos bootstrap que no se solapen para el par top-2. Si ningún tamaño cumple, el subconjunto no es apto para conclusiones; queda como herramienta de desarrollo.

## 5.5 Pseudocódigo

```python
# 1) Perfil estructural
act = interactions.groupby("user").size()
pop = interactions.groupby("item").size()

# 2) Estratos: decil de actividad x campo dominante
users["stratum"] = decile(act).astype(str) + "|" + users["field"]

# 3) Asignación proporcional (o de Neyman) por estrato
target = int(len(users) * frac)
sample = (users.groupby("stratum", group_keys=False)
               .apply(lambda g: g.sample(n=max(1, round(len(g) * target / len(users))),
                                         random_state=seed)))

# 4) Fidelidad con el mismo split y >=3 semillas
for frac in [0.05, 0.10, 0.25, 0.50, 1.00]:
    for model in [KGSeq, SASRecText]:
        metrics[frac][model] = train_eval(sample_users(frac), model, seeds=3)

tau   = spearmanr(rank_full, rank_subset)
delta = max(abs(m_sub - m_full) / m_full for m in models)
```

El código es ilustrativo: la parte crítica no es el `sample`, sino el bucle de fidelidad.

## 5.6 Guía de decisión

| Objetivo | Recomendación |
|---|---|
| Iterar rápido sin publicar conclusiones del subconjunto | N0, semillas fijas, sin ceremonia |
| Benchmark fijo para ablaciones | N1 al tamaño mínimo que pase los criterios |
| OOM o lentitud del HGT | N3 primero; después N1 si hace falta |
| Inferencia sobre cursos concretos | N2, asumiendo que el catálogo cambia |
| Investigación en eficiencia de datos | N4, comparado contra N1 y contra poda aleatoria |

# 6. Discusión

La conclusión central es que "representativo" es una relación, no una propiedad. Un subconjunto es representativo **para una inferencia y un zoo de modelos**. Un subconjunto fiel para KGSeq puede no serlo para SASRecText, y uno fiel para NDCG@10 puede no serlo para Recall@20. Por eso el protocolo incluye ambos modelos y ambas métricas, y por eso la tarjeta del subconjunto debe declarar para qué se validó.

La segunda conclusión es que la evaluación es la parte frágil, no el entrenamiento. La literatura es consistente: muestrear candidatos, reducir objetivos o cambiar el split altera las comparaciones (Krichene & Rendle, 2020; Cañamares & Castells, 2020; Meng et al., 2020; Parajuli et al., 2026). Entrenar con menos datos es una decisión de coste; evaluar con menos datos es una decisión de validez. No son simétricas.

La tercera es que la poda aleatoria es un baseline incómodamente fuerte (Ayed & Hayou, 2023). Cualquier método sofisticado debe superarla, no solo superar al conjunto completo en coste. Y en recomendación la evidencia de coresets es todavía una excepción (Ju et al., 2024), no un cuerpo consolidado.

La cuarta es específica de EDuRec: el grafo de conceptos domina el coste. Si el objetivo es entrenar más rápido, reducir usuarios es la palanca menos eficiente; bajar `emb_dim`, reducir capas o muestrear subgrafos ataca el cuello de botella real. Si el objetivo es un benchmark manejable, muestrear usuarios es la palanca correcta. Distinguir ambas cosas evita meses de ajuste en la dirección equivocada.

# 7. Limitaciones

- Revisión narrativa de un solo revisor, sin doble cribado ni PRISMA; el sesgo de selección es posible.
- No se ejecutó el benchmark sobre MOOCCubeX; el protocolo es una propuesta de diseño, no una validación empírica.
- Parte de la evidencia sobre coresets y poda procede de visión por computador; su transferencia a recomendación no está establecida.
- La evidencia específica de MOOC es escasa; la mayoría de los resultados provienen de recomendación general.
- Los umbrales de aceptación (τ ≥ 0,90; δ ≤ 5%; γ ≥ 90%) son puntos de partida razonables, no estándares de la literatura.
- El informe asume que el grafo de conceptos es información estática de curso, no derivada de logs de usuarios; si no fuera así, habría que revisar el riesgo de leakage.

# 8. Conclusiones y recomendaciones

1. **Definir la fidelidad antes de muestrear.** Fijar ε, δ y γ, el zoo de modelos y las métricas. Sin eso, cualquier subconjunto es defendible a posteriori.
2. **Muestrear usuarios, no ítems ni interacciones.** Conserva el catálogo, los historiales y la comparabilidad con el ranking full-catalog.
3. **Fijar el split antes de muestrear.** Split temporal primero; muestreo dentro de train; test intacto o muestreado por usuario con todos sus positivos.
4. **Usar el hash actual como N0.** Es una línea base válida y ya existe; añadir estratificación solo si la fidelidad no se cumple.
5. **Atacar el grafo por separado.** Si el problema es memoria o velocidad, `emb_dim`, capas y subgraph sampling antes que recortar usuarios.
6. **No evaluar con candidatos muestreados.** Si se hace por coste, usar estimadores corregidos y reportarlo.
7. **Tratar los coresets como experimento.** Prometedores, caros y con evidencia limitada en recomendación; compararlos contra poda aleatoria.
8. **Emitir una tarjeta de subconjunto.** Tamaño, método, sal, cobertura, τ, error de efecto, semillas y fecha. Sin tarjeta, el subconjunto no es reutilizable.

# Referencias

::: {.references}

Ayed, F., & Hayou, S. (2023). *Data pruning and neural scaling laws: Fundamental limitations of score-based algorithms* (arXiv:2302.06960). arXiv. https://doi.org/10.48550/arXiv.2302.06960

Cañamares, R., & Castells, P. (2020). On target item sampling in offline recommender system evaluation. *Proceedings of the 14th ACM Conference on Recommender Systems*. ACM. https://doi.org/10.1145/3383313.3412259

Chiang, W.-L., Liu, X., Si, S., Li, Y., Bengio, S., & Hsieh, C.-J. (2019). Cluster-GCN: An efficient algorithm for training deep and large graph convolutional networks. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM. https://doi.org/10.1145/3292500.3330925

Dacrema, M. F., Boglio, S., Cremonesi, P., & Jannach, D. (2021). A troubling analysis of reproducibility and progress in recommender systems research. *ACM Transactions on Information Systems, 39*(2). https://doi.org/10.1145/3434185

Feldman, D. (2020). Core-sets: An updated survey. *WIREs Data Mining and Knowledge Discovery, 10*(1), e1335. https://doi.org/10.1002/widm.1335

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems, 30*.

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and powering graph convolution network for recommendation. *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*. ACM. https://doi.org/10.1145/3397271.3401063

Hidasi, B., & Czapp, Á. T. (2023). Widespread flaws in offline evaluation of recommender systems. *Proceedings of the 17th ACM Conference on Recommender Systems*. ACM. https://doi.org/10.1145/3604915.3608839

Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph transformer. *Proceedings of The Web Conference 2020*. ACM. https://doi.org/10.1145/3366423.3380027

Ju, Z., Du, H., Tragos, E., Hurley, N., & Lawlor, A. (2024). Exploring coresets for efficient training and consistent evaluation of recommender systems. *Proceedings of the 18th ACM Conference on Recommender Systems*. ACM. https://doi.org/10.1145/3640457.3691716

Krichene, W., & Rendle, S. (2020). On sampled metrics for item recommendation. *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM. https://doi.org/10.1145/3394486.3403226

Lei, S., & Tao, D. (2023). A comprehensive survey of dataset distillation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. https://doi.org/10.1109/TPAMI.2023.3322540

Li, D., Jin, R., Liu, Z., Ren, B., & Gao, J. (2023). On item-sampling evaluation for recommender system. *ACM Transactions on Recommender Systems*. https://doi.org/10.1145/3629171

Meng, Z., McCreadie, R., Macdonald, C., & Ounis, I. (2020). Exploring data splitting strategies for the evaluation of recommendation models. *Proceedings of the 14th ACM Conference on Recommender Systems*. ACM. https://doi.org/10.1145/3383313.3418479

Parajuli, S., Vaez Barenji, S., & Ekstrand, M. D. (2026). On the convergent validity of offline evaluation designs for recommender systems. *Proceedings of the 20th ACM Conference on Recommender Systems*. ACM. https://doi.org/10.1145/3773078.3831818

Paul, M., Ganguli, S., & Dziugaite, G. K. (2021). Deep learning on a data diet: Finding important examples early in training. *Advances in Neural Information Processing Systems, 34*.

Sachdeva, N., & McAuley, J. (2023). *Data distillation: A survey* (arXiv:2301.04272). arXiv. https://doi.org/10.48550/arXiv.2301.04272

Sakai, T. (2014). Statistical reform in information retrieval? *ACM SIGIR Forum, 48*(1), 3–12. https://doi.org/10.1145/2641383.2641385

Sakai, T. (2016). Topic set size design. *Information Retrieval Journal, 19*(3), 256–283. https://doi.org/10.1007/s10791-015-9273-z

Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. S. (2022). Beyond neural scaling laws: Beating power law scaling via data pruning. *Advances in Neural Information Processing Systems, 35*.

Wang, X., He, X., Cao, Y., Liu, M., & Chua, T.-S. (2019a). KGAT: Knowledge graph attention network for recommendation. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM. https://doi.org/10.1145/3292500.3330989

Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S. (2019b). Neural graph collaborative filtering. *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*. ACM. https://doi.org/10.1145/3298689.3347054

Yu, J., Wang, Y., Zhong, Q., Luo, G., Mao, Y., Sun, K., Feng, W., Xu, W. X., Cao, S., Zeng, K., Yao, Z., Hou, L., Lin, Y., Li, P., Zhou, J., Xu, B., Li, J., Tang, J., & Sun, M. (2021). MOOCCubeX: A large knowledge-centered repository for adaptive learning in MOOCs. *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*. ACM. https://doi.org/10.1145/3459637.3482010

Zeng, H., Zhou, H., Srivastava, A., Kannan, R., & Prasanna, V. (2020). GraphSAINT: Graph sampling based inductive learning method. *International Conference on Learning Representations*.

Zhang, J., Xue, R., Fan, W., Xu, X., Li, Q., Pei, J., & Liu, X. (2024). Linear-time graph neural networks for scalable recommendations. *Proceedings of the ACM Web Conference 2024*. ACM. https://doi.org/10.1145/3589334.3645486

:::

# Anexo A. Plantilla de tarjeta de subconjunto

```text
Dataset / versión:        MOOCCubeX, <fecha de descarga>
Filtros aplicados:        <resource, deprecated, max_enrollments, min_connections>
Tamaño:                   <n_usuarios> / <n_cursos> / <n_interacciones> (<% del total>)
Método de muestreo:       <hash | estratificado | cluster | subgraph>
Sal / semilla:            <...>
Cobertura:                <conceptos %> · <campos %> · <cola larga %>
Distancias:               <KS actividad> · <JSD popularidad>
Fidelidad:                <τ Spearman> · <error relativo máx.> · <error de efecto top-2>
Semillas / IC:            <3–5 semillas, bootstrap 95%>
Modelos validados:        <KGSeq, SASRecText>
Veredicto:                <apto para conclusiones | solo desarrollo>
Fecha / responsable:      <...>
```

# Anexo B. Divulgación de IA y control de calidad

**Divulgación de IA.** Este informe se elaboró con asistencia de IA (OpenCode, modelo DeepSeek V4.1 Flash). La búsqueda y los metadatos bibliográficos se verificaron contra OpenAlex y las fuentes primarias; las citas se contrastaron una a una. Los errores residuales son responsabilidad del autor.

**Revisión editorial.** Veredicto: *minor revision*, resuelta. Problemas detectados y corregidos: (i) se presentaba el coreset como opción de primera línea, corregido con la cautela de Ayed y Hayou (2023); (ii) faltaba análisis de potencia, añadido con Sakai (2014, 2016); (iii) se generalizaba evidencia de visión a recomendación, acotado en Limitaciones.

**Revisión ética.** *Cleared.* Datos públicos, sin sujetos humanos nuevos. Riesgo de representación: un subconjunto puede invisibilizar usuarios de baja actividad o cursos de cola larga; se declara en la tarjeta y se mide con cobertura. Sin riesgo de doble uso.

**Devil's Advocate, checkpoint 3.** *PASS.* Contraargumento más fuerte: "el muestreo hash uniforme ya es suficiente y todo esto es sobreingeniería". Respuesta: es suficiente mientras el subconjunto se use solo para desarrollo; en cuanto se publiquen conclusiones basadas en él, la fidelidad debe medirse. El protocolo no obliga a usarlo siempre: obliga a declarar cuándo no se usó.
