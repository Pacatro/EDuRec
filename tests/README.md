# Tests básicos

Ejecutar desde la raíz del proyecto, con las dependencias de desarrollo instaladas:

```sh
uv run python -m pytest -q
```

La suite utiliza datos sintéticos pequeños en CPU, sin descargar datasets ni pesos.
Cubre particiones temporales, historiales y contexto sin fugas de información;
configuración y ablaciones; pooling, codificadores, scoring y pérdida contrastiva;
forward/backward de KGSeq y SASRecText; métricas de ranking y entrada de la CLI.

Se comprueban los contratos esenciales, sin ejecutar entrenamientos completos,
optimizaciones de hiperparámetros ni benchmarks de servicios externos.
