Requisitos previos

Python 3.11 o superior

Git

Acceso a una API key de OpenRouter (gratis con registro en https://openrouter.ai
)

Editor recomendado: VS Code con extensión de Python

Instalación (Windows paso a paso)

Clonar el repositorio

git clone https://github.com/TU-USUARIO/ufro-assistant.git
cd ufro-assistant


Crear el entorno virtual

python -m venv .venv
.venv\Scripts\activate


Instalar dependencias

pip install -r requirements.txt


Configurar variables de entorno

Copiar el archivo .env.example a .env

Editar .env y pegar la API key de OpenRouter:

OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=
OPENROUTER_REFERER=
OPENROUTER_TITLE=

Preparar los datos

Colocar los documentos oficiales en la carpeta data/raw/.
Ejemplo de archivos:

calendario_academico_2025.txt
reglamento_convivencia_2025.pdf
reglamento_regimen_estudios_2023.pdf


Registrar la trazabilidad en data/sources.csv con este formato:

doc_id,title,url,fecha_descarga,vigencia,tipo,observaciones
cal_2025,Calendario Académico 2025,https://...,2025-09-24,2025,TXT,Campus Virtual UFRO
convivencia_2025,Reglamento de Convivencia Universitaria,https://...,2025-09-24,2025,PDF,Versión vigente 2025
reg_estudios_2023,Reglamento de Régimen de Estudios (2023),https://...,2025-09-24,2023,PDF,Revisar si hay actualización

Construir el índice (RAG)

Ingesta y chunking

python rag/ingest.py


Esto genera data/processed/chunks.parquet.
Si algún PDF requiere OCR, aparecerá un aviso.

Embeddings y FAISS

python rag/embed.py


Esto genera:

data/index.faiss

data/processed/chunks_meta.parquet

Uso del asistente

Consulta simple (modo RAG, por defecto)

python app.py "¿Cuándo inicia el semestre 2025?"


Ejemplo de salida:

Respuesta (RAG):
El semestre 2025 comienza el 10 de marzo, según el calendario académico.

Referencias:
[Calendario Académico 2025]


Consulta directa (sin RAG)