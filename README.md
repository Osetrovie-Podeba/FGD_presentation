git clone https://github.com/Osetrovie-Podeba/FGD_presentation

cd FGD_presentation

uv sync

.venv\Scripts\activate.bat

uvicorn app:app --reload --host 0.0.0.0 --port 8000
