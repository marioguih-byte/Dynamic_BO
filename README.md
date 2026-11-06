# Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente **todos os datasets** do catálogo (`dynamical_catalog.list()`) e **todas as variáveis**
de cada um (lidas do próprio dataset). Séries em ponto, mapas, inspetor de catálogo, exportação CSV/PNG.

## Rodar
```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.12
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Streamlit Community Cloud: suba `app.py`, `dynamical_core.py` e `requirements.txt` (Python 3.12+).
Descomente `cartopy` no requirements.txt se quiser costas/fronteiras (precisa de internet para o Natural Earth).
Memória é o limite no plano gratuito: prefira regiões menores e conjuntos com poucos passos nos mapas.
