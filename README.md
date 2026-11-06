# Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente **todos os datasets** do catálogo (`dynamical_catalog.list()`) e **todas as variáveis**
de cada um (lidas do próprio dataset). Séries em ponto, mapas interativos, inspetor de catálogo, exportação CSV/HTML.

## Mapa
Global por padrão (sem local fixo); regiões e resolução opcionais. Plotly interativo com tema escuro/claro, campo suave/pixels/contornos, setas de vento, nomes e **contorno de todos os
países** (Natural Earth, domínio público, embutido em `data/`; sem cartopy). Salve PNG pelo ícone de câmera do gráfico
ou baixe o HTML interativo.

## Rodar
```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.12
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Suba tudo na raiz do repositório (`app.py`, `dynamical_core.py`, `mapview.py`, `data/`, `requirements.txt`),
escolha Python 3.12+ em *Advanced settings* e `app.py` como arquivo principal.
Memória é o limite no plano gratuito: prefira regiões menores e conjuntos com poucos passos nos mapas.
