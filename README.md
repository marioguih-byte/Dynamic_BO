# Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente **todos os datasets** do catálogo (`dynamical_catalog.list()`) e **todas as variáveis**
de cada um (lidas do próprio dataset). Séries em ponto, mapas interativos, inspetor de catálogo, exportação CSV/HTML.

## Horário
Tudo em horário de Brasília (America/Sao_Paulo), sempre com a rodada mais recente de cada modelo. O mapa abre no passo mais próximo de agora e as séries marcam "agora".

## Mapa
Painel quadrado interativo (MapLibre + OpenFreeMap, sem chave de API) como no "Explore" do dynamical.org, com a visão Plotly como alternativa.
Global por padrão (sem local fixo); regiões e resolução opcionais. Plotly interativo com tema escuro/claro, campo suave/pixels/contornos, setas de vento, nomes e **contorno de todos os
países** (Natural Earth, domínio público, embutido em `data/`; sem cartopy). Salve PNG pelo ícone de câmera do gráfico
ou baixe o HTML interativo.

## Unidades
Conversão automática para as unidades usadas no Brasil: °C, mm (e mm/h), hPa, %, W/m², vento em m/s, km/h ou nós.
Temperatura em K → °C; chuva em m/kg·m⁻²/taxas → mm; fração 0–1 → %; J/m² por hora → W/m².

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
