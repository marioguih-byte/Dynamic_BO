# Previsão multi-modelo · dynamical.org (Streamlit)

Descobre automaticamente **todos os datasets** do catálogo (`dynamical_catalog.list()`) e **todas as variáveis**
de cada um (lidas do próprio dataset). Séries em ponto, mapas interativos, inspetor de catálogo, exportação CSV/HTML.

## Horário
Tudo em horário de Brasília (America/Sao_Paulo), sempre com a rodada mais recente de cada modelo. O mapa abre no passo mais próximo de agora e as séries marcam "agora".

## Mapa
Painel quadrado interativo, como o "Explore" do dynamical.org. O motor padrão é um **Canvas 2D próprio**: não precisa de
internet, WebGL, Web Workers nem de servidor de mapa, então funciona em qualquer navegador. Arrastar, zoom (roda, pinça,
botões), terra e contorno de **todos os países** (Natural Earth, domínio público, embutido em `data/`), nomes dos países,
grade de coordenadas, campo da previsão, setas de vento, legenda e valor sob o mouse/clique.

Em *Mais opções → Mapa-base* há também mapas **online** (MapLibre + OpenFreeMap, sem chave de API) com ruas e rótulos.
Se o mapa online não carregar (CDN, WebGL, Worker, rede ou 9 s sem resposta), o painel volta sozinho ao modo offline e
mostra o motivo. A visão Plotly continua como alternativa. Global por padrão (sem local fixo).

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
