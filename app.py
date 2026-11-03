import re
import time
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Buscador eleitoral | POC 2', page_icon='🔎', layout='wide')
BASE = Path(__file__).parent / 'despesas_contratadas_candidatos_2026_AL.parquet'
PREFERIDAS = ['NM_CANDIDATO', 'NR_CPF_CANDIDATO', 'NM_FORNECEDOR', 'NM_FORNECEDOR_RFB', 'NR_CPF_CNPJ_FORNECEDOR']


def normalizar(valor):
    if pd.isna(valor):
        return ''
    s = unicodedata.normalize('NFKD', str(valor).strip().upper())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    if re.fullmatch(r'[\d.\-/\s]+', s):
        s = re.sub(r'\D', '', s)
    return s


@st.cache_data(show_spinner=False)
def carregar_base():
    inicio = time.perf_counter()
    with duckdb.connect(database=':memory:') as con:
        df = con.execute('SELECT * FROM read_parquet(?)', [str(BASE)]).df()
    return df.fillna('').reset_index(drop=True), time.perf_counter() - inicio


@st.cache_data(show_spinner=False)
def preparar_colunas(_df, colunas, assinatura):
    inicio = time.perf_counter()
    return {campo: _df[campo].map(normalizar) for campo in colunas}, time.perf_counter() - inicio


def corresponde(serie, termo, modo):
    if modo == 'Exata':
        return serie.eq(termo)
    if modo == 'Contém trecho':
        return serie.str.contains(termo, regex=False, na=False)
    palavras = termo.split()
    return pd.concat([serie.str.contains(p, regex=False, na=False) for p in palavras], axis=1).all(axis=1)


st.title('🔎 Buscador de despesas eleitorais | POC 2')
st.caption('Alagoas • Eleições 2026 • DuckDB + Parquet • Busca determinística, sem IA')
if not BASE.is_file():
    st.error('Arquivo Parquet não encontrado. Adicione despesas_contratadas_candidatos_2026_AL.parquet ao repositório.')
    st.stop()

try:
    base, tempo_leitura = carregar_base()
except Exception as exc:
    st.error(f'Não foi possível ler a base: {exc}')
    st.stop()

st.info(f'Base disponível: {len(base):,} registros e {len(base.columns)} colunas.'.replace(',', '.'))
modo = st.radio('Tipo de correspondência', ['Exata', 'Contém trecho', 'Todas as palavras'], horizontal=True)
campos = st.multiselect('Campos em que pesquisar', base.columns.tolist(), default=[c for c in PREFERIDAS if c in base.columns])
entrada = st.text_area('Cole nomes, CPFs ou CNPJs (um por linha, separados por ; ou tab)', height=160, placeholder='EDILSON DE LIMA SANTOS\nSILVA\n123.456.789-01')

with st.expander('Diagnóstico de desempenho'):
    st.write(f'Leitura da base (primeira execução do cache): **{tempo_leitura:.3f} s**. Esta medida não inclui o despertar do servidor nem o carregamento do navegador.')
    st.write(f'Tamanho do Parquet: **{BASE.stat().st_size / 1024 / 1024:.2f} MB**.')

if st.button('Buscar na base', type='primary', use_container_width=True):
    if not campos:
        st.warning('Selecione pelo menos uma coluna.')
        st.stop()
    termos = {}
    for original in (x.strip() for x in re.split(r'[;\t\r\n]+', entrada)):
        chave = normalizar(original)
        if chave:
            termos.setdefault(chave, original)
    if not termos:
        st.warning('Informe pelo menos um termo.')
        st.stop()
    if len(termos) > 2000:
        st.warning('Limite de 2.000 termos por consulta.')
        st.stop()

    inicio = time.perf_counter()
    norm_cols, tempo_preparacao = preparar_colunas(base, tuple(campos), str(BASE.stat().st_mtime_ns))
    partes = []
    encontrados = set()
    with st.spinner('Pesquisando...'):
        for campo, serie in norm_cols.items():
            for chave, original in termos.items():
                indices = serie.index[corresponde(serie, chave, modo)]
                if len(indices):
                    encontrados.add(chave)
                    partes.append(pd.DataFrame({'_linha': indices, 'termo_pesquisado': original, 'campo_encontrado': campo}))
        if partes:
            achados = pd.concat(partes, ignore_index=True)
            resultado = pd.concat([achados[['termo_pesquisado', 'campo_encontrado']].reset_index(drop=True), base.iloc[achados['_linha'].to_numpy()].reset_index(drop=True)], axis=1)
        else:
            resultado = pd.DataFrame()
    tempo_busca = time.perf_counter() - inicio
    ausentes = [v for k, v in termos.items() if k not in encontrados]
    st.write(f'**Termos pesquisados:** {len(termos)} | **Correspondências:** {len(resultado)} | **Termos sem resultado:** {len(ausentes)}')
    st.caption(f'Tempo da consulta: {tempo_busca:.3f} s. Preparação inicial de colunas: {tempo_preparacao:.3f} s (cache nas próximas consultas).')
    if not resultado.empty:
        st.subheader('Resultados')
        st.caption('Prévia das primeiras 100 correspondências. O CSV para download contém todos os resultados. Um registro pode aparecer mais de uma vez quando corresponde a diferentes termos ou campos.')
        st.dataframe(resultado.head(100), use_container_width=True, hide_index=True)
        st.download_button('Baixar todos os resultados CSV', resultado.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name='resultados_busca.csv', mime='text/csv')
    else:
        st.info('Nenhuma correspondência encontrada.')
    if ausentes:
        with st.expander('Termos não encontrados'):
            st.write('\n'.join(f'- {x}' for x in ausentes))
            st.download_button('Baixar termos não encontrados', pd.DataFrame({'termo_nao_encontrado': ausentes}).to_csv(index=False).encode('utf-8-sig'), file_name='nao_encontrados.csv', mime='text/csv')

with st.expander('Sobre a busca'):
    st.write('Exata: igualdade integral. Contém trecho: sequência de caracteres. Todas as palavras: cada palavra deve estar presente, em qualquer ordem. Ignora acentos e diferenças entre maiúsculas/minúsculas. Para documentos numéricos, remove a pontuação. Não usa IA nem correspondência aproximada.')
    st.write('Valores como -1 e -4 podem indicar ausência de informação, não documentos válidos. Os dados ficam no servidor.')
