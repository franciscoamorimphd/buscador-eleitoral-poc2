import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Buscador eleitoral | POC',
    page_icon='🔎',
    layout='wide'
)

BASE = Path(__file__).resolve().parent / 'despesas_contratadas_candidatos_2026_AL.parquet'

PREFERIDAS = [
    'NM_CANDIDATO',
    'NR_CPF_CANDIDATO',
    'NM_FORNECEDOR',
    'NM_FORNECEDOR_RFB',
    'NR_CPF_CNPJ_FORNECEDOR'
]

VAZIO = np.array([], dtype=np.int64)


def normalizar(valor):
    if pd.isna(valor):
        return ''

    s = unicodedata.normalize(
        'NFKD', str(valor).strip().upper()
    )
    s = ''.join(
        c for c in s if not unicodedata.combining(c)
    )

    if re.fullmatch(r'[\d.\-/\s]+', s):
        s = re.sub(r'\D', '', s)

    return s


# Metadados: não carrega os registros
@st.cache_data(show_spinner=False)
def metadados():
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(BASE)
    return (
        pf.schema_arrow.names,
        pf.metadata.num_rows
    )


# Base: carregada somente na primeira pesquisa
@st.cache_data(show_spinner=False)
def carregar_base():
    import duckdb

    con = duckdb.connect()
    try:
        cols = [
            r[0] for r in con.execute(
                'DESCRIBE SELECT * FROM read_parquet(?)',
                [str(BASE)]
            ).fetchall()
        ]

        sel = ', '.join(
            'COALESCE(CAST("' +
            c.replace('"', '""') +
            '" AS VARCHAR), \'\') AS "' +
            c.replace('"', '""') + '"'
            for c in cols
        )

        df = con.execute(
            f'SELECT {sel} FROM read_parquet(?)',
            [str(BASE)]
        ).df()

        return df.reset_index(drop=True)
    finally:
        con.close()


# Índices: construídos somente quando necessários
@st.cache_resource(show_spinner=False)
def indice_coluna(campo):
    s = carregar_base()[campo]

    mapa = {
        valor: normalizar(valor)
        for valor in s.unique()
    }

    norm = s.map(mapa)

    return norm.groupby(
        norm, sort=False
    ).indices


def buscar(grupos, termo, modo):
    if modo == 'Exata':
        return grupos.get(termo, VAZIO)

    if modo == 'Contém trecho':
        chaves = [
            k for k in grupos
            if termo in k
        ]
    else:
        palavras = termo.split()
        chaves = [
            k for k in grupos
            if all(p in k for p in palavras)
        ]

    if not chaves:
        return VAZIO

    return np.concatenate([
        grupos[k] for k in chaves
    ])


# ---------------- INTERFACE ----------------

st.title('🔎 Buscador de despesas eleitorais')

st.caption(
    'Prova de conceito • Alagoas • '
    'Eleições 2026 • Busca textual sem IA'
)

if not BASE.is_file():
    st.error(
        'Arquivo da base não encontrado '
        'na pasta do aplicativo.'
    )
    st.stop()

try:
    colunas, n_linhas = metadados()
except Exception as exc:
    st.error(
        f'Não foi possível ler os metadados: {exc}'
    )
    st.stop()

st.info(
    f'Base disponível: {n_linhas:,} registros '
    f'e {len(colunas)} colunas.'.replace(',', '.')
)

modo = st.radio(
    'Tipo de correspondência',
    ['Exata', 'Contém trecho', 'Todas as palavras'],
    horizontal=True
)

campos_padrao = [
    c for c in PREFERIDAS if c in colunas
]

campos = st.multiselect(
    'Campos em que pesquisar',
    options=colunas,
    default=campos_padrao
)

entrada = st.text_area(
    'Cole nomes, CPFs ou CNPJs '
    '(um por linha, separados por ; ou tab)',
    height=190,
    placeholder=(
        'EDILSON DE LIMA SANTOS\n'
        'SILVA\n'
        '123.456.789-01'
    )
)


# ---------------- BUSCA ----------------

if st.button(
    'Buscar na base',
    type='primary',
    use_container_width=True
):
    st.session_state.pop('res', None)

    if not campos:
        st.warning(
            'Selecione pelo menos uma coluna.'
        )
        st.stop()

    termos_originais = [
        x.strip()
        for x in re.split(r'[;\t\r\n]+', entrada)
        if x.strip()
    ]

    termos = {}

    for original in termos_originais:
        chave = normalizar(original)
        if chave:
            termos.setdefault(chave, original)

    if not termos:
        st.warning(
            'Cole pelo menos um nome, termo ou CPF/CNPJ.'
        )
        st.stop()

    if len(termos) > 2000:
        st.warning(
            'Limite de 2.000 termos por consulta nesta POC.'
        )
        st.stop()

    # Só agora os dados são carregados
    try:
        with st.spinner(
            'Preparando a base (somente na primeira pesquisa)...'
        ):
            base = carregar_base()
    except Exception as exc:
        st.error(
            f'Não foi possível carregar a base: {exc}'
        )
        st.stop()

    linhas_idx = []
    termos_col = []
    campos_col = []
    encontrados = set()

    with st.spinner('Pesquisando na base...'):
        for campo in campos:
            grupos = indice_coluna(campo)

            for chave, original in termos.items():
                idx = buscar(grupos, chave, modo)

                if len(idx):
                    encontrados.add(chave)
                    linhas_idx.append(idx)

                    termos_col.append(
                        np.full(
                            len(idx), original, dtype=object
                        )
                    )

                    campos_col.append(
                        np.full(
                            len(idx), campo, dtype=object
                        )
                    )

        if linhas_idx:
            idx_all = np.concatenate(linhas_idx)

            resultado = pd.concat([
                pd.DataFrame({
                    'termo_pesquisado': np.concatenate(
                        termos_col
                    ),
                    'campo_encontrado': np.concatenate(
                        campos_col
                    )
                }),
                base.iloc[idx_all].reset_index(
                    drop=True
                )
            ], axis=1)
        else:
            resultado = pd.DataFrame()

        ausentes = [
            original
            for chave, original in termos.items()
            if chave not in encontrados
        ]

    st.session_state['res'] = {
        'n_termos': len(termos),
        'resultado': resultado,
        'ausentes': ausentes,
        'csv': (
            resultado.to_csv(
                index=False, sep=';'
            ).encode('utf-8-sig')
            if not resultado.empty else None
        ),
        'csv_aus': (
            pd.DataFrame({
                'termo_nao_encontrado': ausentes
            }).to_csv(index=False).encode('utf-8-sig')
            if ausentes else None
        )
    }


# ---------------- RESULTADOS ----------------

r = st.session_state.get('res')

if r:
    resultado = r['resultado']
    ausentes = r['ausentes']

    st.write(
        f'**Termos pesquisados:** {r["n_termos"]}  |  '
        f'**Correspondências:** {len(resultado)}  |  '
        f'**Termos sem resultado:** {len(ausentes)}'
    )

    if not resultado.empty:
        st.subheader('Resultados')

        st.caption(
            'Um registro pode aparecer mais de uma vez '
            'quando corresponde a vários termos ou campos.'
        )

        # Exibe apenas as primeiras 100 correspondências
        st.dataframe(
            resultado.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            'Exibindo até 100 correspondências. '
            'O arquivo CSV contém todos os resultados.'
        )

        st.download_button(
            'Baixar resultados CSV',
            data=r['csv'],
            file_name='resultados_busca.csv',
            mime='text/csv'
        )
    else:
        st.info(
            'Nenhuma correspondência encontrada '
            'para os campos e o modo selecionados.'
        )

    if ausentes:
        with st.expander('Ver termos não encontrados'):
            st.write(
                '\n'.join(f'- {t}' for t in ausentes)
            )

            st.download_button(
                'Baixar termos não encontrados',
                data=r['csv_aus'],
                file_name='nao_encontrados.csv',
                mime='text/csv'
            )


# ---------------- INFORMAÇÕES ----------------

with st.expander('Sobre a busca e os dados'):
    st.write(
        'Exata: igualdade integral do campo. '
        'Contém trecho: sequência de caracteres '
        'em qualquer posição. '
        'Todas as palavras: cada palavra deve '
        'aparecer no campo, em qualquer ordem.'
    )

    st.write(
        'As buscas ignoram diferenças entre '
        'maiúsculas/minúsculas e acentos. '
        'Documentos numéricos são comparados '
        'sem pontuação.'
    )

    st.write(
        'Valores especiais como -1 e -4 podem '
        'aparecer na base e não devem ser '
        'interpretados como CPF válido.'
    )

    st.write(
        'A base permanece no servidor; '
        'os usuários não precisam enviar arquivos.'
    )
