# Bag of Words e Classificador Naive Bayes para Detecção de Spam

Projeto didático de **Aprendizado de Máquina (Machine Learning)** desenvolvido em Python, com foco no processamento de textos, representação numérica de mensagens e classificação automática utilizando algoritmos Naive Bayes.

O projeto demonstra como transformar mensagens de texto em dados numéricos por meio da técnica **Bag of Words**, treinar classificadores para identificar mensagens de spam e avaliar o desempenho dos modelos.

## Objetivos

- Compreender o funcionamento da técnica Bag of Words.
- Transformar textos em vetores numéricos usando `CountVectorizer`.
- Treinar um classificador Naive Bayes para identificar mensagens de spam.
- Avaliar modelos utilizando métricas de classificação.
- Visualizar resultados por meio de matrizes de confusão.
- Comparar os classificadores Multinomial Naive Bayes e Bernoulli Naive Bayes.
- Conhecer a utilização de pipelines com Scikit-learn.
- Explorar a classificação multiclasse utilizando o conjunto de dados de dígitos manuscritos.

## Tecnologias utilizadas

- **Python** — linguagem de programação.
- **UV** — gerenciamento de dependências e ambiente do projeto.
- **Scikit-learn** — vetorização de textos, treinamento e avaliação dos modelos.
- **Pandas** — organização e visualização de dados em tabelas.
- **NumPy** — suporte a operações numéricas.
- **Matplotlib** — visualização gráfica dos resultados.

## Estrutura do projeto

```text
am-naive-bayes-spam/
├── 01_bag_of_words.py
├── 02_naive_bayes_spam.py
├── 03_avaliacao_modelo.py
├── 04_digitos_e_pipeline.py
├── pyproject.toml
├── uv.lock
└── README.md
```

*Os arquivos `pyproject.toml` e `uv.lock` são gerados ou mantidos pelo UV durante a configuração do projeto.*

## Conteúdo dos scripts

### 1. Bag of Words

**Arquivo:** `01_bag_of_words.py`

Apresenta a transformação de mensagens em uma representação numérica baseada na frequência das palavras. Utiliza `CountVectorizer`, remove palavras comuns selecionadas em português e apresenta o vocabulário e a matriz de contagem.

### 2. Classificação de spam com Naive Bayes

**Arquivo:** `02_naive_bayes_spam.py`

Utiliza um conjunto de mensagens rotuladas para treinar um classificador `MultinomialNB`. Separa os dados em conjuntos de treinamento e teste, classifica novas mensagens e apresenta as probabilidades estimadas pelo modelo.

### 3. Avaliação e comparação de modelos

**Arquivo:** `03_avaliacao_modelo.py`

Avalia o desempenho dos classificadores com métricas como precisão, recall e F1-score. Também apresenta uma matriz de confusão e compara os resultados de `MultinomialNB` e `BernoulliNB`.

### 4. Classificação de dígitos e Pipeline

**Arquivo:** `04_digitos_e_pipeline.py`

Explora o conjunto de dados de dígitos manuscritos disponível no Scikit-learn e demonstra uma aplicação de classificação multiclasse. Em seguida, apresenta um pipeline que combina a vetorização dos textos e a classificação de mensagens.

## Pré-requisitos

- Python compatível com as versões das dependências instaladas.
- UV instalado no sistema.
- Git, caso deseje clonar o repositório.

## Instalação

### 1. Clone o repositório

Substitua `URL_DO_SEU_REPOSITORIO` pelo endereço real do repositório no GitHub.

```bash
git clone URL_DO_SEU_REPOSITORIO
cd am-naive-bayes-spam
```

### 2. Instale as dependências

```bash
uv sync
```

Caso ainda não tenha configurado as dependências no projeto, utilize:

```bash
uv add scikit-learn numpy matplotlib pandas
```

## Como executar

Execute os scripts individualmente a partir da pasta raiz do projeto.

**Bag of Words:**

```bash
uv run python 01_bag_of_words.py
```

**Classificação de spam:**

```bash
uv run python 02_naive_bayes_spam.py
```

**Avaliação dos modelos:**

```bash
uv run python 03_avaliacao_modelo.py
```

**Classificação de dígitos e Pipeline:**

```bash
uv run python 04_digitos_e_pipeline.py
```

Os resultados são apresentados no terminal. Os scripts que utilizam gráficos podem abrir janelas de visualização.

## Conceitos abordados

- **Bag of Words:** técnica que representa um texto pela contagem das palavras presentes no vocabulário, sem considerar diretamente a ordem das palavras.
- **Naive Bayes:** família de algoritmos probabilísticos baseada no teorema de Bayes, com uma hipótese simplificadora de independência condicional entre as características.
- **MultinomialNB:** classificador adequado a dados de contagem, como frequências de palavras.
- **BernoulliNB:** classificador que trabalha com características binárias, indicando a presença ou ausência de termos.
- **Métricas de avaliação:** precisão, recall, F1-score e matriz de confusão ajudam a analisar os acertos e erros do classificador.
- **Pipeline:** estrutura que organiza etapas de processamento e classificação em um único fluxo de treinamento e previsão.

## Limitações

Este projeto tem finalidade educacional. O conjunto de mensagens é pequeno e não representa necessariamente a diversidade de mensagens encontradas em situações reais. Portanto, os resultados não devem ser interpretados como uma avaliação de desempenho em produção.

As probabilidades retornadas pelo classificador são estimativas do modelo e não garantem que uma mensagem seja efetivamente spam.

## Possíveis melhorias

- Ampliar e diversificar a base de mensagens.
- Comparar Bag of Words com TF-IDF.
- Avaliar o efeito do balanceamento entre as classes.
- Ajustar os hiperparâmetros dos classificadores.
- Adicionar testes automatizados.
- Criar uma interface simples para testar novas mensagens.
- Avaliar o modelo em dados que não tenham sido utilizados durante o treinamento.

## Finalidade acadêmica

Projeto desenvolvido para estudo prático de conceitos de Aprendizado de Máquina, processamento de linguagem natural e classificação de dados com Python.

## Licença

Se desejar permitir o uso e a modificação do código por outras pessoas, adicione uma licença ao repositório, como a licença MIT, após definir os termos de distribuição.
