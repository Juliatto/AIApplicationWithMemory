# Chatbot com LangChain e API do Groq

Este projeto implementa um chatbot baseado no LangChain, utilizando a API do Groq para processar perguntas e fornecer respostas com base em um dicionário de dados de cidades brasileiras.

## Índice

1. [Descrição do Projeto](#descrição-do-projeto)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação](#instalação)
4. [Explicação do Código](#explicação-do-código)
    - [Importação das Bibliotecas](#importação-das-bibliotecas)
    - [Carregamento de Variáveis de Ambiente](#carregamento-de-variáveis-de-ambiente)
    - [Inicialização do Modelo](#inicialização-do-modelo)
    - [Criação do Prompt Template](#criação-do-prompt-template)
    - [Gerenciamento da Memória do Chatbot](#gerenciamento-da-memória-do-chatbot)
    - [Base de Dados de Cidades](#base-de-dados-de-cidades)
    - [Pipeline de Execução](#pipeline-de-execução)
    - [Exemplos de Interação](#exemplos-de-interação)
5. [Glossário](#glossário)
6. [Como Contribuir](#como-contribuir)

## Descrição do Projeto

O chatbot responde a perguntas sobre a população, pontos turísticos e principais universidades de diversas cidades brasileiras. Utiliza a biblioteca LangChain para estruturar prompts, gerenciar o histórico de mensagens e se conectar à API do Groq.

## Pré-requisitos

- Python 3.10 ou superior
- Git instalado na máquina

## Instalação

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

No Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

No Linux/MacOS:

```bash
python -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Crie um arquivo `.env` com a seguinte estrutura e adicione sua chave da API do Groq:

```env
GROQ_API_KEY=suas-chave-aqui
```

5. Execute o código:

```bash
python main.py
```

## Explicação do Código

### Importação das Bibliotecas

As bibliotecas importadas têm as seguintes funções:

- `os`: Permite acessar variáveis de ambiente.
- `dotenv`: Carrega variáveis de ambiente a partir de um arquivo `.env`.
- `langchain_groq`: Integra o LangChain com a API do Groq.
- `ChatMessageHistory` e `BaseChatMessageHistory`: Gerenciam o histórico de mensagens.
- `RunnableWithMessageHistory`: Executa fluxos com histórico dinâmico.
- `ChatPromptTemplate` e `MessagesPlaceholder`: Criam e manipulam prompts para o modelo.
- `HumanMessage`, `AIMessage`, `SystemMessage`, `trim_messages`: Manipulam mensagens entre usuário, IA e sistema.
- `RunnablePassthrough`: Cria etapas reutilizáveis no pipeline.
- `itemgetter`: Facilita a extração de valores de dicionários.

### Carregamento de Variáveis de Ambiente

```python
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```

Carrega a chave da API do Groq do arquivo `.env` para proteger credenciais sensíveis.

### Inicialização do Modelo

```python
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
```

Inicializa o modelo de IA da API Groq com o nome "Gemma2-9b-It".

### Criação do Prompt Template

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda todas as perguntas com precisão."),
    MessagesPlaceholder(variable_name="messages")
])
```

Define um template de prompt, configurando a personalidade do assistente e permitindo adicionar mensagens dinâmicas.

### Gerenciamento da Memória do Chatbot

```python
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
```

Controla o tamanho do histórico de mensagens para não exceder o limite de tokens do modelo.

### Base de Dados de Cidades

```python
city_data = {
    "São Paulo": {"população": "12,33 milhões", "pontos_turisticos": ["Parque Ibirapuera", "Avenida Paulista"], "universidade": "USP"},
    "Rio de Janeiro": {"população": "6,7 milhões", "pontos_turisticos": ["Cristo Redentor", "Pão de Açúcar"], "universidade": "UFRJ"}
}
```

Contém informações sobre diversas cidades brasileiras, como população, pontos turísticos e principais universidades.

### Pipeline de Execução

```python
chain = (
    RunnablePassthrough.assign(city_data=itemgetter("city_data") | trimmer)
    | prompt
    | model
)
```

Cria um fluxo de execução que prepara os dados, aplica o template e interage com o modelo.

### Exemplos de Interação

```python
response = chain.invoke({
    "city_data": city_data,
    "messages": [HumanMessage(content="Qual é a população de São Paulo?")],
    "language": "Português"
})
print("Resposta final:", response.content)
```

Executa consultas sobre as cidades e exibe as respostas do modelo.

## Glossário

- **LangChain**: Framework para criar agentes conversacionais utilizando modelos de linguagem.
- **Prompt**: Instrução enviada ao modelo para guiar suas respostas.
- **API**: Interface que permite interação entre diferentes sistemas.
- **Tokens**: Unidades de texto que um modelo de linguagem processa.

## Como Contribuir

1. Faça um fork do projeto.
2. Crie uma branch para sua modificação: `git checkout -b minha-modificacao`
3. Envie suas alterações: `git commit -m "Descrição clara da mudança"`
4. Faça o push da branch: `git push origin minha-modificacao`
5. Abra um pull request no repositório original.

