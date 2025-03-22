# Projeto: Chatbot com LangChain e Groq

Este projeto implementa um chatbot utilizando a biblioteca LangChain integrada com a API Groq. O sistema gerencia o histórico de mensagens por sessão, permitindo interações contínuas com o usuário. 

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Configuração do Ambiente](#configuração-do-ambiente)
4. [Explicação do Código](#explicação-do-código)
   - [Importação de Bibliotecas](#importação-de-bibliotecas)
   - [Configuração do Modelo](#configuração-do-modelo)
   - [Exemplo 1: Histórico de Conversas](#exemplo-1-histórico-de-conversas)
   - [Exemplo 2: Uso de Templates e Otimização de Histórico](#exemplo-2-uso-de-templates-e-otimização-de-histórico)
5. [Glossário](#glossário)

## Pré-requisitos

- Python 3.10 ou superior
- Conta na plataforma Groq para acesso à API

## Instalação

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração do Ambiente

1. Crie um arquivo `.env` na raiz do projeto e adicione sua chave da API do Groq:

```
GROQ_API_KEY=your_api_key_here
```

2. Certifique-se de que a chave da API está configurada corretamente.

## Explicação do Código

### Importação de Bibliotecas

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
```

- `os`: Permite acesso a variáveis de ambiente do sistema operacional.
- `dotenv`: Carrega variáveis de ambiente a partir do arquivo `.env`.
- `ChatGroq`: Integra o LangChain com a API do Groq.
- `ChatMessageHistory`: Gerencia o histórico de mensagens.
- `RunnableWithMessageHistory`: Permite gerenciar o histórico de mensagens dinamicamente.
- `ChatPromptTemplate`: Cria templates personalizados para prompts.
- `MessagesPlaceholder`: Permite inserir mensagens dinâmicas no prompt.
- `HumanMessage`, `AIMessage`, `SystemMessage`: Representam diferentes tipos de mensagens no chat.
- `trim_messages`: Limita o número de mensagens em memória.
- `RunnablePassthrough`: Cria pipelines de execução.
- `itemgetter`: Facilita a extração de valores de dicionários.

### Configuração do Modelo

```python
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
```

- `load_dotenv()`: Carrega as variáveis de ambiente do arquivo `.env`.
- `os.getenv`: Obtém a chave da API do Groq.
- `ChatGroq`: Inicializa o modelo de IA.

### Exemplo 1: Histórico de Conversas

```python
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "chat1"}}

response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Eduardo e sou um engenheiro de dados.")],
    config=config
)

print("Resposta do modelo:", response.content)
```

- `store`: Dicionário que armazena o histórico de conversas.
- `get_session_history`: Cria ou recupera o histórico por sessão.
- `RunnableWithMessageHistory`: Conecta o modelo ao histórico de mensagens.
- `invoke`: Executa a interação e retorna a resposta do modelo.

### Exemplo 2: Uso de Templates e Otimização de Histórico

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda todas as perguntas com precisão."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | model

chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Eduardo")]}))

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="Você é um bom assistente"),
    HumanMessage(content="Oi! Meu nome é Bob"),
    AIMessage(content="Oi, Bob! Como posso te ajudar?"),
    HumanMessage(content="Eu gosto de sorvete de baunilha"),
]

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke({
    "messages": messages + [HumanMessage(content="Qual sorvete eu gosto?")],
    "language": "Português"
})

print("Resposta final:", response.content)
```

- `ChatPromptTemplate`: Cria um template com mensagens dinâmicas.
- `trim_messages`: Limita o número de mensagens no histórico.
- `RunnablePassthrough`: Cria pipelines de execução.
- `chain.invoke`: Executa a interação usando o pipeline.

## Glossário

- **API**: Interface para interações entre software.
- **LangChain**: Framework para criação de pipelines com modelos de linguagem.
- **Groq**: Plataforma de modelos de IA.
- **Prompt Template**: Modelo para estruturar a entrada do chatbot.
- **Runnable**: Unidade de execução em LangChain.
- **Session ID**: Identificador único para cada conversa.

---

Agora você está pronto para executar o chatbot e personalizar suas interações.

