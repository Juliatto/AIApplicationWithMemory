# Importação das bibliotecas necessárias
import os
from dotenv import load_dotenv  # Biblioteca para carregar variáveis de ambiente
from langchain_groq import ChatGroq  # Integração do LangChain com Groq
from langchain_community.chat_message_histories import ChatMessageHistory  # Histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory  # Classe base para histórico
from langchain_core.runnables.history import RunnableWithMessageHistory  # Permite gerenciar histórico dinamicamente
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Criação de templates para prompts
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages  # Manipulação de mensagens
from langchain_core.runnables import RunnablePassthrough  # Para criar fluxos de execução reutilizáveis
from operator import itemgetter  # Facilita a extração de valores de dicionários

# Carrega as variáveis de ambiente do arquivo .env (para proteger credenciais)
load_dotenv()

# Obtém a chave da API do Groq armazenada nas variáveis de ambiente
groq_api_key = os.getenv("GROQ_API_KEY")

# Inicializa o modelo de IA utilizando a API do Groq
model = ChatGroq(model="Gemma2-9b-It", groq_api_key = groq_api_key)

# Criação de um prompt template para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente útil. Responda todas as perguntas com precisão."),
        MessagesPlaceholder(variable_name="messages")  # Permite adicionar mensagens dinamicamente
    ]
)

# Conecta o modelo ao template de prompt
chain = prompt | model


# Gerenciamento da memória do chatbot
trimmer = trim_messages(
    max_tokens=45,  # Define um limite máximo de tokens para evitar ultrapassar o contexto
    strategy="last",  # Mantém as últimas mensagens mais recentes
    token_counter=model,  # Usa o modelo para contar os tokens
    include_system=True,  # Inclui a mensagem do sistema no histórico
    allow_partial=False,  # Evita que mensagens fiquem cortadas
    start_on="human"  # Começa a contagem com mensagens humanas
)

# Exemplo de histórico de mensagens
city_data = {
    "São Paulo": {
        "população": "12,33 milhões",
        "pontos_turisticos": ["Parque Ibirapuera", "Avenida Paulista", "Mercado Municipal", "Catedral da Sé"],
        "universidade": "Universidade de São Paulo (USP)"
    },
    "Rio de Janeiro": {
        "população": "6,7 milhões",
        "pontos_turisticos": ["Cristo Redentor", "Pão de Açúcar", "Praia de Copacabana"],
        "universidade": "Universidade Federal do Rio de Janeiro (UFRJ)"
    },
    "Salvador": {
        "população": "2,9 milhões",
        "pontos_turisticos": ["Pelourinho", "Elevador Lacerda", "Farol da Barra"],
        "universidade": "Universidade Federal da Bahia (UFBA)"
    },
    "Belo Horizonte": {
        "população": "2,5 milhões",
        "pontos_turisticos": ["Praça da Liberdade", "Igreja São José", "Museu de Artes e Ofícios"],
        "universidade": "Universidade Federal de Minas Gerais (UFMG)"
    },
    "Fortaleza": {
        "população": "2,7 milhões",
        "pontos_turisticos": ["Praia do Futuro", "Catedral Metropolitana", "Mercado Central"],
        "universidade": "Universidade Federal do Ceará (UFC)"
    },
    "Brasília": {
        "população": "3,1 milhões",
        "pontos_turisticos": ["Congresso Nacional", "Catedral de Brasília", "Palácio do Planalto"],
        "universidade": "Universidade de Brasília (UnB)"
    },
    "Curitiba": {
        "população": "1,9 milhões",
        "pontos_turisticos": ["Jardim Botânico", "Ópera de Arame", "Rua XV de Novembro"],
        "universidade": "Universidade Federal do Paraná (UFPR)"
    },
    "Porto Alegre": {
        "população": "1,5 milhões",
        "pontos_turisticos": ["Parque Redenção", "Caminho dos Antiquários", "Fundação Ibere Camargo"],
        "universidade": "Universidade Federal do Rio Grande do Sul (UFRGS)"
    },
    "Recife": {
        "população": "1,6 milhões",
        "pontos_turisticos": ["Praia de Boa Viagem", "Instituto Ricardo Brennand", "Marco Zero"],
        "universidade": "Universidade Federal de Pernambuco (UFPE)"
    },
    "Manaus": {
        "população": "2,1 milhões",
        "pontos_turisticos": ["Teatro Amazonas", "Encontro das Águas", "Palácio Rio Negro"],
        "universidade": "Universidade Federal do Amazonas (UFAM)"
    },
    "Natal": {
        "população": "1,4 milhões",
        "pontos_turisticos": ["Forte dos Reis Magos", "Praia de Ponta Negra", "Dunas de Genipabu"],
        "universidade": "Universidade Federal do Rio Grande do Norte (UFRN)"
    },
    "Maceió": {
        "população": "1,0 milhão",
        "pontos_turisticos": ["Praia do Francês", "Palácio Marechal Floriano Peixoto", "Igreja de São Gonçalo do Amarante"],
        "universidade": "Universidade Federal de Alagoas (UFAL)"
    },
    "Cuiabá": {
        "população": "620 mil",
        "pontos_turisticos": ["Parque Nacional de Chapada dos Guimarães", "Catedral Basílica do Senhor Bom Jesus", "Museu do Morro da Caixa D'Água"],
        "universidade": "Universidade Federal de Mato Grosso (UFMT)"
    },
    "Aracaju": {
        "população": "650 mil",
        "pontos_turisticos": ["Praia de Atalaia", "Museu Palácio Marechal Floriano Peixoto", "Mercado Municipal"],
        "universidade": "Universidade Federal de Sergipe (UFS)"
    }
}

# Aplica o limitador de memória ao histórico de mensagens
trimmer.invoke(city_data)

# Criando um pipeline de execução para otimizar a passagem de informações
chain = (
    RunnablePassthrough.assign(city_data =itemgetter("city_data") | trimmer)  # Aplica a otimização do histórico
    | prompt  # Passa a entrada pelo template de prompt
    | model  # Envia para o modelo
)

# Exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "city_data": city_data,  # Passa o dicionário separado
        "messages": [HumanMessage(content="Qual é a população de São Paulo?")],  # Lista de mensagens
        "language": "Português"
    }
)

# Exibe a resposta final do modelo
print("Resposta final:", response.content)

# Exemplo 2 de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "city_data": city_data,
        "messages": [HumanMessage(content="Quais são os pontos turísticos de Fortaleza?")],
        "language": "Português"
    }
)

print("Resposta final:", response.content)

# Exemplo 3 de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "city_data": city_data,
        "messages": [HumanMessage(content="Qual é a principal universidade de Recife?")],
        "language": "Português"
    }
)

# Exibe a resposta final do modelo
print("Resposta final:", response.content)