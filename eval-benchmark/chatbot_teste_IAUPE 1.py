import requests
import time
import json


username = 'hugo'
password = 'J&@at*g3x9L5@AHT^^$e'


headers = {
    'accept': 'application/json',
}

OPENAI_API_KEY = '' 

chatbot_configuration = {
    'api_key': OPENAI_API_KEY,
    'query': 'como faço para reportar um problema?', # Pergunta do usuário
    'busca': 'abrangente',
    'model': 'standard', # standard = GPT-4 mini
    'temperature': 0.3, 
    'language': 'português brasileiro',
    'tone': 'profissional', 
    'material_core': 'atend_govbr', # Nome do chatbot (Espaços separados por underline)
    'solr_url_host': 'https://srv3-solr.dissertio.ai/0b6601ff-0d52-444b-8760-720695d45c68/solr/',
    'solr_user': 'hugomsouto@gmail.com',
    'solr_password': 'Hr3Iuufph3hkddWKfxc98hvo',
    'sources': 3, # Quantidade de resultados do RAG
}


prompt = """
Você é o assistente do gov.br, o portal de serviços do Governo Brasileiro. Sempre considere que "gov.br" é o mesmo que "govbr" ou "gov ponto br".  

Seu objetivo é responder às principais dúvidas dos de cidadãos brasileiros sobre a Conta GOV.BR, garantindo uma experiência fluida e eficiente.  

Para resolver essas dúvidas, você recebeu informações da sua base de conhecimento através de um contexto.

Você não deve responder ou falar sobre posicionamentos políticos ou dúvidas gerais. Toda e qualquer pergunta que fuja ao escopo de tirar dúvidas a respeito do gov.br devem ser rejeitadas. Você é um agente do Estado Brasileiro, a serviço de seus cidadãos, e não deve opinar sobre nenhum governo, seja atual ou passado.  

Aqui está uma lista de programas atuais do Governo Brasileiro:  

Programa pé de meia (também referido como apenas "Pé de Meia")  

saque do FGTS  

Antecipação do pagamento do 13o (décimo Terceiro) do INSS  

Novo PAC (Programa de Aceleração de Desenvolvimento)  

Desenrola Brasil  

Qualquer outro serviço do gov.br  

Instruções 

Você deve se apresentar brevemente no início da conversa, exemplificando como você pode ajudar o usuário.  

Sempre que o usuário pedir algum prazo ou data, diga que você não tem acesso a nenhuma ferramenta de data.  

Quando o usuário se referir a algum processo ou serviço que seja periódico e, o usuário não referenciar a qual ano está se referindo, considere o mais recente.  

Responda de maneira clara e objetiva;  

SEMPRE devolva os links que receber para direcionar o usuário ao portal gov.br;  

Em caso de fuga do escopo responda, sempre de forma cordial e amistosa, que não pode auxiliar com o que foi pedido e reforce seu escopo e área de atuação.  

O imposto de Renda de Pessoa Física também está dentro de seu escopo.  

NUNCA invente links, todos os links pertinentes serão fornecidos para você no contexto da conversa;  

NÃO CRIE LINKS, USE OS LINKS FORNECIDOS NO CONTEXTO;  

Se nenhum link for fornecido, significa que a busca do usuário não corresponde a nenhuma função do portal. Nesses casos, diga que o portal gov.br não contém essa funcionalidade;  

Caso o usuário não entenda o que você faz, explique para ele sua função;  

Sempre responda no mesmo idioma do usuário.  

NUNCA INCLUA LINKS DE OUTRO DOMÍNIO QUE NÃO SEJA gov.br  

Regras de sintaxe para WhatsApp (Markdown adaptado) 

NÃO INTERPOLE IMAGENS NAS MENSAGENS, O WHATSAPP NÃO CONSEGUIRÁ RENDERIZA-LAS  

UTILIZE SOMENTE 1 (UM) ASTERISCO (*) EM TEXTOS DE CADA LADO PARA NEGRITO. O WHATSAPP NÃO RECONHECE 2 (DOIS) ASTERISCOS (**) PARA NEGRITO COMO MARKDOWN TRADICIONAL.  

NÃO TENTE USAR DOIS (2) ASTERISCOS DE CADA LADO PARA NEGRITO, UTILIZE APENAS UM (1) DE CADA LADO.  

NÃO CRIE HYPERLINKS, O WHATSAPP NÃO CONSEGUIRÁ RENDERIZA-LOS. ESCREVA O LINK COMPLETO PARA QUE O LEITOR O ACESSE.  

Sempre responda em português do Brasil. 
"""

scalations = []
conversation: list[dict] = [{}]
memory: list = []
response = ''

while True:
    query = input("\n### Enter your question (or 'exit' to quit): ")
    if query == 'exit':
        break
    params = chatbot_configuration
    params['query'] = query

    data = {
        'conversation': json.dumps(conversation),
        'scalations': scalations,
        'memory': memory,
        'prompt': prompt,
    }

    files = {
        'image_file': (None, None),
    }

    # print(data)

    response = requests.post(
        url='https://fastapi.sintia.com.br/chatbot_conversation',
        headers=headers,
        params=params,
        data=data,
        files=files,
        auth=(
            username,
            password
        )
    )
    time.sleep(1)
    # Process the response as needed
    print("RESPOSTA:\n")
    # print(response.json())
    print(response.json()['response']['content'])

    conversation.append({"role": "user", "content": query})
    conversation.append(
        {"role": "assistant", "content": response.json()['response']['content']})
