#####################################
# incluir pacotes Chatterbot, spacy
# pip install chatterbot
# pip install spacy
#####################################


from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

bot = ChatBot("LUCCA Python")

conversa = ['Oi', 'Olá', 'Tudo bem?', 'Tudo ótimo', 
			'Você gosta de programar?', 'Sim, eu programo em Python']

bot.set_trainer(ListTrainer)
bot.train(conversa)

while True:
    pergunta = input("Oi")
    resposta = bot.get_response(pergunta)
    if float(resposta.confidence) > 0.5:
        print('TW Bot: ', resposta)
    else:
        print('TW Bot: Ainda não sei responder esta pergunta')