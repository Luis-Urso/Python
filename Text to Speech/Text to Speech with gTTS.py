from gtts import gTTS
import pygame
import playsound
	

tts = gTTS(text='Oi esse é um teste de fala computacional.....k k k k k ', lang='pt-BR')
tts.save("hello.mp3")

playsound.playsound('hello.mp3', True)

