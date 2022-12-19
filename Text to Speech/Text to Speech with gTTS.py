from gtts import gTTS
import pygame
import playsound
	

tts = gTTS(text='Ratos de Porón son muy buenos', lang='es')
tts.save("hello.mp3")

playsound.playsound('hello.mp3', True)

