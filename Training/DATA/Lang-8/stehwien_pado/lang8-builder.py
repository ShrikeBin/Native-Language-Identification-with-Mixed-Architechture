#-*- coding:utf-8 -*-
import os
import re

'''
Author: Sabrina Stehwien
Date: January 2015
Description: takes output of lang8-scraper.py and sorts entries into directories by native language
'''

def build_corpus():
	directory = "Lang8texts"
	os.mkdir(directory)
	path = directory+"/"

	entryfile = open("lang8_entries.txt", "r").readlines()
	count = 0
	for i in range(0, len(entryfile)):
		line = entryfile[i]
		if line.startswith("^^DOC^^ |||"):
			words = line.split(" ||| ")
			language = words[1]
			url = words[2]
			text = entryfile[i+1]
			text = re.sub('<.*?>', '', text)

			langdir = os.path.join(path, language)
			if not os.path.exists(langdir):
    				os.makedirs(langdir)
		
			name = language[:3]+str(count)
			newfile = name+".txt"
			output = open(os.path.join(langdir, newfile), 'wb')
			output.write(text)
			output.close()
			count+=1

if __name__=="__main__":
	build_corpus()
