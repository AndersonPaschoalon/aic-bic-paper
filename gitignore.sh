#!/bin/bash
#find ./* -size +99M | cat >> .gitignore
#find ./* -size +99M | sed "s/\ /\\ /g"
find ./* -size +49M | sed "s/ /\\\ /g" | cat >> .gitignore


