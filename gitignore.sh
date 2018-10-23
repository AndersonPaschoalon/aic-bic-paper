#!/bin/bash

function init_gitignore
{
	rm .gitignore
	#find * -size +49M | sed "s/ /\\\ /g" | cat >> .gitignore
	find * -size +49M | cat >> .gitignore
}

function main
{
	init_gitignore;
}

if [[ "$1" != "--source" ]]
	then
	main;
fi


