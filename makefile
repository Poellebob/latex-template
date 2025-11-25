.PHONY: latex biber sage latex

sage:
	sage *.sagetex.sage

latex:
	latexmk -pdf -shell-escape

biber:
	biber main
