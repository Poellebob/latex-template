.PHONY: latex biber sage latex

sage:
	sage *.sagetex.sage

latex:
	latexmk -pdf

biber:
	biber main
