pdflatex main.tex

bibtex main

pdflatex main.tex

pdflatex main.tex

rm *.out *.aux *.bbl *.blg *.fdb_latexmk *.fls main.log *.gz

# mv main.pdf ../paper/"A Systematic Characterization of Fine-Tuning Strategies Across CNN, Vision Transformer, and Hybrid Architectures for Medical Image Classification.pdf"

# open ../paper/"A Systematic Characterization of Fine-Tuning Strategies Across CNN, Vision Transformer, and Hybrid Architectures for Medical Image Classification.pdf"

