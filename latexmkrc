$pdf_mode = 1;
@default_files = ('index.tex');
$out_dir = 'output';

$pdflatex = "pdflatex -synctex=1 -halt-on-error %O %S";

$pdf_previewer = "zathura %O %S";
