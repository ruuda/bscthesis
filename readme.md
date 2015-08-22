The Hopf Map in Magnetohydrodynamics
====================================

This repository contains the source of my bachelor’s thesis. I have made the
source available here in the hope that it will be useful to somebody. The output
pdf is [freely accessible][pdf] at the university website anyway. For
convenience the pdf is also included in this repository.

[pdf]: https://www.math.leidenuniv.nl/en/theses/547/

Compiling
---------

You will need the following prerequisites:

 - A recent TeX Live distribution (I tested 2014 and 2015)
 - The Minion Pro font
 - The [`FontPro`][fontpro] package for Minion
 - Python 3 in your path as `python`

To compile, run

    $ latexmk --xelatex --enable-write18

This has been verified to work under Windows as well as Arch Linux.

[fontpro]: https://github.com/sebschub/FontPro

Licence
-------

The content of the thesis is copyrighted; I am merely making available the
source files for inspection. Feel free to use some of the LaTeX macros in your
own documents.

The Python code used to generate the graphics in this thesis is released under
version 3 of the [GNU Public Licence][gplv3].

The seal of Leiden University is included in this repository as `ulzegel.eps`.

[gplv3]: https://gnu.org/licenses/gpl
