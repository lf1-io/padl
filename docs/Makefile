# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

github:
	make clean
	-mkdir _static
	make build

readme:
	-rm ../README.rst
	-rm ../.README.md
	-rm ../.README.rst
	-rm README.rst
	cd .. && tail +9 README.md > .README.md && m2r --overwrite .README.md && rm .README.md && mv .README.rst docs/README.rst
	(echo "====" && echo "Introduction" && echo "====" && echo ".. toctree::" && echo "   :maxdepth: 3" && echo "   :caption: Contents:" && cat README.rst) > .README.rst && mv .README.rst README.rst

build:
	cp -r ../img/* ./img/
	make html
	cp -r _build/html/* ./
	mv _sources sources
	mv _images images
	mv _static static
	sed -i'.bak' -e 's/_sources/sources/g' *.html
	sed -i'.bak' -e 's/_images/images/g' *.html
	sed -i'.bak' -e 's/_static/static/g' *.html
	rm -rf _build
	rm *.bak
	
clean:
	-rm searchindex.js
	-rm *.rst.txt
	-rm *.html
	-rm -rf sources
	-rm -rf images
	-rm -rf static
	-rm -rf _build
	-rm *.bak

