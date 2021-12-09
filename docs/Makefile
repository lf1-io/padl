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
	make build

build:
	cp -r ../img/* ./img/
	-rm -rf _build
	-mkdir _static
	-rm -rf autodocs
	python gen_doc_files.py
	cp custom.css _static/custom.css
	make html
	cp -r _build/html/* ./
	mv _sources sources
	-mv _images images
	-mv _static static
	-sed -i'.bak' -e 's/_sources/sources/g' *.html
	-sed -i'.bak' -e 's/_images/images/g' *.html
	-sed -i'.bak' -e 's/_static/static/g' *.html
	-sed -i'.bak' -e 's/_static/static/g' apidocs/*.html
	-sed -i'.bak' -e 's/_static/static/g' advanced/*.html
	-sed -i'.bak' -e 's/_static/static/g' usage/*.html
	-rm -rf _static
	-rm -rf _build
	-rm *.bak
	-rm apidocs/*.bak
	-rm advanced/*.bak
	-rm usage/*.bak

clean:
	-rm searchindex.js
	-rm *.html
	-rm apidocs/*.html
	-rm advanced/*.html
	-rm usage/*.html
	-rm -rf sources
	-rm -rf images
	-rm -rf static
	-rm -rf build
	-rm -rf _sources
	-rm -rf _images
	-rm -rf _static
	-rm -rf _build
	-rm *.bak
	-rm apidocs/*.bak
	-rm advanced/*.bak
	-rm usage/*.bak

