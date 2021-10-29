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
	make html
	cp -r _build/html/* ./
	mv _sources sources
	mv _static static
	sed -i'.bak' -e 's/_sources/sources/g' *.html
	sed -i'.bak' -e 's/_static/static/g' *.html
	rm -rf _build
	rm *.bak
	
clean:
	-rm -rf sources
	-rm -rf static
	-rm -rf _build
	-rm *.bak

