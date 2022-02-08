# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = src
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
	python gen_doc_files.py
	make html
	mv _build/html unstable
	cp -r src/img unstable/
	cp src/custom.css unstable/_static
	rm -r _build

clean:
	-rm -r unstable
	-rm -r _build

version:
	cp -r unstable $(VERSION)
	-rm latest
	-ln -s $(VERSION) latest
	python gen_versions.py
