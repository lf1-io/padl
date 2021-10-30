clean:
	-rm docs/README_NO_IMAGES.md
	-rm -rf dist/

build:
	cd docs && python3 clean_readme_for_pip.py
	python3 setup.py sdist

release:
	twine upload dist/*
