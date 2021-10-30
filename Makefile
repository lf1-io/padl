clean:
	-rm -rf dist/

build:
	python3 setup.py sdist

release:
	twine upload dist/*
