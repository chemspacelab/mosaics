# K.Karandashev: Largely inspired by the makefile from https://github.com/qmlcode/qmllib
# I made it more lightweight though, giving the developer more leeway on what environment
# to base and test their commit in.
# TODO: at some point add "test:".

all: install

env:
	pip install pre-commit

./.git/hooks/pre-commit: env
	pre-commit install

dev-setup: env ./.git/hooks/pre-commit

review: dev-setup
	pre-commit run --all-files

install:
	pip install .

clean:
	rm -Rf ./build
	rm -Rf ./dist
	rm -Rf ./mosaics.egg-info
