install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main *.py

format:	
	black *.py 

lint:
	pylint --disable=R,C --ignore-patterns=*.py
		
all: install lint test format