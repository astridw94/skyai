set_env:
	pyenv virtualenv 3.10.6 skyenv
	pyenv local skyenv

install_reqs:
	python -m pip install --upgrade pip
	@pip install -r requirements.txt

reinstall_package:
	@pip uninstall -y skypkg || :
	@pip install -e .


clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc

all: reinstall_package clean

run_api:
	cd api && uvicorn fast:app --reload
