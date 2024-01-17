UID := $(shell id -u)
GID := $(shell id -g)

docker-compose := env UID=${UID} GID=${GID} docker compose

build:
	@${docker-compose} build

up:
	@${docker-compose} up

stop:
	@${docker-compose} stop

restart:
	@${docker-compose} restart

estimate_translation:
	@${docker-compose} exec app pipenv run python manage.py estimate_translation

estimate_inference:
	@${docker-compose} exec app pipenv run python manage.py estimate_inference

run_translations:
	@${docker-compose} exec app pipenv run python manage.py run_translations


