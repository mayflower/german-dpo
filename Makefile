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

run_translations:
	@${docker-compose} exec app pipenv run python manage.py run_translations

estimate_metrics:
	@${docker-compose} exec app pipenv run python manage.py estimate_metrics
