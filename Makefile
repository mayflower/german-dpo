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

run:
	@${docker-compose} exec app pipenv run python manage.py run

estimate_total_cost:
	@${docker-compose} exec app pipenv run python manage.py estimate_total_cost
