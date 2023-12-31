RM := rm -fr

ENVFILE	:= .env

GR= \033[32;1m
RE= \033[31;1m
YE= \033[33;1m
CY= \033[36;1m
RC= \033[0m

all: up

up : $(ENVFILE)
	@printf "$(CY)"
	@echo "Starting up..."
	@printf "$(RC)"
	docker compose up -d --build

$(ENVFILE) :
	@printf "$(YE)"
	@printf "$(RC)"

down :
	@printf "$(RE)"
	@echo "Shutting down..."
	@printf "$(RC)"
	docker compose down -v

clean : down

fclean :
	docker compose down -v --rmi all --remove-orphans

re : fclean all

.PHONY: all clean fclean re up down

