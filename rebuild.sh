docker-compose down --volumes --remove-orphans && docker-compose build --build-arg APPUSER_UID=$(id -u)  --build-arg APPUSER_GID=$(id -u)  &&  docker-compose up
