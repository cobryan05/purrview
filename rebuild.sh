docker-compose down --volumes --remove-orphans \
&& BUILD_TARGET=dev TRAINING_BUILD_TARGET=training-dev docker-compose build --build-arg APPUSER_UID=$(id -u) --build-arg APPUSER_GID=$(id -g) \
&& docker-compose up
