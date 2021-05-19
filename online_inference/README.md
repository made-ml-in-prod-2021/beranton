Homework # 2
==============================

Модель как REST сервис + Docker

### Сборка образа
Для сборки образа и последующего запуска необходимо выполнить команды: 
```
 docker build . -f online_inference/app/Dockerfile -t predict_app:v1 

 docker run -p 8080:8080 online_inference/predict_app:v1

```

Внутри контейнера будет запущен сервис, посылать запросы к сервису можно с помощью скрипта make_request.py

### Оптимизация образа

Для сокращения объема образа использовался python:3.9-slim. Он позволил сократить объем образа в 2 раза по сравнению с изначальным объемом
Размер образа на dockerhub - 227 MB

### Получение образа с DockerHub

Собранный образ можно скачать с DockerHub следующими командами:
```
 docker pull beranton/predict_app:v1  

 docker run -p 8080:8080 beranton/predict_app:v1

```

### Запуск скрипта генерации запросов к сервису make_request.py
Для запросов к сервису написан скрипт make_request.py.  
Скрипт принимает 2 параметра на вход: 
```
    --file_path  - адрес файла с данными
    --count - количество строк из файла, на которые нужно получить предсказание. 

    Пример запуска скрипта из папки online_inference:
        python make_request.py --file_path=data.csv

```

### Запуск тестов
```
    pytest tests
```