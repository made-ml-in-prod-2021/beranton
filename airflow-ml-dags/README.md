#### Запуск
```
docker-compose up --build
```

#### Скрины
Список DAG
![alt text][proofs/dags.png]
Загрузка данных
![alt text][proofs/download_data.png]
Тренировочный пайплайн
![alt text][proofs/train_dag.png]

#### Самооценка
1. DAG загрузки данных - 5 баллов.
2. DAG обучения модели - 10 баллов.
3. Все сделано через DockerOperator - 10 баллов.
4. Настроен alert - 3 балла.
5. Самооценка - 1 балл.

В итоге: 29 * 0.6 = 17.4 балла за просрочку мягкого дедлайна.