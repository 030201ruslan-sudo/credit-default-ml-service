# Credit Default ML Service

# Описание проекта
Проект представляет собой сервис для предсказания вероятности дефолта клиента по данным о кредитной карте.

Цель проекта — показать полный цикл внедрения ML-модели:
- обучение модели;
- сохранение модели для инференса;
- создание API на Flask;
- контейнеризация через Docker;
- подготовка плана A/B-тестирования.

# Используемый датасет
UCI Credit Card Default Dataset.

Целевая переменная:
`default.payment.next.month`

# Используемые технологии
- Python
- pandas
- scikit-learn
- Flask
- joblib
- Docker

# Структура проекта
```text
credit-default-ml-service/
├── app/
│   └── api.py
├── models/
│   └── model.joblib
├── src/
│   └── train_model.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── ARCHITECTURE.md
├── ab_test_plan.md
├── requirements.txt
└── .gitignore

# GitHub
https://github.com/030201ruslan-sudo/credit-default-ml-service

# Docker Hub
https://hub.docker.com/r/030201ruslan/credit-default-api