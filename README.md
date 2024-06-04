# FQWB
Данный репозиторий содержит код библиотеки по прореживанию ResNet модели. 
Для прореживания выбираются группы весов соответствующих нейронам промежуточного представления данных между двумя слоями сверток Residual блока архитектуры.

Модуль [inner_data_regularization.py](inner_data_regularization.py)
Позволяет запустить поиск наилучшей конфигурации удаления весов в нейронной сети, где в качестве криетрия выбора используется энтропия промежуточного состояния

Модуль [iteration.py](iteration.py)
Позволяет запустить поиск наилучшей конфигурации удаления Residual блоков 

Модуль [load_and_cut.py](load_and_cut.py)
Итоговое удаление блоков из модели и сохранение результата

Модуль [main_pipeline.py](main_pipeline.py)
Скрипт обучения моделей с разными гиперпараметрами с сохранением результатов

Модуль [metrics.py](metrics.py)
Снятие метрик с полученной модели

# Dependency
- conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install -r requirements.txt
