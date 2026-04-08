# FQWB

Данный репозиторий содержит код библиотеки по прореживанию ResNet модели.
Для прореживания выбираются группы весов соответствующих нейронам промежуточного представления данных между двумя слоями
сверток Residual блока архитектуры.

Модуль [inner_data_regularization.py](inner_data_regularization.py)
Позволяет запустить поиск наилучшей конфигурации удаления весов в нейронной сети, где в качестве криетрия выбора
используется энтропия промежуточного состояния

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

# Logs of learning:

- tensorboard --logdir=/media/kirrog/data/data/fqwb_data/stats

# Datasets:

 Dataset name  | link                                                                                             | max_size  | categories_num | images_num 
---------------|--------------------------------------------------------------------------------------------------|-----------|----------------|------------
 +batterfly    | https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification                       | 224x224   | 75             | 6499       
 +blood_cells  | https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset                           | 366x369   | 17             | 17092      
 breast_hist   | https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images                   | 50x50     | 2              | 277524     
 chest_xray    | https://www.kaggle.com/datasets/pritpal2873/chest-x-ray-dataset-4-categories                     | 5623x4757 | 4              | 7132       
 +crop_desease | https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection                  | 400x400   | 22             | 25220      
 flowers       | https://www.kaggle.com/datasets/imsparsh/flowers-dataset                                         | 1024x442  | 5              | 3670       
 +fourniture   | https://www.kaggle.com/datasets/anthonytherrien/image-classification-32-classes-fourniture       | 512x512   | 32             | 12360      
 house_plant   | https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species?select=house_plant_species | 6720x8021 | 47             | 14288      
 vehicle       | https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification                             | 7993x5995 | 7              | 5504       
 +cifar10      | https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html                 | 32x32     | 10             | 60000      
 +cifar100     | https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html                | 32x32     | 100            | 60000      
 +tinyimagenet | https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet                                       | 64x64     | 200            | 110000     
 imagenet1k    | https://www.kaggle.com/datasets/sautkin/imagenet1k0/data                                         | 9331x6530 | 1000           | 1330297    

Для построения графиков:
1) prunning_results_analysis/comparability_check.py
2) prunning_results_analysis/stats_comparer.py
3) prunning_results_analysis/merge_plots.py

