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

 Dataset name  | link                                                                                                                   | max_size  | categories_num | images_num 
---------------|------------------------------------------------------------------------------------------------------------------------|-----------|----------------|------------
 SVHN          | https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN                   | 0x0       | 0              | 0          
 SUN397        | https://pytorch.org/vision/stable/generated/torchvision.datasets.SUN397.html#torchvision.datasets.SUN397               | 0x0       | 0              | 0          
 STL10         | https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10                 | 0x0       | 0              | 0          
 StanfordCars  | https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html#torchvision.datasets.StanfordCars   | 0x0       | 0              | 0          
 PCAM          | https://pytorch.org/vision/stable/generated/torchvision.datasets.PCAM.html#torchvision.datasets.PCAM                   | 0x0       | 0              | 0          
 Places365     | https://pytorch.org/vision/stable/generated/torchvision.datasets.Places365.html#torchvision.datasets.Places365         | 0x0       | 0              | 0          
 Caltech101    | https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech101.html#torchvision.datasets.Caltech101       | 0x0       | 0              | 0          
 Caltech256    | https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech256.html#caltech256                            | 0x0       | 0              | 0          
 Country211    | https://pytorch.org/vision/stable/generated/torchvision.datasets.Country211.html#torchvision.datasets.Country211       | 0x0       | 0              | 0          
 EuroSAT       | https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT             | 0x0       | 0              | 0          
 FashionMNIST  | https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST   | 0x0       | 0              | 0          
 FGVCAircraft  | https://pytorch.org/vision/stable/generated/torchvision.datasets.FGVCAircraft.html#torchvision.datasets.FGVCAircraft   | 0x0       | 0              | 0          
 Flowers102    | https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html#torchvision.datasets.Flowers102       | 0x0       | 0              | 0          
 Food101       | https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101             | 0x0       | 0              | 0          
 ImageNet      | https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet           | 0x0       | 0              | 0          
 Imagenette    | https://pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html#torchvision.datasets.Imagenette       | 0x0       | 0              | 0          
 KMNIST        | https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST               | 0x0       | 0              | 0          
 OxfordIIITPet | https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html#torchvision.datasets.OxfordIIITPet | 0x0       | 0              | 0          
 batterfly     | --                                                                                                                     | 224x224   | 75             | 6499       
 blood_cells   | --                                                                                                                     | 366x369   | 17             | 17092      
 breast_hist   | --                                                                                                                     | 50x50     | 2              | 277524     
 chest_xray    | --                                                                                                                     | 5623x4757 | 4              | 7132       
 crop_desease  | --                                                                                                                     | 400x400   | 22             | 25220      
 flowers       | --                                                                                                                     | 1024x442  | 5              | 3670       
 fourniture    | --                                                                                                                     | 512x512   | 32             | 12360      
 house_plant   | --                                                                                                                     | 6720x8021 | 47             | 14288      
 vehicle       | --                                                                                                                     | 7993x5995 | 7              | 5504       
