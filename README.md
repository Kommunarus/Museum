# Museum
С сайта госфонда был скачан файл с датасетом data-4-...csv.
Из него было вырезано несколько колонок относящиеся к топологии и описанию.
И сформированы более мелкие файлы для закачивания картинок. Например по 30т примеров каждого класса.
В скрипте download  также есть код для скачивания. 
Еще одним скриптом, img_to_pkl картинки перегонялись сразу в pkl, образались до размера 224*224, что бы быстрее обучались сетки и не тратилось время для открывания фото. Исходники фото удалялись. Гуиды назначались случайные. Никаких поисков для knn не делались.
Обработка текста шла на полном файле с сайта госфонда методом опорных векторов.
самым важным для точности оказалось использование tfidf с такими параметрами min=1, ngram=(1,3):
Более толстые модели уже не влазили в память.
После этого с  текстом я уже не работал. Мне показалось, что точность классификации svm хорошей и давала сразу f1 около 0,5 без картинок. Хотя я пробовал много моделей, но точнее svm ничего не было.
 Также для удобства добавил флаг.
Флаг fitting позволял мне не обучать каждый раз модель, а записать ее и потом постоянно использовать. Svm на мой машине обучался около 2х часов. Именно эти веса берут 14 Гб места на диске
Обучение на картинках шло скриптом script15.
Как видно, в максимуме у меня было закачано 1.8 млн картинок (1,6 трейн, 0,2 тест). Но на таком массиве я обучился только самый последний раз.  Большуя часть картинок я успел закачать когда еще не банили. Потом уже почти не качал. Наверно последние тысяч 400  закачал за последние 2 недели, так как точности соло сеток мне не хватало для топа. Разбивку на тест и трейн делал случайно. Фото с тестом не сопоставлял.  Время обучения – сутки, 7 эпох.
Я искал лучшую сеть и так получилось, что лучше классифицировала efficientnet b3 .
Для предсказания есть два файла: для соло модели и для ансамбля. Точнее показал себя ансамбль.
Как видно, наличие текста однозначно говорило уходить в модель svm, если была картинка – то ансамблю сеток, иначе – что то по умолчанию. Я сначала пытался сделать одну сеть, с двумя входами, но точность была гораздо меньше, чем если использовать для текста и картинок разные методы.
Итоговая точность на паблике ансамбля из 5 моделей вывела меня на 1 место (на утро четверга). Это 4 модели b3  и одна resnet34. 
