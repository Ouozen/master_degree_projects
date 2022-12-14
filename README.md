# Описание скриптов которые лежат в репозитории

## RANSAC.py
Этот скрипт реализовывает алгоритм RANSAC. Алгоритм позволяет находить прямые среди зашумленных данных с помощью случайного, итеративного отбора точек из общей выборки, построения по ним прямой и выбора прямой без зашемленных данных.  
В скрипте 4 функции: функция для генерации данных, для расчета необходимого количества итераций алгоритма, функция расчета максимального расстояния от прямой до точки и функция поиска прямой алгоритмом.

## hough.py
Скрипт реализует алгоритм Хафа для поиска прямых на изображении. Алгоритм переводит каждую точку на изображении в пространство Хафа. Каждая точка в этом пространстве становится синусоидой, если на исходном изображении есть прямая, то в пространстве Хафа все точки этой прямой пересекутся в одной точке. На основе наиболее "ярких" точек и производится поиск прямой.  
Реализовано 2 функции: функция для перевода исходного изображения в пространство Хафа и функуция поиска точек в пространстве Хафа которые являются прямыми на исходном изображении.

## tmux.py  
Скрипт запускает tmux, открывает указанной количество окон и запускает в них jupyter-notebook. Скрипт был выполнен в учебных целях, чтобы научиться запускать несколько окон и параллельно в каждом выполнять необходимый набор команд.

## parallel_prefix.py
Скрипт имитирует работу алгоритма параллельного префикса. Этот алгоритм позволяет распараллелить операцию выполняемую за n шагов, таким образом приводит ее к операции выполняемой за log(n) шагов.
