from math import ceil, log2

def print_prefix(n):
    depth = ceil(log2(n)) # посчитали глубину 

    # создадим словарь который будет считать позицию элементов.
    position = {str(i): [i] for i in range(n)}
    elem_start = n

    for level in range(depth): # начнем проходиться по уровням.
        # добавим элементы которые мы будем обходить на этом уровне.
        diff = n - (n - (2 ** level))
        elem_end = elem_start + (n - (2 ** level))
        for pos, elem in zip(range(diff, n), range(elem_start, elem_end)):
            position[str(pos)].append(elem)
            # теперь пройдемся по новому элементу и распечатаем что подается на вход.
            pred_pos = pos - (2 ** level) # ветка первого элемента зависит от уровня
            
            if len(position[str(pos)]) == len(position[str(pred_pos)]):
                # если длина такая же, то чтобы подать элемент с предыдущей ветки нужно взять -2 элемент.
                input_elem_1 = position[str(pred_pos)][-2]
            else:
                # если длина разная, то элементом с предыдущей ветки будет последний в списке.
                input_elem_1 = position[str(pred_pos)][-1] 
            input_elem_2 = position[str(pos)][-2] # вторым элементом подается предыдущий элемент по этой ветке.

            print(f'GATE {elem} OR {input_elem_1} {input_elem_2}')
        elem_start = elem_end

    # чтобы распечатать output просто пройдемся по словарю и выведем номер ветки и последний элемент в списке.
    for pos, elem in position.items():
        print(f'OUTPUT {pos} {elem[-1]}')


if __name__ == '__main__':
    n = int(input()) # получили значение
    result = print_prefix(n)
    print(result)