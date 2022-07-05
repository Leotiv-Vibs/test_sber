"""
3) Целочисленное разделение n представляет собой слабо убывающий список положительных целых чисел,сумма которых равна n.
Например, есть 3 целочисленных разделов по 3:
[3], [2,1], [1,1,1].
Напишите функцию, которая возвращает количество целочисленных разделов n.
Функция должна уметь находить количество целочисленных разделов n менее 100.
"""


def check_integer_intervals(number: int) -> int:
    """
    the function gets an integer number and returns the number of integer intervals
    :param number: int value for devide intervals
    :return: count of integer intervals of the number
    """
    list_values_parts = [0] * number  # a list in which the current state will be stored
    index = 0  # index to be worked with in the current iteration
    list_values_parts[index] = number  # initialization of the list by the number itself
    count_intervals = 0  # counter initialization

    # loop works as long as the state list does not contain all units
    while list_values_parts != [1 for i in range(number)]:
        insert_value = 0  # insert_value initialization
        while index >= 0 and list_values_parts[index] == 1:  # check index value for equals 1
            insert_value += list_values_parts[index]  # add 1 for insert_value
            index -= 1  # left step index
        list_values_parts[index] -= 1  # change value for next iteration
        insert_value += 1  # add 1 for insert value
        while insert_value > list_values_parts[index]:
            list_values_parts[index + 1] = list_values_parts[index]  # insert this iteration value index++
            insert_value -= list_values_parts[index]  # change value for insert because this value use it
            index += 1  # next index
        list_values_parts[index + 1] = insert_value  # insert iteration value
        count_intervals += 1  # add 1 for count_intervals
        index += 1  # next step index
    count_intervals += 1  # last add 1 for count_intervals

    return count_intervals


def run():
    for number in range(1, 25):
        print(f"Count integer intervals for number {number}: {check_integer_intervals(number)}")


if __name__ == '__main__':
    run()
