from typing import List
from sympy import isprime



def eratosthenes(number: int) -> List:
    """
    Eratosthenes sieve function
    :param number: the number up to which we are looking for primes
    :return: a sorted list of prime numbers up to n
    """
    sieve = list(range(number + 1))
    sieve[1] = 0
    for i in sieve:
        if i > 1:
            for j in range(2 * i, len(sieve), i):
                sieve[j] = 0
    return sorted(list(set(sieve)))[1:]


def is_prime_(number: int):
    """
    checks if the number is prime
    :param number: check digit
    :return: whether the number is a prime [True or False]
    """
    if number > 1:
        for i in range(2, number):
            if (number % i) == 0:
                return False
        else:
            return True
    else:
        return False


def prime_with_eratosthenes(number: int):
    """
    determines whether the number is a prime twin with eratosthenes sieve
    :param number: number to be determined
    :return: whether the number is a prime twin [True or False]
    """
    list_prime_number = eratosthenes(number + 1)
    if number in list_prime_number:
        if number - 2 in list_prime_number and number + 2 in list_prime_number:
            return True
        else:
            return False
    else:
        return False


def prime_with_sympy(number: int):
    """
    determines whether the number is a prime twin with sympy library
    :param number: number to be determined
    :return: whether the number is a prime twin [True or False]
    """
    if isprime(number) and isprime(number - 2) and isprime(number + 2):
        return True
    else:
        return False


def prime(number: int):
    """
    determines whether the number is a prime twin
    :param number: number to be determined
    :return: whether the number is a prime twin [True or False]
    """
    if is_prime_(number) and is_prime_(number - 2) and is_prime_(number + 2):
        return True
    else:
        return False


def run():
    for i in range(1000):
        print(f"{i}: {prime(i)}")


if __name__ == '__main__':
    run()
