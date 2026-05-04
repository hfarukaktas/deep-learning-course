# Section 1: Decorators

print("=" *60)
print("Section 1: Decorators")
print("=" *60)

def myDecorator (func):
    def wrapper():
        print("Wrapper executed")
        func()
        print("Wrapper finished")
    return wrapper

@myDecorator
def helloWorld():
    print("Hello World!")

helloWorld()

# Section 2: Property Decorators

print("=" *60)
print("Section 2: Property Decorators")
print("=" *60)

# data validation, private/public (encapsulation)

class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("name must be a string")
        if len(value) < 2:
            raise ValueError("Name must be longer")
        self.__name = value
    @name.deleter
    def name(self):
        self.__name = None

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise ValueError("Age must an integer!")
        if value < 0:
            raise ValueError("Age has to be a positive number!")
        if value > 150:
            raise ValueError("Age should be less than 150")
        self.__age = value

faruk = Person("Faruk", 24)

print(faruk.name)

faruk.name = "Hasan"

print(faruk.name)

faruk.name = "asd"

print(faruk.name)

del faruk.name

print(faruk.name)

#faruk.name = 70

print(faruk.name)

print(faruk.age)

faruk.age = 30

print(faruk.age)

#faruk.age = 151

print(faruk.age)

# Section 3: Static Methods

print("=" *60)
print("Section 3: Static Methods")
print("=" *60)

class MathOperations:

    @staticmethod
    def add(x,y):
        return x+y
    @staticmethod
    def divide(x,y):
        return x/y
print(MathOperations.add(3,2))
math = MathOperations()
print(math.add(2,3))

print(MathOperations.divide(10,2))

# Section 4: Class Methods

print("=" *60)
print("Section 4: Class Methods")
print("=" *60)

# alternative constructor use case

class Pizza:

    total_pizzas = 0

    def __init__(self, ingredients):
        self.ingredients = ingredients
        Pizza.total_pizzas += 1

    @classmethod
    def margherita(cls):
        return cls(["cheese", "tomato", "sauce"])

    @classmethod
    def pepperoni(cls):
        return cls(["pepperoni", "cheese", "tomato"])

    @classmethod
    def get_total_pizzas(cls):
        return cls.total_pizzas


pizza1 =  print(Pizza.margherita().ingredients)
pizza2 = print(Pizza.pepperoni().ingredients)

print(Pizza.get_total_pizzas())

# Section 5: Abstract Methods

print("=" *60)
print("Section 5: Abstract Methods")
print("=" *60)

from abc import ABC, abstractmethod

class Animal(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def make_sound(self):
        pass
    @abstractmethod
    def move(self):
        pass
    @abstractmethod
    def sleep(self):
        pass


class Dog(Animal):

    def sleep(self):
        print("zzzz")
    def move(self):
        print("dog is moving")
    def make_sound(self):
        print("whof whof")

it = Dog("it")

it.make_sound()

# overloading, overriding, final


# Section 6: Overloading

print("=" *60)
print("Section 6: Overloading")
print("=" *60)

from typing import overload, Union

class Calculator:
    @overload
    def add(self, a: int, b: int) -> int:
        ...

    @overload
    def add(self, a: int, b: int, c: int) -> int:
        ...

    def add(self, a: int, b: int, c: int | None = None)-> int:
        if c is None:
            return a+b
        if isinstance(c, int):
            return a+b+c
        else:
            raise ValueError("You can add 2 or 3 arguments at once!")

    @overload
    def process(self, value: int)-> int:
        ...

    @overload
    def process(self, value: str) -> str:
        ...

    def process(self, value: Union[int, str]) -> Union[int,str]:
        if isinstance(value, str):
            return value.upper()
        elif isinstance(value, int):
            return value * 2
        else:
            raise ValueError("Value must be an integer or a string")



calc = Calculator()

print(calc.add(10, 20, 30))

print(calc.process("messi"))
print(calc.process(10))


# Section 7: Final

print("=" *60)
print("Section 7: Final")
print("=" *60)

from typing import final

class BaseGame:

    def start(self):
        print("game started")

    @final
    def calculate_score(self, points: int) -> int:
        bonus = 100
        return points + bonus

    def end(self):
        print("game over")


class MyGame(BaseGame):
    def start(self):
        #override
        print("my game started")

    def calculate_score(self, points: int) -> int:
        return(points * 2)

myGame = MyGame()

myGame.start()

print(myGame.calculate_score(100))

@final
class SecretAlgorithm:
    def process(self):
        print("secret algorithm used")


# ide gives us a warning about final
class MySecondGame(SecretAlgorithm):
    pass


# Section 8: Override

print("=" *60)
print("Section 8: Override")
print("=" *60)

from typing import override

class Shape:

    def area(self) -> float:
        return 0.0
    def perimeter(self) -> float:
        return 0.0

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    @override
    def area(self) -> float:
        return self.width * self.height

    @override
    def perimeter(self) -> float:
        return (self.width + self.height) * 2


rectangle = Rectangle(5,4)

print(rectangle.area())
print(rectangle.perimeter())



# Section 9: Combining Decorators

print("=" *60)
print("Section 9: Combining Decorators")
print("=" *60)

def multiply_decorator(func):
    def wrapper(x: int):
        return func(x) * 2
    return wrapper

def other_decorator(func):
    def wrapper(x: int):
        return func(x) * 3
    return wrapper

@other_decorator
@multiply_decorator
def calculate(x: int):
    return x * 2

print(calculate(10))