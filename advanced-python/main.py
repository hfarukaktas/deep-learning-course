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






