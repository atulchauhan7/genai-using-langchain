from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    occupation: str

new_person: Person = {
    "name": "Atul",
    "age": 24,
    "occupation": "Software Developer"
}

print(new_person)