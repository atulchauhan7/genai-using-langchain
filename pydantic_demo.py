
from pydantic import BaseModel

class User(BaseModel):
    name: str = "Atul"

user = {}

new_user = User (**user)

print(new_user.name)
     