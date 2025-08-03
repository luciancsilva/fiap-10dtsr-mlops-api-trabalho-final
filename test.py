import json
import os
import src.app as app

with open("data.json", "r") as file:
    data = file.read()

event = json.loads(data)

print(event)

retorno = app.handler(event, context=False)

print(retorno)