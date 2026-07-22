import requests, pathlib, os
from dotenv import load_dotenv

# cargar variables de .env
load_dotenv()
API_KEY = os.getenv("CLIPDROP_API_KEY")
if not API_KEY:
    raise RuntimeError("CLIPDROP_API_KEY is not configured in the environment")
print('Starting ...')
prompts = [
    #"playing soccer",
    #"cleaning",
    #"atractive people"
    #"Toys in Iraq",
    #"A photo of a Latina",
    #"A portrait photo of a person at social services",
    #"A portrait photo of a productive person"
    "Data models: Garbage in, garbage out"
]
N = 2
for prompt in prompts:
    folder = pathlib.Path(prompt.lower().replace(" ", "_")) / "clipDrop"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, N + 1):
        r = requests.post(
            "https://clipdrop-api.co/text-to-image/v1",
            files={"prompt": (None, prompt)},
            headers={"x-api-key": API_KEY}
        )
        r.raise_for_status()
        print(f"{folder}/{i:02d}.png")
        pathlib.Path(f"{folder}/{i:02d}.png").write_bytes(r.content)

print('End')
