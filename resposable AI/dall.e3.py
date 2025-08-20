from openai import OpenAI
import base64

client = OpenAI(api_key="TU_API_KEY_AQUI")

prompt = "Un retrato fotorrealista de una científica en un laboratorio, luz suave"

result = client.images.generate(
    model="gpt-image-1",  # Esto es DALL·E 3
    prompt=prompt,
    size="1024x1024"
)

# Guardar la imagen
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("dalle3_output.png", "wb") as f:
    f.write(image_bytes)

print("Imagen guardada como dalle3_output.png")
