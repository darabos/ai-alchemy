import ctranslate2
import fastapi
import huggingface_hub
import json
import os
import random
import starlette
import transformers
import urllib
import uuid
import websocket


def U(x):
    return {"role": "user", "content": x}


def A(x):
    return {"role": "assistant", "content": x}


model_cache = None


def load_model():
    global model_cache
    if model_cache is None:
        model_dir = huggingface_hub.snapshot_download(
            repo_id="Praise2112/Mistral-7B-Instruct-v0.1-int8-ct2"
        )
        generator = ctranslate2.Generator(model_dir, device="cuda", compute_type="int8")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1"
        )
        model_cache = generator, tokenizer
    return model_cache


def generate(messages, max_length):
    generator, tokenizer = load_model()
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = [
        tokenizer.convert_ids_to_tokens(model_input) for model_input in model_inputs
    ]
    generated_ids = generator.generate_batch(
        model_inputs,
        max_length=max_length,
        sampling_topk=10,
        end_token=".",
        include_prompt_in_result=False,
    )
    decoded = [res.sequences_ids[0] for res in generated_ids]
    decoded = tokenizer.batch_decode(decoded)[0]
    return decoded


def get_merge_result(a, b):
    example_merges = {
        ("Ice", "Fire"): "Water",
        ("Fire", "Water"): "Steam",
        ("Fire", "City"): "Fire station",
        ("Superman", "Batman"): "Superbatman",
        ("Human", "Stone"): "Dwarf",
    }
    messages = []
    for xa, xb in example_merges:
        messages.append(U(f"What do we get if we combine {xa} and {xb}?"))
        messages.append(A(f"{example_merges[xa, xb]}."))
    messages[0]["content"] = (
        "We are playing a game about merging things. " + messages[0]["content"]
    )
    messages.append(U(f"What do we get if we combine {a} and {b}?"))

    meh = set()
    not_new = set(merges.values())
    for i in range(10):
        g = generate(messages, max_length=20)
        if g.startswith("A "):
            g = g.removeprefix("A ").capitalize()
        if g.startswith("An "):
            g = g.removeprefix("An ").capitalize()
        if len(g) < 15 and g not in not_new:
            return g
        meh.add(g)
    return meh.pop() if meh else "Nothing"


def get_image_description(element):
    messages = [
        U('How would you depict "Life" on a card in one sentence?'),
        A("A radiant green heart."),
        U('How would you depict "Fire" on a card in one sentence?'),
        A("A flame."),
        U('How would you depict "Deadly poison" on a card in one sentence?'),
        A("A skull in a puddle of green liquid."),
        U(f'How would you depict "{element}" on a card in one sentence?'),
    ]
    return generate(messages, max_length=100)


class ComfyUI:
    def __init__(self):
        self.server = "http://127.0.0.1:8188/"
        self.client_id = str(uuid.uuid4())

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"{self.server}history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename, subfolder, folder_type):
        url_values = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"{self.server}view?{url_values}") as response:
            return response.read()

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(self.server + "prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_images(self, history):
        output_images = {}
        for o in history["outputs"]:
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    images_output = []
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                output_images[node_id] = images_output
        return output_images

    def generate_image(self, subject):
        print(f'generating image for "{subject}"')
        with open("comfyui_workflow.json") as f:
            prompt = json.loads(f.read())
        prompt["6"]["inputs"]["text"] = prompt["6"]["inputs"]["text"].replace(
            "SUBJECT", subject
        )
        prompt["13"]["inputs"]["noise_seed"] = random.randint(0, 1000000)
        ws = websocket.WebSocket()
        ws.connect(f"ws://localhost:8188/ws?clientId={self.client_id}")
        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        while True:
            out = ws.recv()
            history = self.get_history(prompt_id)
            if prompt_id in history:
                ws.close()
                images = self.get_images(history[prompt_id])
                history = self.get_history(prompt_id)[prompt_id]
                images = self.get_images(history)
                k = list(images.keys())[0]
                return images[k][0]


comfy = ComfyUI()
app = fastapi.FastAPI()

base_cards = ["Water", "Fire"]
merges = {}
unlocks = {
    "Steam": "Love",
    "Passion": "Life",
    "Explosion": "Motion",
    "Pressure": "Time",
    "Happiness": "Human",
    "Death": "Stone",
    "Firestorm": "Wasteland",
    "Fish": "Diamond",
    "God": "Magic",
}


@app.get("/")
def read_root():
    return starlette.responses.FileResponse("index.html")


@app.get("/vue.js")
def read_vue():
    return starlette.responses.FileResponse("vue.js")


@app.get("/info")
def get_info():
    # JSON can't deal with tuple keys.
    ms = {f"{a} + {b}": v for ((a, b), v) in merges.items()}
    return {"base_cards": base_cards, "merges": ms, "unlocks": unlocks}


@app.get("/merge")
def get_merge(a, b):
    [a, b] = sorted([a, b])
    if (a, b) not in merges:
        x = get_merge_result(a, b)
        print(f"Merged {a} and {b} to get {x}")
        if x in unlocks and unlocks[x] not in base_cards:
            base_cards.append(unlocks[x])
        merges[(a, b)] = x
    return {"merged": merges[(a, b)], 'base_cards': base_cards}


@app.post("/set_base")
def set_base(new_base_cards):
    global base_cards
    base_cards = new_base_cards
    return {"status": "ok"}


@app.post("/forget")
def forget(card):
    for k in list(merges.keys()):
        if card == merges[k]:
            del merges[k]
    return {"status": "ok"}


@app.post("/redraw")
def redraw(card):
    imagefile = f"images/{card}.png"
    if os.path.exists(imagefile):
        os.remove(imagefile)
    return {"status": "ok"}


@app.get("/image/{x}")
def image(x):
    os.makedirs("images", exist_ok=True)
    # TODO: Normalize and sanitize x.
    imagefile = f"images/{x}.png"
    if not os.path.exists(imagefile):
        description = get_image_description(x)
        imagedata = comfy.generate_image(description)
        with open(imagefile, "wb") as f:
            f.write(imagedata)
    return starlette.responses.FileResponse(imagefile)
