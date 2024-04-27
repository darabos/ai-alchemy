import ctranslate2
import fastapi
import huggingface_hub
import json
import os
import random
import re
import sqlite3
import starlette
import transformers
import urllib
import uuid
import websocket
import yaml


def U(x):
    return {"role": "user", "content": x}


def A(x):
    return {"role": "assistant", "content": x}


model_cache = None


def load_model():
    global model_cache
    if model_cache is None:
        model_dir = huggingface_hub.snapshot_download(
            repo_id="jncraton/Mistral-7B-Instruct-v0.2-ct2-int8"
        )
        generator = ctranslate2.Generator(model_dir, device="cuda", compute_type="int8")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "jncraton/Mistral-7B-Instruct-v0.2-ct2-int8"
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


def get_merge_result(cfg, a, b):
    db = get_db(cfg)
    m = cfg["text"]["merge"]
    messages = []
    for xa, xb, xc in m["examples"]:
        messages.append(U(m["prompt"].replace("CARD1", xa).replace("CARD2", xb)))
        messages.append(A(f"{xc}."))
    messages[0]["content"] = m["prefix"] + " " + messages[0]["content"]
    messages.append(U(m["prompt"].replace("CARD1", a).replace("CARD2", b)))
    for x in messages:
        print(x)
    meh = set()
    not_new = set(get_merges(db).values())
    for i in range(10):
        g = generate(messages, max_length=20)
        if g.startswith("A "):
            g = g.removeprefix("A ").capitalize()
        if g.startswith("An "):
            g = g.removeprefix("An ").capitalize()
        if len(g) < 15 and g not in not_new and " and " not in g.lower():
            return g
        meh.add(g)
    return sorted(meh, key=lambda x: len(x))[0] if meh else "Nothing"


def get_image_description(cfg, element):
    messages = [
        U(p.replace("ELEMENT", element)) if i % 2 == 0 else A(p)
        for i, p in enumerate(cfg["text"]["art"]["prompt"])
    ]
    return generate(messages, max_length=100)


class ComfyUI:
    def __init__(self):
        self.server = "http://127.0.0.1:8188/"

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"{self.server}history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename, subfolder, folder_type):
        url_values = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"{self.server}view?{url_values}") as response:
            return response.read()

    def queue_prompt(self, prompt, client_id):
        p = {"prompt": prompt, "client_id": client_id}
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

    def generate_image(self, cfg, subject):
        print(f'generating image for "{subject}"')
        with open("comfyui_workflow.json") as f:
            prompt = json.loads(f.read())
        client_id = str(uuid.uuid4())
        prompt["6"]["inputs"]["text"] = cfg["art"]["prompt"].replace("SUBJECT", subject)
        prompt["7"]["inputs"]["text"] = cfg["art"]["negative"]
        prompt["13"]["inputs"]["noise_seed"] = random.randint(0, 1000000)
        ws = websocket.WebSocket()
        ws.connect(f"ws://localhost:8188/ws?clientId={client_id}")
        prompt_id = self.queue_prompt(prompt, client_id)["prompt_id"]
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


def open_db(variant_name):
    os.makedirs("db", exist_ok=True)
    # If this is thread safe, that is by accident.
    db = sqlite3.connect(f"db/{variant_name}.db", check_same_thread=False)
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS merges(a, b, makes)")
    db.commit()
    return db


def get_config(variant):
    with open(f"configs/{variant}.yml") as f:
        cfg = yaml.safe_load(f)
        cfg["variant_name"] = variant
        return cfg


comfy = ComfyUI()
app = fastapi.FastAPI()
load_model()  # Avoid accidentally loading it twice.
dbs = {}


def get_db(cfg):
    if cfg["variant_name"] not in dbs:
        dbs[cfg["variant_name"]] = open_db(cfg["variant_name"])
    return dbs[cfg["variant_name"]]


def get_merges(db):
    cur = db.cursor()
    cur.execute("SELECT a, b, makes FROM merges")
    return {(a, b): makes for (a, b, makes) in cur.fetchall()}


@app.get("/")
def read_root():
    return starlette.responses.FileResponse("index.html")


@app.get("/vue.js")
def read_vue():
    return starlette.responses.FileResponse("vue.js")


@app.get("/info")
def get_info():
    return {
        "variants": [variant.removesuffix(".yml") for variant in os.listdir("configs")]
    }


@app.get("/{variant}/info")
def get_variant_info(variant):
    cfg = get_config(variant)
    db = get_db(cfg)
    # JSON can't deal with tuple keys.
    merges = {f"{a} + {b}": v for ((a, b), v) in get_merges(db).items()}
    return {
        "base_cards": cfg["game"]["base_cards"],
        "unlocks": cfg["game"]["unlocks"],
        "merges": merges,
    }


@app.get("/{variant}/merge")
def get_merge(variant, a, b, store=False):
    cfg = get_config(variant)
    db = get_db(cfg)
    [a, b] = sorted([a, b])
    cur = db.cursor()
    merged = cur.execute(
        "SELECT makes FROM merges WHERE a = ? AND b = ?", (a, b)
    ).fetchone()
    if merged:
        return {"merged": merged[0]}
    merged = get_merge_result(cfg, a, b)
    print(f"Merged {a} and {b} to get {merged}")
    if store:
        cur.execute("INSERT INTO merges VALUES (?, ?, ?)", (a, b, merged))
        db.commit()
    return {"merged": merged}


@app.post("/{variant}/forget")
def forget(variant, a, b, card):
    cfg = get_config(variant)
    db = get_db(cfg)
    [a, b] = sorted([a, b])
    cur = db.cursor()
    cur.execute("DELETE FROM merges WHERE a = ? AND b = ? AND makes = ?", (a, b, card))
    db.commit()
    return {"status": "ok"}


@app.post("/{variant}/redraw")
def redraw(variant, card):
    cfg = get_config(variant)
    imagefile = get_imagefile(cfg, card)
    if os.path.exists(imagefile):
        os.remove(imagefile)
    return {"status": "ok"}


def get_imagefile(cfg, card):
    imgdir = "images/" + cfg["variant_name"]
    os.makedirs(imgdir, exist_ok=True)
    card = re.sub(r"\W", "_", card.lower())
    imagefile = f"{imgdir}/{card}.png"
    return imagefile


@app.get("/{variant}/image/{card}")
def image(variant, card):
    cfg = get_config(variant)
    imagefile = get_imagefile(cfg, card)
    if not os.path.exists(imagefile):
        description = get_image_description(cfg, card)
        imagedata = comfy.generate_image(cfg, f"{card}: {description}")
        with open(imagefile, "wb") as f:
            f.write(imagedata)
    return starlette.responses.FileResponse(imagefile)
