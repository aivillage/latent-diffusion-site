import json
from datetime import datetime
from flask import Flask, render_template, request, url_for, flash, redirect
import os, uuid
import sqlite3
from contextlib import closing
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = "aa57fa721a88635bfc71e1e6a59f9f5ff6117dfb8edc47f9"
images = []

if not os.path.exists("static/files/"):
    os.mkdir("static/files/")

class LdmGenerator():
    def __init__(self, images_path: str, model_path: str, cfg_path: str) -> None:
        print(f"Loading config from {cfg_path}")
        config = OmegaConf.load(cfg_path)  # TODO: Optionally download from same location as ckpt and chnage this logic        
        print(f"Loading model from {model_path}")
        pl_sd = torch.load(model_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
        print("Moving model to GPU")
        
        model = model.half().cuda()
        model.eval()
        self.model = model
        self.plms_sampler = PLMSSampler(model)
        self.ddim_sampler = DDIMSampler(model)
        self.images_path = images_path

    def create_image(self, parameters):
        file_name = str(uuid.uuid1())
        sample_path = os.path.join(self.images_path, file_name)
        base_count = 0
        parameters["path_root"] = sample_path
        if parameters["plms"]:
            sampler = self.plms_sampler
        else:
            sampler = self.ddim_sampler
        all_samples = list()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with self.model.ema_scope():
                    uc = None
                    if parameters["scale"] != 1.0:
                        uc = self.model.get_learned_conditioning(parameters["n_samples"] * [""])
                    for n in trange(parameters["n_iter"], desc="Sampling"):
                        c = self.model.get_learned_conditioning(parameters["n_samples"] * [parameters["prompt"]])
                        shape = [4, parameters["height"]//8, parameters["width"]//8]
                        samples_ddim, _ = sampler.sample(S=parameters["ddim_steps"],
                                                        conditioning=c,
                                                        batch_size=parameters["n_samples"],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=parameters["scale"],
                                                        unconditional_conditioning=uc,
                                                        eta=parameters["ddim_eta"])

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            image_path = f"{sample_path}_{base_count:04}.png"
                            Image.fromarray(x_sample.astype(np.uint8)).save(image_path)
                            base_count += 1
                        all_samples.append(x_samples_ddim)
        parameters["image_count"] = base_count

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid)
        
        # to image  
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(f"{sample_path}.png")
        parameters["file_name"] = f"{file_name}.png"
        parameters["uuid"] = file_name
        return parameters

generator = LdmGenerator("static/", "models/ldm/text2img-large/model.ckpt", "models/ldm/text2img-large/txt2img-1p4B-eval.yaml")

print(f"Loading database")
with closing(sqlite3.connect("static/images.db")) as connection:
    with closing(connection.cursor()) as cursor:
        if len(cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';").fetchall()) == 0:
            cursor.execute("CREATE TABLE images (prompt TEXT, file_name TEXT, ddim_steps INTEGER, plms BOOLEAN, ddim_eta FLOAT, n_iter INTEGER, height INTEGER, width INTEGER, scale FLOAT)")

@app.route('/')
def index():
    images = []
    for file in os.listdir("static/files/"):
        with open(os.path.join("static/files/", file)) as file:
            image = json.load(file)
            image["file_name"] = f"{image['uuid']}.png"
            images.append(image)
    images = sorted(images, key=lambda x: x["timestamp"], reverse=True)
    return render_template('index.html', images=images)

@app.route('/create/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        parameters = {}

        parameters['prompt'] = request.form['prompt']
        try:
            parameters['ddim_steps'] = int(request.form['ddim_steps'])
        except:
            flash('ddim_steps needs to be an integer')
        try:
            parameters['ddim_eta'] = float(request.form['ddim_eta'])
        except:
            flash('ddim_eta needs to be an float')
        try:
            parameters['n_iter'] = int(request.form['n_iter'])
        except:
            flash('n_iter needs to be an integer')
        try:
            parameters['n_samples'] = int(request.form['n_samples'])
        except:
            flash('n_samples needs to be an integer')
        try:
            parameters['height'] = int(request.form['height'])
        except:
            flash('height needs to be an integer')
        try:
            parameters['width'] = int(request.form['width'])
        except:
            flash('width needs to be an integer')
        try:
            parameters['scale'] = float(request.form['scale'])
        except:
            flash('scale needs to be an float')

        if request.form.get('plms') is not None:
            parameters['plms'] = True
        else:
            parameters['plms'] = False

        if not parameters["prompt"]:
            flash('Title is required!')
        else:
            parameters = generator.create_image(parameters)
            parameters["timestamp"] = datetime.now().timestamp()
            with open(f"static/files/{parameters['uuid']}.json", "w+") as file:
                json.dump(parameters, file)
            return redirect(url_for('index'))

    return render_template('create.html')