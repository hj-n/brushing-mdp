import json
import numpy as np
import math

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

import contour as Contour

app = Flask(__name__)
CORS(app)

## Global Variables -- Constant 
DENSITY = None
SIMILARITY = None
EMB = None
METADATA = None
CURRENT_PATH = None
MAX_EMB = None
MIN_EMB = None

## Global Variables -- Modified
CURRENT_EMB = None
CURRENT_SELECTION = None



## Helpers
def parseArgs(request):
    dataset = request.args.get('dataset', 'spheres') ## use spheres as default
    method  = request.args.get('method', 'umap')
    sample  = request.args.get('sample', '5')
    return dataset, method, sample

def checkPath(request, cpath):
    dataset, method, sample = parseArgs(request)
    path = dataset + "/" + method + "/" + sample
    if path != cpath:
        return False
    else:
        return True

@app.route('/')
def test():
    return "test"

@app.route('/init')
def init():
    global CURRENT_PATH
    global SIMILARITY
    global DENSITY
    global EMB
    global METADATA
    global MAX_EMB
    global MIN_EMB

    global CURRENT_EMB
    global CURRENT_SELECTION

    dataset, method, sample = parseArgs(request) 
    ## If no such dataset 
    path = dataset + "/" + method + "/" + sample
    if not Path("./json/" + path + "/snn_density.json").exists():
        return "failed", 400

    if (CURRENT_PATH == path):
        return "success", 200
    else:
        density_file = open("./json/" + path + "/snn_density.json")
        similarity_file = open("./json/" + path + "/snn_similarity.json")
        emb_file = open("./json/" + path + "/emb.json")
        current_emb_file = open("./json/" + path + "/emb.json")
        metadata_file = open("./json/" + path + "/metadata.json")
        DENSITY = json.load(density_file)
        SIMILARITY = json.load(similarity_file)
        EMB = json.load(emb_file)
        CURRENT_EMB = json.load(current_emb_file)
        METADATA = json.load(metadata_file)
        CURRENT_PATH = "path"
        CURRENT_SELECTION = set()

        np_emb = np.array(EMB)
        MAX_EMB = np.max(np_emb, axis = 0)
        MIN_EMB = np.min(np_emb, axis = 0)


    return "success", 200

@app.route('/basic')
def density():
    global CURRENT_PATH
    global DENSITY
    global EMB

    if checkPath(request, CURRENT_PATH):
        return "failed", 400

    return jsonify({"density": DENSITY, "emb": EMB})

@app.route('/similarity')
def similarity():
    global SIMILARITY
    index = request.args.get("index")
    list_similarity = SIMILARITY[int(index)]["similarity"]
    list_similarity[int(index)] = 0
    max_similarity = np.max(np.array(list_similarity))

    return {
        "similarity": list_similarity,
        "max": max_similarity
    }
    
@app.route('/pointlens')
def pointLens():
    global SIMILARITY
    global EMB
    global METADATA
    global CURRENT_EMB

    index  = request.args.get("index")
    radius = float(request.args.get("radius"))
    list_similarity = SIMILARITY[int(index)]["similarity"]
    max_similarity = METADATA["max_snn_similarity"]

    ## naive implementation for the filtering (should be parallelized)
    modified_emb = []
    center_coor = EMB[int(index)]
    for i, coor in enumerate(EMB):
        if i == int(index):
            modified_emb.append(coor)
            continue

        
        distance = math.dist(center_coor, coor)
        if (distance > radius):
            if (list_similarity[i] / max_similarity) < 0.5:
                modified_emb.append(coor)
                continue

    
        direction = np.array(coor) - np.array(center_coor)
        direction = direction / np.linalg.norm(direction)
        direction = direction * radius * (1 - (list_similarity[i] / max_similarity))
        
        new_coor = direction + np.array(center_coor)
        modified_emb.append(new_coor.tolist())

    CURRENT_EMB = modified_emb
    return jsonify(modified_emb)

@app.route('/brushing')
def brushing_status():
    global CURRENT_EMB
    global CURRENT_SELECTION
    global MAX_EMB
    global MIN_EMB

    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    radius = float(request.args.get("r"))
    
    current_length = len(CURRENT_SELECTION)
    ## should be accelerated by parallelization afterward
    for i, coor in enumerate(CURRENT_EMB):
        direction = np.array([coor[0] - x, coor[1] - y])
        distance = np.linalg.norm(direction)
        if (distance < radius) and (i not in CURRENT_SELECTION):
            CURRENT_SELECTION.add(i)

    if(current_length != len(CURRENT_SELECTION)):
        selection_to_list = [0] * len(CURRENT_EMB)
        for idx in CURRENT_SELECTION:
            selection_to_list[idx] = 1
        
        contours = Contour.generate(CURRENT_SELECTION, CURRENT_EMB, MAX_EMB, MIN_EMB)

        print(contours)
        return jsonify({
            "changed": True,
            "contour": contours,
            "selection": selection_to_list
        })
    else:
        return jsonify({
            "changed": False
        })

@app.route('/brushingfinish')
def brushing_finish():
    global CURRENT_SELECTION
    CURRENT_SELECTION = set()
    return "return"


if __name__ == '__main__':
    app.run(host='0.0.0.0')