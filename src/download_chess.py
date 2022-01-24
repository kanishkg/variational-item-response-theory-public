import requests
import json
import random

from subprocess import Popen, PIPE, run


def download(url, filename):
    get_response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


with open("/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/chess/leela.json", "r") as f:
    data_nets = json.load(f)
random.shuffle(data_nets)

elo_ranges = [1650, 1250]
dlinks = []
for e in elo_ranges:
    for d in data_nets:
        if e <= float(d['Elo']) < e + 50:
            dlinks.append((d['download_link'], d['download_file'], d['Elo']))
            break

dlinks = [(l[:27] + l[28:], d, e) for l, d, e in dlinks]
root = "/mnt/fs6/kanishkg/lc0/weights/"
dest_files = []
for l, d, e in dlinks:
    DEST = f"/mnt/fs6/kanishkg/lc0/weights/{d[:-5]}_{int(float(e))}"
    dest_files.append(DEST)
    download(l, DEST)
    print(d, e)

for d in dest_files:
    command = ['lc0', 'describenet', f'--weights={d}']
    result = run(command, stdout=PIPE, stderr=PIPE, text=True)
    if result.returncode != 0:
        print(d)
        command = ['mv', '{f}', f'{d}.txt.gz']
        result = run(command, stdout=PIPE, stderr=PIPE, text=True)
        command = ['gunzip', f'{d}.txt.gz']
        result = run(command, stdout=PIPE, stderr=PIPE, text=True)
