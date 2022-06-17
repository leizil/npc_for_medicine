import json

jsontext={}


def json_data(jsontext):
    jsonpath = '../cfgs/superparam.json'
    jsontext['data_dir']= '/mnt/llz/media/npcMri/cls/npc/'
    jsontext['csv_dir']='/mnt/llz/media/npcMri/cls/cfgs/dataInfo/'
    jsontext['n_splits'] = 3
    jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
    with open(jsonpath, 'w') as f:
        f.write(jsondata)

def open_json():
    jsonpath = '../cfgs/superparam.json'
    with open(jsonpath,'r') as f:
        datadict=json.load(f)
    return datadict

def test():
    json_data(jsontext,jsonpath)

if __name__ == '__main__':
    test()
