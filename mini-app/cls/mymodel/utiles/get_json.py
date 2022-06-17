import json


json_path='../cfgs/superparam.json'

def json_data():
    jsontext = {}
    jsonpath = '/mnt/llz/code/cls/mymodel/cfgs/superparam.json'
    jsontext['data_dir']= '/mnt/llz/media/npcMri/cls/npc/'
    jsontext['csv_dir']='/mnt/llz/media/npcMri/cls/cfgs/dataInfo/'
    jsontext['n_splits'] = 3
    jsontext['num_epochs']=10
    jsontext['fold']=0
    jsontext['resnet-layers']=10
    jsontext['lr']=1e-3



    jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
    with open(jsonpath, 'w') as f:
        f.write(jsondata)

def open_json():
    jsonpath = '/mnt/llz/code/cls/mymodel/cfgs/superparam.json'
    with open(jsonpath,'r') as f:
        datadict=json.load(f)
    return datadict

def test():
    json_data()

if __name__ == '__main__':
    test()
