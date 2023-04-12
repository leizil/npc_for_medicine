import json


json_path='../cfgs/superparam.json'

def json_data():
    jsontext = {}
    jsonpath = '/mnt/llz/code/cls/mymodel/cfgs/superparam.json'
    jsontext['data_dir']= '/mnt/llz/dataset/Project_npc/work_dir/t2/train'
    jsontext['csv_dir']='/mnt/llz/dataset/Project_npc/work_dir/t2/info'
    jsontext['n_splits'] = 3
    jsontext['num_epochs']=50
    jsontext['fold']=0
    jsontext['mode']='cover'
    jsontext['test_mode'] = 'continue'
    # jsontext['resnet-layers']=10
    jsontext['lr']=1e-4
    jsontext['batch_size']=64
    jsontext['cls']=2
    jsontext['num_workers']=8
    jsontext['test_path']='/mnt/llz/dataset/Project_npc/work_dir/t2/test'
    # print('/mnt/llz/media/imagenet/ILSVRC2012/')



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
