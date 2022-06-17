import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging
import shutil

base_path = '/mnt/llz/log'

class MyLog():
    def __init__(self,base_path='/mnt/llz/log'):
        self.base_path=base_path
        self.data = time.strftime("%Y-%m-%d")
        self.path=self.create_dir()
        self.info_dir=self.get_model_info_dir()
        self.boardx_dir=self.get_boardx_dir()
        self.savemodel_dir=self.get_saveModel_dir()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

        self.handler = logging.FileHandler(os.path.join(self.info_dir, 'info.txt'))
        self.handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.console = logging.StreamHandler()
        self.console.setFormatter(formatter)
        self.logger.addHandler(self.console)


    def create_dir(self,mode='cover'):
        path = os.path.join(self.base_path, self.data)
        if not os.path.exists(path):
            os.mkdir(path)
            print(path, ' has created!')
        else:
            print(path, " has existed! would be deleted or create new")
            if mode =='cover':
                shutil.rmtree(path)
                os.mkdir(path)
                print("deleted its contents !!!")
            else:
                path=os.path.join(path,data+'_1')
        return path

    def get_boardx_dir(self):
        boardx_dir=os.path.join(self.path,'tensorboardx')
        if not os.path.exists(boardx_dir):
            os.mkdir(boardx_dir)
            print(boardx_dir, ' has created!')
        return boardx_dir

    def get_saveModel_dir(self):
        save_model_dir=os.path.join(self.path,'save_models')
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
            print(save_model_dir, ' has created!')
        return save_model_dir

    def get_model_info_dir(self):
        info_dir=os.path.join(self.path,'model_info')
        # print('in info func')
        if not os.path.exists(info_dir):
            os.mkdir(info_dir)
            print(info_dir, ' has created!')

        return info_dir

    def write_info(self,info):
        """

        :param type: metrics  epoch  predict
        :param info:
        :return:
        """
        time_hour_minute = time.strftime("%H:%M  %S")
        self.logger.info(time_hour_minute+': '+str(info)+'\n')


        # if type == 'loss':
        #     info = time_hour_minute + "  " + info
        #     loss_log = os.path.join(path, 'loss.txt')
        #     with open(loss_log, 'a+') as fl:
        #         fl.write(info)


def test():
    base_path = '/mnt/llz/log'
    mylog=MyLog(base_path)
    mylog.write_info('hi')

if __name__ == '__main__':
    test()





