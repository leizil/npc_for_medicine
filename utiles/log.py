import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging
import shutil

base_path = '/mnt/llz/log'

class MyLog():
    def __init__(self,base_path='/mnt/llz/log',mode='cover'):
        self.base_path=base_path
        self.data = time.strftime("%Y-%m-%d")
        self.path=self.create_dir(mode)
        self.info_dir=self.get_model_info_dir()
        self.boardx_dir=self.get_boardx_dir()
        self.savemodel_dir=self.get_saveModel_dir()
        self.pred_dir=self.get_pred_dir()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)

        self.handler = logging.FileHandler(os.path.join(self.info_dir, 'info.txt'))
        self.handler.setLevel(logging.INFO)
        FMT = '%(asctime)s  %(message)s'
        DATEFMT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=FMT,datefmt=DATEFMT)
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
        elif mode=='cover':
            print(path, " has existed! would be deleted or create new")
            shutil.rmtree(path)
            os.mkdir(path)
            print("deleted its contents !!!")

        elif mode=='continue':
            print(path," has existed , mode is continue!")
            return path
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

    def get_pred_dir(self):
        pred_dir=os.path.join(self.path,'pred')
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
            print(pred_dir, ' has created!')
        return pred_dir

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
        # time_hour_minute = time.strftime("%Y-%m-%d, %H:%M:%S")
        # self.logger.info(time_hour_minute+'  info : '+'  '+str(info)+'\n')
        self.logger.info( '  info : ' + '  ' + str(info) + '\n')
    def write_info_notime(self,info):
        """

        :param type: metrics  epoch  predict
        :param info:
        :return:
        """
        # time_hour_minute = time.strftime("%H:%M  %S")
        self.logger.info(str(info)+'\n')

    def write_calc(self,calc:list):
        calc=[str(i) for i in calc]
        calc_str=" \n ".join(calc)
        # self.logger.info(calc_str+'\n')
        with open(os.path.join(self.get_pred_dir(),"val_score_calc.txt"),"a+") as f:
            f.write(calc_str)
            f.write("\n")

    def write_calc_info(self,info):
        with open(os.path.join(self.get_pred_dir(),"val_score_calc.txt"),"a+") as f:
            f.write(info)
            f.write("\n")

def test():
    base_path = '/mnt/llz/log'
    mylog=MyLog(base_path)
    mylog.write_info('hi')

if __name__ == '__main__':
    test()





