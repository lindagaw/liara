import os
import shutil
dataroot = "..//datasets//gender_dataset//Training//"

def render(dataroot):
    for category in os.listdir(dataroot):
        try:
            shutil.rmtree(os.path.join(dataroot, category, 'content'))
        except:
            pass

        try:
            os.mkdir(os.path.join(dataroot, category, 'content'))
        except:
            pass

        for image in os.listdir(os.path.join(dataroot, category)):
            try:
                os.rename(os.path.join(dataroot, category, image),
                        os.path.join(dataroot, category, 'content', image))
            except:
                pass

render(dataroot)
