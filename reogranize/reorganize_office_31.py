import os
import shutil
dataroot_amazon = "..//office-31//amazon//images//"
dataroot_dslr = "..//office-31//dslr//images//"
dataroot_webcam = "..//office-31//webcam//images//"

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

render(dataroot_amazon)
render(dataroot_dslr)
render(dataroot_webcam)
