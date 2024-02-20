from detectron2.config import CfgNode as CN

def add_custom_config(cfg: CN):
    _C = cfg
    _C.images_dir = CN()
    _C.images_dir.Name=""

    _C.MODEL_NAME=CN()
    _C.MODEL_NAME.Name=""

    _C.Teacher_Model=CN()
    _C.Teacher_Model.NAME=""

    _C.MTN=CN()
    _C.MTN.NAME=""

    _C.LOSS=CN()
    _C.LOSS.box=1.0
    _C.LOSS.dp=0.6
    _C.LOSS.tr=0.00001
    _C.LOSS.cls = 1.2


 
  


  