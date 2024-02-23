from detectron2.config import CfgNode as CN

def add_custom_config(cfg: CN):
    _C = cfg
    _C.images_dir = CN()
    _C.images_dir.Name=""

    _C.MODEL_NAME=CN()
    _C.MODEL_NAME.Name=""

    _C.Teacher_Model=CN()
    _C.Teacher_Model.NAME=""

    _C.MODEL.Student=CN()
    _C.MODEL.Student.ROI_HEADS = "Student_ROIHead"

  

    _C.MTN=CN()
    _C.MTN.NAME=""

    _C.Student=CN()
    _C.Student.Resume="./output"
    _C.Student.save_interval=None
    _C.loss=CN()
    _C.loss.densepose=None
    _C.loss.cls = None
    _C.loss.box=None
    _C.loss.transfer=None



 
  


  