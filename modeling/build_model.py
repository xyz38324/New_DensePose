from detectron2.utils.registry import Registry
import torch

Model_REGISTRY=Registry("WiFi_DensePose")
Model_REGISTRY.__doc__=""" """

Teacher_Model_REGISTRY=Registry("Teacher_Model")
Teacher_Model_REGISTRY.__doc__=""" """

ModalityTranslationNetwork_REGISTRY=Registry("ModalityTranslationNetwork")
ModalityTranslationNetwork_REGISTRY.__doc__=""" """

Student_ROIHEAD_REGISTRY=Registry("Student_ROIOHead")
Student_ROIHEAD_REGISTRY.__doc__=""" """

Teacher_ROIHEAD_REGISTRY=Registry("Student_ROIOHead")
Teacher_ROIHEAD_REGISTRY.__doc__=""" """

def build_model(cfg):
    name = cfg.MODEL_NAME.Name
    model = Model_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
  
    return model

def build_teacher_model(cfg):
    name = cfg.Teacher_Model.NAME
    model = Teacher_Model_REGISTRY.get(name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
  
    return model


def build_student_roihead(cfg, input_shape):
    name = cfg.MODEL.Student.ROI_HEADS
    model = Student_ROIHEAD_REGISTRY.get(name)(cfg, input_shape)
    model.to(torch.device(cfg.MODEL.DEVICE))

    return model
def build_teacher_roihead(cfg, input_shape):
    name = cfg.MODEL.Teacher.ROI_HEADS
    model = Teacher_ROIHEAD_REGISTRY.get(name)(cfg, input_shape)
    model.to(torch.device(cfg.MODEL.DEVICE))

    return model


def build_mtn(cfg):
    
    name = cfg.MTN.NAME
    

    model  = ModalityTranslationNetwork_REGISTRY.get(name)(cfg)
    
    return model
    