from detectron2.engine import DefaultTrainer
from detectron2.engine import SimpleTrainer
import time
from modeling.build_model import build_model,build_teacher_model
from dataloader.customerdataloader import CustomMMFIDataset,custom_collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader
import weakref
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        teacher_model = MyTrainer.build_teachermodel(cfg)
        teacher_model.eval()
        checkpointer = DetectionCheckpointer(teacher_model)
        checkpointer.load(cfg.MODEL.WEIGHTS)   
        
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        self._trainer = CustomTrainer(cfg, model, data_loader, optimizer,teacher_model=teacher_model)       
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_teachermodel(cls, cfg):
        
        model = build_teacher_model(cfg)

        return model

    @classmethod
    def build_model(cls, cfg):
      
        model = build_model(cfg)
        return model
    
    @classmethod
    def build_train_loader(cls,cfg):
        dataset = CustomMMFIDataset(cfg=cfg,transform=transforms.ToTensor())
       
        data_loader = DataLoader(dataset,shuffle=True, collate_fn=custom_collate_fn,num_workers=cfg.DATALOADER.NUM_WORKERS,batch_size=cfg.SOLVER.IMS_PER_BATCH)
                
        return data_loader
        
    




class CustomTrainer(SimpleTrainer):
    def __init__(self,cfg, model,data_loader, optimizer,teacher_model):
        super().__init__(model, data_loader, optimizer)
        self.loss_cls = cfg.LOSS.cls
        self.loss_box_reg = cfg.LOSS.box
    
        self.loss_transfer = cfg.LOSS.tr
        self.loss_densepose = cfg.LOSS.dp
        self.teacher_model = teacher_model

    def run_step(self):

    
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:

            self.optimizer.zero_grad()

        results=self.teacher_model(data)
        loss_dict = self.model(data)
        
      
        
        losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

       
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj
 
    