from detectron2.engine import DefaultTrainer
from detectron2.engine import SimpleTrainer
import time
from modeling.build_model import build_model,build_teacher_model
from dataloader.customerdataloader import CustomMMFIDataset,custom_collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader
import weakref,torch,os
from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        teacher_model = MyTrainer.build_teachermodel(cfg)
        teacher_model.eval()
        checkpointer = DetectionCheckpointer(teacher_model)
        checkpointer.load(cfg.MODEL.WEIGHTS)   
        
        # cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)     
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )


        if os.path.exists(cfg.MODEL.WEIGHTS):
            self.checkpointer.load(cfg.Student.Resume)
            print("Loaded weights from {}".format(cfg.Student.Resume))
        else:
            print("Weights file {} not found, using random initialization.".format(cfg.Student.Resume))

        

        self._trainer = CustomTrainer(cfg, model, data_loader, optimizer,teacher_model=teacher_model,
                                      checkpointer=self.checkpointer)  

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
    def __init__(self,cfg, model,data_loader, optimizer,teacher_model,checkpointer):
        super().__init__(model, data_loader, optimizer)
        self.loss_cls = cfg.loss.cls
        self.loss_box = cfg.loss.box
        self.loss_transfer = cfg.loss.transfer
        self.loss_densepose = cfg.loss.densepose
        self.teacher_model = teacher_model
        self.checkpointer = checkpointer
        self.save_interval = cfg.Student.save_interval
        self.transfer_only =cfg.transfer.only
    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with torch.no_grad():
            results,features=self.teacher_model(data)
    
        loss_dict = self.model(data,results,features)
        if self.transfer_only:
            losses = self.loss_transfer * loss_dict['loss_transfer']
        else:
            losses = self.loss_box * loss_dict['loss_box'] + self.loss_cls * loss_dict['loss_cls'] +self.loss_densepose * loss_dict['loss_densepose']+ self.loss_transfer * loss_dict['loss_transfer']


        print(self.iter)
        print(f"losses = {losses}")
        if self.iter % self.save_interval==0 and self.iter>0:
            self.checkpointer.save(f"checkpoint_{self.iter}")


        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        losses.backward()

    
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
 
    