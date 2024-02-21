from engine.mytrainer import MyTrainer
from datetime import timedelta
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import DEFAULT_TIMEOUT, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from engine.add_config import add_custom_config
from detectron2.checkpoint import DetectionCheckpointer
def setup(args):
    """
    Create a configuration object from args here.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_custom_config(cfg)
    
    cfg.merge_from_file("configs/wifi_densepose.yaml")  

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="densepose")
    return cfg


def main(args):
    cfg = setup(args)
    PathManager.set_strict_kwargs_checking(False)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    timeout = (
        DEFAULT_TIMEOUT if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE else timedelta(hours=4)
    )
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timeout,
    )
