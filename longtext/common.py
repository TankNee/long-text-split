from loguru import logger


def init_logger(args):
    logger.add(
        f"{args.output_dir}/train.log",
        rotation="10 MB",
    )
    logger.info("init logger successfully")


def launch_debugger():
    try:
        import debugpy

        logger.info("Waiting for debugger attach")
        debugpy.listen(5678)
        debugpy.wait_for_client()
        logger.info("Debugger attached")
    except Exception as e:
        logger.error(e)
