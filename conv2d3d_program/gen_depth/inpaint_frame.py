from components.binocular_calculator import BinocularCalculator
import numpy as np
import timeit as tt
import logging
import queue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def inpaintFrame(inputQ: queue, outputQ: queue, threads=1):
    binoCalculator = BinocularCalculator(threads)
    # end
    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        if qElem == 'EOF':
            break
        t0 = tt.default_timer()
        monoFrm = qElem['monoFrm']
        depth_for_inpaint = qElem['depth_for_inpaint']
        npRefFrm = np.ascontiguousarray(monoFrm)
        depth_for_inpaint = np.ascontiguousarray(depth_for_inpaint)
        # pts  = qElem['present_time']
        # import cv2
        # import os 
        # path = f'/home/zhaohoj/Videos/out/{pts}'
        # if  not os.path.exists(path):
            # os.makedirs(path)
        # cv2.imwrite(os.path.join(path,'input_original.png'),cv2.cvtColor(npRefFrm,cv2.COLOR_RGB2BGR))
        # cv2.imwrite(os.path.join(path,'depth_for_inpaint.png',depth_for_inpaint))
        t1 = tt.default_timer()
        binoCalculator.newTask(qElem, npRefFrm, depth_for_inpaint, occlusionWeightThresh=1.0, interpRatio=3.0,
                               outputMode=0)
        finishedTasks = binoCalculator.getFinishedTasks()
        t2 = tt.default_timer()
        for finished in finishedTasks:
            # import cv2
            # import os 
            # pts  = finished['present_time']
            # frmLR =finished['frmLR']
            # frmLR = np.asarray(frmLR)    
            # path = f'/home/zhaohoj/Videos/out/{pts}'
            # if  not os.path.exists(path):
            #     os.makedirs(path)
            # cv2.imwrite(os.path.join(path,'frmLR.png'),cv2.cvtColor(frmLR,cv2.COLOR_RGB2BGR))
            outputQ.put(finished)
        # logger.info(f'Inpaint frame: {qElem["index"]}, finished={len(finishedTasks)}, t01={(t1 - t0) * 1000:.0f}, t12={(t2 - t1) * 1000:.0f}')
    # end

    logger.info(f'Wait all inpaint task done...')
    binoCalculator.waitAllTaskDoneAndExit()
    finishedTasks = binoCalculator.getFinishedTasks()
    for finished in finishedTasks:
        outputQ.put(finished)
    logger.info(f'Finally finished tasks {len(finishedTasks)}')
    outputQ.put('EOF')