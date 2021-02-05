import torch
import xzh_mediaunit as mu
import numpy as np
import logging
import queue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resizeFrame(inputQ, outputQ, config):
    calcDepthSize = config.calc_depth_size
    outputMonoSize = config.output_mono_size
    crop = config.crop
    calcDepthRszFilter = mu.FilterGraph(
        mu.MediaType.VIDEO,
        f"scale={calcDepthSize[0]}:{calcDepthSize[1]},format=rgb24"
    )
    outputRszFilter = mu.FilterGraph(
        mu.MediaType.VIDEO,
        f"scale={outputMonoSize[0]}:{outputMonoSize[1]},format=rgb24"
    )

    while True:
        try:
            qElem = inputQ.get(timeout=0.1)
        except TimeoutError:
            continue
        except queue.Empty:
            continue
        if qElem == 'EOF':
            break
        try:
            decOutFrm = qElem['original_inpaint']
            if outputMonoSize is not None and decOutFrm.width != outputMonoSize[0] or decOutFrm.height != \
                    outputMonoSize[1]:
                if not outputRszFilter.is_inited:
                    outputRszFilter.initVideoFilterGraphByFrame(decOutFrm)
                monoFrm = outputRszFilter.writeFrame(decOutFrm)
            else:
                monoFrm = mu.MediaFrame.clone(decOutFrm)
                if monoFrm.format != mu.PixelFormat.RGB24:
                    monoFrm.toPixelFormat(mu.PixelFormat.RGB24,
                                          cscSliceCount=8)
                # end
            # end

            if crop is not None:
                top, height, left, width = crop
                height_ratio = monoFrm.height / decOutFrm.height
                width_ratio = monoFrm.width / decOutFrm.width
                top_new, height_new, left_new, width_new = int(
                    top * height_ratio), int(height * height_ratio), int(
                    left * width_ratio), int(width * width_ratio)
                monoFrm = np.asarray(monoFrm)
                monoFrm = monoFrm[top_new:top_new + height_new, left_new:left_new + width_new]
                monoFrm = np.ascontiguousarray(monoFrm)
                monoFrm = mu.MediaFrame.fromRgbArray(monoFrm, mu.PixelFormat.RGB24)

            qElem['monoFrm'] = monoFrm
            # logger.info(f'monoFrm:{monoFrm.width}:{monoFrm.height}')

            decOutFrm = monoFrm

            if calcDepthSize is not None and decOutFrm.width != calcDepthSize[
                0] or decOutFrm.height != calcDepthSize[1]:
                if not calcDepthRszFilter.is_inited:
                    calcDepthRszFilter.initVideoFilterGraphByFrame(
                        decOutFrm)
                calcDepthFrm = calcDepthRszFilter.writeFrame(decOutFrm)
            else:
                calcDepthFrm = mu.MediaFrame.clone(decOutFrm)
                if calcDepthFrm.format != mu.PixelFormat.RGB24:
                    calcDepthFrm.toPixelFormat(mu.PixelFormat.RGB24,
                                               cscSliceCount=8)
                # end
            # end
            calcDepthTen = torch.from_numpy(np.asarray(calcDepthFrm))
            qElem['calcDepthTen'] = calcDepthTen
            logger.info(f'Resize frame {qElem["index"]}, pt={monoFrm.present_time}')
            outputQ.put(qElem)
        except Exception:
            import traceback
            traceback.print_exc()
    # end
    outputQ.put('EOF')
    logger.info('resize Done')

# end
