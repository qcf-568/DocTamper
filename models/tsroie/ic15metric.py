from typing import Sequence, List
from typing import Dict, List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from shapely.geometry import Polygon
from collections import namedtuple
import numpy as np

class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        # print(len(gt))
        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
            if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
            if len(detDontCarePolsNum) > 0 else "\n")

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'hmean': hmean,
            'precision': precision,
            'recall': recall,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        # print(methodRecall, methodPrecision, methodHmean)
        # sys.exit(-1)
        methodMetrics = {
            'hmean': methodHmean,
            'precision': methodPrecision,
            'recall': methodRecall,
        }

        return methodMetrics

@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # set default_prefix

    def __init__(self,
             ann_file: Optional[str] = None,
             metric: Union[str, List[str]] = 'bbox',
             classwise: bool = False,
             proposal_nums: Sequence[int] = (100, 300, 1000),
             iou_thrs: Optional[Union[float, Sequence[float]]] = None,
             metric_items: Optional[Sequence[str]] = None,
             format_only: bool = False,
             outfile_prefix: Optional[str] = None,
             file_client_args: dict = None,
             backend_args: dict = None,
             collect_device: str = 'cpu',
             prefix: Optional[str] = None,
             sort_categories: bool = False) -> None:
             super().__init__(collect_device=collect_device, prefix=prefix)
             self.deteval = DetectionIoUEvaluator()
            # coco evaluation metrics
            # self.metrics = metric if isinstance(metric, list) else [metric]
            # allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
            # for metric in self.metrics:
            #     if metric not in allowed_metrics:
            #         raise KeyError(
            #             "metric should be one of 'bbox', 'segm', 'proposal', "
            #             f"'proposal_fast', but got {metric}.")

            # # do class wise evaluation, default False
            # self.classwise = classwise

            # # proposal_nums used to compute recall or precision.
            # self.proposal_nums = list(proposal_nums)

            # # iou_thrs used to compute recall or precision.
            # if iou_thrs is None:
            #     iou_thrs = np.linspace(
            #         .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            # self.iou_thrs = iou_thrs
            # self.metric_items = metric_items
            # self.format_only = format_only
            # if self.format_only:
            #     assert outfile_prefix is not None, 'outfile_prefix must be not'
            #     'None when format_only is True, otherwise the result files will'
            #     'be saved to a temp directory which will be cleaned up at the end.'

            # self.outfile_prefix = outfile_prefix

            # self.backend_args = backend_args
            # if file_client_args is not None:
            #     raise RuntimeError(
            #         'The `file_client_args` is deprecated, '
            #         'please use `backend_args` instead, please refer to'
            #         'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            #     )

            # # if ann_file is not specified,
            # # initialize coco api with the converted dataset
            # if ann_file is not None:
            #     with get_local_path(
            #             ann_file, backend_args=self.backend_args) as local_path:
            #         self._coco_api = COCO(local_path)
            #         if sort_categories:
            #             # 'categories' list in objects365_train.json and
            #             # objects365_val.json is inconsistent, need sort
            #             # list(or dict) before get cat_ids.
            #             cats = self._coco_api.cats
            #             sorted_cats = {i: cats[i] for i in sorted(cats)}
            #             self._coco_api.cats = sorted_cats
            #             categories = self._coco_api.dataset['categories']
            #             sorted_categories = sorted(
            #                 categories, key=lambda i: i['id'])
            #             self._coco_api.dataset['categories'] = sorted_categories
            # else:
            #     self._coco_api = None

            # # handle dataset lazy init
            # self.cat_ids = None
            # self.img_ids = None

    def np2ic15(self, ipt):
        results = []
        for (y1, x1, y2, x2) in ipt:
            results.append(
                {
                  'points': [(y1, x1), (y1, x2), (y2, x2), (y2, x1)],
                  'ignore':  False,
                }
            )
        return results

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # fetch classification prediction results and category labels
        # print(data_batch)
        for ds in data_samples:
            gt_box = self.np2ic15(ds['gt_instances']['bboxes'].cpu().numpy())
            pred_box = self.np2ic15(ds['pred_instances']['bboxes'].cpu().numpy())
            metric = self.deteval.evaluate_image(gt_box, pred_box)
            self.results.append(metric)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        metrics = self.deteval.combine_results(results)
        return metrics
