from .segment_eval_handler import SegmentEvalHandler


class DummySegmentEval(SegmentEvalHandler):
    def __call__(self, prompt):
        prompt.segmentation_evaluation = [-1] * prompt.segmentation_count
        return prompt
