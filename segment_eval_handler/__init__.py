def get_all_methods():
    from .iou import IOU
    from .dummy import DummySegmentEval
    return [IOU, DummySegmentEval]
