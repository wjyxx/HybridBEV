from mmcv.runner import load_checkpoint


def init_weights_teacher(self, path=None):
    """Load the pretrained model in teacher detector.

    Args:
        pretrained (str, optional): Path to pre-trained weights.
            Defaults to None.
    """
    checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')


def init_weights_student(self, path=None):
    checkpoint = load_checkpoint(self.student, path, map_location='cpu')
