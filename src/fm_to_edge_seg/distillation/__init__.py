"""Teacher cache and knowledge-distillation utilities."""

from fm_to_edge_seg.distillation.cache import (
    TeacherCache,
    TeacherSample,
    create_reference_teacher_cache,
    write_teacher_cache,
)
from fm_to_edge_seg.distillation.sam3_adapter import (
    Sam3TextTeacher,
    create_sam3_teacher_cache,
    merge_sam3_instances,
)

__all__ = [
    "TeacherCache",
    "TeacherSample",
    "create_reference_teacher_cache",
    "write_teacher_cache",
    "Sam3TextTeacher",
    "create_sam3_teacher_cache",
    "merge_sam3_instances",
]
