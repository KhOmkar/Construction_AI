"""
Rule-based auto-caption generator for scenes and worker crops.
Captions are stored in manifest.csv and alongside crop metadata.
"""

def generate_scene_caption(box_classes):
    """
    Generate descriptive caption for a scene given list/set of class IDs.
    Classes: 0: helmet, 1: no_helmet, 2: vest, 3: no_vest, 4: person
    """
    if isinstance(box_classes, str):
        if not box_classes or box_classes.strip() == "":
            classes = set()
        else:
            classes = set(int(x.strip()) for x in box_classes.split(",") if x.strip().isdigit())
    elif isinstance(box_classes, (list, tuple, set)):
        classes = set(box_classes)
    else:
        classes = set()

    has_helmet = 0 in classes
    has_no_helmet = 1 in classes
    has_vest = 2 in classes
    has_no_vest = 3 in classes
    has_person = 4 in classes

    violations = []
    compliant = []

    if has_no_helmet:
        violations.append("missing safety helmet")
    elif has_helmet:
        compliant.append("safety helmet")

    if has_no_vest:
        violations.append("missing safety vest")
    elif has_vest:
        compliant.append("reflective safety vest")

    if violations and compliant:
        return f"a construction site with workers showing safety violation: {', '.join(violations)}, while wearing {', '.join(compliant)}"
    elif violations:
        return f"a construction site showing critical safety violations: {', '.join(violations)}"
    elif compliant:
        return f"a compliant construction site with workers properly equipped with {', '.join(compliant)}"
    elif has_person:
        return "a construction site with workers visible"
    else:
        return "a construction site scene with no active safety equipment detected"


def generate_crop_caption(crop_class_id):
    """
    Generate fine-grained caption for an individual worker crop:
    0: compliant (helmet + vest)
    1: no_helmet violation
    2: no_vest violation
    3: both_missing violation
    """
    if crop_class_id == 0:
        return "a construction worker properly wearing both a protective helmet and a reflective safety vest"
    elif crop_class_id == 1:
        return "a construction worker with a safety violation: head exposed without a safety helmet"
    elif crop_class_id == 2:
        return "a construction worker with a safety violation: upper body without a high-visibility safety vest"
    elif crop_class_id == 3:
        return "a construction worker with severe safety violations: wearing neither a helmet nor a safety vest"
    else:
        return "a construction worker on a job site"
