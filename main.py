from os import listdir
from os.path import isfile, join
from Picture import Picture
from efficiencyevaluation import evaluation


def get_pictures():
    healthy = 'pictures\\healthy'
    healthy_fovmask = 'pictures\\healthy_fovmask'
    healthy_manual = 'pictures\\healthy_manualsegm'
    healthy_paths = [join(healthy, f) for f in listdir(healthy)]
    healthy_fovmask_paths = [join(healthy_fovmask, f) for f in listdir(healthy_fovmask)]
    healthy_manual_paths = [join(healthy_manual, f) for f in listdir(healthy_manual)]
    return [Picture(healthy_paths[i], healthy_fovmask_paths[i], healthy_manual_paths[i])
            for i in range(len(healthy_paths))]


pictures = get_pictures()
pictures[0].process_image()
pictures[0].show_image()
pictures[0].show_target()

evaluation(pictures[0].image, pictures[0].target)

