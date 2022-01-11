from dataloaders.utils import decode_seg_map_sequence

segmentation_classes = [
    'background','aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow','diningtable','dog','horse',
    'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
]

def labels():
  l = {}
  for i, label in enumerate(segmentation_classes):
    l[i] = label
  return l

def wb_mask(bg_img, pred_mask, true_mask):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})


def colorize(value, vmin=None, vmax=None, cmap=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=False) # (nxmx4)
    return value