# Automatic Image Segmentation with SAM and Saliency Maps

Segment the object an image classifier "sees", without any prompt or label:
SAM's automatic mask generator segments **everything** in the image, and a
saliency map of the classifier decides **which segments belong to the object**.

![Image, saliency map, all SAM masks, selection](docs/images/falcon-heavy-boosters-overview.png)

ResNet50 classifies the photo of two landing Falcon Heavy boosters as
*missile*; LayerCAM highlights the boosters; SAM finds 29 masks; the pipeline
selects the clusters that carry the relevance.

## How it works

1. **Classify and explain.** ResNet50 predicts the ImageNet class on the image
   scaled down to at most 640 px (on a full-size photo it would see texture,
   not the object); an XAI method (LayerCAM, GradCAM or LRP) produces a saliency
   map for that class, scaled to the image and to [0, 1].
2. **Segment everything.** `SamAutomaticMaskGenerator` samples a 32×32 point
   grid (plus crops) and returns every mask it finds.
3. **Rank the masks.** Each mask is scored by a *weighted IoU*: its soft IoU with
   the saliency map times the mean relevance inside the mask. Large masks with
   little relevance (sky, ground) rank low.

   ![SAM masks ranked by weighted IoU](docs/images/falcon-heavy-boosters-ranked.png)

4. **Cluster.** Going down the ranking, each mask absorbs the later masks it
   overlaps by more than 20 %, so the parts of one object (booster, landing
   legs, plume) end up in one cluster.

   ![Clusters, selected ones in red, too large ones in gray](docs/images/falcon-heavy-boosters-clusters.png)

5. **Select.** Clusters covering more than 40 % of the image (`max_area`) are
   background and left out; with a weak heatmap around the object their size
   alone can earn them a high score (the sky above). The remaining clusters are
   kept up to the elbow of their score curve; their union is the segmentation.

   ![Cluster scores and elbow](docs/images/falcon-heavy-boosters-scores.png)

## More examples

![Corgi: image, saliency map, all SAM masks, selection](docs/images/corgi-overview.png)

ResNet50 classifies the corgi running through the snow as *Pembroke*; LayerCAM
highlights its head and body; of SAM's 27 masks, the pipeline selects exactly
the dog, without the snow around it.

![Seal: image, saliency map, all SAM masks, selection](docs/images/seal-overview.png)

The harbor seal on the beach is a *sea lion* to ResNet50 (ImageNet has no
harbor seal class); LayerCAM highlights its head, and of SAM's 99 masks the
pipeline selects the whole seal as one cluster, without the sand.

---

Image sources and licenses: [examples/README.md](examples/README.md).

A group project of the Data Science project course at TU Berlin (DAI-Labor),
led by [karray](https://github.com/karray).
