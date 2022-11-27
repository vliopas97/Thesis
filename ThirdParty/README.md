# Third Party Software
## Basic Folder Structure
Within Third Party folder is contained all the third party software used by the Unreal Project.
A very basic preview of the folder structure is:

<pre>
├── boost
├── <b>GM</b>
├── opencv
├── python
├── <b>scripts
│   ├── deep_sort</b>
└── zed
</pre>

Highlighted **folders** contain custom-made vendor code (can be edited) while the rest are third party libraries.

## Notes

The object-tracking algorithm exists in its final form in [scripts/ssd_final.py](scripts/ssd_final.py) which is the file linked to the Unreal Project.
Files like [webcam.py](scripts/webcam.py) and [ssd.py](scripts/ssd.py) were used for testing and the rest are for training the model from scratch.

## References
- The object detection implementation in [scripts](scripts) is based on the [Single Shot Multibox Detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2).

- The object tracking implementation in [scripts/deep_sort](scripts/deep_sort) is based on the [DeepSort](https://github.com/nwojke/deep_sort.git) algorithm and repo.
