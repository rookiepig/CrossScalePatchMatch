Cross-Scale Cost Aggregation for Stereo Matching (CVPR 2014 + PatchMatch stereo submit to TPAMI)
================

## Compilation
### Windows
The code is a Visual Studio 2010 project on Windows x64 platform. To build the project, you need to configure [OpenCV](http://opencv.org/) (version 2.4.6, however, other versions are acceptable by modifying [commfunc.h](/CSPM/commfunc.h)) on your own PC. Besides, to parse command paramters, I adopted the [gflags](https://code.google.com/p/gflags/).
### Other Platforms
The code requires no platform-dependent libraries. Thus, it is easy to compile it on other platforms with OpenCV.

## Usage
Since I adopted gflags, the parameters are totally changed. The following example can demonstrate how to run the original [PatchMatch](#PM) stereo algorithm:
```
--l_img_file="cones_l.png" --r_img_file="cones_r.png" --l_dis_file="cones_ld.png" --r_dis_file="cones_rd.png" --max_dis=60 --dis_scale=4 --cc_name="GRD" --use_cs=false --use_pp=false --reg_lambda=0.0
```

**Hint**: the bool flag *use_cs* indicates the usage of cross-scale cost aggregation; the bool flag *use_pp* indicates the usage of post-processing.

## Citation
Citation is very important for researchers. If you find this code useful, please cite:
```
@inproceedings{CrossScaleStereo,
        author    = {Kang Zhang and Yuqiang Fang  and Dongbo Min and Lifeng Sun and Shiqiang Yang  and Shuicheng Yan and Qi Tian},
        title     = {Cross-Scale Cost Aggregation for Stereo Matching},
        booktitle = {CVPR},
        year     = {2014}
}
```
The PatchMatch stereo algorithm comes from the following paper:
<a name="PM">[PM]</a>: M. Bleyer, C. Rhemann, and C. Rother, “PatchMatch stereo
- stereo matching with slanted support windows,” in
BMVC, 2011.
