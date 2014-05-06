///////////////////////////////////////////////////////
// File: main.cpp
// Desc: PatchMatch Stereo Matching
//       Usage: [l_img_file] [r_img_file] 
//              [l_dis_file] [r_dis_file] 
//              [max_dis] [dis_scale] 
// Author: rookiepig
// Date: 2014/04/04
//
///////////////////////////////////////////////////////
#include"commfunc.h"
#include"cs_patchmatch.h"
#include"plane_cost\grd_pc.h"
#include"plane_cost\cspc.h"

//
// gflags commind line variables
//
DEFINE_string(l_img_file, "l_img.png", "input left image file name");
DEFINE_string(r_img_file, "r_img.png", "input right image file name");
DEFINE_string(l_dis_file, "l_dis.png", 
  "output left disparity file name");
DEFINE_string(r_dis_file, "r_dis.png", 
  "output right disparity file name");
DEFINE_int32(max_dis, 0, "max allowed disparity range");
DEFINE_int32(dis_scale, 0, "disparity re-scaling factor");


int main(int argc, char** argv) {
  cout << "PatchMatch Stereo Matching" << endl;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //
  // Load left right image
  //
  cout << "--------------------------------------------------------\n";
  cout << "Load Image: " << FLAGS_l_img_file << " " << FLAGS_r_img_file 
    << "\n";
  cout << "--------------------------------------------------------\n";
  Mat l_img = imread(FLAGS_l_img_file, CV_LOAD_IMAGE_COLOR);
  Mat r_img = imread(FLAGS_r_img_file, CV_LOAD_IMAGE_COLOR);
  if (!l_img.data || !r_img.data) {
    cout << "Error: can not open image\n";
    cout << "\nPress any key to continue...\n";
    cin.get();
    return EXIT_FAILURE;
  }
  // set image format
  //cvtColor(l_img, l_img, CV_BGR2RGB);
  //cvtColor(r_img, r_img, CV_BGR2RGB);
  //l_img.convertTo(l_img, CV_64F, 1 / 255.0f);
  //r_img.convertTo(r_img, CV_64F, 1 / 255.0f);

  // init time
  double duration = static_cast<double>(getTickCount());

  //
  // Patch Match
  //
  const int max_iter = 3;
  CSPatchMatch* patch_match = new CSPatchMatch(l_img, r_img,
    FLAGS_max_dis, FLAGS_dis_scale);
  const int wnd_size = 35;

  //const double alpha = 0.1;
  //const double tau_color = 10.0 / 255.0;
  //const double tau_grd   = 2.0  / 255.0;
  //const double gamma     = 10.0;

  GrdPC* plane_cost = new GrdPC(l_img, r_img, FLAGS_max_dis, wnd_size);
  // CSPC* plane_cost = new CSPC(l_img, r_img, FLAGS_max_dis, wnd_size, 5);
  // , alpha, tau_color, tau_grd, gamma);
  
  patch_match->PatchMatch(max_iter, plane_cost);

  // record time
  duration = static_cast<double>(getTickCount()) - duration;
  duration /= cv::getTickFrequency(); // the elapsed time in sec
  cout << "--------------------------------------------------------\n";
  cout << "Total Time: " << duration << endl;
  cout << "--------------------------------------------------------\n";

  //
  // Save Output
  //
  Mat l_dis = patch_match->dis(kLeft);
  Mat r_dis = patch_match->dis(kRight);
  imwrite(FLAGS_l_dis_file, l_dis);
  imwrite(FLAGS_r_dis_file, r_dis);

  //cout << "Press any key to continue..." << endl;
  //getchar();
  return EXIT_SUCCESS;
}