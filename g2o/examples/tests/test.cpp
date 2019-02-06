// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <Eigen/StdVector>

#include <unordered_set>

#include <iostream>
#include <stdint.h>

#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

using namespace Eigen;
using namespace std;
using namespace g2o;


class Sample
{
  public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr)
{
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma)
{
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to)
{
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform()
{
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma)
{
  return gauss_rand(0., sigma);
}

g2o::VertexSCam* setUpSCam()
{
  Vector2d focal_length(500,500); // pixels
  Vector2d principal_point(320,240); // 640x480 image
  double baseline = 0.075;      // 7.5 cm baseline
  g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],
                           principal_point[0],principal_point[1],
                           baseline);
  Vector3d trans(0,0,0);
  Eigen:: Quaterniond q;
  q.setIdentity();
  Eigen::Isometry3d pose;
  pose = q;
  pose.translation() = trans;

  g2o::VertexSCam * v_se3 = new g2o::VertexSCam();

  v_se3->setId(0);
  v_se3->setEstimate(pose);
  v_se3->setAll();            // set aux transforms
  return v_se3;
}

g2o::VertexCamRig* setUpCamRig()
{
  Vector2d focal_length(500,500); // pixels
  Vector2d principal_point(320,240); // 640x480 image
  double baseline = 0.075;

  // set up camera params
  for (int i = 0; i<3;++i){
    g2o::VertexCamRig::setKcam(i, focal_length[0],focal_length[1],
	principal_point[0],principal_point[1]); 
    Eigen::Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d pose;
    pose = q;
    Vector3d trans(i*baseline,0,0);
    pose.translation()=trans;
    g2o::VertexCamRig::setCalibration(i, pose);
  } 
  g2o::VertexCamRig * camRig = new g2o::VertexCamRig();

  Eigen::Quaterniond q;
  q.setIdentity();
  Eigen::Isometry3d pose;
  pose = q;
  Vector3d trans(0,0,0);
  camRig->setId(0);
  camRig->setEstimate(pose);
  return camRig;
}

void _assert(bool b, std::string msg="")
{ 
  if (!b){
    std::cout << "failed tests " << msg << std::endl;
    exit(-1);
  }
}

void _assert_small(double x, std::string msg="")
{
  _assert(std::abs(x)<1e-5, msg);
}
//sanity. make sure point is projected to the right quadrant of the image (more or less that center pixel) for sensor 0 and 1
void testProjectionCamRig()
{
  g2o::VertexCamRig * camRig = setUpCamRig();
  
  //test sensor 0
  {
    Vector3d point(0,0,1);
    Vector2d z;
    camRig->mapPoint(z,0, point);
    _assert(z[0]==320 && z[1]==240, "sensor 0");

    std::cout << z.transpose() << std::endl;
    point(1) = 0.2;
    camRig->mapPoint(z,0, point);
    std::cout << z.transpose() << std::endl;
    _assert(z[0]==320 && z[1]>240, "sensor 0");
  }
  //test sensor 1
  {
    Vector3d point(0,0,1);
    Vector2d z;
    camRig->mapPoint(z,1, point);
    _assert(z[0]<320 && z[1]==240, "sensor 1");

    std::cout << z.transpose() << std::endl;
    point(1) = 0.2;
    camRig->mapPoint(z,1, point);
    std::cout << z.transpose() << std::endl;
    _assert(z[0]<320 && z[1]>240, "sensor 1");
  }
}

void testProjectionBaseline()
{
  g2o::VertexCamRig * camRig = setUpCamRig();
  
  Vector3d point(-0.4,0.2,1);
  Vector2d z0, z1;;
  camRig->mapPoint(z0,0, point);
  camRig->mapPoint(z1,1, point);
  _assert_small(z0[1]-z1[1],"y should be equal");

  g2o::VertexSCam* scam = setUpSCam();
  Vector3D z_stereo;
  scam->mapPoint(z_stereo, point);
  std::cout << "stereo projection " << z_stereo.transpose()<< std::endl;
  std::cout << "rig 0  projection " << z0.transpose()<< std::endl;
  std::cout << "rig 1  projection " << z1.transpose()<< std::endl;
  _assert_small(z_stereo[0]-z0[0], "projection on sensor 0 doesn't match");
  _assert_small(z_stereo[1]-z0[1], "projection on sensor 0 doesn't match");
  _assert_small(z_stereo[2]-z1[0], "baseline doesn't match");
}

int main(int argc, const char* argv[])
{
  testProjectionCamRig();
  testProjectionBaseline();
  exit(-1);
  if (argc<2)
  {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] [STRUCTURE_ONLY] [DENSE]" << endl;
    cout << endl;
    cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
    cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)" << endl;
    cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)" << endl;
    cout << "STRUCTURE_ONLY: performe structure-only BA to get better point initializations (0 or 1; default: 0==false)" << endl;
    cout << "DENSE: Use dense solver (0 or 1; default: 0==false)" << endl;
    cout << endl;
    cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to 1==true." << endl;
    cout << endl;
    exit(0);
  }

  double PIXEL_NOISE = atof(argv[1]);

  double OUTLIER_RATIO = 0.0;

  if (argc>2)
  {
    OUTLIER_RATIO = atof(argv[2]);
  }

  bool ROBUST_KERNEL = false;
  if (argc>3)
  {
    ROBUST_KERNEL = atoi(argv[3]) != 0;
  }
  bool STRUCTURE_ONLY = false;
  if (argc>4)
  {
    STRUCTURE_ONLY = atoi(argv[4]) != 0;
  }

  bool DENSE = false;
  if (argc>5)
  {
    DENSE = atoi(argv[5]) != 0;
  }

  cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << endl;
  cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  endl;
  cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
  cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
  cout << "DENSE: "<<  DENSE << endl;



  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
  if (DENSE)
  {
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    cerr << "Using DENSE" << endl;
  }
  else
  {
#ifdef G2O_HAVE_CHOLMOD
    cerr << "Using CHOLMOD" << endl;
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
#elif defined G2O_HAVE_CSPARSE
    linearSolver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
    cerr << "Using CSPARSE" << endl;
#else
#error neither CSparse nor Cholmod are available
#endif
  }

  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

  optimizer.setAlgorithm(solver);

  // set up 500 points
  vector<Vector3d> true_points;
  for (size_t i=0;i<500; ++i)
  {
    true_points.push_back(Vector3d((Sample::uniform()-0.5)*3,
	  Sample::uniform()-0.5,
	  Sample::uniform()+3));
  }


  Vector2d focal_length(500,500); // pixels
  Vector2d principal_point(320,240); // 640x480 image


  vector<Eigen::Isometry3d,
    aligned_allocator<Eigen::Isometry3d> > true_poses;

  // set up camera params
  for (int i = 0; i<3;++i){
    g2o::VertexCamRig::setKcam(i, focal_length[0],focal_length[1],
	principal_point[0],principal_point[1]); 
    Eigen:: Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d pose;
    pose = q;
    Vector3d trans((i-1)*0.075,0,0);
    pose.translation() = trans;
    g2o::VertexCamRig::setCalibration(i, pose);
  } 
  // set up 5 vertices, first 2 fixed
  int vertex_id = 0;
  for (size_t i=0; i<5; ++i)
  {
    Vector3d trans(i*0.04-1.,0,0);

    Eigen:: Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d pose;
    pose = q;
    pose.translation() = trans;


    g2o::VertexCamRig * v_se3
      = new g2o::VertexCamRig();

    v_se3->setId(vertex_id);
    v_se3->setEstimate(pose);
    //v_se3->setAll();            // set aux transforms

    if (i<2)
      v_se3->setFixed(true);

    optimizer.addVertex(v_se3);
    true_poses.push_back(pose);
    vertex_id++;
  }

  int point_id=vertex_id;
  int point_num = 0;
  double sum_diff2 = 0;

  cout << endl;
  unordered_map<int,int> pointid_2_trueid;
  unordered_set<int> inliers;

  // add point projections to this vertex
  for (size_t i=0; i<true_points.size(); ++i)
  {
    g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
    v_p->setId(point_id);
    v_p->setMarginalized(true);
    v_p->setEstimate(true_points.at(i) + Vector3d(Sample::gaussian(1), 
	  Sample::gaussian(1), Sample::gaussian(1)));

    int num_obs = 0;

    for (size_t j=0; j<true_poses.size(); ++j)
    {
      Vector2d z;
      auto camrig = dynamic_cast<g2o::VertexCamRig*> (optimizer.vertices().find(j)->second);
      for (int frame_id=0; frame_id<3 && num_obs<2; frame_id++){
	camrig->mapPoint(z,frame_id, true_points.at(i)); 
	if (0<=z[0] && z[0]<640 && 0<=z[1] && z[1]<480) {
	  ++num_obs;
	}
      }
    }

    if (num_obs>=2)
    {
      optimizer.addVertex(v_p);

      bool inlier = true;
      for (size_t j=0; j<true_poses.size(); ++j)
      {
	Vector2d z;
	auto camrig = dynamic_cast<g2o::VertexCamRig*> (optimizer.vertices().find(j)->second);
	for (int frame_id=0; frame_id<3 ; frame_id++){
	  camrig->mapPoint(z,frame_id, true_points.at(i)); 

	  if (0<=z[0] && z[0]<640 && 0<=z[1] && z[1]<480) {
	    double sam = Sample::uniform();
	    if (sam<OUTLIER_RATIO)
	    {
	      z = Vector2d(Sample::uniform(64,640), Sample::uniform(0,480));
	      inlier= false;
	    } 
	    z += Vector2d(Sample::gaussian(PIXEL_NOISE), 
		Sample::gaussian(PIXEL_NOISE)); 
	    g2o::Edge_XYZ_VRIG * e = new g2o::Edge_XYZ_VRIG();

	    e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);
	    e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
	      (optimizer.vertices().find(j)->second);
	    Vector3D mes;
	    mes.head<2>()=z;
	    mes(2)=frame_id;
	    e->setMeasurement(mes);
	    //e->inverseMeasurement() = -z;
	    e->information() = Matrix3d::Identity();
	    if (ROBUST_KERNEL) {
	      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	      e->setRobustKernel(rk);
	    } 
	    optimizer.addEdge(e);
	  }
	}
      }

      if (inlier)
      {
	inliers.insert(point_id);
	Vector3D diff = v_p->estimate() - true_points[i];

	sum_diff2 += diff.dot(diff);
      }
      // else
      //   cout << "Point: " << point_id <<  "has at least one spurious observation" <<endl;

      pointid_2_trueid.insert(make_pair(point_id,i));

      ++point_id;
      ++point_num;
    }

  }

  cout << endl;
  optimizer.initializeOptimization();

  optimizer.setVerbose(true);

  if (STRUCTURE_ONLY)
  {
    cout << "Performing structure-only BA:"   << endl;
    g2o::StructureOnlySolver<3> structure_only_ba;
    g2o::OptimizableGraph::VertexContainer points;
    for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() == 3)
	points.push_back(v);
    }

    structure_only_ba.calc(points, 10);
  }

  cout << endl;
  cout << "Performing full BA:" << endl;
  optimizer.optimize(10);

  cout << endl;
  cout << "Point error before optimisation (inliers only): " 
    << sqrt(sum_diff2/inliers.size()) << endl;

  point_num = 0;
  sum_diff2 = 0;

  for (unordered_map<int,int>::iterator it=pointid_2_trueid.begin();
      it!=pointid_2_trueid.end(); ++it)
  {
    g2o::HyperGraph::VertexIDMap::iterator v_it
      = optimizer.vertices().find(it->first);

    if (v_it==optimizer.vertices().end())
    {
      cerr << "Vertex " << it->first << " not in graph!" << endl;
      exit(-1);
    }

    g2o::VertexSBAPointXYZ * v_p
      = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);

    if (v_p==0)
    {
      cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
      exit(-1);
    }

    Vector3D diff = v_p->estimate()-true_points[it->second];
    if (inliers.find(it->first)==inliers.end())
      continue;

    sum_diff2 += diff.dot(diff);
    ++point_num;
  }

  cout << "Point error after optimisation (inliers only): " << sqrt(sum_diff2/inliers.size()) << endl;
  cout << endl;
  sum_diff2=0;
  Vector3D sum_diff(0);
  for (int i=0; i<5; ++i)
  {
    auto rig = dynamic_cast<g2o::VertexCamRig*>(optimizer.vertices().find(i)->second);
    assert (rig != 0);
    auto pose = rig->estimate();
    Vector3D diff = pose.translation() - true_poses[i].translation();
    sum_diff+=diff.cwiseAbs();
    sum_diff2 += diff.dot(diff);
  }
  cout << "Pose loc error after optimisation : " << sqrt(sum_diff2/3) << endl;
  cout << "Pose loc error after optimisation : " << sum_diff.transpose()/3<< endl;
}
